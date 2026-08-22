"""Limpeza, chunking e montagem dos metadados de cada trecho.

## Por que o chunk é o par Q&A, e não 600 caracteres de prosa

O guia de transição sugere `RecursiveCharacterTextSplitter` com 600 chars e
overlap 80. Esse é um bom default para prosa corrida em PDF, mas o nosso corpus
não é prosa: são 15.272 pares pergunta/resposta atômicos e autocontidos.

Aplicar 600/80 aqui tem dois defeitos medidos:

- **Órfãos de contexto.** A partir do segundo pedaço, o trecho perde a pergunta
  que lhe dá sentido. Um chunk como "...deve ser administrado por via oral" é
  inútil para o retriever e péssimo como evidência citada.
- **Fragmentação desnecessária.** 600/80 produz 39.985 chunks; tratar o par Q&A
  como unidade e só dividir o que excede o limite produz 23.763 (-41%), com
  62,9% dos pares preservados inteiros.

O limite de 1.200 caracteres não foi escolhido por gosto. Medindo com o
tokenizer real do modelo (`artifacts/.../tokenizer.json`), o corpus tem 4,78
caracteres por token — logo 1.200 chars ≈ 251 tokens, e `k=4` monta um contexto
de ~1.005 tokens, dentro da faixa de 900–1.200 recomendada pelo orçamento da
LLM (2.048 de janela − 250 de geração − ~90 de template/pergunta).

Quando um documento precisa ser dividido, a pergunta é repetida no cabeçalho de
cada parte, de modo que todo chunk continue autocontido e citável.
"""

from __future__ import annotations

from dataclasses import dataclass

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from .loaders import DocumentoBruto
from .schemas import (
    FONTE_PRONTUARIO,
    MetadadosChunk,
    checksum,
    id_chunk,
    id_documento,
    resolver_fonte,
    validar_metadados,
)

# Hipóteses de configuração, não constantes definitivas: a etapa de avaliação
# compara 1.200/k=4 com 1.500/k=3 e registra o resultado no README.
LIMITE_CHARS_CHUNK = 1200
OVERLAP_CHARS = 120


@dataclass(frozen=True)
class VersaoCorpus:
    """Identifica a versão do corpus que originou os chunks.

    Vai para os metadados de todo chunk (`version`/`updated_at`) para que uma
    resposta citada possa ser rastreada até a extração exata que a gerou.
    """

    version: str
    updated_at: str  # ISO-8601, data da extração


def _cabecalho(doc: DocumentoBruto) -> str:
    """Prefixo repetido em toda parte, para que o chunk seja autocontido.

    O formato muda com o tipo de documento porque o cabeçalho é lido pelo
    modelo de embedding *e* acaba no prompt da LLM. Rotular um prontuário como
    "Question:/Answer:" ensinaria os dois a tratá-lo como FAQ — e o risco
    concreto é a LLM responder sobre o paciente errado por analogia com uma
    pergunta genérica parecida.
    """
    if doc.document_type == "prontuario_sintetico":
        return f"Prontuário {doc.titulo}\n"
    return f"Question: {doc.pergunta}\nAnswer: "


def _montar_metadados(
    doc: DocumentoBruto,
    document_id: str,
    ordem: int,
    total_partes: int,
    versao: VersaoCorpus,
) -> MetadadosChunk:
    e_prontuario = doc.document_type == "prontuario_sintetico"
    rotulo = "registro" if e_prontuario else "resposta"
    secao = rotulo if total_partes == 1 else f"{rotulo} parte {ordem + 1}/{total_partes}"

    if e_prontuario:
        fonte = FONTE_PRONTUARIO
        # A URL carrega o próprio identificador do paciente: a citação vira
        # `prontuario-sintetico://PAC-0007`, que é rastreável até a linha exata
        # do JSONL sem fingir ser um endereço público.
        url = f"{fonte.url}{doc.patient_id}"
    else:
        fonte = resolver_fonte(doc.group_name)
        url = fonte.url

    return MetadadosChunk(
        chunk_id=id_chunk(document_id, ordem),
        document_id=document_id,
        # `source` aponta para a instituição de origem, não para o JSONL local:
        # é o que sustenta a citação exigida pela camada de explainability.
        source=url,
        title=doc.pergunta,
        document_type=doc.document_type,
        page_or_section=secao,
        language=doc.language,
        version=versao.version,
        updated_at=versao.updated_at,
        is_synthetic=doc.is_synthetic,
        checksum=checksum(doc.resposta),
        source_group=doc.group_name or "desconhecido",
        institution=fonte.instituicao,
        patient_id=doc.patient_id,
    )


def dividir_documento(
    doc: DocumentoBruto,
    versao: VersaoCorpus,
    limite_chars: int = LIMITE_CHARS_CHUNK,
    overlap_chars: int = OVERLAP_CHARS,
) -> list[Document]:
    """Converte um par Q&A em um ou mais `Document` prontos para indexação."""
    document_id = id_documento(doc.pergunta, doc.resposta)
    cabecalho = _cabecalho(doc)

    # O cabeçalho é repetido em toda parte, então o espaço disponível para a
    # resposta é o limite menos o cabeçalho. Perguntas muito longas poderiam
    # zerar esse espaço; o piso de 300 chars evita gerar centenas de partes
    # minúsculas para um caso patológico.
    espaco_resposta = max(limite_chars - len(cabecalho), 300)

    if len(doc.resposta) <= espaco_resposta:
        partes = [doc.resposta]
    else:
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=espaco_resposta,
            chunk_overlap=min(overlap_chars, espaco_resposta // 4),
            separators=["\n\n", "\n", ". ", "? ", "! ", "; ", ", ", " ", ""],
            length_function=len,
        )
        partes = splitter.split_text(doc.resposta)

    documentos: list[Document] = []
    for ordem, parte in enumerate(partes):
        parte = parte.strip()
        if not parte:
            continue
        metadados = _montar_metadados(doc, document_id, ordem, len(partes), versao)
        validar_metadados(metadados.to_dict())
        documentos.append(
            Document(
                page_content=cabecalho + parte,
                metadata=metadados.to_dict(),
            )
        )
    return documentos


def dividir_todos(
    docs: list[DocumentoBruto],
    versao: VersaoCorpus,
    limite_chars: int = LIMITE_CHARS_CHUNK,
    overlap_chars: int = OVERLAP_CHARS,
) -> list[Document]:
    chunks: list[Document] = []
    for doc in docs:
        chunks.extend(dividir_documento(doc, versao, limite_chars, overlap_chars))
    return chunks
