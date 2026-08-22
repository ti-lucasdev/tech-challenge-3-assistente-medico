"""Leitura das fontes brutas do corpus e validação mínima.

Duas fontes hoje:

- `data/medquad_rag_pool.jsonl` — as linhas do MedQuAD que o Integrante 1 *não*
  usou no fine-tuning. Usar linhas de treino aqui inflaria as métricas de
  recuperação por memorização — ver `README.md` §4.4.
- `data/prontuarios_sinteticos.jsonl` — 30 registros clínicos fabricados para
  este projeto. O enunciado pede consultar "base de dados estruturadas (como
  prontuários e registros)" e contextualizar a resposta com dados do paciente;
  o MedQuAD sozinho é FAQ pública e não tem paciente nenhum.

O módulo devolve `DocumentoBruto`, não `Document` do LangChain: o chunking e a
montagem de metadados são responsabilidade do `preprocess.py`. Aqui só se lê,
valida e reporta o que foi descartado.

## Sobre a anonimização dos prontuários

O enunciado exige "anonimização" dos dados. A anonimização mais forte possível é
**não gerar o identificador**: os registros não têm nome, CPF, telefone,
endereço nem data de nascimento — só `PAC-NNNN`, idade e sexo, que é o mínimo
clinicamente necessário. Não há dado a remover porque não há dado. Isso também
mantém `tests/test_privacidade.py` verde por construção, e não por allowlist.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path

from .schemas import NAO_APLICAVEL, Idioma, TipoDocumento, id_documento

RAIZ_PROJETO = Path(__file__).resolve().parent.parent.parent
CAMINHO_POOL_MEDQUAD = RAIZ_PROJETO / "data" / "medquad_rag_pool.jsonl"
CAMINHO_PRONTUARIOS = RAIZ_PROJETO / "data" / "prontuarios_sinteticos.jsonl"


@dataclass(frozen=True)
class DocumentoBruto:
    """Um documento lido da fonte, antes de qualquer chunking.

    Os campos `pergunta`/`resposta` nasceram do MedQuAD, que é Q&A puro. Para um
    prontuário eles carregam título e corpo do registro — daí os aliases
    `titulo`/`corpo`, que deixam o código de prontuário legível sem duplicar a
    estrutura nem forçar um `Union` no pipeline inteiro.

    Os campos com default preservam exatamente o comportamento anterior do
    MedQuAD, então nenhum chamador existente precisou mudar.
    """

    pergunta: str
    resposta: str
    group_name: str
    origem: str  # caminho do arquivo de onde veio, para rastreabilidade
    document_type: TipoDocumento = "faq_medica"
    language: Idioma = "en"
    is_synthetic: bool = False
    patient_id: str = NAO_APLICAVEL

    @property
    def titulo(self) -> str:
        return self.pergunta

    @property
    def corpo(self) -> str:
        return self.resposta


@dataclass
class RelatorioCarga:
    """O que entrou e o que foi descartado, e por quê.

    Existe porque o critério de aceite do guia exige que arquivos vazios ou
    inválidos "falhem com mensagem clara" — e porque descartar documentos em
    silêncio é a forma mais fácil de um corpus ficar menor do que se pensa.
    """

    lidos: int = 0
    aceitos: int = 0
    descartes: dict[str, int] = field(default_factory=dict)

    def descartar(self, motivo: str) -> None:
        self.descartes[motivo] = self.descartes.get(motivo, 0) + 1

    def resumo(self) -> str:
        if not self.descartes:
            return f"{self.aceitos}/{self.lidos} documentos aceitos (nenhum descarte)"
        detalhe = ", ".join(f"{m}: {n}" for m, n in sorted(self.descartes.items()))
        return f"{self.aceitos}/{self.lidos} documentos aceitos (descartes -> {detalhe})"


# Respostas muito curtas não são evidência utilizável: no pool há registros com
# 6 caracteres. Indexá-los só polui o topo-k com ruído de alta similaridade.
MIN_CHARS_RESPOSTA = 40
MIN_CHARS_PERGUNTA = 10


def carregar_medquad(
    caminho: Path | str = CAMINHO_POOL_MEDQUAD,
    limite: int | None = None,
) -> tuple[list[DocumentoBruto], RelatorioCarga]:
    """Lê o pool do MedQuAD em JSONL.

    `limite` corta a leitura nas primeiras N linhas válidas — usado na fatia
    fina de desenvolvimento, para iterar sobre 500 documentos em vez de 15 mil.
    """
    caminho = Path(caminho)
    if not caminho.exists():
        raise FileNotFoundError(
            f"Corpus não encontrado: {caminho}. "
            "Rode `python scripts/export_medquad_dataset.py` para regenerá-lo."
        )

    documentos: list[DocumentoBruto] = []
    relatorio = RelatorioCarga()
    # O pool contém 47 linhas que repetem, byte a byte, um par Q&A já visto
    # (sobretudo registros do NIDDK, repetidos de 2 a 4 vezes). Indexá-las
    # geraria chunks idênticos disputando o mesmo top-k, empurrando evidência
    # distinta para fora. Verificado: são duplicatas reais, não colisão de hash.
    vistos: set[str] = set()

    for bruto in _ler_jsonl(caminho, relatorio):
        pergunta = (bruto.get("query") or "").strip()
        resposta = (bruto.get("answers") or "").strip()

        if not pergunta or len(pergunta) < MIN_CHARS_PERGUNTA:
            relatorio.descartar("pergunta ausente ou curta demais")
            continue
        if not resposta:
            relatorio.descartar("resposta vazia")
            continue
        if len(resposta) < MIN_CHARS_RESPOSTA:
            relatorio.descartar(f"resposta com menos de {MIN_CHARS_RESPOSTA} chars")
            continue

        identidade = id_documento(pergunta, resposta)
        if identidade in vistos:
            relatorio.descartar("duplicata exata (mesma pergunta e mesma resposta)")
            continue
        vistos.add(identidade)

        documentos.append(
            DocumentoBruto(
                pergunta=pergunta,
                resposta=resposta,
                group_name=(bruto.get("group_name") or "").strip(),
                origem=_caminho_legivel(caminho),
            )
        )
        relatorio.aceitos += 1

        if limite is not None and len(documentos) >= limite:
            break

    if not documentos:
        raise ValueError(f"Nenhum documento válido em {caminho}: {relatorio.resumo()}")

    return documentos, relatorio


CAMPOS_PRONTUARIO = ("patient_id", "titulo", "texto", "especialidade", "atualizado_em")


def carregar_prontuarios(
    caminho: Path | str = CAMINHO_PRONTUARIOS,
    limite: int | None = None,
) -> tuple[list[DocumentoBruto], RelatorioCarga]:
    """Lê os prontuários sintéticos em JSONL.

    Diferente do MedQuAD, aqui todo campo é obrigatório e um registro
    malformado é erro, não descarte: o arquivo é nosso, tem 30 linhas e é
    revisado à mão. Descartar em silêncio um prontuário faria o assistente
    responder "não há informação sobre este paciente" quando na verdade houve
    falha de carga — o pior modo de erro possível nesta camada.
    """
    caminho = Path(caminho)
    if not caminho.exists():
        raise FileNotFoundError(
            f"Prontuários sintéticos não encontrados: {caminho}. "
            "O arquivo é versionado; confira se o checkout está completo."
        )

    documentos: list[DocumentoBruto] = []
    relatorio = RelatorioCarga()
    identificadores: set[str] = set()

    for numero, bruto in enumerate(_ler_jsonl(caminho, relatorio), start=1):
        faltando = [c for c in CAMPOS_PRONTUARIO if not str(bruto.get(c) or "").strip()]
        if faltando:
            raise ValueError(
                f"{caminho.name}, linha {numero}: campos ausentes ou vazios: "
                f"{', '.join(faltando)}"
            )

        patient_id = str(bruto["patient_id"]).strip()
        if patient_id in identificadores:
            raise ValueError(
                f"{caminho.name}, linha {numero}: patient_id '{patient_id}' repetido. "
                "Um paciente por registro — repetir o ID faria um filtro por "
                "paciente devolver dois prontuários distintos como se fossem um."
            )
        identificadores.add(patient_id)

        documentos.append(
            DocumentoBruto(
                pergunta=f"{patient_id} — {str(bruto['titulo']).strip()}",
                resposta=str(bruto["texto"]).strip(),
                group_name=str(bruto["especialidade"]).strip(),
                origem=_caminho_legivel(caminho),
                document_type="prontuario_sintetico",
                language="pt-BR",
                is_synthetic=True,
                patient_id=patient_id,
            )
        )
        relatorio.aceitos += 1

        if limite is not None and len(documentos) >= limite:
            break

    if not documentos:
        raise ValueError(f"Nenhum prontuário válido em {caminho}: {relatorio.resumo()}")

    return documentos, relatorio


def _caminho_legivel(caminho: Path) -> str:
    """Caminho relativo à raiz do projeto quando possível, absoluto caso contrário.

    Fontes fora da árvore do projeto (um diretório temporário de teste, um
    corpus montado em outro volume) não têm caminho relativo — `relative_to`
    levanta `ValueError` nesse caso.
    """
    try:
        return caminho.relative_to(RAIZ_PROJETO).as_posix()
    except ValueError:
        return caminho.as_posix()


def _ler_jsonl(caminho: Path, relatorio: RelatorioCarga) -> Iterator[dict]:
    with caminho.open(encoding="utf-8") as arquivo:
        for numero, linha in enumerate(arquivo, start=1):
            linha = linha.strip()
            if not linha:
                continue
            relatorio.lidos += 1
            try:
                yield json.loads(linha)
            except json.JSONDecodeError as erro:
                raise ValueError(
                    f"JSON inválido em {caminho}, linha {numero}: {erro}"
                ) from erro
