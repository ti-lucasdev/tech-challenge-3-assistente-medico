"""Indexação idempotente no Chroma e manifesto de proveniência.

## O que garante a idempotência

Os `chunk_id` derivam do conteúdo (ver `schemas.id_documento`), então reingerir
o mesmo corpus faz *upsert* sobre os mesmos IDs: a contagem não muda. Isso é
verificável e está coberto por teste.

## O que a idempotência NÃO resolve

Mudar a configuração (tamanho de chunk, modelo de embedding) gera IDs
diferentes para o mesmo conteúdo. Um `add` por cima da coleção antiga não
substituiria nada — apenas *somaria* uma segunda versão do corpus ao índice, e a
busca passaria a devolver duplicatas quase idênticas com proveniência
inconsistente. É a falha "Duplicação na base" listada no guia.

Por isso o manifesto guarda a assinatura da configuração e a ingestão se recusa
a escrever numa coleção construída com configuração diferente, a menos que se
peça a reconstrução explicitamente.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import date
from pathlib import Path

from langchain_chroma import Chroma
from langchain_core.documents import Document

from .embeddings import MODELO_PADRAO, EmbeddingsE5
from .loaders import (
    CAMINHO_PRONTUARIOS,
    RAIZ_PROJETO,
    RelatorioCarga,
    carregar_medquad,
    carregar_prontuarios,
)
from .preprocess import LIMITE_CHARS_CHUNK, OVERLAP_CHARS, VersaoCorpus, dividir_todos
from .schemas import checksum

DIRETORIO_VECTORSTORE = RAIZ_PROJETO / "vectorstore"
CAMINHO_MANIFESTO = RAIZ_PROJETO / "data" / "manifest.json"
COLECAO_PADRAO = "medical_kb_v1"

# O Chroma tem limite por chamada; 1.000 mantém o uso de memória estável e dá
# granularidade para relatar progresso numa ingestão de ~25 mil chunks.
LOTE_INSERCAO = 1000


@dataclass(frozen=True)
class ConfiguracaoIngestao:
    """Tudo que, se mudar, invalida um índice já construído."""

    modelo_embedding: str = MODELO_PADRAO
    limite_chars_chunk: int = LIMITE_CHARS_CHUNK
    overlap_chars: int = OVERLAP_CHARS
    colecao: str = COLECAO_PADRAO

    def assinatura(self) -> str:
        return checksum(json.dumps(asdict(self), sort_keys=True))[:16]


def abrir_store(
    config: ConfiguracaoIngestao,
    persist_directory: Path | str = DIRETORIO_VECTORSTORE,
    embeddings: EmbeddingsE5 | None = None,
) -> Chroma:
    """Abre (ou cria) a coleção persistente.

    O caminho é resolvido a partir da raiz do projeto, não do diretório atual:
    abrir a coleção de um `cwd` diferente é a causa clássica de "a coleção
    apareceu vazia" citada no guia.
    """
    persist_directory = Path(persist_directory)
    persist_directory.mkdir(parents=True, exist_ok=True)
    return Chroma(
        collection_name=config.colecao,
        embedding_function=embeddings or EmbeddingsE5(modelo=config.modelo_embedding),
        persist_directory=str(persist_directory),
        # O padrão do Chroma é distância L2. Com embeddings normalizados o
        # *ranking* de L2 e cosseno coincide, mas o *valor* do score só é
        # interpretável em cosseno — e precisamos interpretá-lo para calibrar
        # o limiar de abstenção ("não há evidência suficiente").
        collection_metadata={"hnsw:space": "cosine"},
    )


def ingerir(
    chunks: list[Document],
    config: ConfiguracaoIngestao,
    persist_directory: Path | str = DIRETORIO_VECTORSTORE,
    embeddings: EmbeddingsE5 | None = None,
    recriar: bool = False,
    verboso: bool = True,
) -> Chroma:
    """Indexa os chunks, fazendo upsert por `chunk_id`."""
    store = abrir_store(config, persist_directory, embeddings)

    existentes = store._collection.count()
    if recriar and existentes:
        if verboso:
            print(f"  recriando coleção '{config.colecao}' ({existentes} registros descartados)")
        store.reset_collection()
    elif existentes:
        _conferir_configuracao(config, existentes)

    ids = [c.metadata["chunk_id"] for c in chunks]
    if len(set(ids)) != len(ids):
        raise ValueError(
            f"{len(ids) - len(set(ids))} chunk_id duplicados entre os documentos a indexar; "
            "isso indicaria colisão de hash ou documento repetido na fonte"
        )

    for inicio in range(0, len(chunks), LOTE_INSERCAO):
        lote = chunks[inicio : inicio + LOTE_INSERCAO]
        store.add_documents(lote, ids=ids[inicio : inicio + LOTE_INSERCAO])
        if verboso:
            print(f"  {min(inicio + LOTE_INSERCAO, len(chunks)):>6}/{len(chunks)} chunks", end="\r")

    if verboso:
        print(f"  {len(chunks)}/{len(chunks)} chunks indexados      ")
    return store


def _conferir_configuracao(config: ConfiguracaoIngestao, existentes: int) -> None:
    """Impede somar um corpus reconfigurado por cima de um índice antigo."""
    if not CAMINHO_MANIFESTO.exists():
        return
    manifesto = json.loads(CAMINHO_MANIFESTO.read_text(encoding="utf-8"))
    anterior = manifesto.get("configuracao", {}).get("assinatura")
    if anterior and anterior != config.assinatura():
        raise RuntimeError(
            f"A coleção '{config.colecao}' tem {existentes} registros construídos com outra "
            f"configuração (assinatura {anterior}, atual {config.assinatura()}).\n"
            "Indexar por cima duplicaria o corpus com proveniência inconsistente.\n"
            "Use --recriar para reconstruir o índice do zero."
        )


@dataclass(frozen=True)
class FonteCorpus:
    """Uma origem de documentos que entrou no índice, para o manifesto."""

    nome: str
    caminho: Path
    relatorio: RelatorioCarga
    observacao: str


def _descrever_fonte(fonte: FonteCorpus) -> dict:
    conteudo = fonte.caminho.read_bytes()
    return {
        "arquivo": fonte.caminho.relative_to(RAIZ_PROJETO).as_posix(),
        "checksum_sha256": checksum(conteudo.decode("utf-8", errors="replace")),
        "bytes": len(conteudo),
        "linhas_lidas": fonte.relatorio.lidos,
        "documentos_aceitos": fonte.relatorio.aceitos,
        "descartes": fonte.relatorio.descartes,
        "observacao": fonte.observacao,
    }


def _somar_descartes(fontes: list[FonteCorpus]) -> dict[str, int]:
    total: dict[str, int] = {}
    for fonte in fontes:
        for motivo, quantidade in fonte.relatorio.descartes.items():
            total[motivo] = total.get(motivo, 0) + quantidade
    return total


def escrever_manifesto(
    config: ConfiguracaoIngestao,
    versao: VersaoCorpus,
    fontes: list[FonteCorpus],
    total_chunks: int,
    caminho: Path = CAMINHO_MANIFESTO,
) -> None:
    """Registra a proveniência do índice — é evidência acadêmica, não log.

    Guarda o checksum de cada arquivo-fonte para que se possa afirmar *qual*
    extração gerou o índice, e a assinatura da configuração para a checagem
    acima. São várias fontes desde que os prontuários sintéticos entraram: uma
    resposta pode citar uma FAQ pública do NIH ou um registro fabricado, e a
    auditoria precisa distinguir as duas coisas.
    """
    manifesto = {
        "gerado_em": date.today().isoformat(),
        "corpus": {
            "versao": versao.version,
            "atualizado_em": versao.updated_at,
            "dataset_upstream": "mukulb/clustered_MEDQUAD_dataset_with_groups",
            "medquad_original": "https://github.com/abachaa/MedQuAD",
            "fontes": {f.nome: _descrever_fonte(f) for f in fontes},
        },
        "carga": {
            "linhas_lidas": sum(f.relatorio.lidos for f in fontes),
            "documentos_aceitos": sum(f.relatorio.aceitos for f in fontes),
            # Somado, não sobrescrito: duas fontes podem descartar pelo mesmo
            # motivo, e um `dict` montado por compreensão perderia a contagem
            # da primeira em silêncio.
            "descartes": _somar_descartes(fontes),
        },
        "indice": {
            "chunks": total_chunks,
            "colecao": config.colecao,
        },
        "configuracao": {**asdict(config), "assinatura": config.assinatura()},
    }
    caminho.write_text(
        json.dumps(manifesto, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def construir_indice(
    config: ConfiguracaoIngestao | None = None,
    limite: int | None = None,
    persist_directory: Path | str = DIRETORIO_VECTORSTORE,
    recriar: bool = False,
    verboso: bool = True,
    incluir_prontuarios: bool = True,
) -> tuple[Chroma, dict]:
    """Pipeline completo: carregar -> chunk -> indexar -> manifesto.

    `limite` corta apenas o MedQuAD. Os 30 prontuários entram sempre inteiros:
    são a fatia que dá ao assistente um paciente sobre o qual falar, e indexar
    um pedaço deles tornaria a fatia de desenvolvimento enganosa — o retriever
    responderia "não há registro deste paciente" por corte de `--limite`, não
    por ausência real.
    """
    from .loaders import CAMINHO_POOL_MEDQUAD

    config = config or ConfiguracaoIngestao()
    versao = VersaoCorpus(version="medquad-pool-v1", updated_at=date.today().isoformat())

    if verboso:
        print(f"[1/4] carregando {CAMINHO_POOL_MEDQUAD.name}...")
    docs, relatorio = carregar_medquad(limite=limite)
    if verboso:
        print(f"      {relatorio.resumo()}")

    fontes = [
        FonteCorpus(
            nome="medquad_pool",
            caminho=CAMINHO_POOL_MEDQUAD,
            relatorio=relatorio,
            observacao=(
                "Somente linhas NÃO usadas no fine-tuning do Integrante 1. "
                "Ver README.md §4.4 para o critério de exclusão (por texto da "
                "pergunta, não só por índice)."
            ),
        )
    ]

    if incluir_prontuarios:
        prontuarios, relatorio_prontuarios = carregar_prontuarios()
        if verboso:
            print(f"      + {relatorio_prontuarios.resumo()} (prontuários sintéticos)")
        docs = docs + prontuarios
        fontes.append(
            FonteCorpus(
                nome="prontuarios_sinteticos",
                caminho=CAMINHO_PRONTUARIOS,
                relatorio=relatorio_prontuarios,
                observacao=(
                    "Registros clínicos FICTÍCIOS, escritos para este projeto. "
                    "Não correspondem a paciente real e não contêm PII: sem nome, "
                    "CPF, telefone, endereço ou data de nascimento."
                ),
            )
        )

    if verboso:
        print("[2/4] dividindo em chunks...")
    chunks = dividir_todos(docs, versao, config.limite_chars_chunk, config.overlap_chars)
    if verboso:
        inteiros = len({
            c.metadata["document_id"] for c in chunks
            if c.metadata["page_or_section"] in ("resposta", "registro")
        })
        print(f"      {len(chunks)} chunks ({inteiros}/{len(docs)} documentos não divididos)")

    if verboso:
        print(f"[3/4] indexando em '{config.colecao}' (assinatura {config.assinatura()})...")
    store = ingerir(chunks, config, persist_directory, recriar=recriar, verboso=verboso)

    if verboso:
        print("[4/4] escrevendo manifesto...")
    escrever_manifesto(config, versao, fontes, len(chunks))

    resumo = {
        "documentos": len(docs),
        "chunks": len(chunks),
        "prontuarios": sum(1 for d in docs if d.document_type == "prontuario_sintetico"),
        "registros_no_indice": store._collection.count(),
        "colecao": config.colecao,
    }
    return store, resumo
