"""Testes do contrato do retriever: filtros, score, abstenção e formatação.

A qualidade semântica da busca **não** é testada aqui — ela é medida pelo
conjunto de avaliação (Recall@k/MRR), que exige o modelo real e a base
completa. Estes testes usam embeddings falsos e verificam o contrato, que é o
que os Integrantes 3 e 4 consomem.
"""

from __future__ import annotations

from dataclasses import replace

import pytest
from langchain_chroma import Chroma
from langchain_core.documents import Document

from src.rag.ingest import ConfiguracaoIngestao
from src.rag.preprocess import dividir_todos
from src.rag.retriever import (
    MedicalRetriever,
    ResultadoRecuperacao,
    _passa_filtro,
    _traduzir_filtros,
)


@pytest.fixture
def retriever(tmp_path, documentos, versao, embeddings) -> MedicalRetriever:
    config = ConfiguracaoIngestao(colecao="teste_retriever")
    chunks = dividir_todos(documentos, versao)
    store = Chroma(
        collection_name=config.colecao,
        embedding_function=embeddings,
        persist_directory=str(tmp_path / "vs"),
        collection_metadata={"hnsw:space": "cosine"},
    )
    store.add_documents(chunks, ids=[c.metadata["chunk_id"] for c in chunks])
    return MedicalRetriever(config=config, store=store)


# --------------------------------------------------------------------------
# Contrato básico
# --------------------------------------------------------------------------


def test_retrieve_respeita_k(retriever):
    assert len(retriever.retrieve("diabetes", k=2)) == 2


def test_retrieve_devolve_rank_sequencial(retriever):
    resultados = retriever.retrieve("diabetes", k=3)
    assert [r.rank for r in resultados] == [1, 2, 3]


def test_retrieve_traz_score(retriever):
    """O score precisa vir junto: a abstenção do Integrante 4 depende dele."""
    for r in retriever.retrieve("diabetes", k=3):
        assert isinstance(r.score, float)
        assert -1.0 <= r.score <= 1.0


def test_retrieve_rejeita_consulta_vazia(retriever):
    with pytest.raises(ValueError, match="vazia"):
        retriever.retrieve("   ")


def test_recuperar_documentos_devolve_document(retriever):
    docs = retriever.recuperar_documentos("diabetes", k=2)
    assert all(isinstance(d, Document) for d in docs)


def test_todo_resultado_tem_fonte_para_citar(retriever):
    """Sem isso, a camada de explainability não tem o que exibir."""
    for r in retriever.retrieve("diabetes", k=3):
        citacao = r.citacao()
        assert r.metadados["title"] in citacao
        assert r.metadados["institution"] in citacao
        assert r.metadados["source"] in citacao


def test_colecao_vazia_falha_com_mensagem_acionavel(tmp_path, embeddings):
    config = ConfiguracaoIngestao(colecao="vazia")
    store = Chroma(
        collection_name=config.colecao,
        embedding_function=embeddings,
        persist_directory=str(tmp_path / "vs"),
    )
    with pytest.raises(RuntimeError, match="build_vector_store"):
        MedicalRetriever(config=config, store=store)


# --------------------------------------------------------------------------
# Filtros
# --------------------------------------------------------------------------


def test_filtro_por_fonte_nao_vaza_outras_fontes(retriever):
    resultados = retriever.retrieve("prevention", k=5, filters={"source_group": "9_CDC_QA"})
    assert resultados
    assert all(r.metadados["source_group"] == "9_CDC_QA" for r in resultados)


def test_filtro_aceita_lista(retriever):
    alvos = ["9_CDC_QA", "5_NIDDK_QA"]
    resultados = retriever.retrieve("x", k=10, filters={"source_group": alvos})
    assert resultados
    assert all(r.metadados["source_group"] in alvos for r in resultados)


def test_filtro_por_tipo_de_documento(retriever):
    resultados = retriever.retrieve("x", k=5, filters={"document_type": "faq_medica"})
    assert all(r.metadados["document_type"] == "faq_medica" for r in resultados)


def test_traduzir_filtros_usa_and_para_multiplas_condicoes():
    """O Chroma exige `$and` explícito; duas chaves soltas é erro silencioso."""
    traduzido = _traduzir_filtros({"language": "en", "document_type": "faq_medica"})
    assert "$and" in traduzido
    assert len(traduzido["$and"]) == 2


def test_traduzir_filtros_vazio_e_none():
    assert _traduzir_filtros(None) is None
    assert _traduzir_filtros({}) is None


def test_filtro_em_memoria_tem_mesma_semantica():
    doc = Document(page_content="x", metadata={"language": "en", "source_group": "9_CDC_QA"})
    assert _passa_filtro(doc, {"language": "en"})
    assert not _passa_filtro(doc, {"language": "pt-BR"})
    assert _passa_filtro(doc, {"source_group": ["9_CDC_QA", "2_GARD_QA"]})
    assert not _passa_filtro(doc, {"source_group": ["2_GARD_QA"]})


# --------------------------------------------------------------------------
# Abstenção
# --------------------------------------------------------------------------


def _resultado(score: float, rank: int = 1) -> ResultadoRecuperacao:
    return ResultadoRecuperacao(
        documento=Document(page_content="x", metadata={"chunk_id": f"c{rank}"}),
        score=score,
        rank=rank,
    )


def test_sem_resultados_nao_ha_evidencia(retriever):
    assert retriever.tem_evidencia_suficiente([]) is False


def test_score_abaixo_do_limiar_nao_ha_evidencia(retriever):
    assert retriever.tem_evidencia_suficiente([_resultado(0.40)], limiar=0.82) is False


def test_score_acima_do_limiar_ha_evidencia(retriever):
    assert retriever.tem_evidencia_suficiente([_resultado(0.91)], limiar=0.82) is True


# --------------------------------------------------------------------------
# Formatação de contexto
# --------------------------------------------------------------------------


def test_format_context_respeita_orcamento(retriever):
    """O corpus de teste é todo em inglês, então a razão aplicada é a de `en`."""
    from src.rag.retriever import CHARS_POR_TOKEN_POR_IDIOMA

    resultados = retriever.retrieve("diabetes", k=4)
    contexto = retriever.format_context(resultados, max_tokens=60)
    assert len(contexto) / CHARS_POR_TOKEN_POR_IDIOMA["en"] <= 60 * 1.05


def test_format_context_usa_razao_do_idioma_do_chunk(retriever):
    """Português custa mais tokens por caractere, e o orçamento precisa saber.

    Com uma razão única (a antiga, de 4,78), um bloco em pt-BR entrava no
    contexto por ~60% menos do que custa de verdade — folga suficiente para o
    prompt estourar a janela da LLM e perder o final do contexto em silêncio.
    """
    from src.rag.retriever import CHARS_POR_TOKEN_POR_IDIOMA

    assert CHARS_POR_TOKEN_POR_IDIOMA["pt-BR"] < CHARS_POR_TOKEN_POR_IDIOMA["en"]

    ingles, portugues = [], []
    for r in retriever.retrieve("diabetes", k=2):
        ingles.append(r)
        documento = Document(
            page_content=r.texto, metadata={**r.metadados, "language": "pt-BR"}
        )
        portugues.append(replace(r, documento=documento))

    # Mesmo texto, mesmo orçamento: o bloco tratado como pt-BR é considerado
    # mais caro, então cabe menos coisa.
    orcamento = 80
    assert len(retriever.format_context(portugues, max_tokens=orcamento)) <= len(
        retriever.format_context(ingles, max_tokens=orcamento)
    )


def test_contar_tokens_explicito_tem_precedencia(retriever):
    """Passar o tokenizer real deve desligar a estimativa, não competir com ela."""
    resultados = retriever.retrieve("diabetes", k=3)
    chamadas = []

    def contar(texto: str) -> int:
        chamadas.append(texto)
        return 10

    contexto = retriever.format_context(resultados, max_tokens=25, contar_tokens=contar)
    assert chamadas, "a função de contagem não foi usada"
    assert contexto.count("[FONTE") == 2  # 25 // 10 = 2 blocos de custo 10


def test_format_context_preserva_marcadores_e_fontes(retriever):
    resultados = retriever.retrieve("diabetes", k=2)
    contexto = retriever.format_context(resultados, max_tokens=4000)
    assert "[FONTE 1]" in contexto
    for r in resultados:
        assert r.metadados["source"] in contexto


def test_format_context_vazio_quando_nao_ha_resultados(retriever):
    assert retriever.format_context([]) == ""


def test_format_context_usa_contador_de_tokens_informado(retriever):
    resultados = retriever.retrieve("diabetes", k=4)
    # Contador que declara custo altíssimo: nada deve caber.
    assert retriever.format_context(resultados, max_tokens=10, contar_tokens=lambda t: 999) == ""
