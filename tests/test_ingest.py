"""Testes de carga, chunking, metadados e idempotência da ingestão."""

from __future__ import annotations

import json

import pytest
from langchain_chroma import Chroma

from src.rag.ingest import ConfiguracaoIngestao, ingerir
from src.rag.loaders import MIN_CHARS_RESPOSTA, DocumentoBruto, carregar_medquad
from src.rag.preprocess import dividir_documento, dividir_todos
from src.rag.schemas import (
    CAMPOS_OBRIGATORIOS,
    MetadadosInvalidos,
    id_documento,
    normalizar_pergunta,
    resolver_fonte,
    validar_metadados,
)

# --------------------------------------------------------------------------
# Identificadores
# --------------------------------------------------------------------------


def test_id_documento_e_deterministico():
    a = id_documento("What causes X ?", "Because of Y.")
    b = id_documento("What causes X ?", "Because of Y.")
    assert a == b


def test_id_documento_distingue_mesma_pergunta_com_respostas_diferentes():
    """O MedQuAD repete perguntas com respostas de fontes distintas."""
    a = id_documento("What is (are) Gallbladder Cancer ?", "Resposta da fonte A.")
    b = id_documento("What is (are) Gallbladder Cancer ?", "Resposta da fonte B.")
    assert a != b


def test_id_documento_nao_confunde_fronteira_pergunta_resposta():
    """Sem separador, ("ab","c") e ("a","bc") colidiriam."""
    assert id_documento("ab", "c") != id_documento("a", "bc")


def test_normalizar_pergunta_agrupa_variacoes():
    assert normalizar_pergunta("What is (are)  Diabetes ?") == normalizar_pergunta(
        "what is are diabetes?"
    )


# --------------------------------------------------------------------------
# Metadados
# --------------------------------------------------------------------------


def test_validar_metadados_rejeita_nulo():
    dados = {c: "x" for c in CAMPOS_OBRIGATORIOS}
    dados["is_synthetic"] = False
    dados["source"] = None
    with pytest.raises(MetadadosInvalidos, match="None"):
        validar_metadados(dados)


def test_validar_metadados_rejeita_tipo_nao_escalar():
    dados = {c: "x" for c in CAMPOS_OBRIGATORIOS}
    dados["is_synthetic"] = False
    dados["title"] = ["uma", "lista"]
    with pytest.raises(MetadadosInvalidos, match="Chroma"):
        validar_metadados(dados)


def test_validar_metadados_rejeita_campo_ausente():
    dados = {c: "x" for c in CAMPOS_OBRIGATORIOS if c != "source"}
    dados["is_synthetic"] = False
    with pytest.raises(MetadadosInvalidos, match="source"):
        validar_metadados(dados)


def test_grupo_desconhecido_degrada_sem_quebrar():
    fonte = resolver_fonte("99_GRUPO_INEXISTENTE")
    assert fonte.instituicao
    assert fonte.url


def test_todo_chunk_tem_metadados_completos(documentos, versao):
    for chunk in dividir_todos(documentos, versao):
        for campo in CAMPOS_OBRIGATORIOS:
            assert campo in chunk.metadata, f"faltou {campo}"
        validar_metadados(chunk.metadata)


# --------------------------------------------------------------------------
# Chunking
# --------------------------------------------------------------------------


def test_resposta_curta_vira_um_unico_chunk(documentos, versao):
    chunks = dividir_documento(documentos[0], versao)
    assert len(chunks) == 1
    assert chunks[0].metadata["page_or_section"] == "resposta"


def test_resposta_longa_e_dividida(documentos, versao):
    chunks = dividir_documento(documentos[1], versao)
    assert len(chunks) > 1
    assert all("parte" in c.metadata["page_or_section"] for c in chunks)


def test_pergunta_e_repetida_em_toda_parte(documentos, versao):
    """Sem isso, do 2º pedaço em diante o trecho fica órfão de contexto."""
    doc = documentos[1]
    chunks = dividir_documento(doc, versao)
    assert len(chunks) > 1
    for chunk in chunks:
        assert doc.pergunta in chunk.page_content


def test_chunks_respeitam_o_limite(documentos, versao):
    for chunk in dividir_todos(documentos, versao, limite_chars=1200):
        assert len(chunk.page_content) <= 1200


def test_chunk_ids_sao_unicos(documentos, versao):
    chunks = dividir_todos(documentos, versao)
    ids = [c.metadata["chunk_id"] for c in chunks]
    assert len(ids) == len(set(ids))


# --------------------------------------------------------------------------
# Carga
# --------------------------------------------------------------------------


def _escrever_jsonl(caminho, linhas):
    caminho.write_text(
        "\n".join(json.dumps(linha, ensure_ascii=False) for linha in linhas),
        encoding="utf-8",
    )


def test_carga_descarta_respostas_curtas_e_vazias(tmp_path):
    arquivo = tmp_path / "corpus.jsonl"
    _escrever_jsonl(arquivo, [
        {"query": "What causes Alpha thalassemia ?", "answers": "A" * 200, "group_name": "2_GARD_QA"},
        {"query": "What causes Beta thalassemia ?", "answers": "curta", "group_name": "2_GARD_QA"},
        {"query": "What causes Gamma condition ?", "answers": "", "group_name": "2_GARD_QA"},
    ])
    docs, relatorio = carregar_medquad(arquivo)
    assert len(docs) == 1
    assert relatorio.lidos == 3
    assert sum(relatorio.descartes.values()) == 2


def test_carga_remove_duplicata_exata(tmp_path):
    """O pool tem 47 linhas que repetem, byte a byte, um par Q&A anterior."""
    arquivo = tmp_path / "corpus.jsonl"
    linha = {"query": "What causes Acromegaly ?", "answers": "B" * 200, "group_name": "5_NIDDK_QA"}
    _escrever_jsonl(arquivo, [linha, linha, linha])
    docs, relatorio = carregar_medquad(arquivo)
    assert len(docs) == 1
    assert relatorio.descartes["duplicata exata (mesma pergunta e mesma resposta)"] == 2


def test_carga_falha_com_mensagem_clara_se_arquivo_ausente(tmp_path):
    with pytest.raises(FileNotFoundError, match="export_medquad_dataset"):
        carregar_medquad(tmp_path / "nao_existe.jsonl")


def test_carga_falha_em_json_invalido(tmp_path):
    arquivo = tmp_path / "corpus.jsonl"
    arquivo.write_text('{"query": "ok", "answers": "x"}\n{quebrado\n', encoding="utf-8")
    with pytest.raises(ValueError, match="linha 2"):
        carregar_medquad(arquivo)


def test_carga_respeita_limite(tmp_path):
    arquivo = tmp_path / "corpus.jsonl"
    _escrever_jsonl(arquivo, [
        {"query": f"What causes condition number {i} ?", "answers": "C" * 200,
         "group_name": "2_GARD_QA"}
        for i in range(10)
    ])
    docs, _ = carregar_medquad(arquivo, limite=4)
    assert len(docs) == 4


# --------------------------------------------------------------------------
# Idempotência e persistência
# --------------------------------------------------------------------------


def _store(tmp_path, embeddings, config):
    return Chroma(
        collection_name=config.colecao,
        embedding_function=embeddings,
        persist_directory=str(tmp_path / "vs"),
        collection_metadata={"hnsw:space": "cosine"},
    )


def test_reingestao_identica_nao_duplica(tmp_path, documentos, versao, embeddings):
    config = ConfiguracaoIngestao(colecao="teste_idempotencia")
    chunks = dividir_todos(documentos, versao)

    store = _store(tmp_path, embeddings, config)
    store.add_documents(chunks, ids=[c.metadata["chunk_id"] for c in chunks])
    primeira = store._collection.count()

    store.add_documents(chunks, ids=[c.metadata["chunk_id"] for c in chunks])
    assert store._collection.count() == primeira == len(chunks)


def test_ingestao_rejeita_ids_duplicados(tmp_path, documentos, versao, embeddings):
    config = ConfiguracaoIngestao(colecao="teste_dup")
    chunks = dividir_todos(documentos, versao)
    with pytest.raises(ValueError, match="duplicados"):
        ingerir(
            chunks + chunks[:1],
            config,
            persist_directory=tmp_path / "vs",
            embeddings=embeddings,
            verboso=False,
        )


def test_colecao_persiste_entre_processos(tmp_path, documentos, versao, embeddings):
    """Reabrir a coleção deve encontrar os dados, sem reindexar."""
    config = ConfiguracaoIngestao(colecao="teste_persistencia")
    chunks = dividir_todos(documentos, versao)

    primeiro = _store(tmp_path, embeddings, config)
    primeiro.add_documents(chunks, ids=[c.metadata["chunk_id"] for c in chunks])
    esperado = primeiro._collection.count()
    del primeiro

    segundo = _store(tmp_path, embeddings, config)
    assert segundo._collection.count() == esperado


def test_assinatura_muda_com_a_configuracao():
    a = ConfiguracaoIngestao(limite_chars_chunk=1200)
    b = ConfiguracaoIngestao(limite_chars_chunk=1500)
    assert a.assinatura() != b.assinatura()
    assert a.assinatura() == ConfiguracaoIngestao(limite_chars_chunk=1200).assinatura()
