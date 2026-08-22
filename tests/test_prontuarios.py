"""Camada de registros clínicos sintéticos.

O que se testa aqui é o que faz o prontuário ser utilizável pelas camadas de
cima: que o recorte por paciente funcione, que a natureza fabricada do dado
esteja visível na citação, e que o conteúdo continue livre de PII. Qualidade
semântica da recuperação não é testada aqui (ver `scripts/avaliar_retriever.py`).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

RAIZ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ))

from src.rag.ingest import ConfiguracaoIngestao, ingerir  # noqa: E402
from src.rag.loaders import (  # noqa: E402
    CAMINHO_PRONTUARIOS,
    DocumentoBruto,
    carregar_prontuarios,
)
from src.rag.preprocess import dividir_todos  # noqa: E402
from src.rag.retriever import MedicalRetriever  # noqa: E402
from src.rag.schemas import NAO_APLICAVEL  # noqa: E402

MINIMO_PRONTUARIOS = 30


@pytest.fixture(scope="module")
def prontuarios() -> list[DocumentoBruto]:
    documentos, _ = carregar_prontuarios()
    return documentos


# -- carga -----------------------------------------------------------------


def test_carrega_todos_os_prontuarios(prontuarios):
    assert len(prontuarios) >= MINIMO_PRONTUARIOS


def test_identificadores_de_paciente_sao_unicos(prontuarios):
    identificadores = [d.patient_id for d in prontuarios]
    assert len(set(identificadores)) == len(identificadores)


def test_todo_prontuario_e_marcado_como_sintetico(prontuarios):
    """`is_synthetic` é o que impede um registro fabricado ser citado como real."""
    assert all(d.is_synthetic for d in prontuarios)
    assert all(d.document_type == "prontuario_sintetico" for d in prontuarios)
    assert all(d.language == "pt-BR" for d in prontuarios)


def test_registro_malformado_falha_alto(tmp_path):
    """Descartar um prontuário em silêncio é o pior modo de erro desta camada.

    O assistente responderia "não há registro deste paciente" quando o que
    houve foi falha de carga — indistinguível, para quem lê, de uma ausência
    real de informação.
    """
    arquivo = tmp_path / "prontuarios.jsonl"
    arquivo.write_text(
        json.dumps({"patient_id": "PAC-9001", "titulo": "x", "texto": "y"}) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="campos ausentes"):
        carregar_prontuarios(arquivo)


def test_patient_id_repetido_falha(tmp_path):
    registro = {
        "patient_id": "PAC-9001",
        "titulo": "Consulta",
        "texto": "Texto clínico do registro sintético para o teste.",
        "especialidade": "clinica_medica",
        "atualizado_em": "2026-08-22",
    }
    arquivo = tmp_path / "prontuarios.jsonl"
    arquivo.write_text(
        json.dumps(registro) + "\n" + json.dumps(registro) + "\n", encoding="utf-8"
    )
    with pytest.raises(ValueError, match="repetido"):
        carregar_prontuarios(arquivo)


# -- metadados e citação ---------------------------------------------------


def test_metadados_permitem_recortar_e_citar(prontuarios, versao):
    chunks = dividir_todos(prontuarios, versao)
    assert chunks

    for chunk in chunks:
        m = chunk.metadata
        assert m["document_type"] == "prontuario_sintetico"
        assert m["is_synthetic"] is True
        assert m["patient_id"].startswith("PAC-")
        # A citação precisa ser rastreável até o paciente, e visivelmente não
        # ser uma URL pública: um prontuário fabricado citado como se fosse
        # uma página do NIH é exatamente o erro que a explainability deve evitar.
        assert m["source"] == f"prontuario-sintetico://{m['patient_id']}"
        assert "sintético" in m["institution"]


def test_chunk_de_prontuario_nao_se_disfarca_de_faq(prontuarios, versao):
    """O cabeçalho Q&A do MedQuAD não pode vazar para um registro clínico."""
    chunks = dividir_todos(prontuarios, versao)
    for chunk in chunks:
        assert chunk.page_content.startswith("Prontuário PAC-")
        assert not chunk.page_content.startswith("Question:")


def test_faq_do_medquad_nao_ganha_paciente(documentos, versao):
    """O campo novo não pode contaminar o corpus que já existia."""
    chunks = dividir_todos(documentos, versao)
    assert chunks
    for chunk in chunks:
        assert chunk.metadata["patient_id"] == NAO_APLICAVEL
        assert chunk.metadata["is_synthetic"] is False
        assert chunk.metadata["document_type"] == "faq_medica"


def test_todo_prontuario_declara_exames_pendentes(prontuarios):
    """O fluxo clínico do enunciado ("verificar exames pendentes") depende disso.

    Sem esse campo em todo registro, o nó de decisão do StateGraph passa a
    depender de o texto por acaso mencionar exames — e o fluxo fica não
    determinístico por motivo errado.
    """
    sem_campo = [d.patient_id for d in prontuarios if "EXAMES PENDENTES:" not in d.corpo]
    assert not sem_campo, f"prontuários sem seção de exames pendentes: {sem_campo}"


# -- recuperação com filtro por paciente -----------------------------------


def test_filtro_por_paciente_isola_o_prontuario(prontuarios, versao, embeddings, tmp_path):
    """O requisito de "contextualizar com dados do paciente" vive aqui.

    Sem esse recorte, uma pergunta sobre o paciente A pode trazer o registro do
    paciente B por similaridade clínica — dois diabéticos descompensados são
    textos muito parecidos.
    """
    chunks = dividir_todos(prontuarios, versao)
    config = ConfiguracaoIngestao(colecao="teste_prontuarios")
    store = ingerir(
        chunks, config, tmp_path, embeddings=embeddings, recriar=True, verboso=False
    )
    retriever = MedicalRetriever(config=config, store=store)

    alvo = prontuarios[0].patient_id
    resultados = retriever.retrieve(
        "exames pendentes e conduta", k=5, filters={"patient_id": alvo}
    )

    assert resultados
    assert {r.metadados["patient_id"] for r in resultados} == {alvo}


def test_filtro_por_tipo_separa_prontuario_de_faq(
    prontuarios, documentos, versao, embeddings, tmp_path
):
    chunks = dividir_todos(prontuarios + documentos, versao)
    config = ConfiguracaoIngestao(colecao="teste_misto")
    store = ingerir(
        chunks, config, tmp_path, embeddings=embeddings, recriar=True, verboso=False
    )
    retriever = MedicalRetriever(config=config, store=store)

    apenas_faq = retriever.retrieve(
        "diabetes", k=5, filters={"document_type": "faq_medica"}
    )
    assert apenas_faq
    assert all(r.metadados["is_synthetic"] is False for r in apenas_faq)

    apenas_registros = retriever.retrieve(
        "diabetes", k=5, filters={"document_type": "prontuario_sintetico"}
    )
    assert apenas_registros
    assert all(r.metadados["is_synthetic"] is True for r in apenas_registros)


# -- índice real (opt-in) --------------------------------------------------

DIRETORIO_VECTORSTORE = RAIZ / "vectorstore"

precisa_do_indice = pytest.mark.skipif(
    not DIRETORIO_VECTORSTORE.exists(),
    reason="exige a base vetorial real; rode scripts/build_vector_store.py",
)


@precisa_do_indice
def test_filtro_por_paciente_e_confiavel_no_indice_real():
    """Recuperação por paciente contra o índice de verdade, com o E5.

    Diferente dos testes acima, este usa o modelo de embedding real e a coleção
    persistida. Fica opt-in — pula quando a base não foi construída — para não
    quebrar a regra de que a suíte roda sem GPU e sem rede.
    """
    from src.rag.retriever import MedicalRetriever

    retriever = MedicalRetriever()
    documentos, _ = carregar_prontuarios()
    perguntas = [
        "Quais exames estao pendentes?",
        "Qual a conduta sugerida para este paciente?",
        "Quais medicacoes o paciente usa?",
    ]

    erros = []
    for documento in documentos:
        for pergunta in perguntas:
            resultados = retriever.retrieve(
                pergunta, k=1, filters={"patient_id": documento.patient_id}
            )
            if not resultados:
                erros.append(f"{documento.patient_id}/{pergunta!r}: nada recuperado")
            elif resultados[0].metadados["patient_id"] != documento.patient_id:
                erros.append(
                    f"{documento.patient_id}/{pergunta!r}: veio "
                    f"{resultados[0].metadados['patient_id']}"
                )

    assert not erros, "\n".join(erros[:10])


@precisa_do_indice
def test_abstencao_por_score_nao_serve_para_prontuario():
    """Regressão de um resultado medido, não de uma suposição.

    Em 90 consultas com filtro por paciente, a recuperação acertou 100% e o
    limiar de 0,73 rejeitaria 100%. Este teste trava esse fato: se alguém ligar
    `tem_evidencia_suficiente` no caminho do paciente, o assistente passa a
    responder "não há evidência suficiente" sobre todo paciente do sistema.

    Se um dia a recalibração acontecer e os scores subirem, este teste falha —
    e falhar é o comportamento certo: significa que a orientação no handoff
    precisa ser reescrita.
    """
    from src.rag.retriever import LIMIAR_EVIDENCIA, MedicalRetriever

    retriever = MedicalRetriever()
    documentos, _ = carregar_prontuarios()

    scores = []
    for documento in documentos:
        resultados = retriever.retrieve(
            "Quais exames estao pendentes?", k=1,
            filters={"patient_id": documento.patient_id},
        )
        assert resultados
        scores.append(resultados[0].score)

    assert max(scores) < LIMIAR_EVIDENCIA, (
        f"o maior score com filtro por paciente ({max(scores):.4f}) passou do "
        f"limiar de {LIMIAR_EVIDENCIA}. A orientação em README.md §5.7 — "
        "de não usar abstenção por score nesse caminho — foi escrita com base "
        "no oposto disso e precisa ser revista."
    )


# -- privacidade -----------------------------------------------------------


def test_prontuarios_nao_tem_identificadores_pessoais():
    """A anonimização aqui é por construção: o dado não existe para ser removido.

    `tests/test_privacidade.py` já varre `data/*.jsonl` com padrões de CPF,
    telefone e e-mail. Este teste cobre o que aquele não pega: nome próprio e
    data de nascimento, que não têm formato fixo o bastante para regex confiável,
    mas cujos *rótulos* podem ser proibidos no arquivo que nós escrevemos.
    """
    texto = CAMINHO_PRONTUARIOS.read_text(encoding="utf-8").lower()
    rotulos_proibidos = [
        "nome:",
        "nome completo",
        "data de nascimento",
        "cpf",
        "rg:",
        "telefone",
        "endereco:",
        "endereço:",
        "e-mail",
        "cartao nacional de saude",
    ]
    encontrados = [r for r in rotulos_proibidos if r in texto]
    assert not encontrados, (
        f"rótulo de dado pessoal no arquivo de prontuários: {encontrados}. "
        "Estes registros são sintéticos e devem permanecer sem identificação "
        "direta, mesmo fabricada."
    )
