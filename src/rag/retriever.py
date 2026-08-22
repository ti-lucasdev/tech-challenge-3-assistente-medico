"""Interface de busca consumida pelo Integrante 3 (orquestração) e pelo 4 (guardrails).

## Por que o retorno não é `list[Document]`

O guia propõe `retrieve() -> list[Document]`, com o score exposto "num método
separado, se o vector store fornecer". Aqui o score volta junto, dentro de um
tipo próprio, por uma razão concreta: **a abstenção depende dele**.

O enunciado exige que o assistente não improvise quando não há evidência. Quem
decide isso é a camada de guardrails (Integrante 4), e essa decisão precisa do
score do topo-k. Se o score sair por outra porta, ou a decisão migra para a
camada errada, ou cada consumidor reimplementa a sua — e passam a divergir.

`recuperar_documentos()` continua devolvendo `list[Document]` puro, para quem
quiser o formato canônico do LangChain sem os extras.

## Sobre o limiar de abstenção — leia antes de confiar nele

Medido em 200 sondas parafraseadas contra 30 perguntas sem suporte no corpus
(`scripts/avaliar_retriever.py`), o resultado foi **negativo**: as duas
distribuições de score do topo-1 se **sobrepõem**. Positivos têm p05 = 0,718;
negativos chegam a 0,726. Não há corte limpo.

Sinais alternativos foram testados e são piores. O gap entre topo-1 e topo-5
separa ao contrário: negativos têm gap *maior* que positivos (p95 = 0,040
contra p05 = 0,016). O motivo é estrutural — numa pergunta legítima há vários
chunks quase igualmente bons (partes do mesmo documento, mais documentos com a
mesma pergunta vindos de outras instituições), então o topo-1 não se destaca.
Numa pergunta sem suporte, um chunk arbitrário vence por uma margem maior.

O valor de 0,73 é, portanto, uma escolha de **assimetria de custo**, não um
ponto de separação: num assistente clínico, apresentar resposta sem suporte é
pior do que abster-se à toa. Medido neste conjunto:

| limiar | abstenção falsa | negativo aceito |
|--------|-----------------|-----------------|
| 0,710  | 3,5%            | 6,7%            |
| 0,720  | 6,0%            | 3,3%            |
| 0,730  | 9,0%            | 0,0%            |
| 0,750  | 17,0%           | 0,0%            |

Os 0,0% valem para 30 negativos — com n=30, o limite superior do intervalo de
confiança fica perto de 10%. **Isto é um sinal de baixa confiança, não uma
garantia.** A camada de guardrails não deve tratá-lo como suficiente: convém
exigir também que a resposta gerada cite alguma fonte recuperada.

Recalibrar é obrigatório se o modelo de embedding ou o corpus mudarem.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final, Literal

from langchain_chroma import Chroma
from langchain_core.documents import Document

from .embeddings import EmbeddingsE5
from .ingest import DIRETORIO_VECTORSTORE, ConfiguracaoIngestao, abrir_store

ModoBusca = Literal["denso", "bm25", "hibrido"]

# Ver a seção sobre abstenção no topo deste arquivo: as distribuições de
# positivos e negativos se sobrepõem, então este valor é uma escolha de
# assimetria de custo (9% de abstenção falsa, 0/30 negativos aceitos), não um
# ponto de separação limpo.
LIMIAR_EVIDENCIA: Final[float] = 0.73

# Constante padrão da Reciprocal Rank Fusion. Amortece o peso das primeiras
# posições, evitando que um único ranking domine a fusão.
K_RRF: Final[int] = 60

# Razões medidas com o tokenizer real do modelo, por idioma. Usadas só para
# estimar o orçamento de contexto quando não se passa um contador de tokens.
#
# A diferença entre elas não é detalhe: o tokenizer do Llama-3 foi treinado
# majoritariamente em inglês, então o português se fragmenta em muito mais
# tokens por caractere. Medido neste corpus com o tokenizer do adaptador:
# MedQuAD em inglês dá 4,58 chars/token; os prontuários em pt-BR dão 3,00.
#
# Usar 4,78 para tudo subestimava o custo do português em ~60%: `k=4`
# prontuários somam ~1.599 tokens reais, e a estimativa antiga dizia 1.004 —
# folga suficiente para o prompt estourar a janela de 2.048 sem ninguém notar,
# porque o corte silencioso acontece dentro da LLM e o que se perde é o final
# do contexto.
CHARS_POR_TOKEN_POR_IDIOMA: Final[dict[str, float]] = {
    "en": 4.58,
    "pt-BR": 3.00,
}

# Fallback para idioma não medido: o menor valor conhecido, porque errar para
# baixo aqui significa admitir contexto demais e truncar dentro da LLM.
CHARS_POR_TOKEN: Final[float] = min(CHARS_POR_TOKEN_POR_IDIOMA.values())


@dataclass(frozen=True)
class ResultadoRecuperacao:
    """Um trecho recuperado, com a informação necessária para citá-lo."""

    documento: Document
    score: float  # similaridade de cosseno, 0..1 — maior é mais relevante
    rank: int  # posição no ranking final, começando em 1

    @property
    def texto(self) -> str:
        return self.documento.page_content

    @property
    def metadados(self) -> dict[str, Any]:
        return self.documento.metadata

    @property
    def document_id(self) -> str:
        return self.documento.metadata["document_id"]

    def citacao(self) -> str:
        """Referência curta e legível, para a resposta final do assistente."""
        m = self.documento.metadata
        return f"{m['title']} — {m['institution']} ({m['source']})"


class MedicalRetriever:
    """Busca semântica sobre a base vetorial médica.

    Uso típico (Integrante 3):

        retriever = MedicalRetriever()
        resultados = retriever.retrieve("What causes hypothyroidism?", k=4)
        if not retriever.tem_evidencia_suficiente(resultados):
            ...  # responder que não há suporte, em vez de improvisar
        contexto = retriever.format_context(resultados)
    """

    def __init__(
        self,
        config: ConfiguracaoIngestao | None = None,
        persist_directory: Path | str = DIRETORIO_VECTORSTORE,
        embeddings: EmbeddingsE5 | None = None,
        modo: ModoBusca = "denso",
        store: Chroma | None = None,
    ) -> None:
        self.config = config or ConfiguracaoIngestao()
        self.modo = modo
        self._store = store or abrir_store(self.config, persist_directory, embeddings)
        self._bm25: Any | None = None
        self._corpus_bm25: list[Document] = []

        if self._store._collection.count() == 0:
            raise RuntimeError(
                f"A coleção '{self.config.colecao}' está vazia em "
                f"{Path(persist_directory).resolve()}.\n"
                "Rode `python scripts/build_vector_store.py` antes de consultar."
            )

    # -- busca ------------------------------------------------------------

    def retrieve(
        self,
        query: str,
        k: int = 4,
        filters: dict[str, Any] | None = None,
    ) -> list[ResultadoRecuperacao]:
        """Recupera os `k` trechos mais relevantes, ordenados por relevância."""
        if not query or not query.strip():
            raise ValueError("consulta vazia")

        if self.modo == "denso":
            return self._buscar_denso(query, k, filters)
        if self.modo == "bm25":
            return self._buscar_bm25(query, k, filters)
        return self._buscar_hibrido(query, k, filters)

    def recuperar_documentos(
        self,
        query: str,
        k: int = 4,
        filters: dict[str, Any] | None = None,
    ) -> list[Document]:
        """Mesma busca, devolvendo `Document` puro do LangChain."""
        return [r.documento for r in self.retrieve(query, k, filters)]

    def _buscar_denso(
        self, query: str, k: int, filters: dict[str, Any] | None
    ) -> list[ResultadoRecuperacao]:
        pares = self._store.similarity_search_with_score(
            query, k=k, filter=_traduzir_filtros(filters)
        )
        # O Chroma devolve *distância*; com a coleção em espaço de cosseno,
        # similaridade = 1 - distância.
        return [
            ResultadoRecuperacao(documento=doc, score=1.0 - float(dist), rank=i)
            for i, (doc, dist) in enumerate(pares, start=1)
        ]

    def _buscar_bm25(
        self, query: str, k: int, filters: dict[str, Any] | None
    ) -> list[ResultadoRecuperacao]:
        from rank_bm25 import BM25Okapi

        if self._bm25 is None:
            self._carregar_bm25()

        candidatos = [
            (doc, s)
            for doc, s in zip(self._corpus_bm25, self._bm25.get_scores(_tokenizar(query)))
            if _passa_filtro(doc, filters)
        ]
        candidatos.sort(key=lambda p: p[1], reverse=True)
        maior = candidatos[0][1] if candidatos and candidatos[0][1] > 0 else 1.0
        return [
            # BM25 não é limitado a [0,1]; normaliza-se pelo topo apenas para
            # que o valor seja comparável dentro da mesma consulta. NÃO é
            # similaridade de cosseno e não deve ser comparado ao limiar.
            ResultadoRecuperacao(documento=doc, score=float(s) / maior, rank=i)
            for i, (doc, s) in enumerate(candidatos[:k], start=1)
        ]

    def _buscar_hibrido(
        self, query: str, k: int, filters: dict[str, Any] | None
    ) -> list[ResultadoRecuperacao]:
        """Funde denso e BM25 por Reciprocal Rank Fusion.

        Embeddings densos casam intenção mas tropeçam em entidade rara; o BM25
        acerta o nome exato da condição e ignora a intenção. A RRF combina os
        rankings sem exigir que as duas escalas de score sejam comparáveis —
        que é justamente o que elas não são.
        """
        profundidade = max(k * 5, 20)
        densos = self._buscar_denso(query, profundidade, filters)
        lexicais = self._buscar_bm25(query, profundidade, filters)

        pontos: dict[str, float] = {}
        por_chunk: dict[str, Document] = {}
        for lista in (densos, lexicais):
            for r in lista:
                cid = r.metadados["chunk_id"]
                pontos[cid] = pontos.get(cid, 0.0) + 1.0 / (K_RRF + r.rank)
                por_chunk[cid] = r.documento

        # O score de cosseno do lado denso é preservado quando existe, para que
        # o limiar de abstenção continue interpretável no modo híbrido.
        cosseno = {r.metadados["chunk_id"]: r.score for r in densos}
        ordenados = sorted(pontos.items(), key=lambda p: p[1], reverse=True)[:k]
        return [
            ResultadoRecuperacao(
                documento=por_chunk[cid],
                score=cosseno.get(cid, 0.0),
                rank=i,
            )
            for i, (cid, _) in enumerate(ordenados, start=1)
        ]

    def _carregar_bm25(self) -> None:
        from rank_bm25 import BM25Okapi

        dados = self._store._collection.get(include=["documents", "metadatas"])
        self._corpus_bm25 = [
            Document(page_content=t, metadata=m)
            for t, m in zip(dados["documents"], dados["metadatas"])
        ]
        self._bm25 = BM25Okapi([_tokenizar(d.page_content) for d in self._corpus_bm25])

    # -- abstenção --------------------------------------------------------

    def tem_evidencia_suficiente(
        self,
        resultados: list[ResultadoRecuperacao],
        limiar: float = LIMIAR_EVIDENCIA,
    ) -> bool:
        """Sinaliza se o topo-k sustenta uma resposta.

        Falso significa "responda que não há suporte no corpus", não "os
        documentos são irrelevantes". A decisão final é da camada de guardrails.

        Heurística de baixa confiança: as distribuições de score de perguntas
        cobertas e não cobertas se sobrepõem neste corpus (ver a seção sobre
        abstenção no topo do módulo). Não use isoladamente como garantia de
        que a resposta tem suporte.
        """
        if not resultados:
            return False
        return resultados[0].score >= limiar

    # -- formatação -------------------------------------------------------

    def format_context(
        self,
        resultados: list[ResultadoRecuperacao],
        max_tokens: int = 1200,
        contar_tokens: Any | None = None,
    ) -> str:
        """Monta o bloco de contexto para o prompt da LLM, dentro do orçamento.

        Separado de `retrieve()` de propósito: a busca não deve saber nada sobre
        o formato de prompt da LLM, que o Integrante 3 pode versionar.

        `contar_tokens` recebe uma função `str -> int`; sem ela, estima pela
        razão de chars/token medida por idioma (ver `CHARS_POR_TOKEN_POR_IDIOMA`).
        **Para o prompt final, passe o tokenizer real do modelo** — a estimativa
        serve para planejar, não para garantir que cabe:

            contexto = retriever.format_context(
                resultados,
                contar_tokens=lambda t: len(tokenizer.encode(t, add_special_tokens=False)),
            )
        """
        blocos: list[str] = []
        usados = 0
        for r in resultados:
            m = r.metadados
            bloco = (
                f"[FONTE {r.rank}]\n"
                f"title: {m['title']}\n"
                f"source: {m['source']}\n"
                f"institution: {m['institution']}\n"
                f"section: {m['page_or_section']}\n"
                f"content: {r.texto}"
            )
            # A razão vem do idioma do próprio chunk: um contexto pode misturar
            # FAQ em inglês e prontuário em pt-BR, e uma razão única erraria em
            # um dos dois. Um `contar_tokens` explícito sempre tem precedência.
            if contar_tokens is not None:
                custo = contar_tokens(bloco)
            else:
                razao = CHARS_POR_TOKEN_POR_IDIOMA.get(
                    m.get("language", ""), CHARS_POR_TOKEN
                )
                custo = int(len(bloco) / razao)
            if usados + custo > max_tokens:
                # Trunca antes de estourar: um contexto cortado no meio pela
                # janela da LLM perde o final silenciosamente, e é justamente
                # o final que costuma carregar a conclusão.
                break
            blocos.append(bloco)
            usados += custo

        return "\n\n".join(blocos)


# --------------------------------------------------------------------------
# Filtros
# --------------------------------------------------------------------------


def _traduzir_filtros(filters: dict[str, Any] | None) -> dict[str, Any] | None:
    """Converte um dicionário simples para a sintaxe `where` do Chroma.

    Aceita `{"document_type": "faq_medica"}` e `{"source_group": ["9_CDC_QA",
    "5_NIDDK_QA"]}`. O Chroma exige `$and` explícito quando há mais de uma
    condição — passar duas chaves soltas é erro silencioso em algumas versões.
    """
    if not filters:
        return None

    condicoes = [
        {campo: {"$in": list(valor)}} if isinstance(valor, (list, tuple, set))
        else {campo: {"$eq": valor}}
        for campo, valor in filters.items()
    ]
    return condicoes[0] if len(condicoes) == 1 else {"$and": condicoes}


def _passa_filtro(documento: Document, filters: dict[str, Any] | None) -> bool:
    """Mesma semântica do filtro do Chroma, aplicada em memória para o BM25."""
    if not filters:
        return True
    for campo, valor in filters.items():
        atual = documento.metadata.get(campo)
        if isinstance(valor, (list, tuple, set)):
            if atual not in valor:
                return False
        elif atual != valor:
            return False
    return True


def _tokenizar(texto: str) -> list[str]:
    return texto.lower().split()
