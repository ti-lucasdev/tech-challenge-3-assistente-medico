"""Fixtures compartilhadas.

Os testes unitários não baixam o modelo de embedding (1,1 GB) nem dependem de
GPU: usam um embedding determinístico e trivial. Isso é proposital — o que se
testa aqui é o encanamento (IDs, idempotência, metadados, filtros, orçamento de
contexto), não a qualidade semântica. Qualidade de recuperação é medida pelo
conjunto de avaliação com Recall@k/MRR, que é outra coisa e roda separado.
"""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import pytest
from langchain_core.embeddings import Embeddings

RAIZ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ))

from src.rag.loaders import DocumentoBruto  # noqa: E402
from src.rag.preprocess import VersaoCorpus  # noqa: E402

DIMENSAO = 32


class EmbeddingsFalsos(Embeddings):
    """Embedding determinístico por hash — sem modelo, sem rede, sem GPU.

    Não preserva semântica alguma: textos parecidos ficam distantes. Serve para
    exercitar o caminho de indexação e filtro, não a busca por significado.
    """

    def _vetor(self, texto: str) -> list[float]:
        digest = hashlib.sha256(texto.encode("utf-8")).digest()
        bruto = [digest[i % len(digest)] / 255.0 for i in range(DIMENSAO)]
        norma = sum(v * v for v in bruto) ** 0.5 or 1.0
        return [v / norma for v in bruto]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._vetor(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._vetor(text)


@pytest.fixture
def versao() -> VersaoCorpus:
    return VersaoCorpus(version="teste-v1", updated_at="2026-08-22")


@pytest.fixture
def documentos() -> list[DocumentoBruto]:
    """Corpus mínimo: um curto, um longo o bastante para dividir, outra fonte."""
    return [
        DocumentoBruto(
            pergunta="What are the symptoms of Ankylosing spondylitis ?",
            resposta="Back pain and stiffness that worsen with rest and improve with exercise.",
            group_name="2_GARD_QA",
            origem="tests/fixtures",
        ),
        DocumentoBruto(
            pergunta="What causes Type 2 diabetes ?",
            resposta=(
                "Insulin resistance combined with progressive beta-cell dysfunction. "
                * 40
            ),
            group_name="5_NIDDK_QA",
            origem="tests/fixtures",
        ),
        DocumentoBruto(
            pergunta="How to prevent influenza ?",
            resposta="Annual vaccination remains the most effective preventive measure.",
            group_name="9_CDC_QA",
            origem="tests/fixtures",
        ),
    ]


@pytest.fixture
def embeddings() -> EmbeddingsFalsos:
    return EmbeddingsFalsos()
