"""Modelo de embedding e a convenção de prefixos do E5.

## Por que um modelo separado da LLM ajustada

O adaptador LoRA do Integrante 1 gera texto; ele não produz vetores compatíveis
para busca. Usar seus pesos ou seu tokenizer como se fossem embeddings degrada a
recuperação. O adaptador só entra depois que o retriever devolveu o contexto.

## A regra do E5 (não é opcional)

A família `intfloat/*-e5-*` foi treinada com prefixos assimétricos: passagens
indexadas como `passage: <texto>` e consultas como `query: <pergunta>`. Omitir
os prefixos, ou trocá-los entre si, derruba a qualidade da busca — a
documentação do modelo é explícita. Os embeddings também são normalizados, para
que similaridade de cosseno e produto interno coincidam.

Encapsular isso numa classe evita o erro mais provável da camada: aplicar o
prefixo na ingestão e esquecer na consulta (ou vice-versa), o que produz uma
busca silenciosamente ruim, sem erro nenhum.

## Precisão

fp16 é o padrão em GPU. Medido nesta máquina: 25,6 -> 149,9 chunks/s (5,9x),
com similaridade de cosseno mínima de 0,99999 contra os vetores em fp32 numa
amostra de 50 chunks. A diferença é numericamente irrelevante para ranking e
derruba a ingestão completa de ~18 min para ~3 min.
"""

from __future__ import annotations

from typing import Final

from langchain_core.embeddings import Embeddings
from langchain_huggingface import HuggingFaceEmbeddings

MODELO_PADRAO: Final[str] = "intfloat/multilingual-e5-base"
PREFIXO_PASSAGEM: Final[str] = "passage: "
PREFIXO_CONSULTA: Final[str] = "query: "

# Medido na MX570 (4,3 GB): a vazão fica em ~26 chunks/s para batch 32, 64 ou
# 128 — o gargalo é a GPU, não o lote. 64 é o ponto onde a VRAM ainda sobra.
BATCH_SIZE_PADRAO: Final[int] = 64


def _detectar_dispositivo() -> str:
    try:
        import torch
    except ImportError:  # pragma: no cover - torch é dependência dura
        return "cpu"
    return "cuda" if torch.cuda.is_available() else "cpu"


class EmbeddingsE5(Embeddings):
    """Envelopa o E5 aplicando os prefixos corretos em cada lado da busca.

    Implementa a interface `Embeddings` do LangChain, então pode ser passada
    direto para o Chroma — que chamará `embed_documents` na ingestão e
    `embed_query` na consulta, garantindo o prefixo certo em cada caso.
    """

    def __init__(
        self,
        modelo: str = MODELO_PADRAO,
        dispositivo: str | None = None,
        batch_size: int = BATCH_SIZE_PADRAO,
        normalizar: bool = True,
        fp16: bool | None = None,
    ) -> None:
        self.modelo = modelo
        self.dispositivo = dispositivo or _detectar_dispositivo()
        # fp16 só em GPU: em CPU a meia precisão é emulada e fica mais lenta.
        self.fp16 = self.dispositivo.startswith("cuda") if fp16 is None else fp16

        model_kwargs: dict = {"device": self.dispositivo}
        if self.fp16:
            model_kwargs["model_kwargs"] = {"torch_dtype": "float16"}

        self._base = HuggingFaceEmbeddings(
            model_name=modelo,
            model_kwargs=model_kwargs,
            encode_kwargs={
                "normalize_embeddings": normalizar,
                "batch_size": batch_size,
            },
        )

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self._base.embed_documents([PREFIXO_PASSAGEM + t for t in texts])

    def embed_query(self, text: str) -> list[float]:
        return self._base.embed_query(PREFIXO_CONSULTA + text)

    def __repr__(self) -> str:  # pragma: no cover - conveniência de log
        return f"EmbeddingsE5(modelo={self.modelo!r}, dispositivo={self.dispositivo!r})"
