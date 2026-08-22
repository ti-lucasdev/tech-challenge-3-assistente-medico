"""Confere se o ambiente atende às duas metades do projeto.

O assistente precisa, no mesmo processo, recuperar evidência (LangChain +
Chroma + sentence-transformers) e gerar resposta (unsloth + adaptador LoRA).
Este script verifica as duas pilhas separadamente e diz qual delas falhou —
distinção que importa, porque as causas e os responsáveis são diferentes.

Uso:
    .venv\\Scripts\\python.exe scripts\\verificar_ambiente.py
    .venv\\Scripts\\python.exe scripts\\verificar_ambiente.py --pular-llm
"""

from __future__ import annotations

import argparse
import sys
import traceback
from pathlib import Path

RAIZ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ))

CAMINHO_ADAPTADOR = RAIZ / "artifacts" / "adaptador_medquad_lora_final"

OK = "  [OK]  "
FALHA = "  [FALHA]"
AVISO = "  [AVISO]"


def secao(titulo: str) -> None:
    print(f"\n{titulo}\n{'-' * len(titulo)}")


def verificar_base() -> bool:
    secao("Plataforma")
    print(f"{OK} Python {sys.version.split()[0]}")
    try:
        import torch
    except Exception:
        print(f"{FALHA} torch não importa")
        traceback.print_exc()
        return False

    print(f"{OK} torch {torch.__version__}")
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        vram = props.total_memory / 1e9
        print(f"{OK} CUDA disponível — {props.name} ({vram:.1f} GB)")
        if vram < 6.0:
            print(
                f"{AVISO} {vram:.1f} GB de VRAM. O Llama-3-8B em 4-bit precisa de "
                "~5,5 GB só de pesos; a geração pode falhar por falta de memória "
                "nesta máquina, independentemente das versões instaladas."
            )
    else:
        print(f"{AVISO} CUDA indisponível — a ingestão roda em CPU (~10x mais lenta)")
    return True


def verificar_rag(consulta: str) -> bool:
    secao("Camada de recuperação (Integrante 2)")
    try:
        import chromadb  # noqa: F401
        import sentence_transformers  # noqa: F401
        from langchain_chroma import Chroma  # noqa: F401
        from langchain_core.documents import Document  # noqa: F401
    except Exception:
        print(f"{FALHA} imports do stack de RAG")
        traceback.print_exc()
        return False
    print(f"{OK} imports (langchain-core, langchain-chroma, chromadb, sentence-transformers)")

    try:
        from src.rag.retriever import MedicalRetriever
    except Exception:
        print(f"{FALHA} import de src.rag.retriever")
        traceback.print_exc()
        return False

    try:
        retriever = MedicalRetriever()
    except RuntimeError as erro:
        print(f"{AVISO} base vetorial indisponível: {erro}")
        print(f"{AVISO} rode `python scripts/build_vector_store.py` — o stack em si está sadio")
        return True
    except Exception:
        print(f"{FALHA} não foi possível abrir a base vetorial")
        traceback.print_exc()
        return False

    try:
        resultados = retriever.retrieve(consulta, k=3)
    except Exception:
        print(f"{FALHA} consulta ao retriever")
        traceback.print_exc()
        return False

    if not resultados:
        print(f"{FALHA} a consulta não devolveu nenhum resultado")
        return False

    print(f"{OK} consulta devolveu {len(resultados)} resultados")
    print(f"         top-1 score={resultados[0].score:.4f}")
    print(f"         {resultados[0].metadados['title'][:66]}")
    return True


def verificar_llm() -> bool:
    secao("Camada de geração (Integrante 1)")
    try:
        from unsloth import FastLanguageModel
    except Exception:
        print(f"{FALHA} import do unsloth")
        print("         Esta é a incompatibilidade mais provável numa troca de")
        print("         versão do Python: o unsloth reescreve internas do")
        print("         transformers em tempo de importação.")
        traceback.print_exc()
        return False
    print(f"{OK} import do unsloth")

    if not CAMINHO_ADAPTADOR.exists():
        print(f"{AVISO} adaptador ausente em {CAMINHO_ADAPTADOR.relative_to(RAIZ)}")
        print(f"{AVISO} baixe conforme o README — o stack em si está sadio")
        return True

    try:
        modelo, _ = FastLanguageModel.from_pretrained(
            model_name=str(CAMINHO_ADAPTADOR),
            max_seq_length=2048,
            dtype=None,
            load_in_4bit=True,
        )
    except Exception as erro:
        texto = f"{type(erro).__name__}: {erro}".lower()
        # O bitsandbytes não levanta "out of memory" quando o modelo não cabe:
        # ele tenta distribuir camadas para CPU/disco e recusa. A mensagem real
        # é "Some modules are dispatched on the CPU or the disk".
        sintomas_vram = (
            "out of memory",
            "cuda oom",
            "enough gpu ram",
            "dispatched on the cpu",
        )
        if any(s in texto for s in sintomas_vram):
            print(f"{AVISO} o modelo não cabe na VRAM desta GPU.")
            print("         Limite de hardware, NÃO incompatibilidade de versão —")
            print("         o mesmo erro ocorre em qualquer versão de Python.")
            print("         Para gerar respostas, use uma GPU com 6 GB+ ou o Colab.")
            return True
        print(f"{FALHA} carregamento do modelo base + adaptador")
        traceback.print_exc()
        return False

    print(f"{OK} modelo base + adaptador LoRA carregados")
    del modelo
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pular-llm", action="store_true")
    parser.add_argument(
        "--consulta", default="Which clinical manifestations suggest an underactive thyroid?"
    )
    args = parser.parse_args()

    resultados = {"plataforma": verificar_base()}
    resultados["rag"] = verificar_rag(args.consulta)
    if not args.pular_llm:
        resultados["llm"] = verificar_llm()

    secao("Resumo")
    for nome, ok in resultados.items():
        print(f"  {nome:14} {'OK' if ok else 'FALHOU'}")

    return 0 if all(resultados.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
