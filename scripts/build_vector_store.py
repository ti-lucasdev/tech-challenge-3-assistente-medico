"""Constrói (ou atualiza) a base vetorial do assistente médico.

Uso:
    .venv\\Scripts\\python.exe scripts\\build_vector_store.py
    .venv\\Scripts\\python.exe scripts\\build_vector_store.py --limite 500
    .venv\\Scripts\\python.exe scripts\\build_vector_store.py --recriar

Rodar duas vezes seguidas sem `--recriar` é seguro e não duplica registros: os
IDs derivam do conteúdo, então a segunda execução é um upsert sobre os mesmos
chunks. Ver `src/rag/ingest.py` para o que acontece quando a *configuração* muda.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.rag.ingest import (  # noqa: E402
    COLECAO_PADRAO,
    DIRETORIO_VECTORSTORE,
    ConfiguracaoIngestao,
    construir_indice,
)
from src.rag.preprocess import LIMITE_CHARS_CHUNK  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--limite", type=int, default=None,
        help="indexa apenas os N primeiros documentos (desenvolvimento)",
    )
    parser.add_argument(
        "--recriar", action="store_true",
        help="apaga a coleção existente antes de indexar",
    )
    parser.add_argument(
        "--sem-prontuarios", action="store_true",
        help="indexa só o MedQuAD, sem os 30 registros clínicos sintéticos",
    )
    parser.add_argument("--colecao", default=COLECAO_PADRAO)
    parser.add_argument("--chunk-chars", type=int, default=LIMITE_CHARS_CHUNK)
    parser.add_argument("--persist-dir", default=str(DIRETORIO_VECTORSTORE))
    args = parser.parse_args()

    config = ConfiguracaoIngestao(
        colecao=args.colecao,
        limite_chars_chunk=args.chunk_chars,
    )

    inicio = time.time()
    try:
        _, resumo = construir_indice(
            config=config,
            limite=args.limite,
            persist_directory=args.persist_dir,
            recriar=args.recriar,
            incluir_prontuarios=not args.sem_prontuarios,
        )
    except RuntimeError as erro:
        print(f"\nERRO: {erro}", file=sys.stderr)
        return 1

    duracao = time.time() - inicio
    print()
    print(f"Coleção .............. {resumo['colecao']}")
    print(f"Documentos ........... {resumo['documentos']}")
    print(f"  dos quais prontuários {resumo['prontuarios']}")
    print(f"Chunks ............... {resumo['chunks']}")
    print(f"Registros no índice .. {resumo['registros_no_indice']}")
    print(f"Persistido em ........ {Path(args.persist_dir).resolve()}")
    print(f"Tempo ................ {duracao/60:.1f} min")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
