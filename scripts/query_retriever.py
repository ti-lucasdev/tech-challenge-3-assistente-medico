"""Consulta a base vetorial pela linha de comando.

Serve como exemplo executável de entrada/saída para o Integrante 3 e como
ferramenta de depuração: mostra score, fonte e o bloco de contexto exatamente
como ele chegaria à LLM.

Uso:
    .venv\\Scripts\\python.exe scripts\\query_retriever.py "What causes hypothyroidism?"
    .venv\\Scripts\\python.exe scripts\\query_retriever.py "..." --k 6 --modo hibrido
    .venv\\Scripts\\python.exe scripts\\query_retriever.py "..." --filtro source_group=9_CDC_QA

Recorte por paciente (registros clínicos sintéticos):

    .venv\\Scripts\\python.exe scripts\\query_retriever.py "exames pendentes" --filtro patient_id=PAC-0007
    .venv\\Scripts\\python.exe scripts\\query_retriever.py "..." --filtro document_type=faq_medica
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.rag.retriever import MedicalRetriever  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("consulta")
    parser.add_argument("--k", type=int, default=4)
    parser.add_argument("--modo", choices=["denso", "bm25", "hibrido"], default="denso")
    parser.add_argument(
        "--filtro", action="append", default=[], metavar="CAMPO=VALOR",
        help="repetível; ex.: --filtro language=en --filtro source_group=9_CDC_QA",
    )
    parser.add_argument("--max-tokens", type=int, default=1200)
    parser.add_argument("--sem-contexto", action="store_true",
                        help="não imprime o bloco formatado para a LLM")
    args = parser.parse_args()

    filtros = {}
    for item in args.filtro:
        if "=" not in item:
            print(f"ERRO: filtro inválido {item!r}; use CAMPO=VALOR", file=sys.stderr)
            return 2
        campo, valor = item.split("=", 1)
        filtros[campo] = valor

    try:
        retriever = MedicalRetriever(modo=args.modo)
    except RuntimeError as erro:
        print(f"ERRO: {erro}", file=sys.stderr)
        return 1

    resultados = retriever.retrieve(args.consulta, k=args.k, filters=filtros or None)

    print(f"\nConsulta: {args.consulta!r}   (modo={args.modo}, k={args.k})")
    if filtros:
        print(f"Filtros:  {filtros}")

    suficiente = retriever.tem_evidencia_suficiente(resultados)
    print(f"Evidência suficiente: {'SIM' if suficiente else 'NÃO — responder que falta suporte'}")
    print()

    for r in resultados:
        m = r.metadados
        print(f"  [{r.rank}] score={r.score:.4f}  {m['source_group']}  ({m['page_or_section']})")
        print(f"      {m['title']}")
        print(f"      {m['institution']}")
        print(f"      {r.texto[:160].replace(chr(10), ' ')}...")
        print()

    if not args.sem_contexto:
        print("-" * 70)
        print("Contexto que seria enviado à LLM:")
        print("-" * 70)
        print(retriever.format_context(resultados, max_tokens=args.max_tokens))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
