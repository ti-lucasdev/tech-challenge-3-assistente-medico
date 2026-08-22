"""Mede a qualidade da recuperação e calibra o limiar de abstenção.

Compara os três modos de busca sobre o mesmo índice e o mesmo conjunto de
sondas parafraseadas, e depois usa os casos negativos para escolher o limiar
que separa "há evidência" de "não há".

Uso:
    .venv\\Scripts\\python.exe scripts\\avaliar_retriever.py
    .venv\\Scripts\\python.exe scripts\\avaliar_retriever.py --casos 300
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.rag.evaluation import (  # noqa: E402
    NEGATIVOS,
    calcular_metricas,
    construir_casos,
)
from src.rag.loaders import carregar_medquad  # noqa: E402
from src.rag.retriever import MedicalRetriever  # noqa: E402

MODOS = ("denso", "bm25", "hibrido")
KS = (1, 3, 5, 10)
CAMINHO_SAIDA = Path(__file__).resolve().parent.parent / "data" / "avaliacao_retriever.json"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--casos", type=int, default=200)
    parser.add_argument("--k", type=int, default=10, help="profundidade de recuperação")
    parser.add_argument("--semente", type=int, default=42)
    parser.add_argument(
        "--modos", nargs="+", choices=MODOS, default=list(MODOS),
        help="modos a avaliar; use apenas 'denso' para recalibrar rápido",
    )
    args = parser.parse_args()

    print("Montando conjunto de avaliação...")
    docs, _ = carregar_medquad()
    casos = construir_casos(docs, quantidade=args.casos, semente=args.semente)
    print(f"  {len(casos)} sondas parafraseadas + {len(NEGATIVOS)} negativos\n")

    resultados_por_modo: dict[str, dict] = {}

    for modo in args.modos:
        print(f"[{modo}] recuperando...", end=" ", flush=True)
        retriever = MedicalRetriever(modo=modo)
        inicio = time.time()

        recuperados = []
        scores_topo = []
        for caso in casos:
            achados = retriever.retrieve(caso.consulta, k=args.k)
            recuperados.append([r.document_id for r in achados])
            scores_topo.append(achados[0].score if achados else 0.0)

        scores_negativos = []
        for negativo in NEGATIVOS:
            achados = retriever.retrieve(negativo.consulta, k=args.k)
            scores_negativos.append(achados[0].score if achados else 0.0)

        duracao = time.time() - inicio
        metricas = calcular_metricas(recuperados, casos, ks=KS)
        por_consulta = duracao / (len(casos) + len(NEGATIVOS))
        print(f"{metricas}   ({por_consulta*1000:.0f} ms/consulta)")

        resultados_por_modo[modo] = {
            "metricas": {
                "n": metricas.n,
                "recall": {str(k): round(v, 4) for k, v in metricas.recall_por_k.items()},
                "mrr": round(metricas.mrr, 4),
            },
            "ms_por_consulta": round(por_consulta * 1000, 1),
            "scores_positivos": scores_topo,
            "scores_negativos": scores_negativos,
        }
        del retriever

    _tabela(resultados_por_modo)
    limiar = _calibrar_limiar(resultados_por_modo)

    CAMINHO_SAIDA.write_text(
        json.dumps(
            {
                "configuracao": {
                    "casos": len(casos),
                    "negativos": len(NEGATIVOS),
                    "k": args.k,
                    "semente": args.semente,
                },
                "modos": {
                    m: {c: v for c, v in d.items() if not c.startswith("scores_")}
                    for m, d in resultados_por_modo.items()
                },
                "limiar_sugerido": limiar,
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"\nResultados salvos em {CAMINHO_SAIDA.relative_to(Path.cwd())}")
    return 0


def _tabela(resultados: dict[str, dict]) -> None:
    print("\n" + "=" * 72)
    print(f"{'modo':10} {'R@1':>8} {'R@3':>8} {'R@5':>8} {'R@10':>8} {'MRR':>8} {'ms/q':>8}")
    print("-" * 72)
    for modo, dados in resultados.items():
        r = dados["metricas"]["recall"]
        print(
            f"{modo:10} {r['1']:>8.3f} {r['3']:>8.3f} {r['5']:>8.3f} "
            f"{r['10']:>8.3f} {dados['metricas']['mrr']:>8.3f} "
            f"{dados['ms_por_consulta']:>8.0f}"
        )
    print("=" * 72)


def _calibrar_limiar(resultados: dict[str, dict]) -> dict:
    """Escolhe o limiar de abstenção a partir da separação positivos × negativos.

    Não existe limiar universal de cosseno: a escala depende do modelo e do
    corpus. O que se mede aqui é a sobreposição entre as duas distribuições de
    score do topo-1. Quanto menor a sobreposição, mais confiável a abstenção.
    """
    print("\nCalibração do limiar de abstenção (score do topo-1):")
    saida = {}
    for modo, dados in resultados.items():
        positivos = sorted(dados["scores_positivos"])
        negativos = sorted(dados["scores_negativos"])
        if not positivos or not negativos:
            continue

        p05 = positivos[max(0, int(len(positivos) * 0.05) - 1)]
        neg_max = negativos[-1]
        neg_p50 = negativos[len(negativos) // 2]
        # Ponto médio entre o pior positivo típico e o pior negativo: separa as
        # duas distribuições sem colar em nenhum dos extremos.
        sugerido = round((p05 + neg_max) / 2, 3)

        # As duas taxas que importam na prática. Num assistente clínico, errar
        # para o lado da abstenção é preferível a responder sem suporte — mas
        # abstenção falsa alta torna o sistema inútil, então o número precisa
        # estar à vista, não escondido atrás de "separável: sim".
        abstencao_falsa = sum(1 for s in positivos if s < sugerido) / len(positivos)
        negativo_aceito = sum(1 for s in negativos if s >= sugerido) / len(negativos)
        margem = p05 - neg_max

        print(
            f"  {modo:10} positivos p05={p05:.3f}  |  negativos mediana={neg_p50:.3f} "
            f"máx={neg_max:.3f}  ->  limiar {sugerido:.3f}"
        )
        print(
            f"  {'':10}   margem={margem:+.3f}  "
            f"abstenção falsa={abstencao_falsa:.1%}  negativo aceito={negativo_aceito:.1%}"
            + ("" if margem > 0 else "   [SOBREPOSIÇÃO: abstenção não é confiável]")
        )
        saida[modo] = {
            "positivos_p05": round(p05, 4),
            "positivos_mediana": round(positivos[len(positivos) // 2], 4),
            "positivos_min": round(positivos[0], 4),
            "negativos_max": round(neg_max, 4),
            "negativos_mediana": round(neg_p50, 4),
            "limiar": sugerido,
            "margem": round(margem, 4),
            "taxa_abstencao_falsa": round(abstencao_falsa, 4),
            "taxa_negativo_aceito": round(negativo_aceito, 4),
            "separavel": bool(margem > 0),
            # Distribuições completas: sem elas o limiar não é auditável nem
            # recalculável sem reexecutar a avaliação inteira.
            "scores_positivos": [round(s, 4) for s in positivos],
            "scores_negativos": [round(s, 4) for s in negativos],
        }
    return saida


if __name__ == "__main__":
    raise SystemExit(main())
