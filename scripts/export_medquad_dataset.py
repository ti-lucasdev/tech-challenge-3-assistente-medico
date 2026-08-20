"""Exporta para /data o exato subconjunto do MedQuAD usado no fine-tuning do Integrante 1.

Reproduz a seleção feita em `notebooks/Techchallenge3_executado_final.ipynb`
(dataset bruto -> primeiras 1000 linhas + linha 11718) e salva três arquivos:

- data/medquad_raw.jsonl: as linhas selecionadas, com as colunas originais do
  dataset (antes de qualquer remapeamento).
- data/medquad_processed.jsonl: as mesmas linhas já no formato
  instruction/input/output usado para treinar o adaptador LoRA.
- data/medquad_rag_pool.jsonl: todas as demais linhas do dataset (o
  complemento das linhas de treino), excluindo também qualquer linha cuja
  "query" seja idêntica a uma pergunta de treino — o dataset tem perguntas
  repetidas em índices diferentes (respondidas de forma diferente), então
  excluir só por índice não bastava. Candidatas seguras para compor o
  corpus/golden-set do RAG, sem contaminação com o que o modelo já viu.

Existe para satisfazer o item 2.4 do brief ("Datasets utilizados... com
indicação clara no relatório") e para dar rastreabilidade ao dataset de
fine-tuning, que hoje só existe embutido no notebook.
"""

import json
from pathlib import Path

from datasets import concatenate_datasets, load_dataset

DATASET_NAME = "mukulb/clustered_MEDQUAD_dataset_with_groups"
DATA_DIR = Path(__file__).resolve().parent.parent / "data"
INSTRUCTION = "Responda à pergunta médica com base em informações clínicas confiáveis."


def salvar_jsonl(dataset, caminho: Path) -> None:
    with caminho.open("w", encoding="utf-8") as f:
        for exemplo in dataset:
            f.write(json.dumps(exemplo, ensure_ascii=False) + "\n")


def main() -> None:
    DATA_DIR.mkdir(exist_ok=True)

    dataset_bruto = load_dataset(DATASET_NAME, split="train")

    indices_treino = set(range(1000)) | {11718}
    dataset_selecionado = concatenate_datasets([
        dataset_bruto.select(range(1000)),
        dataset_bruto.select([11718]),
    ])

    salvar_jsonl(dataset_selecionado, DATA_DIR / "medquad_raw.jsonl")

    dataset_processado = dataset_selecionado.map(
        lambda exemplo: {
            "instruction": INSTRUCTION,
            "input": exemplo["query"],
            "output": exemplo["answers"],
        },
        remove_columns=dataset_selecionado.column_names,
    )
    salvar_jsonl(dataset_processado, DATA_DIR / "medquad_processed.jsonl")

    queries_treino = set(dataset_selecionado["query"])
    indices_pool = [
        i
        for i in range(len(dataset_bruto))
        if i not in indices_treino and dataset_bruto[i]["query"] not in queries_treino
    ]
    dataset_pool = dataset_bruto.select(indices_pool)
    salvar_jsonl(dataset_pool, DATA_DIR / "medquad_rag_pool.jsonl")

    print(f"Linhas de treino exportadas: {len(dataset_selecionado)}")
    print(f"Linhas do pool (sem contaminação) exportadas: {len(dataset_pool)}")
    print(f"Colunas originais: {dataset_selecionado.column_names}")
    print(f"Salvo em: {DATA_DIR / 'medquad_raw.jsonl'}")
    print(f"Salvo em: {DATA_DIR / 'medquad_processed.jsonl'}")
    print(f"Salvo em: {DATA_DIR / 'medquad_rag_pool.jsonl'}")


if __name__ == "__main__":
    main()
