# Dataset de fine-tuning

Cópia local do subconjunto do [MedQuAD](https://huggingface.co/datasets/mukulb/clustered_MEDQUAD_dataset_with_groups)
usado pelo Integrante 1 para treinar o adaptador LoRA em
`notebooks/Techchallenge3_executado_final.ipynb` (célula "Carregando Dataset
bruto de treinamento"). Gerado por `scripts/export_medquad_dataset.py`, que
reproduz exatamente a seleção do notebook: as primeiras 1000 linhas do split
`train` + a linha de índice 11718 (1001 linhas no total, de um total de
16407 no dataset original).

## Arquivos

- `medquad_raw.jsonl` — linhas selecionadas com as colunas originais do
  dataset (`text`, `query`, `answers`, `topic_embeddings`, `group_name`).
- `medquad_processed.jsonl` — as mesmas linhas já remapeadas para o formato
  `instruction`/`input`/`output` usado no template Alpaca de treino (campo
  `input` = `query`, campo `output` = `answers`).
- `medquad_rag_pool.jsonl` — as 15.272 linhas restantes do dataset (16407 −
  1001 de treino − 134 removidas por duplicação, ver abaixo), sem
  sobreposição com o treino. Base segura para montar o corpus/golden-set do
  RAG.

Para regenerar (ex.: se o dataset upstream mudar de versão):

```powershell
.venv\Scripts\python.exe scripts\export_medquad_dataset.py
```

## Nota para o RAG (Integrante 2)

Estas 1001 linhas de treino já foram vistas pelo modelo durante o
fine-tuning. Não usá-las como corpus/golden-set de avaliação do
retriever — o modelo acertaria por memorização, não por recuperação, o que
inflaciona artificialmente as métricas de Recall@k/MRR. Usar
`medquad_rag_pool.jsonl` como fonte para o corpus vetorial, ou documentos
sintéticos novos.

**Cuidado com duplicatas por texto, não só por índice**: o dataset
`clustered_MEDQUAD_dataset_with_groups` tem a mesma pergunta (`query`)
repetida em índices diferentes, com respostas diferentes (ex.: "What is
(are) Gallbladder Cancer ?" aparece mais de uma vez, vinda de fontes
distintas). Excluir apenas os índices de treino não bastava — 57 perguntas
de treino reapareciam em outras 134 linhas do dataset. `medquad_rag_pool.jsonl`
já exclui essas linhas por comparação de texto de `query`, não só por
índice. Se o corpus for expandido a partir do dataset bruto por outro
caminho (não via este script), refazer essa checagem de texto — comparar só
os índices não garante ausência de contaminação.
