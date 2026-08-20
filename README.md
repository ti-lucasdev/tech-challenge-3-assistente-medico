# Tech Challenge 3 — Assistente Médico

## Instalação

Requisitos: **Python 3.11** e uma GPU NVIDIA com **CUDA 12.8** (as dependências incluem builds `+cu128` de PyTorch, sem fallback para CPU).

1. Crie e ative o ambiente virtual:

   ```powershell
   py -3.11 -m venv .venv
   .venv\Scripts\activate
   ```

2. Instale as dependências. Como `torch`, `torchvision` e `torchaudio` são fixados em builds `+cu128`, que não existem no PyPI público, é necessário apontar para o índice de wheels da PyTorch:

   ```powershell
   pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu128
   ```

   Sem o `--extra-index-url`, o `pip install -r requirements.txt` falha com `Could not find a version that satisfies the requirement torch==2.11.0+cu128`.

## Adaptador LoRA treinado

O adaptador foi gerado pelo notebook [`notebooks/Techchallenge3_executado_final.ipynb`](notebooks/Techchallenge3_executado_final.ipynb), executado no **Google Colab**. O notebook carrega o modelo base `unsloth/llama-3-8b-Instruct-bnb-4bit` em 4-bit via Unsloth, treina um adaptador LoRA (r=16, alpha=16, módulos de atenção e MLP) sobre o dataset `mukulb/clustered_MEDQUAD_dataset_with_groups` usando `trl.SFTTrainer`, e salva o resultado no Google Drive em `/content/drive/MyDrive/TechChallenge3/adaptador_medquad_lora_final`. Esse diretório foi então compactado e disponibilizado para download abaixo — é o mesmo artifact esperado por `src/inferencia.py`.

Download do adaptador LoRA/QLoRA:

[Baixar adaptador_medquad_lora_final.zip](https://drive.google.com/file/d/1Zf67WsQ4_4UAsq93_5KPJmTLr8y8qx1L/view?usp=sharing)

Após baixar, extraia a pasta `adaptador_medquad_lora_final` dentro de:

```text
artifacts/
```

## Executando a inferência local

Com o ambiente instalado e o adaptador extraído em `artifacts/adaptador_medquad_lora_final`:

```powershell
.venv\Scripts\python.exe src\inferencia.py
```
