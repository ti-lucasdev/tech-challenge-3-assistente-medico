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
