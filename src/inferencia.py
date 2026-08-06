from pathlib import Path

import torch
from unsloth import FastLanguageModel

CAMINHO_ADAPTADOR = Path("artifacts/adaptador_medquad_lora_final")

if not CAMINHO_ADAPTADOR.exists():
    raise FileNotFoundError(f"Adaptador não encontrado: {CAMINHO_ADAPTADOR}")

print("Adaptador localizado:", CAMINHO_ADAPTADOR.resolve())
print("GPU disponível:", torch.cuda.is_available())

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=str(CAMINHO_ADAPTADOR),
    max_seq_length=2048,
    dtype=None,
    load_in_4bit=True,
)

print("Modelo-base e adaptador LoRA carregados.")

FastLanguageModel.for_inference(model)

prompt_style = """ abaixo está uma instrução que descreve uma tarefa, juntamente com uma entrada que fornece contexto adicional. Escreva uma resposta que complete adequadamente o pedido.

### Instrução:
{}

### Entrada:
{}

### Resposta:
{}"""

pergunta = "When should a patient with chest pain seek emergency care?"

prompt = prompt_style.format(
    "Responda à pergunta médica com base em informações clínicas confiáveis.",
    pergunta,
    "",
)

inputs = tokenizer([prompt], return_tensors="pt").to("cuda")

outputs = model.generate(
    **inputs,
    max_new_tokens=250,
    do_sample=False,
    repetition_penalty=1.1,
)

resposta = tokenizer.decode(
    outputs[0][inputs["input_ids"].shape[1]:],
    skip_special_tokens=True,
)

print("\nResposta do modelo:\n")
print(resposta)