"""
Wrapper de inferência com guardrails e logging (Parte 4).
"""
from unsloth import FastLanguageModel
from pathlib import Path
import sys


sys.path.append(str(Path(__file__).resolve().parents[2]))
from src.governance.guardrails import aplicar_guardrail
from src.logging.logger import registrar_interacao

CAMINHO_ADAPTADOR = Path("artifacts/adaptador_medquad_lora_final")

PROMPT_STYLE = """abaixo está uma instrução que descreve uma tarefa, juntamente com uma entrada que fornece contexto adicional. Escreva uma resposta que complete adequadamente o pedido.

### Instrução:
{}

### Entrada:
{}

### Resposta:
{}"""

_model = None
_tokenizer = None

def carregar_modelo():
    global _model, _tokenizer
    if _model is not None:
        return _model, _tokenizer
    if not CAMINHO_ADAPTADOR.exists():
        raise FileNotFoundError(f"Adaptador não encontrado: {CAMINHO_ADAPTADOR}")
    _model, _tokenizer = FastLanguageModel.from_pretrained(
        model_name=str(CAMINHO_ADAPTADOR),
        max_seq_length=2048,
        dtype=None,
        load_in_4bit=True,
    )
    FastLanguageModel.for_inference(_model)
    return _model, _tokenizer

def gerar_resposta_segura(pergunta: str, contexto: str = "") -> str:
    model, tokenizer = carregar_modelo()
    prompt = PROMPT_STYLE.format(
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
    resposta_bruta = tokenizer.decode(
        outputs[0][inputs["input_ids"].shape[1]:],
        skip_special_tokens=True,
    ).strip()

    # TODO: repassar contexto/fontes recuperados pelo RAG (Parte 3, src/orchestration/) assim
    # que este wrapper passar a consumi-los de verdade, em vez de sempre None.
    resultado = aplicar_guardrail(resposta_bruta, contexto, fontes=None)
    registrar_interacao(
        pergunta=pergunta,
        resposta_bruta=resposta_bruta,
        resposta_final=resultado["resposta_final"],
        bloqueado=resultado["bloqueado"],
        fontes=None,
    )
    return resultado["resposta_final"]


if __name__ == "__main__":
    pergunta = "When should a patient with chest pain seek emergency care?"
    print("\nResposta segura do modelo:\n")
    print(gerar_resposta_segura(pergunta))
