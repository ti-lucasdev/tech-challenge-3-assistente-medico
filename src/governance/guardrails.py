"""
Guardrails de segurança para respostas do assistente médico.
Bloqueia menções a dosagem/prescrição direta e garante aviso de validação humana.
"""
import re

AVISO_PADRAO = (
    "\n\n⚠️ Esta resposta foi gerada por um sistema de IA e tem caráter "
    "informativo. Não substitui avaliação, diagnóstico ou prescrição por "
    "um profissional de saúde qualificado."
)

MENSAGEM_SEM_CITACAO = (
    "Esta resposta não referenciou nenhuma das fontes recuperadas e deve ser "
    "tratada como não verificada."
)

PADRAO_CITACAO_FONTE = r"\[FONTE\s*\d+\]"

PADROES_PRESCRICAO = [
    r"\d+\s*(-\s*\d+\s*)?\s*(mg|ml|mcg|g|iu|units?|gotas?|ampolas?|comprimidos?|cp|c[aá]psulas?|colheres?)\b",
    r"\bdose\b.{0,30}\d",
    r"\btome\s+\d",
    r"\btake\s+\d",
    r"\badministre\s+\d",
    r"\binjete\s+\d",
    r"\baplique\s+\d",
    r"\bingira\s+\d",
    r"\bpingue\s+\d",
    r"\bbeba\s+\d",
]

def contem_prescricao_direta(texto: str) -> bool:
    texto_lower = texto.lower()
    return any(re.search(p, texto_lower) for p in PADROES_PRESCRICAO)

def _trecho_prescricao(texto: str) -> str | None:
    texto_lower = texto.lower()
    for p in PADROES_PRESCRICAO:
        m = re.search(p, texto_lower)
        if m:
            return m.group(0)
    return None

def resposta_cita_fonte(resposta: str) -> bool:
    return re.search(PADRAO_CITACAO_FONTE, resposta, re.IGNORECASE) is not None

def aplicar_guardrail(resposta: str, contexto: str = "", fontes: list[str] | None = None) -> dict:
    trecho = _trecho_prescricao(resposta)
    bloqueado = trecho is not None and trecho not in contexto.lower()
    if bloqueado:
        resposta_final = (
            "Não posso fornecer instruções de dosagem ou prescrição direta. "
            "Por favor, consulte um profissional de saúde." + AVISO_PADRAO
        )
    else:
        resposta_final = resposta.strip() + AVISO_PADRAO

    sem_citacao = False
    if fontes and not bloqueado and not resposta_cita_fonte(resposta):
        sem_citacao = True
        resposta_final = MENSAGEM_SEM_CITACAO + AVISO_PADRAO

    return {"resposta_final": resposta_final, "bloqueado": bloqueado, "sem_citacao": sem_citacao}
