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

PADROES_PRESCRICAO = [
    r"\d+\s*(-\s*\d+\s*)?\s*(mg|ml|mcg|g|iu|units?)\b",
    r"\bdose\b.{0,30}\d",
    r"\btome\s+\d",
    r"\btake\s+\d",
    r"\badministre\s+\d",
]

def contem_prescricao_direta(texto: str) -> bool:
    texto_lower = texto.lower()
    return any(re.search(p, texto_lower) for p in PADROES_PRESCRICAO)

def aplicar_guardrail(resposta: str) -> dict:
    bloqueado = contem_prescricao_direta(resposta)
    if bloqueado:
        resposta_final = (
            "Não posso fornecer instruções de dosagem ou prescrição direta. "
            "Por favor, consulte um profissional de saúde." + AVISO_PADRAO
        )
    else:
        resposta_final = resposta.strip() + AVISO_PADRAO
    return {"resposta_final": resposta_final, "bloqueado": bloqueado}
