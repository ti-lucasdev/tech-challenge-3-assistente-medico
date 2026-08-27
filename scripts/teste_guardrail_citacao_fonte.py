"""
Teste rápido (sem GPU, sem carregar modelo) do guardrail de citação de fonte:
verifica que uma resposta com fontes disponíveis mas sem citar [FONTE n] é
marcada como não verificada, que citar a fonte evita isso, e que sem a lista
de fontes o comportamento antigo é preservado.
"""
from src.governance.guardrails import aplicar_guardrail, resposta_cita_fonte

fontes = ["FAQ MedQuAD — What is hypothyroidism?", "Prontuário PAC-0007"]

# Caso (a): resposta cita [FONTE 1] e fontes não vazias -> sem_citacao=False
resposta_com_citacao = "O hipotireoidismo é tratado com reposição hormonal [FONTE 1]."
assert resposta_cita_fonte(resposta_com_citacao) is True
resultado_a = aplicar_guardrail(resposta_com_citacao, fontes=fontes)
assert resultado_a["sem_citacao"] is False, resultado_a
assert resultado_a["bloqueado"] is False, resultado_a
print("OK (a): resposta cita [FONTE 1] com fontes disponíveis -> sem_citacao=False.")
print(f"  resposta_final: {resultado_a['resposta_final']!r}\n")

# Caso (b): resposta não cita nenhuma fonte, mas fontes não vazias -> sem_citacao=True
resposta_sem_citacao = "O hipotireoidismo é tratado com reposição hormonal."
assert resposta_cita_fonte(resposta_sem_citacao) is False
resultado_b = aplicar_guardrail(resposta_sem_citacao, fontes=fontes)
assert resultado_b["sem_citacao"] is True, resultado_b
assert resultado_b["bloqueado"] is False, resultado_b
assert "não referenciou nenhuma das fontes" in resultado_b["resposta_final"]
print("OK (b): resposta sem citação, com fontes disponíveis -> sem_citacao=True, resposta_final substituída.")
print(f"  resposta_final: {resultado_b['resposta_final']!r}\n")

# Caso (c): fontes=None ou [] -> comportamento antigo preservado, independente de citação
resultado_c1 = aplicar_guardrail(resposta_sem_citacao, fontes=None)
assert resultado_c1["sem_citacao"] is False, resultado_c1
assert resultado_c1["resposta_final"] == resposta_sem_citacao.strip() + "\n\n⚠️ Esta resposta foi gerada por um sistema de IA e tem caráter informativo. Não substitui avaliação, diagnóstico ou prescrição por um profissional de saúde qualificado."

resultado_c2 = aplicar_guardrail(resposta_sem_citacao, fontes=[])
assert resultado_c2["sem_citacao"] is False, resultado_c2
assert resultado_c2["resposta_final"] == resultado_c1["resposta_final"]

resultado_c3 = aplicar_guardrail(resposta_sem_citacao)  # nem contexto nem fontes passados
assert resultado_c3["sem_citacao"] is False, resultado_c3
assert resultado_c3["resposta_final"] == resultado_c1["resposta_final"]

print("OK (c): fontes=None, fontes=[] e chamada sem o parâmetro -> comportamento antigo preservado (sem_citacao=False).")

print("\nTodos os testes passaram.")
