"""
Teste rápido (sem GPU, sem carregar modelo) do guardrail com contexto:
verifica que dosagem citada do contexto recuperado não é bloqueada,
mas dosagem "inventada" pelo modelo (fora do contexto) continua sendo.
"""
from src.governance.guardrails import aplicar_guardrail

contexto = "prontuário: paciente em uso de metformina 850 mg"

# Caso 1: resposta cita a mesma dosagem do contexto -> não deve bloquear
resposta_citacao = "Segundo o prontuário, o paciente está em uso de metformina 850 mg."
resultado_citacao = aplicar_guardrail(resposta_citacao, contexto)
assert resultado_citacao["bloqueado"] is False, "citação de dosagem do contexto foi bloqueada indevidamente"
print("OK: dosagem citada do contexto NÃO foi bloqueada.")
print(f"  resposta_final: {resultado_citacao['resposta_final']!r}\n")

# Caso 2: resposta traz dosagem que não está no contexto -> deve bloquear
resposta_inventada = "Tome metformina 2000 mg por dia."
resultado_inventado = aplicar_guardrail(resposta_inventada, contexto)
assert resultado_inventado["bloqueado"] is True, "dosagem fora do contexto não foi bloqueada"
print("OK: dosagem fora do contexto FOI bloqueada.")
print(f"  resposta_final: {resultado_inventado['resposta_final']!r}\n")

# Caso 3 (regressão): sem contexto, comportamento antigo é preservado (bloqueia)
resultado_sem_contexto = aplicar_guardrail(resposta_citacao)
assert resultado_sem_contexto["bloqueado"] is True, "sem contexto, dosagem deveria continuar bloqueada"
print("OK: sem contexto (comportamento antigo), a mesma resposta É bloqueada.")

print("\nTodos os testes passaram.")
