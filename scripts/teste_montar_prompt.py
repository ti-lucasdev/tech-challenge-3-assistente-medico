"""
Teste rápido (sem GPU, sem carregar modelo) de _montar_prompt() em inferencia_segura.py:
confirma que, com contexto, o formato bate exatamente com o do nó de geração da
Camada 3 (src/orchestration/nodes.py::no_gerar_resposta), e que sem contexto o
PROMPT_STYLE (Alpaca) original é preservado sem mudanças.

Não importa nodes.py diretamente: seu import de topo carrega o modelo real
(via src.inferencia) e exige GPU. Por isso o formato esperado é replicado aqui
como string literal — o mesmo texto que está hoje em nodes.py:142-161 — para
comparação byte a byte.

Esta máquina não tem unsloth instalado (sem GPU, ver CLAUDE.md), e
inferencia_segura.py faz "from unsloth import FastLanguageModel" no topo do
arquivo. Como este teste só precisa de _montar_prompt() (que não usa unsloth
em nenhuma linha), colocamos um stub mínimo em sys.modules antes do import
para viabilizá-lo sem instalar a lib nem carregar modelo nenhum.
"""
import sys
import types

if "unsloth" not in sys.modules:
    _stub_unsloth = types.ModuleType("unsloth")
    _stub_unsloth.FastLanguageModel = object
    sys.modules["unsloth"] = _stub_unsloth

from src.governance.inferencia_segura import _montar_prompt, PROMPT_STYLE

pergunta = "Quais exames estão pendentes para o PAC-0007?"
contexto = "Prontuário:\nPaciente com diabetes descompensado.\n\nLiteratura/FAQ:\nControle glicêmico é essencial."

# --- Caso 1: com contexto -> deve bater exatamente com o formato de nodes.py ---
prompt_com_contexto = _montar_prompt(pergunta, contexto)

# Réplica literal do formato em src/orchestration/nodes.py::no_gerar_resposta (linhas ~155-161)
prompt_esperado_nodes_py = (
    f"Abaixo está um contexto clínico. Com base nele, responda à pergunta de forma direta, "
    f"objetiva e em português. Não repita instruções.\n\n"
    f"Contexto:\n{contexto}\n\n"
    f"Pergunta: {pergunta}\n\n"
    f"Resposta:"
)

assert prompt_com_contexto == prompt_esperado_nodes_py, (
    "prompt com contexto NÃO bate com o formato de nodes.py::no_gerar_resposta"
)
print("OK: prompt COM contexto bate exatamente com o formato de nodes.py::no_gerar_resposta.\n")
print("--- prompt COM contexto (para conferência visual) ---")
print(prompt_com_contexto)
print("-" * 60)

# --- Caso 2: sem contexto -> mantém o PROMPT_STYLE (Alpaca) original ---
prompt_sem_contexto = _montar_prompt(pergunta, contexto="")

prompt_esperado_alpaca = PROMPT_STYLE.format(
    "Responda à pergunta médica com base em informações clínicas confiáveis.",
    pergunta,
    "",
)

assert prompt_sem_contexto == prompt_esperado_alpaca, (
    "prompt sem contexto mudou em relação ao PROMPT_STYLE (Alpaca) original"
)
print("\nOK: prompt SEM contexto preserva o PROMPT_STYLE (Alpaca) original, sem mudanças.\n")
print("--- prompt SEM contexto (para conferência visual) ---")
print(prompt_sem_contexto)
print("-" * 60)

print("\nTodos os testes passaram.")
