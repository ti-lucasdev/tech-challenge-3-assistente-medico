"""
Teste rápido (sem GPU, sem carregar modelo) de registrar_interacao() com o
campo opcional "fontes": confirma que fontes=[...] é gravado corretamente no
JSONL, e que a chamada sem fontes continua gravando "fontes": [] sem quebrar
o comportamento anterior.
"""
import json
import tempfile
from pathlib import Path

import src.logging.logger as logger_module

# Redireciona o log para um arquivo temporário, para não sujar logs/interacoes.jsonl
with tempfile.TemporaryDirectory() as tmp:
    logger_module.CAMINHO_LOG = Path(tmp) / "interacoes_teste.jsonl"

    # Caso 1: chamada com fontes explícitas
    logger_module.registrar_interacao(
        pergunta="Quais exames estão pendentes para o PAC-0007?",
        resposta_bruta="resposta bruta 1",
        resposta_final="resposta final 1",
        bloqueado=False,
        fontes=["Fonte A", "Fonte B"],
    )

    # Caso 2: chamada sem fontes (comportamento anterior, sem quebrar)
    logger_module.registrar_interacao(
        pergunta="O que é hipotireoidismo?",
        resposta_bruta="resposta bruta 2",
        resposta_final="resposta final 2",
        bloqueado=False,
    )

    linhas = logger_module.CAMINHO_LOG.read_text(encoding="utf-8").strip().splitlines()
    assert len(linhas) == 2, f"esperava 2 linhas no JSONL, achei {len(linhas)}"

    registro_1 = json.loads(linhas[0])
    registro_2 = json.loads(linhas[1])

    assert registro_1["fontes"] == ["Fonte A", "Fonte B"], registro_1
    print("OK: chamada com fontes=[...] gravou o campo 'fontes' corretamente.")
    print(f"  registro: {registro_1}\n")

    assert registro_2["fontes"] == [], registro_2
    print("OK: chamada sem fontes gravou 'fontes': [] (comportamento anterior preservado).")
    print(f"  registro: {registro_2}\n")

print("Todos os testes passaram.")
