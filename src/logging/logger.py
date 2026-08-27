"""
Registra cada interação com o modelo em formato JSONL,
para rastreabilidade (explainability) e auditoria posterior.
"""
import json
from datetime import datetime, timezone
from pathlib import Path

CAMINHO_LOG = Path("logs/interacoes.jsonl")

def registrar_interacao(pergunta: str, resposta_bruta: str, resposta_final: str, bloqueado: bool, fontes: list[str] = None):
    CAMINHO_LOG.parent.mkdir(parents=True, exist_ok=True)
    registro = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "pergunta": pergunta,
        "resposta_bruta": resposta_bruta,
        "resposta_final": resposta_final,
        "guardrail_bloqueou": bloqueado,
        "fontes": fontes if fontes is not None else [],
    }
    with CAMINHO_LOG.open("a", encoding="utf-8") as f:
        f.write(json.dumps(registro, ensure_ascii=False) + "\n")
