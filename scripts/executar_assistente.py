import sys
from pathlib import Path

# Adiciona a raiz do projeto ao sys.path para garantir portabilidade de importação
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.orchestration.workflow import app_assistente


def main():
    """
    Script de execução em lote / linha de comando para testes automatizados ou consultas diretas.
    Recebe um argumento via CLI ou executa uma pergunta padrão pré-definida.
    Invoca o grafo do assistente e formata a exibição da resposta final e das fontes.
    Facilita testes rápidos de integração contínua ou auditoria de consultas específicas.
    """
    pergunta = (
        sys.argv[1]
        if len(sys.argv) > 1
        else "Quais exames estao pendentes para o PAC-0007?"
    )
    print(f"\nProcessando Pergunta: {pergunta}\n" + "-" * 60)

    resultado = app_assistente.invoke({"pergunta": pergunta})

    print("\nRESPOSTA FINAL DO ASSISTENTE:")
    print(resultado["resposta_final"])

    if resultado.get("fontes"):
        print("\nFontes Consultadas:")
        for fonte in resultado["fontes"]:
            print(f"  - {fonte}")


if __name__ == "__main__":
    main()