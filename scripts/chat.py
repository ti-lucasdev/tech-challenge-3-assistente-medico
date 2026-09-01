import os
import certifi
os.environ["SSL_CERT_FILE"] = certifi.where()

import sys
from pathlib import Path

# Adiciona a raiz do projeto ao sys.path
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import warnings
warnings.filterwarnings("ignore")

import logging
from transformers import logging as hf_logging

# Define o nível de log para ERROR, silenciando warnings desnecessários
hf_logging.set_verbosity_error()

# Aviso visual prévio para o usuário saber que o sistema está carregando
print("Inicializando o ambiente do Assistente Médico. Por favor, aguarde...")

from src.orchestration.workflow import app_assistente

def iniciar_chat():
    """
    Interface de chat interativa via terminal para validação do Assistente Médico.
    Exibe feedback detalhado das etapas do fluxo.
    """
    print("="*60)
    print("Assistente Médico Carregado com Sucesso. (Digite 'sair' para encerrar)")
    print("="*60)

    while True:
        try:
            pergunta = input("\nVocê: ")
            
            if pergunta.lower().strip() in ['sair', 'exit', 'quit', 'q']:
                print("\nEncerrando o assistente. Até logo!")
                break
                
            if not pergunta.strip():
                continue

            # Feedback detalhado das etapas de bastidores
            print("\n[Etapa 1/4] Identificando contexto e paciente...")
            estado_inicial = {"pergunta": pergunta}
            
            print("[Etapa 2/4] Consultando prontuário e base de conhecimento (RAG)...")
            print("[Etapa 3/4] Analisando exames críticos e gerando resposta...")
            print("[Etapa 4/4] Aplicando governança, guardrails e auditoria...")
            
            # Execução do grafo LangGraph
            resultado = app_assistente.invoke(estado_inicial)
            
            # Exibe a resposta final processada
            print("\nAssistente:")
            print(resultado.get("resposta_final", "Nenhuma resposta gerada."))
            
            # Exibe de forma transparente as fontes consultadas
            fontes = resultado.get("fontes", [])
            if fontes:
                print("\nFontes Consultadas:")
                fontes_unicas = list(dict.fromkeys(fontes))
                for fonte in fontes_unicas:
                    print(f"  - {fonte}")
                    
            print("-" * 60)
            
        except KeyboardInterrupt:
            print("\nEncerrando o assistente. Até logo!")
            break
        except Exception as e:
            print(f"\nOcorreu um erro durante o processamento: {e}")

if __name__ == "__main__":
    iniciar_chat()