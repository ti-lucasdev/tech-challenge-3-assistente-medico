from langgraph.graph import StateGraph, END
from src.orchestration.state import EstadoClinico
from src.orchestration.nodes import (
    no_identificar_paciente,
    no_recuperar_contexto,
    no_verificar_exames_e_alertas,
    no_gerar_resposta,
    no_governanca_e_auditoria
)

def verificar_evidencia(state: EstadoClinico) -> str:
    """
    Função condicional (Router) para tomada de decisão no LangGraph.
    Avalia se o RAG encontrou evidências suficientes para prosseguir.
    Verifica a flag booleana 'evidencia_suficiente' no estado clínico.
    Implementa o curto-circuito de abstenção, evitando que a LLM alucine caso o contexto venha vazio.
    """
    if state.get("evidencia_suficiente"):
        # Se achou contexto, segue o fluxo normal para análise de exames e geração
        return "verificar_exames"
    
    # Se NÃO achou evidência, pula a LLM e vai direto para a governança emitir abstenção
    return "governanca"

def construir_grafo():
    """
    Constrói e compila o fluxo direcionado do assistente médico.
    Mapeia os nós de execução, pontos de entrada e arestas condicionais.
    Utiliza o StateGraph do LangGraph encadeando funções e roteadores.
    Orquestra de maneira modular e determinística o ciclo de vida de cada consulta médica.
    """
    workflow = StateGraph(EstadoClinico)

    # Adição dos nós modulares ao grafo
    workflow.add_node("identificar_paciente", no_identificar_paciente)
    workflow.add_node("recuperar_contexto", no_recuperar_contexto)
    workflow.add_node("verificar_exames", no_verificar_exames_e_alertas)
    workflow.add_node("gerar_resposta", no_gerar_resposta)
    workflow.add_node("governanca", no_governanca_e_auditoria)

    # Configuração do ponto de entrada e fluxo sequencial inicial
    workflow.set_entry_point("identificar_paciente")
    workflow.add_edge("identificar_paciente", "recuperar_contexto")

    # Aresta Condicional: Roteamento baseado na suficiência das evidências recuperadas
    workflow.add_conditional_edges(
        "recuperar_contexto",
        verificar_evidencia,
        {
            "verificar_exames": "verificar_exames",
            "governanca": "governanca"
        }
    )

    # Encadeamento do restante do fluxo padrão
    workflow.add_edge("verificar_exames", "gerar_resposta")
    workflow.add_edge("gerar_resposta", "governanca")
    workflow.add_edge("governanca", END)

    return workflow.compile()

# Instância compilada e exportada do grafo para consumo das interfaces
app_assistente = construir_grafo()