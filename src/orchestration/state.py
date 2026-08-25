from typing import TypedDict, List, Optional, Dict, Any

class EstadoClinico(TypedDict):
    """
    Define a estrutura tipada de dados compartilhada entre os nós do LangGraph.
    Armazena o fluxo de variáveis clínicas, metadados e histórico de execução.
    Utiliza TypedDict para garantir consistência de tipos no ecossistema Python.
    Permite que cada nó acesse ou modifique o estado de forma segura e previsível.
    """
    pergunta: str                        # Pergunta de entrada do usuário
    patient_id: Optional[str]            # ID estruturado do paciente (ex: PAC-0007)
    
    # Contextos recuperados pelo RAG
    contexto_prontuario: Optional[str]
    contexto_faq: Optional[str]
    fontes: List[str]
    evidencia_suficiente: bool
    
    # Alertas e Exames
    exames_pendentes: Optional[str]
    alerta_equipe: bool
    mensagem_alerta: Optional[str]
    
    # Saídas da LLM e Governança
    resposta_bruta: Optional[str]
    resposta_final: Optional[str]
    bloqueado: bool