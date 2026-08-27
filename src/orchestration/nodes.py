import os
import sys
import re
from pathlib import Path
from typing import Dict, Any
from src.orchestration.state import EstadoClinico
from src.rag.retriever import MedicalRetriever

# Bloco para carregar a Camada 1 (LLM) suprimindo prints desnecessários de inicialização do Unsloth
with open(os.devnull, 'w') as f:
    old_stdout = sys.stdout
    sys.stdout = f
    try:
        from src.inferencia import model, tokenizer
    finally:
        sys.stdout = old_stdout

from src.governance.guardrails import aplicar_guardrail
from src.logging.logger import registrar_interacao

# Instanciação global do retriever da Camada 2
retriever = MedicalRetriever()


def gerar_resposta_llm(prompt_formatado: str) -> str:
    """
    Função ponte para invocar a inferência do modelo Llama-3 (Camada 1).
    Converte o prompt em tensores na GPU e gera a resposta restrita a tokens novos.
    Utiliza o tokenizador e o modelo carregados previamente via Unsloth.
    Isola a lógica de chamada da LLM para manter o nó de geração limpo e reutilizável.
    """
    inputs = tokenizer([prompt_formatado], return_tensors="pt").to("cuda")
    
    outputs = model.generate(
        **inputs,
        max_new_tokens=250,
        do_sample=False,
        repetition_penalty=1.1,
    )
    
    # Decodifica apenas os tokens gerados, ignorando o prompt de entrada
    resposta = tokenizer.decode(
        outputs[0][inputs["input_ids"].shape[1]:],
        skip_special_tokens=True,
    )
    return resposta.strip()


def no_identificar_paciente(state: EstadoClinico) -> Dict[str, Any]:
    """
    Nó responsável por extrair o ID do paciente da pergunta do usuário.
    Varre o texto em busca de padrões como PAC-XXXX ou 4 dígitos numéricos isolados.
    Utiliza expressões regulares (re.search) com tratamento flexível de formato.
    Permite direcionar a busca do RAG para o prontuário específico do paciente correto.
    """
    pergunta = state["pergunta"]
    
    # Procura por 'PAC-0000' ou sequências de 4 números isolados
    match = re.search(r'(PAC-\d{4}|\b\d{4}\b)', pergunta, re.IGNORECASE)
    
    if match:
        encontrado = match.group(0)
        # Padroniza para o formato oficial PAC-XXXX caso o usuário digite apenas os números
        if not encontrado.startswith("PAC-"):
            encontrado = f"PAC-{encontrado}"
        return {"patient_id": encontrado}
        
    return {"patient_id": None}


def no_recuperar_contexto(state: EstadoClinico) -> Dict[str, Any]:
    """
    Nó de recuperação de contexto híbrido com filtro de relevância (Camada 2 - RAG).
    Busca trechos no prontuário e na FAQ, avaliando se há evidência real para a pergunta.
    Valida IDs de pacientes e aplica checagem de relevância para filtrar perguntas fora do escopo.
    Impede que perguntas absurdas (ex: receitas culinárias) acionem a LLM de forma indevida.
    """
    pergunta = state["pergunta"]
    patient_id = state.get("patient_id")
    
    contexto_prontuario = ""
    contexto_faq = ""
    fontes = []
    evidencia_suficiente = True

    # Palavras-chave básicas para checar se a pergunta possui algum mínimo escopo médico/saúde
    # (Evita que perguntas totalmente leigas ou aleatórias passem batido pelo RAG genérico)
    termos_proibidos_ou_irrelevantes = ['bolo', 'chocolate', 'receita', 'futebol', 'carro', 'filme', 'fórmula 1']
    if any(termo in pergunta.lower() for termo in termos_proibidos_ou_irrelevantes) and not patient_id:
        return {
            "contexto_prontuario": "",
            "contexto_faq": "",
            "fontes": [],
            "evidencia_suficiente": False
        }

    # 1. Recuperação no prontuário restrita ao ID do paciente (se identificado)
    if patient_id:
        res_prontuario = retriever.retrieve(pergunta, k=3, filters={"patient_id": patient_id})
        if res_prontuario:
            contexto_prontuario = retriever.format_context(res_prontuario, max_tokens=600)
            if hasattr(res_prontuario[0], 'citacao'): 
                fontes.extend([r.citacao() for r in res_prontuario])
        else:
            evidencia_suficiente = False

    # 2. Recuperação na base de literatura médica e FAQ (geral)
    res_faq = retriever.retrieve(pergunta, k=3, filters={"document_type": "faq_medica"})
    if res_faq and retriever.tem_evidencia_suficiente(res_faq):
        contexto_faq = retriever.format_context(res_faq, max_tokens=400)
        if hasattr(res_faq[0], 'citacao'):
            fontes.extend([r.citacao() for r in res_faq])
    elif not patient_id:
        evidencia_suficiente = False
        
    return {
        "contexto_prontuario": contexto_prontuario,
        "contexto_faq": contexto_faq,
        "fontes": fontes,
        "evidencia_suficiente": evidencia_suficiente
    }

def no_verificar_exames_e_alertas(state: EstadoClinico) -> Dict[str, Any]:
    """
    Nó de triagem e varredura de termos críticos clínicos.
    Analisa os contextos recuperados em busca de exames críticos ou marcadores sensíveis.
    Realiza checagem de substrings por palavras-chave (ex: 'troponina', 'gasometria').
    Garante que o assistente emita alertas imediatos de segurança para a equipe médica.
    """
    contexto = f"{state.get('contexto_prontuario', '')} {state.get('contexto_faq', '')}".lower()
    alerta = False
    mensagem = ""
    
    termos_criticos = ['troponina', 'tc', 'angiotomografia', 'gasometria']
    if any(termo in contexto for termo in termos_criticos):
        alerta = True
        mensagem = "ALERTA MÉDICO: O paciente possui exames críticos pendentes ou alterados. Acione a equipe de plantão imediatamente."
        
    return {"alerta_equipe": alerta, "mensagem_alerta": mensagem}


def no_gerar_resposta(state: EstadoClinico) -> Dict[str, Any]:
    """
    Nó gerador de resposta estruturada com prompt simplificado para evitar loops do Llama-3.
    Formata o contexto do prontuário e da FAQ em um prompt direto.
    Utiliza um template objetivo em português, instruindo o modelo a abster-se de repetir o prompt.
    Evita o fenômeno de eco e repetição textual comum em modelos ajustados com dataset Alpaca.
    """
    contexto_prontuario = state.get('contexto_prontuario', '')
    contexto_faq = state.get('contexto_faq', '')
    
    contexto_completo = f"Prontuário:\n{contexto_prontuario}\n\nLiteratura/FAQ:\n{contexto_faq}"
    
    # Prompt simplificado e direto, evitando que o modelo ecoe a estrutura interna de treino
    prompt = (
        f"Abaixo está um contexto clínico. Com base nele, responda à pergunta de forma direta, "
        f"objetiva e em português. Não repita instruções.\n\n"
        f"Contexto:\n{contexto_completo}\n\n"
        f"Pergunta: {state['pergunta']}\n\n"
        f"Resposta:"
    )
    
    # Invocação do modelo com parâmetros restritivos contra repetição
    inputs = tokenizer([prompt], return_tensors="pt").to("cuda")
    
    outputs = model.generate(
        **inputs,
        max_new_tokens=250,
        do_sample=False,
        repetition_penalty=1.2,     # Penaliza repetições de palavras
        no_repeat_ngram_size=3,     # Impede frases repetidas em loop
    )
    
    resposta = tokenizer.decode(
        outputs[0][inputs["input_ids"].shape[1]:],
        skip_special_tokens=True,
    )
    
    return {"resposta_bruta": resposta.strip()}


def no_governanca_e_auditoria(state: EstadoClinico) -> Dict[str, Any]:
    """
    Nó final de Governança, Guardrails e Auditoria (Camada 4).
    Valida a resposta gerada, aplica filtros de segurança, injeta alertas e registra a interação.
    Verifica se houve evidência suficiente; caso contrário, abstenha-se com segurança. Aplica guardrails.
    Cumpre os requisitos legais de compliance, rastreabilidade e inserção do disclaimer obrigatório.
    """
    resposta_bruta = state.get("resposta_bruta", "")
    
    # Curto-circuito de abstenção se o RAG não encontrou suporte documental
    if not state.get("evidencia_suficiente", True):
        resposta_final = "Não encontrei informações suficientes no prontuário ou na literatura para responder com segurança."
        registrar_interacao(state["pergunta"], resposta_bruta, resposta_final, bloqueado=True)
        return {"resposta_final": resposta_final, "bloqueado": False}

    resposta_final = resposta_bruta
    bloqueado = False

    # Aplicação de guardrails de moderação caso a resposta passe sem bloqueio prévio
    if "⚠️" not in resposta_final:
        contexto_recuperado = f"{state.get('contexto_prontuario', '')} {state.get('contexto_faq', '')}"
        resultado_guardrail = aplicar_guardrail(resposta_bruta, contexto_recuperado)
        resposta_final = resultado_guardrail["resposta_final"]
        bloqueado = resultado_guardrail.get("bloqueado", False)

    # Injeção prioritária do alerta médico crítico no topo, se aplicável
    if state.get("alerta_equipe") and state.get("mensagem_alerta"):
        resposta_final = f"{state['mensagem_alerta']}\n\n{resposta_final}"

    # Registro formal da auditoria da interação
    registrar_interacao(
        pergunta=state["pergunta"],
        resposta_bruta=resposta_bruta,
        resposta_final=resposta_final,
        bloqueado=bloqueado
    )

    return {"resposta_final": resposta_final, "bloqueado": bloqueado}