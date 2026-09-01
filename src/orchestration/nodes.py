import os
import certifi
os.environ["SSL_CERT_FILE"] = certifi.where()

import sys
import re
from pathlib import Path
from typing import Dict, Any
from src.orchestration.state import EstadoClinico
from src.rag.retriever import MedicalRetriever
from src.governance.guardrails import aplicar_guardrail
from src.logging.logger import registrar_interacao

# Instanciação global do retriever da Camada 2
retriever = MedicalRetriever()


def no_identificar_paciente(state: EstadoClinico) -> Dict[str, Any]:
    pergunta = state["pergunta"]
    match = re.search(r'(PAC-\d{4}|\b\d{4}\b)', pergunta, re.IGNORECASE)
    
    if match:
        encontrado = match.group(0)
        if not encontrado.startswith("PAC-"):
            encontrado = f"PAC-{encontrado}"
        return {"patient_id": encontrado}
        
    return {"patient_id": None}


def no_recuperar_contexto(state: EstadoClinico) -> Dict[str, Any]:
    pergunta = state["pergunta"]
    patient_id = state.get("patient_id")
    
    contexto_prontuario = ""
    contexto_faq = ""
    fontes = []
    evidencia_suficiente = True

    termos_proibidos_ou_irrelevantes = ['bolo', 'chocolate', 'receita', 'futebol', 'carro', 'filme', 'fórmula 1']
    if any(termo in pergunta.lower() for termo in termos_proibidos_ou_irrelevantes) and not patient_id:
        return {
            "contexto_prontuario": "",
            "contexto_faq": "",
            "fontes": [],
            "evidencia_suficiente": False
        }

    if patient_id:
        res_prontuario = retriever.retrieve(pergunta, k=3, filters={"patient_id": patient_id})
        if res_prontuario:
            contexto_prontuario = retriever.format_context(res_prontuario, max_tokens=600)
            if hasattr(res_prontuario[0], 'citacao'): 
                fontes.extend([r.citacao() for r in res_prontuario])
        else:
            evidencia_suficiente = False

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
    contexto = f"{state.get('contexto_prontuario', '')} {state.get('contexto_faq', '')}".lower()
    alerta = False
    mensagem = ""
    
    termos_criticos = ['troponina', 'tc', 'angiotomografia', 'gasometria']
    if any(termo in contexto for termo in termos_criticos):
        alerta = True
        mensagem = "ALERTA MÉDICO: O paciente possui exames críticos pendentes ou alterados. Acione a equipe de plantão imediatamente."
        
    return {"alerta_equipe": alerta, "mensagem_alerta": mensagem}


def no_gerar_resposta(state: EstadoClinico) -> Dict[str, Any]:
    # Se não há evidência suficiente, encerra aqui sem precisar carregar a LLM na GPU
    if not state.get("evidencia_suficiente", True):
        mensagem_abstencao = "Não encontrei informações suficientes no prontuário ou na literatura para responder com segurança."
        return {"resposta_final": mensagem_abstencao}

    # Carregamento sob demanda (Lazy Loading), reaproveitando o carregador de modelo
    # já implementado em src/governance/inferencia_segura.py em vez de
    # reimplementar o carregamento, suprimindo os prints de inicialização do Unsloth.
    with open(os.devnull, 'w') as f:
        old_stdout = sys.stdout
        sys.stdout = f
        try:
            from src.governance.inferencia_segura import carregar_modelo
            model, tokenizer = carregar_modelo()
        finally:
            sys.stdout = old_stdout

    contexto_prontuario = state.get('contexto_prontuario', '')
    contexto_faq = state.get('contexto_faq', '')
    contexto_completo = f"Prontuário:\n{contexto_prontuario}\n\nLiteratura/FAQ:\n{contexto_faq}"
    
    prompt = (
        f"Abaixo está um contexto clínico. Com base nele, responda à pergunta de forma direta, "
        f"objetiva e em português. Não repita instruções.\n\n"
        f"Contexto:\n{contexto_completo}\n\n"
        f"Pergunta: {state['pergunta']}\n\n"
        f"Resposta:"
    )
    
    inputs = tokenizer([prompt], return_tensors="pt").to("cuda")
    
    outputs = model.generate(
        **inputs,
        max_new_tokens=250,
        do_sample=False,
        repetition_penalty=1.2,
        no_repeat_ngram_size=3,
    )
    
    resposta = tokenizer.decode(
        outputs[0][inputs["input_ids"].shape[1]:],
        skip_special_tokens=True,
    )
    
    return {"resposta_bruta": resposta.strip()}


def no_governanca_e_auditoria(state: EstadoClinico) -> Dict[str, Any]:
    # Se já veio a resposta final da abstenção, registra o log e retorna
    if not state.get("evidencia_suficiente", True):
        resposta_final = state.get("resposta_final", "Não encontrei informações suficientes.")
        registrar_interacao(state["pergunta"], "", resposta_final, bloqueado=True)
        return {"resposta_final": resposta_final, "bloqueado": False}

    resposta_bruta = state.get("resposta_bruta", "")
    resposta_final = resposta_bruta
    bloqueado = False

    # Aplicação do guardrail com o contexto recuperado
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