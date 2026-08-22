"""Construção do conjunto de avaliação de recuperação e cálculo de Recall@k / MRR.

## Por que gerar em vez de escrever à mão

O guia sugere 10–20 perguntas escritas manualmente. Com essa quantidade, a
diferença entre Recall@5 de 0,80 e 0,87 é uma única pergunta — não dá para
comparar configurações (denso × BM25 × híbrido, 1.200/k=4 × 1.500/k=3) com
ruído desse tamanho.

O MedQuAD dá o gabarito de graça: o `query` de cada linha é, por construção,
uma pergunta correta para o `answers` daquela linha. Isso permite centenas de
casos sem anotação manual.

## As duas armadilhas, e como este módulo as evita

**1. Usar o `query` original como sonda mede casamento lexical, não semântica.**
As perguntas do MedQuAD são altamente templatizadas (89% em 12 padrões), então
dá para parafrasear de forma determinística trocando o padrão por uma formulação
clínica equivalente — "What causes X ?" vira "What is the etiology of X?".
O nome da condição permanece como âncora, o que é realista (o médico digita o
nome da doença) e ainda deixa o teste honesto: nenhuma palavra de conteúdo além
da entidade é compartilhada com o texto indexado.

**2. Cobrar o `document_id` exato pune acertos legítimos.**
O dataset repete a mesma pergunta em linhas diferentes, respondida por
instituições distintas (comportamento documentado em `README.md` §4.4). Trazer
a resposta da CDC quando o "gabarito" era a do NIDDK é acerto, não erro. Por
isso cada caso carrega um *conjunto* de IDs aceitos — todos os documentos cuja
pergunta normalizada coincide.
"""

from __future__ import annotations

import random
import re
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Final

from .loaders import DocumentoBruto
from .schemas import id_documento, normalizar_pergunta

# --------------------------------------------------------------------------
# Paráfrases determinísticas por template
# --------------------------------------------------------------------------

# Cada entrada: (regex com a entidade capturada, formulação clínica equivalente).
# A ordem importa — padrões mais específicos primeiro.
PARAFRASES: Final[tuple[tuple[str, str], ...]] = (
    (r"^What are the symptoms of (.+?)\s*\?$",
     "Which clinical manifestations are associated with {}?"),
    (r"^What are the treatments for (.+?)\s*\?$",
     "How is {} managed therapeutically?"),
    (r"^What are the complications of (.+?)\s*\?$",
     "Which adverse outcomes may arise from {}?"),
    (r"^What are the genetic changes related to (.+?)\s*\?$",
     "Which genetic alterations underlie {}?"),
    (r"^What are the stages of (.+?)\s*\?$",
     "How is the progression of {} classified?"),
    (r"^What causes (.+?)\s*\?$",
     "What is the etiology of {}?"),
    (r"^What is \(are\) (.+?)\s*\?$",
     "Provide a clinical overview of {}."),
    (r"^How to diagnose (.+?)\s*\?$",
     "Which examinations establish a diagnosis of {}?"),
    (r"^How to prevent (.+?)\s*\?$",
     "Which measures reduce the likelihood of developing {}?"),
    (r"^How many people are affected by (.+?)\s*\?$",
     "What is the prevalence of {} in the population?"),
    (r"^Is (.+?) inherited\s*\?$",
     "Does {} follow a hereditary transmission pattern?"),
    (r"^Who is at risk for (.+?)\s*\??\s*\?$",
     "Which groups present elevated susceptibility to {}?"),
    (r"^What to do for (.+?)\s*\?$",
     "What is the recommended course of action for {}?"),
    (r"^Do you have information about (.+?)\s*$",
     "Provide information regarding {}."),
)

_COMPILADOS: Final[tuple[tuple[re.Pattern[str], str], ...]] = tuple(
    (re.compile(p, re.IGNORECASE), t) for p, t in PARAFRASES
)


def parafrasear(pergunta: str) -> tuple[str, str] | None:
    """Devolve (paráfrase, nome_do_template) ou `None` se nenhum padrão casar.

    Perguntas fora dos templates conhecidos são simplesmente ignoradas na
    construção do conjunto — é preferível a um conjunto menor e limpo do que a
    paráfrases forçadas que não medem nada.
    """
    limpa = re.sub(r"\s+", " ", pergunta).strip()
    for padrao, modelo in _COMPILADOS:
        casamento = padrao.match(limpa)
        if casamento:
            entidade = casamento.group(1).strip()
            if not entidade:
                return None
            return modelo.format(entidade), padrao.pattern
    return None


# --------------------------------------------------------------------------
# Casos de avaliação
# --------------------------------------------------------------------------


@dataclass
class CasoAvaliacao:
    """Uma sonda e o conjunto de documentos que contam como acerto."""

    consulta: str  # a paráfrase, é o que vai ao retriever
    pergunta_original: str
    template: str
    documentos_aceitos: list[str]  # document_ids que respondem a mesma pergunta
    group_name: str


@dataclass
class CasoNegativo:
    """Pergunta sem suporte no corpus: o retriever deve sinalizar baixa confiança."""

    consulta: str
    justificativa: str


# Negativos escritos à mão: temas plausíveis num contexto clínico mas ausentes
# do MedQuAD, mais alguns claramente fora de domínio. Servem para calibrar o
# limiar de abstenção — sem eles, não há como distinguir "achou evidência" de
# "achou o vizinho menos distante".
NEGATIVOS: Final[tuple[CasoNegativo, ...]] = (
    # --- dados do paciente / operação do hospital (a lacuna real do corpus) --
    CasoNegativo("What medication is patient 001 currently taking?",
                 "consulta a prontuário — não há registros de paciente indexados"),
    CasoNegativo("Which lab results are still pending for the patient admitted yesterday?",
                 "exames pendentes exigem prontuário, ausente do corpus"),
    CasoNegativo("Qual o CID-10 correspondente à condição do paciente atual?",
                 "depende de contexto de paciente ausente do corpus"),
    CasoNegativo("Summarize the internal antibiotic protocol of this hospital.",
                 "protocolo interno inexistente: o corpus não contém documentos do hospital"),
    CasoNegativo("What is the discharge checklist used in our cardiology ward?",
                 "procedimento interno, não presente no corpus"),
    CasoNegativo("Which hospital in São Paulo has the shortest waiting list?",
                 "informação operacional local, inexistente no corpus"),
    CasoNegativo("How many beds are available in the ICU right now?",
                 "estado operacional em tempo real, fora de qualquer corpus estático"),
    CasoNegativo("Show me the imaging report from last Tuesday's CT scan.",
                 "documento de paciente, não indexado"),
    # --- prescrição e posologia (devem ser barrados também pelo guardrail) ---
    CasoNegativo("What is the recommended dosage of amoxicillin for my patient?",
                 "pedido de prescrição — o corpus não traz posologia"),
    CasoNegativo("How many milligrams of warfarin should I prescribe for this case?",
                 "pedido de prescrição direta com dose"),
    CasoNegativo("Can I combine metformin and this patient's current medication?",
                 "interação medicamentosa individualizada, exige prontuário"),
    # --- administrativo, jurídico, comercial ---
    CasoNegativo("What is the current price of insulin in Brazilian pharmacies?",
                 "informação comercial/regional, fora do escopo"),
    CasoNegativo("Which insurance plans cover bariatric surgery in Brazil?",
                 "cobertura de convênio, fora do escopo"),
    CasoNegativo("What does Brazilian law say about mandatory notification of dengue?",
                 "legislação brasileira, ausente de um corpus do NIH/CDC"),
    CasoNegativo("How do I bill CPT code 99213 correctly?",
                 "faturamento médico norte-americano, ausente do corpus"),
    CasoNegativo("What is the salary of a cardiologist in São Paulo?",
                 "informação de mercado de trabalho"),
    # --- médico, porém fora da cobertura do MedQuAD -------------------------
    CasoNegativo("What were the primary endpoints of the 2026 SPRINT-3 trial?",
                 "ensaio clínico inventado; nenhum documento pode sustentá-lo"),
    CasoNegativo("Which mRNA vaccine platforms entered phase III in 2026?",
                 "posterior ao corpus; não há evidência indexada"),
    CasoNegativo("What is the recommended anesthesia protocol for robotic prostatectomy?",
                 "protocolo cirúrgico especializado, fora da cobertura de FAQs"),
    CasoNegativo("Which suture technique minimizes scarring in pediatric facial lacerations?",
                 "técnica cirúrgica, fora da cobertura do corpus"),
    CasoNegativo("What are the calibration steps for a Siemens MRI scanner?",
                 "engenharia de equipamento, não conteúdo clínico"),
    CasoNegativo("How should a hospital negotiate a contract with a device supplier?",
                 "gestão hospitalar, fora do escopo clínico"),
    # --- claramente fora de domínio ----------------------------------------
    CasoNegativo("What were the results of the 2026 World Cup final?",
                 "fora de domínio, não médico"),
    CasoNegativo("How do I configure a Kubernetes ingress controller?",
                 "fora de domínio, técnico"),
    CasoNegativo("Write a Python function that reverses a linked list.",
                 "fora de domínio, programação"),
    CasoNegativo("What is the best route from Rio de Janeiro to Belo Horizonte?",
                 "fora de domínio, logística"),
    CasoNegativo("Recommend a good restaurant near the hospital.",
                 "fora de domínio, cotidiano"),
    CasoNegativo("Qual a previsão do tempo para amanhã em Curitiba?",
                 "fora de domínio, meteorologia (em português, para testar o multilíngue)"),
    CasoNegativo("Explique a teoria da relatividade restrita.",
                 "fora de domínio, física (em português)"),
    CasoNegativo("Quem escreveu Grande Sertão: Veredas?",
                 "fora de domínio, literatura (em português)"),
)


def construir_casos(
    documentos: list[DocumentoBruto],
    quantidade: int = 200,
    semente: int = 42,
) -> list[CasoAvaliacao]:
    """Amostra documentos do corpus e gera uma sonda parafraseada para cada um.

    Todos os documentos permanecem indexados — a amostragem escolhe *quais*
    servem de sonda, não remove nada do índice. Retirá-los tornaria o gabarito
    irrecuperável por construção.
    """
    # Agrupa por pergunta normalizada para montar o conjunto de IDs aceitos.
    por_pergunta: dict[str, list[str]] = defaultdict(list)
    for doc in documentos:
        por_pergunta[normalizar_pergunta(doc.pergunta)].append(
            id_documento(doc.pergunta, doc.resposta)
        )

    candidatos = []
    for doc in documentos:
        resultado = parafrasear(doc.pergunta)
        if resultado is None:
            continue
        consulta, template = resultado
        candidatos.append((doc, consulta, template))

    rng = random.Random(semente)
    rng.shuffle(candidatos)

    casos: list[CasoAvaliacao] = []
    templates_usados: dict[str, int] = defaultdict(int)
    # Teto por template para o conjunto não virar 40% de "What is (are) X",
    # que é o padrão mais frequente do corpus.
    teto = max(1, quantidade // 6)

    for doc, consulta, template in candidatos:
        if len(casos) >= quantidade:
            break
        if templates_usados[template] >= teto:
            continue
        templates_usados[template] += 1
        casos.append(
            CasoAvaliacao(
                consulta=consulta,
                pergunta_original=doc.pergunta,
                template=template,
                documentos_aceitos=sorted(
                    set(por_pergunta[normalizar_pergunta(doc.pergunta)])
                ),
                group_name=doc.group_name,
            )
        )
    return casos


# --------------------------------------------------------------------------
# Métricas
# --------------------------------------------------------------------------


@dataclass
class Metricas:
    """Resultado agregado de uma configuração de recuperação."""

    n: int = 0
    recall_por_k: dict[int, float] = field(default_factory=dict)
    mrr: float = 0.0

    def __str__(self) -> str:
        recalls = "  ".join(f"R@{k}={v:.3f}" for k, v in sorted(self.recall_por_k.items()))
        return f"n={self.n}  {recalls}  MRR={self.mrr:.3f}"


def calcular_metricas(
    recuperados_por_caso: list[list[str]],
    casos: list[CasoAvaliacao],
    ks: tuple[int, ...] = (1, 3, 5, 10),
) -> Metricas:
    """Recall@k e MRR sobre os `document_id` devolvidos pelo retriever.

    `recuperados_por_caso[i]` é a lista ordenada de `document_id` retornada para
    `casos[i]`. Duplicatas (vários chunks do mesmo documento) são colapsadas
    preservando a ordem, para que o ranking seja por documento e não por trecho.
    """
    if len(recuperados_por_caso) != len(casos):
        raise ValueError(
            f"{len(recuperados_por_caso)} listas de resultados para {len(casos)} casos"
        )

    acertos = {k: 0 for k in ks}
    soma_rr = 0.0

    for recuperados, caso in zip(recuperados_por_caso, casos):
        ordenados = _unicos_preservando_ordem(recuperados)
        aceitos = set(caso.documentos_aceitos)

        posicao = next(
            (i for i, doc_id in enumerate(ordenados, start=1) if doc_id in aceitos),
            None,
        )
        if posicao is not None:
            soma_rr += 1.0 / posicao
            for k in ks:
                if posicao <= k:
                    acertos[k] += 1

    total = len(casos)
    return Metricas(
        n=total,
        recall_por_k={k: acertos[k] / total for k in ks},
        mrr=soma_rr / total,
    )


def _unicos_preservando_ordem(itens: list[str]) -> list[str]:
    vistos: set[str] = set()
    saida: list[str] = []
    for item in itens:
        if item not in vistos:
            vistos.add(item)
            saida.append(item)
    return saida
