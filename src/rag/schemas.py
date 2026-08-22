"""Contrato de metadados, proveniência das fontes e identificadores determinísticos.

Este módulo é a única fonte de verdade sobre *quais* metadados todo chunk
carrega. Os Integrantes 3 e 4 dependem desse contrato: o 3 para rotear no
StateGraph, o 4 para citar a fonte na resposta (explainability) e auditar.

Duas restrições moldam o desenho:

1. **Chroma só aceita escalares em metadados** (`str`, `int`, `float`, `bool`)
   e rejeita `None`. Validar na origem evita quebrar no meio de uma ingestão
   de 15 mil documentos.
2. **Os IDs precisam ser determinísticos por conteúdo**, não por posição.
   O arquivo `data/medquad_rag_pool.jsonl` não preserva o índice original do
   dataset, e IDs por conteúdo dão idempotência de graça: reingerir o mesmo
   texto gera o mesmo ID, então o upsert não duplica.
"""

from __future__ import annotations

import hashlib
import re
import unicodedata
from dataclasses import dataclass, asdict
from typing import Final, Literal

# --------------------------------------------------------------------------
# Proveniência das fontes
# --------------------------------------------------------------------------

# O MedQuAD é uma coletânea: cada resposta vem de um site institucional real
# do NIH/CDC, e o dataset preserva essa origem no campo `group_name`. Mapear
# esse campo de volta para a instituição é o que permite responder "de onde
# veio essa informação?" com algo melhor do que o caminho do arquivo local.
#
# Nota: o MedQuAD original tem 12 subconjuntos, mas os autores removeram as
# respostas de alguns deles (entre eles o CancerGov) por restrição de direito
# autoral. Por isso o pool tem 8 grupos, numerados de 2 a 9 — a numeração é
# a das pastas do repositório original, não uma sequência contínua.


@dataclass(frozen=True)
class Fonte:
    """Origem institucional de um documento do corpus."""

    sigla: str
    instituicao: str
    url: str


FONTES_MEDQUAD: Final[dict[str, Fonte]] = {
    "2_GARD_QA": Fonte(
        sigla="GARD",
        instituicao="Genetic and Rare Diseases Information Center (NIH/NCATS)",
        url="https://rarediseases.info.nih.gov/",
    ),
    "3_GHR_QA": Fonte(
        sigla="GHR",
        instituicao="Genetics Home Reference (NIH/NLM) — hoje MedlinePlus Genetics",
        url="https://medlineplus.gov/genetics/",
    ),
    "4_MPlus_Health_Topics_QA": Fonte(
        sigla="MedlinePlus",
        instituicao="MedlinePlus Health Topics (NIH/NLM)",
        url="https://medlineplus.gov/healthtopics.html",
    ),
    "5_NIDDK_QA": Fonte(
        sigla="NIDDK",
        instituicao="National Institute of Diabetes and Digestive and Kidney Diseases (NIH)",
        url="https://www.niddk.nih.gov/health-information",
    ),
    "6_NINDS_QA": Fonte(
        sigla="NINDS",
        instituicao="National Institute of Neurological Disorders and Stroke (NIH)",
        url="https://www.ninds.nih.gov/health-information",
    ),
    "7_SeniorHealth_QA": Fonte(
        sigla="NIHSeniorHealth",
        instituicao="NIHSeniorHealth (NIH) — descontinuado, incorporado ao MedlinePlus",
        url="https://medlineplus.gov/",
    ),
    "8_NHLBI_QA_XML": Fonte(
        sigla="NHLBI",
        instituicao="National Heart, Lung, and Blood Institute (NIH)",
        url="https://www.nhlbi.nih.gov/health-topics",
    ),
    "9_CDC_QA": Fonte(
        sigla="CDC",
        instituicao="Centers for Disease Control and Prevention",
        url="https://www.cdc.gov/",
    ),
}

FONTE_DESCONHECIDA: Final[Fonte] = Fonte(
    sigla="MedQuAD",
    instituicao="MedQuAD (origem institucional não identificada no dataset)",
    url="https://github.com/abachaa/MedQuAD",
)

# Prontuários não têm URL institucional, e inventar uma seria pior do que não
# ter: a citação passaria a apontar para um endereço que não sustenta o que foi
# dito. O esquema `prontuario-sintetico://` é deliberadamente não navegável, de
# modo que quem lê a citação veja de imediato que a evidência é interna e
# fabricada — o oposto de uma FAQ do NIH, que é pública e verificável.
FONTE_PRONTUARIO: Final[Fonte] = Fonte(
    sigla="PRONTUARIO",
    instituicao="Registro clínico sintético (dados fictícios, gerados para este projeto)",
    url="prontuario-sintetico://",
)


def resolver_fonte(group_name: str | None) -> Fonte:
    """Traduz o `group_name` do dataset para a fonte institucional.

    Nunca levanta exceção: um grupo desconhecido (por exemplo, se o dataset
    upstream ganhar um subconjunto novo) degrada para `FONTE_DESCONHECIDA` em
    vez de interromper a ingestão. A perda de proveniência fica visível no
    metadado, que é o comportamento desejável.
    """
    if not group_name:
        return FONTE_DESCONHECIDA
    return FONTES_MEDQUAD.get(group_name, FONTE_DESCONHECIDA)


# --------------------------------------------------------------------------
# Tipos de documento e idiomas
# --------------------------------------------------------------------------

# `faq_medica` e `prontuario_sintetico` são os tipos em uso. Os demais estão
# declarados porque o contrato precisa aceitá-los sem alteração se novas fontes
# entrarem no corpus — ver `README.md` §5.2.
TipoDocumento = Literal[
    "faq_medica",
    "prontuario_sintetico",
    "protocolo",
    "diretriz",
]

Idioma = Literal["en", "pt-BR"]

# Marcador para campos que não se aplicam a um tipo de documento. O contrato
# proíbe `None` e string vazia (ver `validar_metadados`), então a ausência
# precisa ser um valor legível: numa citação, "nao_aplicavel" é visivelmente
# diferente de um dado faltando por engano.
NAO_APLICAVEL: Final[str] = "nao_aplicavel"


# --------------------------------------------------------------------------
# Identificadores determinísticos
# --------------------------------------------------------------------------

_TAMANHO_ID: Final[int] = 16


def _sha256(texto: str) -> str:
    return hashlib.sha256(texto.encode("utf-8")).hexdigest()


def checksum(texto: str) -> str:
    """Checksum completo do texto de origem, para detectar mudança da fonte."""
    return _sha256(texto)


def id_documento(pergunta: str, resposta: str, prefixo: str = "medquad") -> str:
    """ID estável de um documento, derivado do seu conteúdo.

    Deriva de pergunta + resposta porque o dataset repete a mesma pergunta em
    linhas diferentes, com respostas de fontes distintas (comportamento já
    documentado em `README.md` §4.4). Usar só a pergunta colapsaria documentos
    legitimamente diferentes num único ID.
    """
    # Separador nulo entre pergunta e resposta para que a concatenação seja
    # injetiva: sem ele, ("ab", "c") e ("a", "bc") gerariam o mesmo hash.
    bruto = pergunta + "\x00" + resposta
    return f"{prefixo}-{_sha256(bruto)[:_TAMANHO_ID]}"


def id_chunk(document_id: str, ordem: int) -> str:
    """ID estável de um trecho dentro do documento."""
    return f"{document_id}-c{ordem:03d}"


def normalizar_pergunta(pergunta: str) -> str:
    """Forma canônica de uma pergunta, para agrupar duplicatas na avaliação.

    O MedQuAD tem a mesma pergunta em linhas diferentes, respondida por fontes
    distintas. Na avaliação de recuperação, trazer a resposta da CDC quando o
    "gabarito" era a do NIDDK é um acerto, não um erro — então o avaliador
    compara perguntas normalizadas, não `document_id`. Sem isso, as métricas
    de Recall@k/MRR ficam artificialmente deprimidas.
    """
    texto = unicodedata.normalize("NFKD", pergunta.strip().lower())
    texto = "".join(c for c in texto if not unicodedata.combining(c))
    texto = re.sub(r"[^\w\s]", " ", texto)
    return re.sub(r"\s+", " ", texto).strip()


# --------------------------------------------------------------------------
# Contrato de metadados
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class MetadadosChunk:
    """Metadados obrigatórios em todo chunk indexado.

    Todo campo é escalar e não-nulo, por exigência do Chroma. Campos que
    "não se aplicam" usam um valor explícito (string vazia nunca; preferir um
    marcador legível), para que a ausência seja visível na citação em vez de
    silenciosa.
    """

    chunk_id: str
    document_id: str
    source: str
    title: str
    document_type: TipoDocumento
    page_or_section: str
    language: Idioma
    version: str
    updated_at: str
    is_synthetic: bool
    checksum: str
    # Campos adicionais úteis para filtro e para a citação, fora do mínimo
    # exigido pelo guia:
    source_group: str  # `group_name` original, para filtrar por subconjunto
    institution: str  # nome legível da instituição, para exibir na resposta
    # Identificador do paciente nos registros clínicos sintéticos. É o que
    # permite `filters={"patient_id": "PAC-0007"}` recortar o contexto para um
    # paciente — o requisito de "contextualizar a resposta com informações do
    # paciente" do enunciado. Vale `NAO_APLICAVEL` em todo chunk de FAQ médica,
    # que não pertence a paciente nenhum.
    patient_id: str = NAO_APLICAVEL

    def to_dict(self) -> dict[str, str | bool]:
        """Serializa no formato aceito pelo Chroma (só escalares, sem `None`)."""
        return asdict(self)


CAMPOS_OBRIGATORIOS: Final[tuple[str, ...]] = tuple(
    f.name for f in MetadadosChunk.__dataclass_fields__.values()  # type: ignore[attr-defined]
)


class MetadadosInvalidos(ValueError):
    """Metadados que quebrariam a ingestão ou a citação da fonte."""


def validar_metadados(metadados: dict[str, object]) -> None:
    """Falha alto e cedo se um chunk não puder ser indexado ou citado.

    Chamado na ingestão, antes de enviar qualquer coisa ao Chroma. É mais
    barato rejeitar um documento malformado aqui do que descobrir, na consulta,
    que um resultado não tem fonte para citar.
    """
    faltando = [c for c in CAMPOS_OBRIGATORIOS if c not in metadados]
    if faltando:
        raise MetadadosInvalidos(f"campos ausentes: {', '.join(faltando)}")

    for chave, valor in metadados.items():
        if valor is None:
            raise MetadadosInvalidos(
                f"campo '{chave}' é None; o Chroma rejeita metadados nulos"
            )
        if not isinstance(valor, (str, int, float, bool)):
            raise MetadadosInvalidos(
                f"campo '{chave}' tem tipo {type(valor).__name__}; "
                "o Chroma só aceita str, int, float ou bool"
            )
        if isinstance(valor, str) and not valor.strip():
            raise MetadadosInvalidos(f"campo '{chave}' está vazio")
