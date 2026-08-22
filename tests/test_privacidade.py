"""Varredura de PII no corpus versionado.

Critério de aceite do guia: "busca automatizada por padrões de CPF, e-mail e
telefone não encontra PII real no corpus versionado".

O corpus atual é público (MedQuAD, publicações do NIH/CDC) e não deveria conter
dado pessoal. Este teste existe para que isso continue verdadeiro quando a
camada de prontuários sintéticos for adicionada — é aí que o risco aparece, e é
justamente quando ninguém se lembra de conferir.

Observação sobre falsos positivos: textos médicos citam números de estudo,
faixas de dosagem e códigos CID à vontade. Os padrões abaixo são deliberadamente
específicos (CPF formatado, e-mail com TLD, telefone com DDD entre parênteses)
para não transformar o teste em ruído — o preço é não detectar PII escrita em
formato livre, que é responsabilidade da revisão humana.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

RAIZ = Path(__file__).resolve().parent.parent
DIRETORIO_DADOS = RAIZ / "data"

PADROES = {
    "CPF": re.compile(r"\b\d{3}\.\d{3}\.\d{3}-\d{2}\b"),
    "CNPJ": re.compile(r"\b\d{2}\.\d{3}\.\d{3}/\d{4}-\d{2}\b"),
    # O domínio precisa ser capturado inteiro (`nei.nih.gov`, não `nei.nih`),
    # senão a comparação com a allowlist abaixo falha por truncamento.
    "e-mail": re.compile(r"\b[\w.+-]+@[\w-]+(?:\.[\w-]+)+\b"),
    "telefone BR": re.compile(r"\(\d{2}\)\s?9?\d{4}-\d{4}"),
    "cartão de crédito": re.compile(r"\b(?:\d[ -]?){13,16}\b"),
}

# Allowlist revisada manualmente. São os 8 únicos endereços do corpus, todos
# baseados em função (não em pessoa) e publicados pelas próprias instituições
# nas páginas de contato do NIH/FDA e de ONGs de acessibilidade. Não são dado
# pessoal, e não são nossos para remover.
#
# A lista é explícita de propósito, em vez de um curinga por domínio: assim,
# qualquer endereço NOVO que entre no corpus faz o teste falhar e passa por
# revisão humana. É a diferença entre uma verificação e um carimbo.
EMAILS_INSTITUCIONAIS_REVISADOS = frozenset({
    "2020@nei.nih.gov",          # National Eye Institute
    "adear@nia.nih.gov",         # Alzheimer's Disease Education and Referral Center
    "webmail@oc.fda.gov",        # FDA
    "NIAMSinfo@mail.nih.gov",    # NIAMS
    "prpl@mail.cc.nih.gov",      # NIH Clinical Center
    "cancergovstaff@mail.nih.gov",
    "atainfo@ataccess.org",      # Alliance for Technology Access
    "resnaTA@resna.org",         # RESNA
})
_EMAILS_NORMALIZADOS = {e.lower() for e in EMAILS_INSTITUCIONAIS_REVISADOS}


def _luhn_valido(digitos: str) -> bool:
    """Checksum de Luhn — todo cartão real passa, quase nenhum número aleatório passa.

    Sem isso o padrão de cartão casa CEP colado a telefone
    ("22314 1-703-894-1805"), que é o que de fato aparece nos endereços de
    contato do NIH. O teste viraria ruído e seria desativado por irritação,
    que é a pior forma de uma verificação de segurança morrer.
    """
    numeros = [int(c) for c in digitos if c.isdigit()]
    if not 13 <= len(numeros) <= 19:
        return False
    total = 0
    for posicao, valor in enumerate(reversed(numeros)):
        if posicao % 2 == 1:
            valor *= 2
            if valor > 9:
                valor -= 9
        total += valor
    return total % 10 == 0


def _arquivos_de_dados() -> list[Path]:
    return sorted(p for p in DIRETORIO_DADOS.glob("*.jsonl"))


def _texto_de(caminho: Path):
    with caminho.open(encoding="utf-8") as arquivo:
        for numero, linha in enumerate(arquivo, start=1):
            linha = linha.strip()
            if not linha:
                continue
            registro = json.loads(linha)
            yield numero, " ".join(
                str(v) for v in registro.values() if isinstance(v, str)
            )


def test_existem_arquivos_para_varrer():
    assert _arquivos_de_dados(), "nenhum .jsonl em data/ — a varredura seria vácua"


@pytest.mark.parametrize("nome_padrao", sorted(PADROES))
def test_corpus_sem_pii(nome_padrao: str):
    padrao = PADROES[nome_padrao]
    achados: list[str] = []

    for caminho in _arquivos_de_dados():
        for numero, texto in _texto_de(caminho):
            for ocorrencia in padrao.findall(texto):
                valor = ocorrencia if isinstance(ocorrencia, str) else ocorrencia[0]
                if nome_padrao == "e-mail" and valor.lower() in _EMAILS_NORMALIZADOS:
                    continue
                if nome_padrao == "cartão de crédito" and not _luhn_valido(valor):
                    continue
                achados.append(f"{caminho.name}:{numero} -> {valor!r}")

    assert not achados, (
        f"{len(achados)} ocorrência(s) de {nome_padrao} no corpus versionado:\n  "
        + "\n  ".join(achados[:15])
    )


def test_allowlist_de_emails_continua_necessaria():
    """Se um endereço da allowlist sumir do corpus, a entrada deve sair da lista.

    Uma allowlist que acumula entradas obsoletas vira permissão silenciosa para
    o que ela originalmente descrevia.
    """
    encontrados: set[str] = set()
    padrao = PADROES["e-mail"]
    for caminho in _arquivos_de_dados():
        for _, texto in _texto_de(caminho):
            encontrados.update(v.lower() for v in padrao.findall(texto))

    obsoletos = _EMAILS_NORMALIZADOS - encontrados
    assert not obsoletos, f"remover da allowlist (não estão mais no corpus): {sorted(obsoletos)}"


def test_manifesto_declara_ausencia_de_dado_real():
    """O manifesto é evidência acadêmica de proveniência.

    Desde que os prontuários sintéticos entraram, o índice tem mais de uma
    fonte e cada uma precisa do próprio checksum: numa auditoria, "a resposta
    veio de uma FAQ pública do NIH" e "a resposta veio de um registro que nós
    fabricamos" são afirmações muito diferentes.
    """
    manifesto = DIRETORIO_DADOS / "manifest.json"
    if not manifesto.exists():
        pytest.skip("manifesto ainda não gerado; rode scripts/build_vector_store.py")
    dados = json.loads(manifesto.read_text(encoding="utf-8"))
    assert dados["corpus"]["dataset_upstream"]

    fontes = dados["corpus"]["fontes"]
    assert fontes, "o manifesto não declara nenhuma fonte"
    for nome, fonte in fontes.items():
        assert fonte["checksum_sha256"], f"fonte '{nome}' sem checksum"
        assert fonte["observacao"], f"fonte '{nome}' sem observação de proveniência"

    if "prontuarios_sinteticos" in fontes:
        observacao = fontes["prontuarios_sinteticos"]["observacao"].lower()
        assert "fictício" in observacao or "ficticio" in observacao, (
            "a natureza fabricada dos prontuários precisa estar explícita no "
            "manifesto — é o que impede alguém tratá-los como dado clínico real"
        )
