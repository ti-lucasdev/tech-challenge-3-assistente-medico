"""Spike de integração: recuperação -> prompt -> adaptador LoRA -> guardrail.

**Não é o nó do LangGraph.** Não há grafo, estado nem roteamento — isso é entrega
do Integrante 3. Este script existe para derrubar, com número em vez de
suposição, três hipóteses que hoje sustentam o desenho das camadas 2 e 4 e que
nunca foram exercidas juntas:

1. **O contexto cabe.** O orçamento (2.048 de janela − 250 de geração − template)
   foi calculado com uma razão média de 4,78 chars/token medida no corpus. Média
   não é garantia: aqui se conta com o tokenizer real, no prompt real.

2. **O adaptador usa o contexto — ou o ignora.** O LoRA foi treinado com
   `input` = pergunta crua e `output` = resposta do MedQuAD, isto é, para
   responder *de memória*. Injetar evidência recuperada no campo `Entrada` é
   fora da distribuição de treino. Se o modelo base for mais fiel ao contexto
   que o ajustado, isso é um achado do projeto, não um defeito do RAG.

3. **A citação sobrevive.** O guardrail recomendado ao Integrante 4 exige que a
   resposta referencie uma `[FONTE n]` efetivamente recuperada. Isso só é
   viável se o marcador chegar até a saída.

Uso:
    # sem GPU e sem baixar pesos: valida só a hipótese 1
    .venv\\Scripts\\python.exe scripts\\spike_rag_geracao.py --so-prompt

    # ponta a ponta (exige GPU com ~6 GB e o adaptador em artifacts/)
    .venv\\Scripts\\python.exe scripts\\spike_rag_geracao.py

    # o experimento que importa: mesma pergunta, mesmo contexto, com e sem LoRA
    .venv\\Scripts\\python.exe scripts\\spike_rag_geracao.py --comparar

    # contextualização por paciente (registros clínicos sintéticos)
    .venv\\Scripts\\python.exe scripts\\spike_rag_geracao.py --paciente PAC-0007 \\
        --pergunta "Quais exames estão pendentes e qual a conduta sugerida?"
"""

from __future__ import annotations

import argparse
import re
import sys
from contextlib import contextmanager
from pathlib import Path

RAIZ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ))

CAMINHO_ADAPTADOR = RAIZ / "artifacts" / "adaptador_medquad_lora_final"

# Janela e teto de geração do adaptador (ver notebooks/ e src/inferencia.py).
# Estão aqui como constantes locais de propósito: se o Integrante 1 retreinar
# com outra janela, este spike deve falhar de forma visível, não se ajustar em
# silêncio a um orçamento que a camada de RAG não conhece.
JANELA_MAXIMA = 2048
MAX_NEW_TOKENS = 250

# O template de treino, replicado byte a byte. Um espaço a mais aqui degrada o
# adaptador sem levantar erro nenhum — é a falha mais silenciosa do conjunto.
# Já existe duplicado no notebook, em src/inferencia.py e em
# src/governance/inferencia_segura.py; esta é a quarta cópia, e o argumento
# para extrair um módulo único de prompt.
PROMPT_STYLE = """ abaixo está uma instrução que descreve uma tarefa, juntamente com uma entrada que fornece contexto adicional. Escreva uma resposta que complete adequadamente o pedido.

### Instrução:
{}

### Entrada:
{}

### Resposta:
{}"""

# A instrução de treino era constante nas 1001 amostras, então o modelo teve
# pouco sinal para tratá-la como comando. Trocá-la por uma instrução de RAG é
# parte do que este spike testa: se o adaptador a ignorar, aparece na métrica
# de ancoragem abaixo.
INSTRUCAO_RAG = (
    "Responda à pergunta médica usando somente as fontes fornecidas na entrada. "
    "Cite a fonte usada no formato [FONTE n]. Se as fontes não contiverem a "
    "resposta, diga que não há informação suficiente."
)
INSTRUCAO_TREINO = "Responda à pergunta médica com base em informações clínicas confiáveis."

PERGUNTA_PADRAO = "What are the treatment options for hypothyroidism?"


# --------------------------------------------------------------------------
# Medição de ancoragem (grounding)
# --------------------------------------------------------------------------

# Heurística deliberadamente simples e transparente. NÃO mede correção clínica
# nem fidelidade semântica: mede quanto do vocabulário de conteúdo da resposta
# aparece no contexto recuperado. Serve para comparar duas gerações sob o mesmo
# contexto — que é exatamente o experimento aqui — e não vale como métrica
# absoluta de qualidade.
_PALAVRAS_VAZIAS = frozenset("""
about above after also among and any are because been before being between both
but can could does during each from had has have here how into its more most not
only other over should some such than that the their them then there these they
this those through under until very was were what when where which while who will
with would your
""".split())


def _termos_de_conteudo(texto: str) -> set[str]:
    return {
        palavra
        for palavra in re.findall(r"[a-zà-ÿ]{5,}", texto.lower())
        if palavra not in _PALAVRAS_VAZIAS
    }


def ancoragem(resposta: str, contexto: str) -> tuple[float, int]:
    """Fração do vocabulário de conteúdo da resposta que aparece no contexto."""
    termos = _termos_de_conteudo(resposta)
    if not termos:
        return 0.0, 0
    do_contexto = _termos_de_conteudo(contexto)
    return len(termos & do_contexto) / len(termos), len(termos)


def fontes_citadas(resposta: str) -> list[int]:
    return sorted({int(n) for n in re.findall(r"\[?FONTE\s*(\d+)\]?", resposta.upper())})


# --------------------------------------------------------------------------
# Montagem do prompt
# --------------------------------------------------------------------------


def montar_prompt(contexto: str, pergunta: str, instrucao: str) -> str:
    """Insere contexto e pergunta no campo `Entrada`.

    É o caminho de menor risco recomendado no handoff: preserva o template de
    treino intacto e não inventa um campo que o adaptador nunca viu.
    """
    entrada = f"{contexto}\n\nPergunta: {pergunta}" if contexto else pergunta
    return PROMPT_STYLE.format(instrucao, entrada, "")


def contar(tokenizer, texto: str) -> int:
    if tokenizer is None:
        from src.rag.retriever import CHARS_POR_TOKEN

        return int(len(texto) / CHARS_POR_TOKEN)  # fallback conservador
    return len(tokenizer.encode(texto, add_special_tokens=False))


def relatar_orcamento(tokenizer, prompt: str, contexto: str) -> bool:
    """Imprime a contabilidade de tokens e devolve se o prompt cabe."""
    total = contar(tokenizer, prompt)
    tokens_contexto = contar(tokenizer, contexto) if contexto else 0
    teto = JANELA_MAXIMA - MAX_NEW_TOKENS
    cabe = total <= teto

    origem = "tokenizer real do adaptador" if tokenizer else "ESTIMATIVA (sem tokenizer)"
    print(f"\n  Orçamento de tokens ({origem})")
    print(f"    contexto recuperado ..... {tokens_contexto}")
    print(f"    prompt completo ......... {total}")
    print(f"    reservado para geração .. {MAX_NEW_TOKENS}")
    print(f"    teto da janela .......... {JANELA_MAXIMA}")
    print(f"    folga ................... {teto - total}")
    print(f"    veredito ................ {'CABE' if cabe else 'ESTOURA — o contexto seria truncado'}")
    if tokenizer is not None and contexto:
        from src.rag.retriever import CHARS_POR_TOKEN

        # Comparação contra o fallback conservador de `retriever.py` — o valor
        # usado quando não há tokenizer nem idioma conhecido. Ele erra para cima
        # de propósito em inglês; o que importa vigiar é o sinal contrário.
        estimado = int(len(contexto) / CHARS_POR_TOKEN)
        desvio = tokens_contexto - estimado
        print(f"    fallback conservador .... {estimado} tokens "
              f"({desvio:+d} vs os {tokens_contexto} reais)")
        if desvio > 0:
            print("      ^ ATENÇÃO: até o fallback conservador SUBESTIMOU o custo.")
            print("        Passe o tokenizer real em format_context(contar_tokens=...).")
    return cabe


# --------------------------------------------------------------------------
# Carregamento
# --------------------------------------------------------------------------


def carregar_tokenizer():
    """Carrega só o tokenizer — sem pesos, sem GPU, sem 5,5 GB de VRAM.

    É o que permite validar a hipótese 1 na máquina de desenvolvimento, que não
    tem VRAM para o modelo. O diretório do adaptador traz os arquivos de
    tokenizer salvos pelo notebook do Integrante 1.
    """
    if not CAMINHO_ADAPTADOR.exists():
        return None
    try:
        from transformers import AutoTokenizer

        return AutoTokenizer.from_pretrained(str(CAMINHO_ADAPTADOR))
    except Exception as erro:  # noqa: BLE001 - degradar é melhor que abortar o spike
        print(f"  [aviso] tokenizer real indisponível ({type(erro).__name__}); usando estimativa")
        return None


def carregar_modelo():
    from unsloth import FastLanguageModel

    if not CAMINHO_ADAPTADOR.exists():
        raise FileNotFoundError(
            f"Adaptador não encontrado em {CAMINHO_ADAPTADOR}.\n"
            "Baixe conforme o README, ou rode com --so-prompt para validar apenas "
            "o orçamento de contexto."
        )
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=str(CAMINHO_ADAPTADOR),
        max_seq_length=JANELA_MAXIMA,
        dtype=None,
        load_in_4bit=True,
    )
    FastLanguageModel.for_inference(model)
    return model, tokenizer


@contextmanager
def adaptador_desligado(model):
    """Desliga o LoRA sem recarregar o modelo base.

    Um segundo `from_pretrained` do modelo base custaria outros ~5,5 GB de VRAM
    e tornaria a comparação impossível na mesma máquina. O `disable_adapter` do
    PEFT dá o A/B honesto: mesmos pesos base, mesmo prompt, mesma seed.
    """
    desligar = getattr(model, "disable_adapter", None)
    if desligar is None:
        raise RuntimeError(
            "O modelo carregado não expõe `disable_adapter` — não é um PeftModel. "
            "A comparação com/sem adaptador não é possível assim."
        )
    with desligar():
        yield


def gerar(model, tokenizer, prompt: str) -> str:
    inputs = tokenizer([prompt], return_tensors="pt").to("cuda")
    saida = model.generate(
        **inputs,
        max_new_tokens=MAX_NEW_TOKENS,
        do_sample=False,  # greedy: a comparação com/sem LoRA precisa ser determinística
        repetition_penalty=1.1,
    )
    return tokenizer.decode(
        saida[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True
    ).strip()


# --------------------------------------------------------------------------
# Relato de uma geração
# --------------------------------------------------------------------------


def relatar_geracao(rotulo: str, resposta: str, contexto: str, fontes_disponiveis: int) -> dict:
    from src.governance.guardrails import aplicar_guardrail

    proporcao, termos = ancoragem(resposta, contexto)
    citadas = fontes_citadas(resposta)
    validas = [n for n in citadas if 1 <= n <= fontes_disponiveis]
    veredito = aplicar_guardrail(resposta)

    print(f"\n{'=' * 74}\n{rotulo}\n{'=' * 74}")
    print(resposta or "(resposta vazia)")
    print(f"\n  ancoragem no contexto ... {proporcao:.1%} de {termos} termos de conteúdo")
    print(f"  fontes citadas .......... {citadas or 'nenhuma'}"
          f"{'' if len(validas) == len(citadas) else f'  <- {len(citadas) - len(validas)} inexistente(s)'}")
    print(f"  guardrail ............... {'BLOQUEADA' if veredito['bloqueado'] else 'liberada'}")

    return {
        "rotulo": rotulo,
        "ancoragem": proporcao,
        "citou_fonte_valida": bool(validas),
        "bloqueada": veredito["bloqueado"],
    }


def concluir(medicoes: list[dict], cabe: bool) -> None:
    print(f"\n{'=' * 74}\nCONCLUSÕES DO SPIKE\n{'=' * 74}")

    print(f"\n1. O contexto cabe na janela: {'SIM' if cabe else 'NÃO'}")

    if not medicoes:
        print("\n2. Uso do contexto: não avaliado (rodou com --so-prompt).")
        print("3. Sobrevivência da citação: não avaliada.")
        print("\nPara fechar 2 e 3 é preciso rodar numa GPU com ~6 GB ou no Colab.")
        return

    com = next((m for m in medicoes if "com LoRA" in m["rotulo"]), None)
    sem = next((m for m in medicoes if "sem LoRA" in m["rotulo"]), None)

    print("\n2. Uso do contexto:")
    for m in medicoes:
        print(f"   {m['rotulo']:<38} ancoragem {m['ancoragem']:.1%}")
    if com and sem:
        delta = com["ancoragem"] - sem["ancoragem"]
        if delta < -0.05:
            print("   -> O modelo BASE ancora mais que o ajustado. O fine-tuning está")
            print("      puxando a resposta para a memória paramétrica, contra o RAG.")
        elif delta > 0.05:
            print("   -> O adaptador ancora mais que o base. O fine-tuning não")
            print("      atrapalha a leitura do contexto neste caso.")
        else:
            print("   -> Diferença dentro do ruído; nenhum dos dois se destaca.")

    print("\n3. Sobrevivência da citação:")
    for m in medicoes:
        estado = "citou [FONTE n] válida" if m["citou_fonte_valida"] else "NÃO citou fonte válida"
        print(f"   {m['rotulo']:<38} {estado}")
    if not any(m["citou_fonte_valida"] for m in medicoes):
        print("   -> Nenhuma geração citou fonte. O guardrail de citação recomendado")
        print("      ao Integrante 4 rejeitaria toda resposta. Ajustar a instrução")
        print("      antes de adotá-lo como critério de bloqueio.")

    if any(m["bloqueada"] for m in medicoes):
        print("\n4. Guardrail: houve bloqueio. Conferir se foi prescrição de fato ou")
        print("   se a resposta apenas reproduziu uma dosagem presente na fonte —")
        print("   o falso bloqueio previsto quando o RAG entrou no caminho.")


# --------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--pergunta", default=PERGUNTA_PADRAO)
    parser.add_argument("--k", type=int, default=4)
    parser.add_argument("--max-tokens-contexto", type=int, default=1200)
    parser.add_argument("--paciente", default=None, metavar="PAC-NNNN",
                        help="recorta o contexto para um paciente do corpus sintético")
    parser.add_argument("--so-prompt", action="store_true",
                        help="não carrega a LLM; valida só o orçamento de contexto")
    parser.add_argument("--comparar", action="store_true",
                        help="gera com e sem o adaptador LoRA, sob o mesmo contexto")
    parser.add_argument("--sem-contexto", action="store_true",
                        help="baseline: gera sem RAG, com a instrução de treino")
    args = parser.parse_args()

    from src.rag.retriever import MedicalRetriever

    print("=" * 74)
    print("SPIKE — recuperação + adaptador LoRA no mesmo processo")
    print("=" * 74)
    print(f"\nPergunta: {args.pergunta!r}")

    # -- recuperação -------------------------------------------------------

    contexto = ""
    fontes_disponiveis = 0

    if not args.sem_contexto:
        filtros = {"patient_id": args.paciente} if args.paciente else None
        try:
            retriever = MedicalRetriever()
        except RuntimeError as erro:
            print(f"\nERRO: {erro}", file=sys.stderr)
            return 1

        resultados = retriever.retrieve(args.pergunta, k=args.k, filters=filtros)
        suficiente = retriever.tem_evidencia_suficiente(resultados)

        print(f"Filtros:  {filtros or 'nenhum'}")
        print(f"\n  {len(resultados)} trechos recuperados"
              f"   evidência suficiente: {'SIM' if suficiente else 'NÃO'}")
        for r in resultados:
            m = r.metadados
            print(f"    [{r.rank}] {r.score:.4f}  {m['document_type']:<22} {m['title'][:44]}")

        if not suficiente:
            print("\n  O retriever abstém-se. Num fluxo real a LLM NÃO seria chamada —")
            print("  é o guardrail mais barato do sistema. Seguindo assim mesmo,")
            print("  porque o objetivo aqui é medir o comportamento do gerador.")

        tokenizer_medicao = carregar_tokenizer()
        contexto = retriever.format_context(
            resultados,
            max_tokens=args.max_tokens_contexto,
            contar_tokens=(lambda t: contar(tokenizer_medicao, t)),
        )
        fontes_disponiveis = len(re.findall(r"\[FONTE \d+\]", contexto))
        print(f"\n  {fontes_disponiveis} de {len(resultados)} trechos couberam no bloco de contexto")
    else:
        tokenizer_medicao = carregar_tokenizer()
        print("\n  Baseline sem RAG: nenhum contexto recuperado.")

    # -- orçamento ---------------------------------------------------------

    instrucao = INSTRUCAO_TREINO if args.sem_contexto else INSTRUCAO_RAG
    prompt = montar_prompt(contexto, args.pergunta, instrucao)
    cabe = relatar_orcamento(tokenizer_medicao, prompt, contexto)

    if args.so_prompt:
        print(f"\n{'-' * 74}\nPrompt que seria enviado à LLM:\n{'-' * 74}")
        print(prompt)
        concluir([], cabe)
        return 0 if cabe else 1

    # -- geração -----------------------------------------------------------

    try:
        model, tokenizer = carregar_modelo()
    except FileNotFoundError as erro:
        print(f"\nERRO: {erro}", file=sys.stderr)
        return 1
    except Exception as erro:  # noqa: BLE001
        texto = f"{type(erro).__name__}: {erro}".lower()
        if any(s in texto for s in ("out of memory", "dispatched on the cpu", "enough gpu ram")):
            print("\nERRO: o modelo não cabe na VRAM desta GPU (~5,5 GB necessários).")
            print("Limite de hardware. Rode com --so-prompt aqui, e o spike completo")
            print("numa GPU maior ou no Colab.", file=sys.stderr)
            return 1
        raise

    medicoes = []
    resposta = gerar(model, tokenizer, prompt)
    rotulo = "SEM RAG (baseline de fine-tuning)" if args.sem_contexto else "COM RAG, com LoRA"
    medicoes.append(relatar_geracao(rotulo, resposta, contexto, fontes_disponiveis))

    if args.comparar:
        with adaptador_desligado(model):
            resposta_base = gerar(model, tokenizer, prompt)
        medicoes.append(
            relatar_geracao("COM RAG, sem LoRA (modelo base)", resposta_base,
                            contexto, fontes_disponiveis)
        )

    concluir(medicoes, cabe)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
