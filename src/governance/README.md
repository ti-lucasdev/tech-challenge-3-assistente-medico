# Governança (Parte 4)

Camada de segurança aplicada às respostas do modelo antes de chegarem ao
usuário final.

## `guardrails.py`

Filtro de conteúdo aplicado à resposta bruta do modelo:

- `contem_prescricao_direta(texto)` — verifica, por regex, se a resposta
  contém indícios de dosagem/prescrição direta (ex.: `"500 mg"`, `"dose ...
  10"`, `"tome 2"`, `"take 3"`, `"administre 5"`).
- `aplicar_guardrail(resposta)` — orquestra o filtro:
  - Se detectar prescrição direta, **bloqueia** a resposta original e a
    substitui por uma mensagem padrão recusando fornecer dosagem/prescrição.
  - Caso contrário, mantém a resposta do modelo, apenas anexando o aviso
    padrão (`AVISO_PADRAO`) de que se trata de conteúdo informativo gerado
    por IA, que não substitui avaliação profissional.
  - Retorna `{"resposta_final": str, "bloqueado": bool}`.

## `inferencia_segura.py`

Wrapper de inferência que liga o modelo treinado (adaptador LoRA), o
guardrail acima e o logging de auditoria:

- `carregar_modelo()` — carrega (uma única vez, em cache nas variáveis
  globais `_model`/`_tokenizer`) o modelo base + adaptador LoRA a partir de
  `artifacts/adaptador_medquad_lora_final` via `unsloth.FastLanguageModel`,
  em 4-bit, e o coloca em modo de inferência.
- `gerar_resposta_segura(pergunta)` — fluxo completo de uma pergunta:
  1. Monta o prompt no mesmo template Alpaca usado no treino.
  2. Gera a resposta bruta do modelo (`generate`, sem sampling).
  3. Passa a resposta por `aplicar_guardrail` (`guardrails.py`).
  4. Registra a interação (pergunta, resposta bruta, resposta final e se
     foi bloqueada) via `registrar_interacao` (`src/logging/logger.py`),
     para rastreabilidade/auditoria.
  5. Retorna apenas a `resposta_final` (já filtrada e com aviso).
- Executado diretamente (`python -m src.governance.inferencia_segura`),
  roda um exemplo fixo de pergunta médica e imprime a resposta segura.

Esse módulo é o ponto de entrada recomendado para uso do assistente em
produção/demo, em vez de chamar `src/inferencia.py` diretamente, pois
adiciona a camada de segurança e o log de auditoria.
