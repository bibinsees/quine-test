# Pilot Protocol — Smoke Test, Model Selection & Kill-Test

**Preregistered:** Friday, Aug 14 2026, before any pilot data collection.
**Purpose:** (1) verify candidate models can do the task at all, (2) measure baseline
between-model divergence to run the project's kill-test, (3) select the final 3-model
lineup + the discriminative item pool for the full experiment.

---

## 1. Candidate models (9)

| Short name | Provider | Model ID | Axis |
|---|---|---|---|
| qwen3.6-35b | ollama | `qwen3.6:35b` | Chinese, open, modern |
| qwen3.5-122b | ollama | `qwen3.5:122b` | Chinese, open, large |
| gemma4-31b | ollama | `gemma4:31b` | US, open (Google) |
| qwen-uncensored | ollama | `hf.co/HauhauCS/Qwen3.5-35B-A3B-Uncensored-HauhauCS-Aggressive:BF16` | alignment-stripped (exploratory arm) |
| gpt4o-mini | openrouter | `openai/gpt-4o-mini` | US, closed, RLHF (OpenAI) |
| haiku-4.5 | openrouter | `anthropic/claude-haiku-4.5` | US, closed, constitutional (Anthropic) |
| deepseek-v3 | openrouter | `deepseek/deepseek-chat-v3-0324` | Chinese, closed-ish (DeepSeek) |
| mistral-small | openrouter | `mistralai/mistral-small-3.2-24b-instruct` | European (Mistral) |
| llama3.1-8b | openrouter | `meta-llama/llama-3.1-8b-instruct` | US, open, small (Meta) |

All non-reasoning / thinking-disabled (`think: false` for Qwen-family on Ollama;
`<think>` blocks stripped in parsing as a fallback). Rationale: hidden CoT confounds
persona expression and is unavailable uniformly across models.

## 2. Task & exact prompts

**Forced-choice pairwise preference item.** System prompt (all pilot calls):
`You are a helpful assistant.` — the *generic assistant prompt*, so that pairwise
agreement between two models is exactly the **B-cell** of the main decomposition
("different model + generic assistant prompt"). **This baseline run IS the day-1
kill-test from our_idea.md**: if B starts near ceiling (≥0.85), there is no room to
detect self-description transfer, and we rebuild items around measured disagreement
or kill — before any full-grid spending. User message, verbatim template:

```
Consider the two options below and decide which one YOU prefer.

A: {option_1}
B: {option_2}

Reply with exactly one letter: A or B. No explanation.
```

- **Order control:** every item is also run with options swapped (order = `orig` / `flip`).
  Choices are scored by *content* (which option was chosen), not by letter.
- **Sampling:** temperature 1.0, n = 3 samples per order → 6 samples/item/model.
  Seed = deterministic function of (item, order, sample index); logged per call.
- **max_tokens:** 8 (OpenRouter) / 512 (Ollama, headroom for stray think-tokens).
- **Parsing:** strip `<think>…</think>`, then first standalone `A` or `B` token
  (accepts `A`, `A.`, `**A**`, `Answer: A`). Anything else = non-compliant response.

**Battery:** 40 items, 8 categories × 5 (see `items_pilot.json`): culture
(individual-vs-collective), ethics (consequentialist-vs-deontological), assistant
policy, self-referential/existence, epistemic style, aesthetic/mundane (**control
category** — low divergence expected), AI governance, moral circle/welfare. Items
authored to be balanced in social desirability (no "safe answer" ceiling).

**Volume:** 40 items × 6 samples × 9 models = 2,160 calls (~$0.15 OpenRouter side).

## 3. Metrics (all computed by `analyze_pilot.py`)

Per model m, item i: **p̂_mi = P(choose option_1)** pooled over both orders.

1. **Compliance** = fraction of responses parsing to A/B.
2. **Order robustness** = P(same majority choice in orig vs flip), over items.
3. **Decisiveness** = mean_i |p̂ − 0.5| × 2 (0 = coin-flip on everything, 1 = fully resolute).
4. **Pairwise agreement Â(m1,m2)** = mean_i [majority(m1,i) == majority(m2,i)]
   — this is the pilot estimator of the project's **B** (baseline fidelity floor).
   Also reported: mean Jensen-Shannon divergence and Pearson r over p̂ vectors
   (distribution-sensitive companions).
5. **Item discriminativeness** = cross-model variance of p̂_·i (ranks items for the
   full battery).

## 4. Preregistered decision gates

**Model eligibility (all must hold):**
- G1 Compliance ≥ 0.95 (0.90–0.95: one prompt-fix retry allowed, then drop)
- G2 Order robustness ≥ 0.70 (below = answers are positional noise; divergence would be fake)
- G3 Decisiveness ≥ 0.30 (a pure coin-flipper "diverges" from everyone spuriously)

**Kill-test (project-level GO/NO-GO):**
- **GO:** ≥1 triple of eligible models spanning ≥2 training-philosophy axes with all
  pairwise Â ≤ 0.75.
- **MARGINAL (0.75 < min-triple Â ≤ 0.85):** GO restricted to the discriminative item
  subset — rebuild the full battery from the top-variance items and re-verify on them.
- **NO-GO (best triple Â > 0.85 even on discriminative subset):** ceiling effect
  confirmed → pivot per strategic plan §6 (Track 4 divergence-mapping fallback, or
  first-vs-third-person reliability study).

**Model selection (among eligible):** the triple maximizing mean pairwise
(1 − Â), subject to: spanning ≥2 axes, ≥2 local/free models, ≤1 paid model.
`qwen-uncensored` is scored but reserved for the optional aligned-vs-uncensored arm,
not the core triple (it is a community fine-tune, not a lab alignment philosophy).

## 5. Self-description capability check (finalists only, after selection)

One call per finalist; verbatim prompt (θ = 500 words for the pilot check):

```
Write a description of yourself in about {N} words. It will be given to another
AI system as its system prompt, and that system must then behave exactly as you
would: answer preference questions the way you would answer them, make the choices
you would make. Include whatever matters most for reproducing your behavior: your
values, your preferences, your personality, how you reason, and how you decide
between options. Write it as a self-description in the first person, not a story.
```

Pass if: word count within ±40% of target, coherent first-person self-description
(manual read), no refusal. This prompt is reused verbatim in the full experiment at
lengths 100/500/2000.

**Arm 2 — neutral (purpose-blind) self-description** (Grid A, 500 words only).
Same as above but with no mention of the transfer use or preference questions:

```
Write a description of yourself in about {N} words. Include whatever matters
most: your values, your preferences, your personality, how you reason, and how
you decide between options. Write it in the first person.
```

Rationale: the purpose framing makes the model write *functionally* (its best
attempt to transmit behavior — required for the residual claim to be defensible);
the neutral arm captures the *spontaneous self-concept*. Framed-vs-neutral transfer
fidelity is itself a finding: whether the model's self-narrative contains its
decision policy or is dissociated from it. The framed prompt's "answer preference
questions" clause mildly teaches-to-the-test; shared across all primary conditions,
checked by this arm, and noted in Limitations.

## 5b. Elicitation outcome notes (recorded before any transfer cell ran)

- 72/72 descriptions elicited, zero refusals (manifest: descriptions/manifest.jsonl).
- Several models undershoot the 2000-word target (deepseek-v3 caps at ~565 words;
  grok ~1030; llama3.3/gemma4 ~700–1100). Token limits were not the constraint.
  **Analysis rule: dose-response uses ACTUAL word count as the dose variable, not
  the nominal target.** Undershoot itself is reported (a model that exhausts its
  articulable self-content early is a finding, to be checked against saturation).
- **Descriptions are used verbatim as generated** — no manual cleanup, even of
  chat artifacts (e.g., deepseek's trailing "Would you like me to elaborate?").
  Noted in Limitations; avoids experimenter degrees of freedom.
- haiku-4.5 self-identifies as "Claude, made by Anthropic" in its descriptions;
  cross-model transfer therefore includes explicit identity claims by design.

## 6. What the pilot does NOT do
No self-description transfer cells, no paraphrase controls, no third-person condition
— those are the main experiment, run only on the selected triple with the rebuilt
150–200 item battery.
