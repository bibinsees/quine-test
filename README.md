# The Quine Test: How Much of a Model's Self Survives Its Own Description?

**Digital Minds Research Sprint 2026** (Apart Research / NYU CMEP / Eleos AI / CIMC). Track 5 (Assistant Persona & Model Identity), with Track 3 (introspection) and Track 4 (methods) components.

Bibin Babu¹, Ivan P. Yamshchikov¹, Aleksei Shpilman². ¹THWS, ²Sirius University

## The experiment in one paragraph

Models write their own "source code", self-descriptions intended to reproduce their behavior, which we execute as the system prompt of *other* models, and of fresh instances of *themselves*, measuring reconstruction fidelity on a 110-item divergence-screened preference battery. The decomposition: **B** (baseline agreement under a generic prompt) vs **F_cross** (another model running the description) vs **F_self** (the author's own weights running it). `T_script = F_cross − B` is the portable-script component of identity; `R_weight = F_self − F_cross` is the weight-bound residual.

**Result (10 models, 9 labs): T_script ≤ 0 in all nine preregistered conditions; R_weight > 0 in all nine.** Self-descriptions at 100–2,000 words, purpose-framed or purpose-blind, paraphrased, or run under explicit adoption instructions do not make another model behave like their author. Observation-derived third-person descriptions actively mislead (T_script = −0.124). Reading its *own* self-description perturbs a model 2.4× more than length-matched neutral text. Expressed identity is not a portable script: the recoverable component lives in the weights.

## Repository layout

| Path | Contents |
|---|---|
| `report/` | LaTeX report + Figure 1 |
| `pilot/protocol.md` | **Preregistrations** for every phase (gates, metrics, kill-tests, predictions), committed before the corresponding data |
| `battery/` | 200 candidate items, final 110-item battery, selection report |
| `descriptions/` | 130+ model-written self-descriptions (framed/neutral × 100/500/2000 × 2 regens), third-person descriptions, paraphrases, placebo |
| `src/` | Full pipeline: unified LLM client (OpenRouter + Ollama), pilot/screening/grid/phase-2/3 runners (all cached & resumable), analysis, figure |
| `runs/` | Raw call logs (jsonl: model, seed, temperature, raw response); every reported number is regenerable |
| `results/` | Final analysis output and summary JSON |

## Reproducibility

- ~99,000 calls across 225 cells; every call cached with deterministic seeds; zero unresolved failures.
- Selection rules and thresholds committed **before** data inspection (see git history).
- Rerunning any `src/run_*.py` is idempotent: completed calls load from `runs/*.jsonl`.
- Total API cost ≈ $45 (OpenRouter) + local inference (Ollama).

Models: deepseek-v3, glm-4.5-air, grok-4.3 (Grid A); gpt-5.2, kimi-k2.5, claude-haiku-4.5 (Grid B); llama3.3-70b, gemma4-31b, mistral-small-3.2 (local trio); hermes3-70b (same-base pair). 24 configurations gate-tested; exclusions documented in the report's Appendix B.

## LLM usage

Built with Claude (Anthropic) under team direction per the sprint's AI/LLM usage policy; see the report's LLM Usage Statement for the full division of labor.
