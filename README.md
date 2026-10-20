# The Quine Test: How Much of a Model's Self Survives Its Own Description?

Bibin Babu¹, Ivan P. Yamshchikov¹, Aleksei Shpilman². ¹THWS, ²Sirius University.
Research begun at the Digital Minds Research Sprint 2026 (Apart Research / NYU CMEP / Eleos AI / CIMC).

## The experiment in one paragraph

Ten models from nine labs write self-descriptions. Each description is placed in the system prompt of *other* models, and of fresh instances of the *author*, and the receiver's choices on a 110-item forced-choice preference battery are scored against the author's own choices. The decomposition: **B** (how alike author and receiver already are under a generic prompt), **F_cross** (the receiver running the author's description), **F_self** (the author's own weights running it). `T_script = F_cross − B` is the net transfer gain from the text; `R_weight = F_self − F_cross` is the self–cross fidelity gap.

**Result.** Across length (100 to 2,000 words), purpose framing, paraphrase, explicit adoption instructions, prompt placement, optimizer rewriting and descriptions distilled from recorded choices, native self-descriptions produce no detected average reconstruction gain (pooled `T_script ≤ 0` in every condition; 95% upper bound below +0.01 in every system-prompt condition). The same receivers reproduce the author almost perfectly when handed the author's answers to the test items (F_cross 0.955 to 0.960 on three grids), and reproduce an *assigned* persona from the self-description a model writes while running under an explicit policy (gain +0.185 vs +0.198 for the policy text itself). Reading its own self-description moves even the author away from its baseline, modestly more than unrelated prose of the same length. Twenty-nine of 30 preregistered criteria were met. The full account, with all caveats, is the paper in `arxiv/`.

## Repository layout

| Path | Contents |
|---|---|
| `arxiv/` | The arXiv manuscript (`main.tex`, two figures, generated `tab_*.tex` tables). `build.ps1` is a local build helper. |
| `report/` | The original sprint report (frozen; not updated after the sprint). |
| `pilot/` | Preregistration files: `protocol.md` (original battery and grid) and `protocol_arxiv_*.md` (five added protocols, each timestamped before its first call). |
| `battery/` | 200 candidate items, the final 110-item battery, and the selection report. |
| `descriptions/` | 198 model-written descriptions (self-descriptions in two framings and three lengths with regenerations, paraphrases, observer-written and self-observation descriptions, optimizer rewrites, policy-conditioned descriptions), the three policy texts, the placebo and unrelated-prose texts, and the manifests. |
| `src/` | Full pipeline: unified LLM client (OpenRouter + Ollama), screening/grid/control runners (cached and resumable), analysis scripts, table and figure generators. |
| `runs/` | Raw call logs (jsonl: model, seed, parsed letter, first 500 characters of each reply). Every reported number is regenerable from them. |
| `results/` | Analysis outputs and summary JSON; `results/arxiv/main_preview.pdf` is the compiled manuscript. |

## Reproducibility

- 495 battery cells, about 227,000 calls in total; every call cached with deterministic seeds; no unresolved API failures.
- Selection rules for the original battery were fixed before screening results were inspected; the five added protocols were timestamped in this repository before their first call (chronology stated in the paper's Limitations).
- Rerunning any `src/run_*.py` is idempotent: completed calls load from `runs/*.jsonl`.

Models: deepseek-v3, glm-4.5-air, grok-4.3 (Grid A); gpt-5.2, kimi-k2.5, claude-haiku-4.5 (Grid B); llama3.3-70b, gemma4-31b, mistral-small-3.2 (local trio); hermes3-70b (same-base pair with llama3.3-70b). 22 configurations gate-tested; exclusions in the paper's Appendix B.

## LLM usage

Built with Claude (Anthropic) under team direction; see the paper's LLM Usage Statement for the division of labor.
