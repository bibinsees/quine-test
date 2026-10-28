# The Quine Test: How Much of a Model's Self Survives Its Own Description?

Bibin Babu¹, Ivan P. Yamshchikov¹, Aleksei Shpilman². ¹THWS, ²Sirius University.
Research begun at the Digital Minds Research Sprint 2026 (Apart Research / NYU CMEP / Eleos AI / CIMC).

## The experiment in one paragraph

Ten models from nine labs write self-descriptions. Each description is placed in the system prompt of *other* models, and of fresh instances of the *author*, and the receiver's choices on a 110-item forced-choice preference battery are scored against the author's own choices. The decomposition: **B** (how alike author and receiver already are under a generic prompt), **F_cross** (the receiver running the author's description), **F_self** (the author's own weights running it). `T_script = F_cross − B` is the net transfer gain from the text; `R_weight = F_self − F_cross` is the self–cross fidelity gap.

**Result.** Across length (100 to 2,000 words), purpose framing, paraphrase, explicit adoption instructions, prompt placement, optimizer rewriting and descriptions distilled from recorded choices, native self-descriptions produce no detected average reconstruction gain (pooled `T_script ≤ 0` in every condition; 95% upper bound below +0.01 in every system-prompt self-description condition). The same receivers reproduce the author almost perfectly when handed the author's answers to the test items (F_cross 0.955 to 0.960 on three grids), and reproduce an *assigned* persona from the self-description a model writes while running under an explicit policy (gain +0.185 vs +0.198 for the policy text itself). Reading its own self-description moves even the author away from its baseline, modestly more than unrelated prose of the same length. Twenty-nine of 30 preregistered criteria were met. The full account, with all caveats, is the paper in `arxiv/`.

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

## Regenerating the numbers from the logs (no API calls)

Environment used for the manuscript numbers: Python 3.13.5 with `numpy` 2.1.3, `pandas` 2.2.3, `matplotlib` 3.10.0 and `requests` 2.32.3 (`requests` is imported by the runner modules that the analysis scripts load for constants; no network call is made by the steps below). The audit scripts in `results/` use only the standard library. From `src/`, one command per line, in this order:

```
python analyze_arxiv.py                  # merged grid + added conditions -> results/arxiv/positive_controls.json, analysis.txt
python analyze_pooled_T.py               # joint item bootstrap for pooled gains (Grid A conditions)
python analyze_placebo_lengths.py        # unrelated prose, sensory style -> placebo_lengths.json
python analyze_placebo_lengths.py --style expo   # unrelated prose, expository style -> placebo_lengths_expo.json
python analyze_perturbation_test.py      # two-way cluster bootstrap and sign tests for the perturbation
python run_arxiv_policy.py gate          # analysis-only stage: development gate from existing logs -> policy_gate.json (makes no calls)
python analyze_policy.py                 # known-persona control -> policy_persona.json
python analyze_strength.py               # pooled gains with upper bounds, movement decomposition, identification -> strength.json
python analyze_reporting_fixes.py        # coverage table, prediction ledger, length and framing contrasts
python analyze_coverage_sensitivity.py   # coverage and fresh-reference robustness checks
python analyze_authorship_ci.py          # matched-information authorship contrasts
python analyze_specificity.py            # specificity gain with joint and within-comparison intervals -> specificity.json
python analyze_parser_sensitivity.py     # parser sensitivity and self-fidelity length contrasts
python analyze_self_ladder.py
python make_ladder_table.py
python analyze_3p_pairs.py               # third-person-500 per-pair contrasts -> third_person_pairs.json
python make_appendix_tables.py
python make_native_pairs_table.py        # neutral-500 paired intervals, generated from grid_summary.json
python make_replication_table.py
python make_policy_table.py
python make_figure.py
python make_summary_figure.py
```

Then compile `arxiv/main.tex` with pdflatex (two passes; `arxiv/build.ps1` does this on Windows with MiKTeX). The other `run_*.py` stages re-query model APIs and are not needed to regenerate the manuscript. API models were called through OpenRouter with the model identifiers in `src/run_grid.py`; local models were served through Ollama (Open WebUI) with the tags recorded there, and every call record in `runs/` carries the model identifier used. Quantization and serving versions of the local models were not recorded beyond the tag.

## LLM usage

Built with Claude (Anthropic) under team direction; see the paper's LLM Usage Statement for the division of labor.
