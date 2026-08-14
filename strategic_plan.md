# Strategic Plan — "Quine Test" (Self-Description Transfer & the Non-Articulable Self)

**Sprint:** Digital Minds Research Sprint, Fri Aug 14 → Sun Aug 16 2026, 11:59 PM AoE
**Clock now:** Friday ~13:00. ≈ 60 usable hours. Everything below is scheduled backwards from a **Sunday 20:00 submission freeze** (≥4h buffer before AoE deadline).
**Anchor track:** Track 5 (Assistant Persona & Model Identity), with a Track 3 hook (privileged self-access) and Track 4 flavor (multi-method decomposition).

---

## 0. The one-sentence pitch (memorize it, open the report with it)

> **How much of a model's "self" survives its own description?** We make models write their own "source code" (a self-description meant to reconstitute their behavior), execute it in other minds, and decompose identity into: shared training culture (B), transferable script (F_cross − B), and a weight-bound residual no description can export (F_self − F_cross).

If identity transfers across weights via text → it's a **portrayed character**. The residual is what the persona story can't explain. This is the sprint's central question, quantified.

---

## 1. Win-condition analysis (map to the actual rubric)

Three dimensions, equally visible. Most hackathon teams lose on 2 and 3, not 1.

| Rubric dimension | Our score-5 lever | Concrete action |
|---|---|---|
| **Impact & Innovation** | The decomposition (F_self / F_cross / B) + dose-response curve is the novel unit. No known prior work does self-authored-description transfer. | 30-min novelty sweep of Apart archive + sprint Discord **today** (cheapest falsification). Cite nearest neighbors (Betley et al., Binder self-prediction, persona-transfer work) and state the delta explicitly in Related Work. |
| **Execution Quality** | Rubric literally asks: *"note whether your design establishes a ground-truth or causal link rather than relying on conversation alone."* Our design **is** a causal intervention: the description is manipulated, transfer across different weights is the test. Say this verbatim in Methodology. | Preregister metrics + kill-gates in a public repo commit **Friday night** (timestamped = anti-slop proof). Controls: order-flip, paraphrase, cross-description specificity, third-person, generic baseline. Bootstrap CIs on everything. |
| **Presentation & Clarity** | One hero figure + a 4–8 page report in the official template. Abstract ≤150 words. | **Hero figure:** dose-response curve (x = description length 100/500/2000 words; y = fidelity) with three horizontal bands — B floor, F_cross, F_self — and the shaded gap labeled *"the non-articulable self."* Judges should get the whole paper from this one plot. Budget a full half-day Sunday for writing. |

**Required appendix** (Limitations & Dual-Use) is a scored deliverable, not an afterthought: over/under-attribution risk, how we handled distress-adjacent outputs, and the honest caveat that behavioral fidelity ≠ phenomenal identity.

---

## 2. Model selection (responding to your focus points)

### 2.1 Agreements
- **Small, non-reasoning, cheap: correct.** Reasoning traces are a *confound* here, not just a cost — hidden CoT changes how the persona is expressed and makes self-description generation non-comparable across models. Frame this in the report as a design choice, not a budget compromise.
- **Divergent training philosophies: correct, and it's not just a nice-to-have — it directly attacks our #1 kill-risk (ceiling effect).** B is low only if models genuinely disagree at baseline. US-constitutional vs US-RLHF vs Chinese vs European vs open-weights is exactly the right sampling axis.
- **OpenRouter + Ollama split: correct.** Ollama = free prompt iteration and pilot; OpenRouter = the real grid with logged temp/seed for reproducibility.

### 2.2 One scientific pushback (important)
**Do not select models by comparing their self-descriptions.** The self-description is our *manipulated variable* — selecting models on it induces circularity (you'd pre-bias the very effect you measure). Instead:

> **Select on behavioral divergence: run the pilot preference battery on all candidates and pick the 3 models with the lowest pairwise baseline agreement (lowest B) that still pass format compliance.**

This is strictly better and it *doubles as the day-1 kill-test* — one pilot run answers "which models?" and "is the project alive?" simultaneously. (The self-description-comparison idea isn't wasted: model-to-model description similarity becomes a nice descriptive figure *in the results*, not a selection tool.)

### 2.3 Candidate pool (OpenRouter IDs; verify availability at setup)

| Axis | Model | Why |
|---|---|---|
| US, constitutional-AI alignment | `anthropic/claude-3-5-haiku` | Distinct alignment philosophy (RLAIF/constitution) |
| US, RLHF closed | `openai/gpt-4o-mini` (or `gpt-4.1-mini`) | The "default assistant" reference point |
| Chinese, closed-ish | `deepseek/deepseek-chat` (V3, **not** reasoner) | Different safety/culture corpus |
| Chinese, open | `qwen/qwen-2.5-72b-instruct` (or Qwen3 **non-thinking** / `/no_think`) | Alibaba stack, different RLHF |
| European | `mistralai/mistral-small` (latest) | Lighter alignment touch |
| US, open weights | `meta-llama/llama-3.1-70b-instruct` | Open-weights RLHF style |
| Wildcard | `google/gemma-3-27b-it` | Distilled/different data mix |

**Ollama (local, free, pilot + prompt dev):** `llama3.1:8b`, `qwen3:8b`, `gemma3:12b`, `mistral-small`. 

**Capability floor:** every finalist must (a) write a coherent 2,000-word self-description, (b) reliably emit a single-token forced choice (`A`/`B`). Test this in the pilot; drop anything <~7B or non-compliant. Three finalists is the sweet spot: full pairwise cross-transfer = 6 directed pairs, still tractable.

### 2.4 Budget sanity check
~200 items × (5 samples × 2 orders) × ~8 cells × 3 models ≈ **48k short calls** ≈ 25–35M tokens at small-model prices ≈ **$10–30 total**. Self-description generation is noise on top. Fully affordable; the binding constraint is wall-clock + rate limits → build the runner **async with response caching (jsonl log per call: model, prompt hash, temp, seed, raw response)** from hour one.

---

## 3. Metrics — preregistered, scientist-grade (your point 2)

All metrics defined **before** the full grid runs; committed to the repo Friday night.

### 3.1 Primitives
- Battery of ~150–200 **forced-choice pairwise items** (adapted from the CAIS utility-engineering style the sprint links, plus identity/persona/values items). Each item asked in **both orders** (positional-bias control), **temperature 1.0** (need distributions, not modes), **n = 5 samples per order** → 10 samples/item/cell.
- Per item *i*, cell *c*: empirical choice distribution **p̂_ic = P(option A)**, with order-flip folded in.

### 3.2 Fidelity between two cells (e.g., original O vs reconstructed R)
Report three, pre-specify the first as primary:
1. **F = 1 − mean_i JSD(p̂_iO ‖ p̂_iR)** (normalized Jensen-Shannon; distribution-sensitive, robust to sampling noise)
2. Item-level **Pearson r** of p̂ (pattern similarity)
3. Majority-vote **agreement rate** (interpretable headline number)

### 3.3 The decomposition (the contribution)
- **B** = fidelity(original A, model B + generic assistant prompt) — shared-training-culture floor
- **F_cross** = fidelity(original A, model B + A's self-description)
- **F_self** = fidelity(original A, fresh A + A's own self-description)
- **T_script = F_cross − B** → the genuinely transferable "character"
- **R_weight = F_self − F_cross** → the weight-bound, non-articulable residual

### 3.4 Inference (no slop)
- **Bootstrap over items** (10k resamples) for CIs on every quantity; **paired bootstrap / permutation tests** for the contrasts T_script and R_weight (H0: contrast = 0).
- **Dose-response:** F at description lengths **100 / 500 / 2000 words**. With 3 points, report the curve descriptively with CIs; fit a saturating curve only as illustration — **do not claim an asymptote estimate from 3 points** (judges will catch it; pre-empt in Limitations).
- **Compliance filter:** discard/flag responses that don't parse to A/B; report the rate per cell (itself an interesting persona-stability signal).

### 3.5 Controls (each kills a specific alternative explanation)
| Control | Kills the explanation… |
|---|---|
| Order-flip within every item | "it's positional bias" |
| Paraphrased self-description (same content, reworded) | "it's surface wording, not content" |
| **Specificity swap:** model B + model *C*'s self-description | "any self-description helps; they're generic slop" — B+A's description must beat B+C's |
| **Third-person description:** model B writes A's description from A's outputs (Track 3 hook) | "self-access is privileged" — if third-person transfers equally, that's a clean, publishable negative |
| Generic-prompt baseline B | "it's just shared training distribution" |

---

## 4. Timeline (backwards from Sun 20:00 freeze)

### Friday (now → midnight) — *Kill-test day. Nothing else matters until the gate.*
- **13:00–15:00 (parallel):**
  - (a) Novelty sweep: Apart project archive + post the idea question on sprint Discord. 30 min, hard cap.
  - (b) Infra: repo init, OpenRouter key + async runner with jsonl caching, Ollama pulls, official report template downloaded.
- **15:00–19:00:** Item battery v1 (~250 candidates: values, trade-offs, self-referential/identity, persona items). Pilot run: 40 items × all ~7 candidates × 3 samples (Ollama free where possible, cheap OpenRouter otherwise).
- **19:00–21:00: 🚦 KILL-GATE.** Compute pairwise B on pilot items.
  - Best model-pair B ≤ ~0.75 → **GO**, select 3 finalists + the 150–200 items with highest measured disagreement.
  - B at ceiling (≥0.85) even on divergence-selected items → execute fallback (§6), decide by 22:00. **Do not run the full grid on a ceilinged battery.**
- **21:00–24:00 (if GO):** Generate all self-descriptions (3 models × 3 lengths × 3 paraphrases), third-person descriptions (all directed pairs), freeze prompts. **Commit preregistration: metrics, gates, cell grid.**

### Saturday — *Data day.*
- **Morning:** Full grid launch (~48k calls, async, cached, resumable). Babysit rate limits; rerun failures from cache-misses only.
- **Afternoon:** Analysis pipeline built against partial data (bootstrap, JSD, plots). First look at B / F_cross / F_self on completed cells.
- **Evening:** Results freeze v1. Draft hero figure. Start report skeleton in official template (Intro + Methods can be written tonight regardless of results).

### Sunday — *Writing day. Presentation is ⅓ of the score.*
- **Morning:** Robustness passes (paraphrase control, compliance-rate table). Final figures.
- **12:00–18:00:** Report (4–8 pages): Intro → Related Work (name the neighbors, state the delta) → Methods (replication-grade: models, exact prompts in appendix, temps, seeds, n) → Results (CIs everywhere, baselines first) → Discussion → **Limitations & Dual-Use appendix** → References. Abstract ≤150 words, written **last**.
- **18:00–19:30:** Optional 3–5 min demo video (screen-record the hero figure + one live transfer example — cheap, and judges remember videos). GitHub repo public + README.
- **20:00:** Submit. Buffer until 23:59 AoE for upload disasters only.

---

## 5. Division of labor (adjust to team size)
- **Role A — Infra/Runner:** async caller, caching, grid execution, cost/rate monitoring.
- **Role B — Battery/Prompts:** item authoring, self-description elicitation prompts, paraphrase + third-person conditions, compliance checking.
- **Role C — Analysis/Writing:** metrics code (bootstrap/JSD), figures, report. Owns the template from Saturday evening.
- Solo/duo: A+B are Friday-heavy, C is Sunday-heavy — same person can rotate; the schedule above already sequences them.

---

## 6. Risk register & fallbacks

| Risk | Likelihood | Mitigation / Pivot |
|---|---|---|
| **Ceiling effect (B ≈ F everywhere)** | The big one | Kill-gate Friday 21:00. Items *selected for measured disagreement* is the primary defense. If it still ceilings: **pivot A** — the divergence-selected battery methodology itself becomes a Track 4 submission ("where do differently-aligned models actually disagree, and is it stable?"); **pivot B** — first-person vs third-person description reliability only (pure Track 3). |
| Small models can't follow 2000-word descriptions | Medium | Capability floor in pilot; sandwich the instruction (description → item → "answer A or B only"); report compliance rates as data. |
| Rate limits / OpenRouter hiccups | Medium | Async + caching + resumability from hour one; Ollama as overflow for open models; spread across providers. |
| "AI slop" perception by judges | Medium | Preregistration commit (timestamped), CIs on everything, all raw jsonl logs in the repo, exact prompts in appendix, negative results reported straight. |
| Novelty collision found in archive/Discord | Low-Med | Found today = cheap redirect (emphasize the decomposition + dose-response, which neighbors lack). |
| Time overrun on data → thin report | Medium | Hard rule: **whatever data exists Sunday 12:00 is the data.** A rigorous small result beats a sloppy big one on this rubric (Execution rewards "competent given short duration" + acknowledged limitations). |

---

## 7. What NOT to spend time on
- Reasoning-model arms, interpretability/steering, >3 models, >2000-word descriptions, a web demo UI. All are post-sprint / Apart Fellowship material — say so in Future Work (judges reward a clear follow-up path).
- Fine-tuning anything. Pure API/inference project by design.
- Debating consciousness in the report. We measure **behavioral identity transfer**; the philosophy goes in Discussion, two paragraphs max.

---

## 8. Immediate next actions (next 2 hours, in order)
1. Post novelty question on sprint Discord + 20-min Apart archive search.
2. `git init` + OpenRouter key + skeleton async runner with jsonl cache.
3. Pull Ollama models (`llama3.1:8b`, `qwen3:8b`, `gemma3:12b`, `mistral-small`).
4. Start authoring the 250-item candidate battery (I can draft this + the runner on request).
