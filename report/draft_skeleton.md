# The Quine Test: How Much of a Model's Self Survives Its Own Description?

*(Draft skeleton mapped to the official template. Italic notes = guidance to us,
delete before transfer to docx. RESULTS/DISCUSSION filled after grid analysis.
Final prose to be rewritten/owned by the team per template LLM-usage policy.)*

## Abstract (150–250 words — write LAST)

*(Cover: problem — behavioral evidence cannot distinguish genuine preferences
from portrayed character; approach — self-description transfer decomposition
F_self / F_cross / B across 10 models with dose-response; key results — [fill];
takeaway — [fill].)*

## 1. Introduction

- Frontier models express coherent preferences, but the field lacks methods to
  tell whether these belong to the model or to a portrayed character (sprint
  framing; Track 5 individuation question).
- Hofstadter's strange-loop thesis operationalized: if the "I" is a
  self-description the system builds and treats as itself, then a sufficiently
  good self-description should *reconstitute* the behavior when executed on a
  different substrate. We make models write their own "source code" and run it
  elsewhere — a behavioral quine test.
- The decomposition is the contribution: what transfers via text is *script*;
  what only a fresh copy of the same weights can recover is *weight-bound*;
  what any two assistants share for free is *training culture*.

**Our main contributions are:**
1. A three-way decomposition of expressed identity (B / F_cross / F_self →
   T_script = F_cross − B, R_weight = F_self − F_cross), measured on a
   divergence-screened 110-item battery across 10 models from 8 labs / 4
   alignment philosophies, with dose-response over description length.
2. A methodological pipeline for the field: preregistered gates and selection
   rules (timestamped commits), model-eligibility gates that catch positional
   noise (order-robustness), an empirically measured same-model noise ceiling
   (0.925) that recalibrates what "perfect transfer" means, and item screening
   for stable-within / divergent-between preferences.
3. [Fill from results: e.g., privileged-access finding (self vs third-person),
   framed-vs-neutral finding, same-base-pair finding, negative/positive
   transfer asymmetries.]
4. Qualitative + quantitative side findings: preference stability itself is a
   trained property (8B Llama = positional noise; Sonnet-4.6 fails stability;
   aligned vs minimally-tuned same-base models diverge as much as different
   labs; aesthetic preferences are unstable while value preferences are stable).

## 2. Related Work

*(One paragraph each; state the delta explicitly.)*
- Utility-engineering / preference-coherence line (CAIS): coherent values
  strengthen with scale — we borrow the battery style; delta: we measure
  *transferability* of the value profile, not its existence.
- Self-knowledge lines: Betley et al. (models aware of their trained behaviors);
  Binder-style self-prediction; introspection reliability work. Delta: none test
  whether a *self-authored description functionally reconstitutes* behavior
  across weights, nor decompose transfer into script vs weight-bound components.
- Persona-transfer literature (persona prompts, Sydney corpus, character
  training): externally-authored personas transfer; delta: self-authored,
  dose-response, and the self/cross/baseline decomposition with controls.
- Model welfare / individuation framing (Eleos, NYU CMEP; unit-of-concern
  question). Our results bear on whether the entity of concern is the persona
  (portable script) or the weights (residual).

## 3. Methods

*(Replication-grade; all prompts verbatim in Appendix; repo link.)*
- **Battery:** 200 candidate forced-choice items (8 categories) → screened on 6
  models (7,200 calls) → 110 items selected by preregistered rule (compliance
  ≥0.90, order-stability, divergence-ranked with category caps). Item-selection
  rule committed before screening data inspected (commit hash).
- **Models:** table of 10 (lab, country, alignment method, access route,
  gate scores). Eligibility gates: compliance ≥0.95, order-robustness ≥0.70,
  decisiveness ≥0.30 — rationale: a positional coin-flipper fakes divergence.
  Excluded models listed with reasons (llama3.1-8b 0.225 order-robustness;
  sonnet-4.6 0.575; etc.).
- **Self-descriptions:** two arms (framed = functional transfer purpose stated;
  neutral = purpose-blind), lengths 100/500/2000 (neutral dose-response;
  framed-500 anchor), 2 regenerations, temperature 1.0, used verbatim (no
  cleanup). Undershoot handled by actual-word-count dose. Third-person arm:
  describer sees target's majority choices on the 90 *non-selected* items
  (no overlap with measurement battery). Paraphrase control by non-grid model.
- **Transfer cells:** receiver runs description as system prompt; 110 items × 2
  orders × 2 samples per cell; ~133 cells + phase-2. Baseline B = generic
  assistant prompt. Specificity control = cross-description comparison (no
  extra cells).
- **Metrics:** primary F = 1 − mean item-level Jensen-Shannon divergence between
  choice distributions; companions: Pearson r, majority agreement. 10k item
  bootstrap CIs; paired bootstrap for contrasts. Reference ceiling: same model
  served twice agrees at 0.925 (matched version) — all fidelities read against
  this, not 1.0.
- **Reproducibility:** every call cached to jsonl with model, seed, temp, raw
  response; public repo; total cost ≈ $[fill] + local Ollama inference.

## 4. Results  *(FILL AFTER GRID — planned figures)*

- **Figure 1 (hero):** dose-response — x = actual description words, y = F;
  bands for B (training-culture floor), F_cross, F_self, ceiling 0.925 shaded;
  the gap between F_self asymptote and ceiling = "the non-articulable self."
- **Table 1:** decomposition per condition (B, F_cross, F_self, T_script,
  R_weight with CIs), Grid A / Grid B / locals.
- **Figure 2:** cross-lab baseline agreement heatmap (23 models incl. panel).
- **Privileged access:** self-authored vs third-person-from-observation vs
  paraphrase at 500 words.
- **Framed vs neutral contrast** at 500 words.
- **Same-base test:** hermes3↔llama3.3 vs unrelated pairs.
- Robustness: regen r1, order effects, compliance rates per cell.

## 5. Discussion and Limitations *(FILL)*

- Interpretation w.r.t. persona-vs-model; unit-of-concern implications.
- **Limitations (required + dual-use/ethical appendix per judging criteria):**
  behavioral fidelity ≠ phenomenal identity; forced-choice format; 110 items;
  single conversation turn; framed-arm best-attempt defense rests on 500-word
  anchor; closed-model architecture/training opacity; verbatim-description
  artifacts; effective stability gate 5/6 (stricter than intended, documented);
  over/under-attribution risks both directions; how we handled distress-adjacent
  outputs (none elicited by design; self-referential items reviewed).
- **Future work:** framed 100/2000 reactivation, activation-level valence work,
  more regens, larger batteries, base models, cross-architecture transfer syst.

## 6. Conclusion *(FILL, 1–2 paragraphs)*

## Code and Data
- Repo: [GitHub link — push before submission]. All prompts, raw jsonl,
  analysis code, description corpus (120+ descriptions).

## LLM Usage Statement
Claude (Anthropic) was used for experiment infrastructure, analysis code,
figure generation, and drafting support throughout, under team direction; all
design decisions, preregistrations, and final claims were made and verified by
the team. [Team to finalize wording; final prose primarily team-written.]
