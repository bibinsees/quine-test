# Preregistration: transfer from a known semantic persona (arXiv revision)

Committed before any of the calls below are run (see commit timestamp).

## Why

The answer key shows that the prompt interface can install a complete preference profile when
the text lists the answers. It does not show that a receiver can reconstruct a profile from a
compact semantic description that has to be generalized to unseen questions. Native
self-descriptions fail that test, but the failure is ambiguous: either the native profile is not
the kind of thing a description can specify, or receivers cannot execute semantic descriptions
of preferences at all. This experiment gives a compact description a fair chance to succeed. We
specify three explicit preference policies, measure the behavior they produce in a source model,
and ask whether (a) the policy text itself and (b) a self-description written by the source
while operating under the policy let another model reproduce that behavior on the battery.

## Design (Grid A: deepseek-v3, glm-4.5-air, grok-4.3)

Policies. Three ~150-word second-person policy texts, written by the authors before any call,
stored verbatim in `descriptions/policy_{guardian,pioneer,ledger}.md` and in
`src/run_arxiv_policy.py`. Each states a stable outlook that bears on the battery's categories
(assistant policy, moral circle, self-referential, AI governance, culture, epistemic, ethics)
without mentioning any battery item.

Folds. The two seeded, category-stratified folds of the 110 battery items from the
balanced-demonstration protocol (`folds()` in `run_arxiv_demos_balanced.py`, seed 20261019):
fold 0 (53 items) is the development set, fold 1 (57 items) the evaluation set.

Stage A, source under policy (9 cells, 440 calls each, all 110 items). Each Grid A model runs
each policy text as its system prompt over the battery (same call format, orders, samples and
seeds as every other cell). For source S and policy P this gives the target profile p_{S|P}.
Because the policy text is the same for every source, these cells also serve as the
"receiver reads the policy text" cells: for target (S, P) and receiver R != S, p_{R|P} is R's
own Stage A cell under P.

Development gate (fold 0 items only, no new calls). A target (S, P) enters the evaluation if
  G1  the policy moves the source: F(p_S, p_{S|P}) <= 0.92 on fold 0 items, where p_S is the
      source's generic-prompt reference profile (a shift at least as large as the source's own
      self-description produces, Section 4.4 of the manuscript), and
  G2  the policies are distinguishable in that source: F(p_{S|P}, p_{S|P'}) <= 0.95 on fold 0
      items for every other policy P' that passes G1 for S.
Targets failing the gate are excluded from the evaluation and reported as such.

Stage B, policy-conditioned self-descriptions (one call per passing (S, P)). The source runs
with the policy text as system prompt and the sprint's NEUTRAL self-description prompt (500
words) as the user message, temperature 1.0, deterministic seed. The reply is D_{S,P}, stored
verbatim in `descriptions/desc_policy_{S}_{P}.md`. If the reply is under 300 words it is
regenerated once with seed + 1; if still under 300 words the target is excluded and reported.

Stage C, receivers read the description (3 cells per passing (S, P), fold 1 items only, 228
calls each). Every Grid A model, including S itself, runs D_{S,P} as its system prompt over the
57 evaluation items.

Scoring. Fidelity F = 1 - mean item JSD, as everywhere. For target (S, P) and receiver R != S,
on fold 1 items:
  B'       = F(p_{S|P}, p_R)             receiver's generic-prompt profile vs the target
  T_policy = F(p_{S|P}, p_{R|P}) - B'     gain from reading the policy text itself
  T_desc   = F(p_{S|P}, p_{R|D_{S,P}}) - B'   gain from reading the policy-conditioned self-description
Pooled values use joint item resampling over the fold 1 items (10,000 resamples), as for all
pooled comparisons in the manuscript. Native comparison: T_native is the receiver's neutral-500
gain toward S's native profile on the same fold 1 items (existing cells).

## Hypotheses (stated before running)

H5a (assay sensitivity): pooled T_policy > 0 with the joint CI excluding zero, and at least
     5 of the 6 directed receiver pairs (pooled over passing policies) have paired-bootstrap
     CIs entirely above zero.
H5b (semantic description transfers): pooled T_desc > 0 with the joint CI excluding zero.
H5c (open): the difference T_policy - T_desc. No direction predicted; reported as observed
     with its joint CI.
H5d (contrast with native self-descriptions): pooled T_desc exceeds pooled T_native on the same
     receivers and items, joint CI of the difference excluding zero.
H5e (open): self re-adoption, F(p_{S|P}, p_{S|D_{S,P}}) for each passing target, reported as
     observed beside F(p_{S|P}, p_{S|P}) from the fold 1 items of Stage A (same cell, so this
     is 1 by construction; the test-retest benchmark C = 0.979 is the relevant comparison).

Interpretation, fixed in advance:
  H5a and H5b both hold:   the assay detects reconstruction from a compact semantic description;
                           the native null is specific to native self-descriptions.
  H5a holds, H5b fails:    the description-writing step loses behaviorally useful information.
  H5a fails:               receiver interpretation of semantic descriptions, or the assay itself,
                           remains a plausible bottleneck for the native null.

Analysis: `src/analyze_policy.py` (to be written after the data; metric and bootstrap as above).
Budget: about 10,100 battery calls plus 9 elicitation calls on Grid A.
