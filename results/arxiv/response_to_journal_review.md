# Response to the journal-readiness review of 2026-10-19

Thank you for the review and for the audit scripts; the independent reconstruction of the
metric was especially useful. We re-verified every quantitative claim in the review against
the logs (`src/verify_review_claims.py`) and found them correct, with one exception noted
under point 4 and one under Related Work. We then revised the manuscript and ran four
further experiments, each under a protocol committed before the first call
(`pilot/protocol_arxiv_demos_balanced.md`, `pilot/protocol_arxiv_obs_stable.md`). The
revised manuscript is `arxiv/main.tex` at commit `52e8563`; all new data are in
`runs/arxiv_demos_balanced.jsonl` and `runs/arxiv_obs_stable.jsonl`. Point by point:

## 1. The "weight-bound residual" does not identify a weight-bound component

Agreed. We renamed R = F_self - F_cross the "self-cross fidelity gap" throughout
(contributions, Figure 1, Design overview, Table 2, Figure 2) and added the
baseline-adjusted comparison you asked for. Measured against each side's own reference,
reading the description costs the author more than it costs a stranger: F_self - C is -0.04
to -0.06 in every self-description condition, F_cross - B is -0.01 to -0.03. Section 4.1 now
states that a positive gap reflects the author instance's pre-existing self-similarity (0.979
against a cross-model baseline of 0.892), not information recovered from the text, and the
Design overview carries the same caution. The "unpacking" sentence in the Discussion is
removed.

## 2. The demonstration experiments have two serious confounds

Both confirmed and both tested.

- Positional regularity. Verified: first-option rate 49.5% under the generic prompt, 62.7%
  under demonstrations-30 (Grid B 60.7%, local 59.5%), 50.6% under the answer key.
- Source items. Verified: the 90 held-out items are exactly the gate failures (0 compliance,
  90 stability); deepseek-v3 has tied labels on 30 of 90, glm-4.5-air 18, grok-4.3 12.

New experiments (Grid A, committed protocol, 18 cells):
- demonstrations-30 with randomised option order and untied items only (same held-out
  source): first-option rate 51.4%; T_script = -0.043 (3 of 6 pairs significantly negative,
  1 positive).
- demonstrations-55 with randomised order and stable source items (the author's untied
  choices on one half of the battery, tested on the other half, two folds): first-option
  rate 50.6%; F_cross = 0.855 vs B = 0.863, T_script = -0.008 (1 pair significantly
  positive, 1 negative, 4 at zero); F_self = 0.869.

So the original -0.06 to -0.09 decomposes into a positional artifact (about -0.02) and the
effect of unstable, partly tied source items (about -0.04); the corrected result is the same
null as for self-descriptions. The abstract, contribution 3, Methods 3.4, Results 4.5 and
Limitations now say exactly this, and the original format is retained only as the
demonstration of the artifact.

## 3. Missing answers change the estimand

Agreed. Methods 3.6 now reports coverage (97.7% of calls parsed; 366 of 426 original cells at
or above 95%, 25 below 90%), the two failure modes (empty replies after hidden reasoning,
preference refusals), the item-drop rule, and the 80-item denominator of the pooled
perturbation tests. A new appendix table lists every cell below 90%, including gemma4-31b's
refusals under its own framed description, which Limitations now describes as a behavioral
effect of the description that fidelity on parsed answers does not capture. "Four parsed
answers per item" and "zero unresolved failures" are reworded (nominally four; no unresolved
API failures).

## 4. "Descriptions are interchangeable" is not supported

Agreed on the conclusion; the point estimate (+0.006) reproduces. On the interval: resampling
the 107 common items jointly across the 18 comparisons gives [+0.001, +0.011] as you report;
resampling items within each comparison and averaging gives [-0.003, +0.014]. We report both
and have replaced the heading with "Descriptions carry little author-specific signal", stating
that a small author-specific signal may be present at the edge of detectability and coexists
with negative net transfer. Your distinction between the information a description carries
and the net effect of executing it is now made explicitly in that paragraph.

## 5. The philosophical conclusions are too strong

Agreed. "The portrayed-character account fails its own best test" is replaced by a paragraph
stating what the account must now say and why the result does not refute it (a character can
be implemented in weights, its self-report can be inaccurate, interpreters can differ). "Does
not compress into anything shorter than itself" is removed; the compression paragraph now
states that no information-theoretic limit is claimed and that a 110-item majority vector is
a short message in principle. "Is false" became "is not supported"; "the assistant is more
than the character" became "the assistant's measured preferences are not recoverable from the
character it describes itself to be"; "no articulation, however long" is gone from the
Introduction. The abstract's bounded final sentence is now the register of the Discussion.

## 6. Four replacement cells are undisclosed; Table 2 is stale

Agreed. The merged analysis (second-draft descriptions) is now the primary analysis; Table 2's
framed-500 and framed-2000 rows are regenerated from it (0.924 / 0.875 / 0.892 / -0.017 /
+0.049 and 0.918 / 0.869 / 0.884 / -0.014 / +0.049). Methods 3.4 discloses that hermes3-70b
refused the framed prompt at 500 and 2,000 words on its first draw (57 and 46 words of
refusal text), that the second compliant draw is used for the four affected cells, and that
the substitution is flagged in the log; the sprint values are kept in a footnote. The refusal
itself is reported as a datum.

## 7. Preregistration wording and prediction ledger

Agreed. The protocol commit is 2026-08-14 16:00:40 UTC; the first pilot call is 13:13:16 UTC
and the first screening call 15:59:12 UTC. The manuscript now claims only what the history
supports: selection rules were committed before screening results were inspected, not before
every call (Limitations; contribution 4 reworded). The prediction accounting is replaced by a
complete ledger (new appendix table): all 30 criteria registered across the four committed
protocols, each with its outcome; 28 are met. The two misses are one local-trio answer-key
pair short of significance (CI [-0.002, +0.071]) and one significantly positive pair in the
balanced demonstrations, both reported as such.

## 8. Secondary conclusions

- Length: paired differences over the 20 cross pairs are +0.001 [-0.006, +0.008] (2,000 vs
  100), -0.004 [-0.010, +0.003] (500 vs 100) and +0.005 [-0.002, +0.012] (2,000 vs 500); any
  length effect on transfer is bounded to about +-0.01, and the text says so.
- Framing: now reported for cross cells (+0.001, n = 20) and self cells (+0.007, n = 9)
  separately.
- Same-base pair: Section 4.3 is retitled "Shared lineage raises baseline similarity, not
  transfer" and reports that the pair's higher cross-fidelity is entirely baseline similarity
  (B = 0.915 vs 0.890) while its transfer gains (-0.031, -0.042) are more negative than the
  other pairs' mean (-0.016).
- Hero figure: the caption already stated the pooled baseline; it now also says that F_self
  sits above F_cross by about the margin a fresh author instance has before reading anything.
- Test-retest: described as an empirical repeatability benchmark, not a hard bound, wherever
  it appears.
- One passage per style: we added a second, expository style at all three lengths (109, 529,
  1,971 words) alongside the sensory texts; Results 4.4 reports both and the pooled tests per
  style (specificity of the perturbation is significant for expository text at 500 and 2,000
  words, borderline for sensory text, absent at 100 words).
- Configurations: Methods 3.3 now lists the exact setting per model (kimi reasoning disabled,
  gpt-5.2 reasoning effort minimal, gemma thinking off, others at provider defaults; glm's
  default reasoning produced empty replies in 7% of main-grid calls).
- Logs: described as holding the first 500 characters of each reply.

## Additional experiment prompted by point 2

Because the observer-written and self-observation descriptions were built from the same
unstable held-out items, we also rebuilt both from stable items (committed protocol; 15 Grid A
cells): observer-written descriptions then score F_self = 0.846 (vs 0.771) and self-written
ones 0.852 (vs 0.781); the authorship difference stays at 0.006; cross transfer of the
self-written stable-item descriptions is T_script = -0.031 (vs -0.116). The original
"observation-derived descriptions mislead" effect was therefore mostly noisy source data, and
Section 4.4, contribution 4, the Discussion and Limitations now say so.

## Related Work

We could not locate a paper titled "How Far Do Persona Effects Generalize in Language Models?"
on arXiv. If you can supply the identifier we will add a substantive comparison; the closest
work we found is Luz de Araujo and Roth, "Helpful assistant or fruitful facilitator?
Investigating how personas affect language model behavior" (arXiv:2407.02099), which we are
considering for the Related Work section.

## Suggested central claim

We have adopted it. The abstract's conclusion reads: "For the preferences measured here,
self-descriptions influence behavior without reliably specifying how to reproduce it in
another model."
