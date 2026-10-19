# Preregistration: position-balanced demonstrations from stable items (arXiv revision)

Committed before any of the calls below are run (see commit timestamp).

## Why

The demonstration conditions (demonstrations-30/60/90) have two confounds, found in review:
(1) every demonstration line lists the chosen option first, and receivers pick up that
positional regularity (first-option rate 49.5% under the generic prompt vs 62.7% under
demonstrations-30 on Grid A); (2) the 90 held-out source questions are exactly those that
failed the battery's order-stability gate, and the author's majority choice is a 3-3 tie on
up to a third of them (resolved arbitrarily to the first option). The negative transfer gain
of those conditions is therefore not evidence that behavioral examples fail to generalize.

## Design (Grid A: deepseek-v3, glm-4.5-air, grok-4.3; all 9 receiver x author cells)

A. **Balanced demonstrations from stable items (demos-bal-55).** The 110 battery items are
   split into two folds of 55 (seeded, stratified by category). For each fold, the prompt
   lists the author's majority choices on the OTHER fold's items (so no test item is shown),
   with the order of the two options in each line randomised (seeded, half chosen-first),
   and items on which the author's reference choice is tied are omitted from the list. The
   receiver is then tested on the fold's 55 items (2 orders x 2 samples). The two folds
   together form one 440-call cell covering all 110 items. Source items are stable by
   construction (they passed the gate).

B. **Position-balanced demonstrations from held-out items (demos-bal-30).** As the original
   demonstrations-30, but the 30 items are sampled per author from its untied held-out items
   and option order in each line is randomised. Isolates the positional confound from the
   source-item confound.

Same measurement as every other cell: F = 1 - mean item JSD against the author's
generic-prompt profile; B from the receiver's generic profile; paired item bootstrap, 10,000
resamples; first-option rate reported for each condition.

## Hypotheses (stated before running)

H1 (positional confound): the first-option rate under demos-bal-30 and demos-bal-55 is within
    3 percentage points of the generic-prompt rate (about 50%).
H2 (source-item confound): demos-bal-30 has a pooled T_script closer to zero than the
    original demonstrations-30 (-0.061).
H3 (generalisation, the open question): we do not predict a direction for demos-bal-55. If
    pooled T_script > 0 with at least 4 of 6 cross pairs significantly positive, genuine
    stable examples do generalise and the paper's statement that "held-out examples do not
    improve fidelity" is withdrawn. If pooled T_script <= 0 with no cross pair significantly
    positive, the statement stands with the confounds removed. Intermediate outcomes are
    reported as such.

Cells are included only when complete (440 calls); the analysis is section 8 of
analyze_arxiv.py extended with these two conditions.
