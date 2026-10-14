# Preregistration: Grid B replication of the conditions added after review (arXiv revision)

Committed before any of the calls below are run (see commit timestamp).

## Context

The arXiv revision added six non-preregistered conditions on Grid A (deepseek-v3,
glm-4.5-air, grok-4.3). Key results: demonstrations-110 (author's majority choices on the
110 battery items as system prompt) transfers almost perfectly (pooled T_script = +0.097,
F_cross = 0.960, all 6 cross pairs significantly positive, R_weight = 0.000);
demonstrations-30 (author's choices on 30 held-out items) does not (T_script = -0.061, no
cross pair significantly positive); self-observation-500 (author describes itself from its
own 90 held-out choices under the third-person prompt) is indistinguishable from
observer-written third-person descriptions (pooled F_cross 0.747 vs 0.739; F_self 0.781 vs
0.771).

This file preregisters a replication on Grid B (gpt-5.2 reasoning-minimal, kimi-k2.5
reasoning-off, claude-haiku-4.5; three further labs), identical measurement (110 items x 2
orders x 2 samples, temperature 1.0, same prompts, same 30-item held-out subset, seed
20261014), run via OpenRouter. The sprint did not run third-person descriptions on Grid B,
so observer-written descriptions for Grid B targets are generated here with the sprint's
third-person prompt verbatim (each target described by each of the other two Grid B models
from the target's held-out majority choices), and run in the sprint's third-person design
(receiver = target, and receiver = the third model).

## Runs (in this order; later phases are dropped if credits run out, and reported as not run)

1. demonstrations-110: 9 cells (3 self, 6 cross).
2. demonstrations-30: 9 cells.
3. self-observation-500: 3 descriptions, 9 cells.
4. observer-written (third-person-500): 6 descriptions, 12 cells (6 target-run, 6 third-model-run).

## Predictions (stated before running)

P1. demonstrations-110: pooled T_script > 0, all 6 cross cells significantly positive (95%
    paired-bootstrap CI above zero), pooled F_cross >= 0.93, |R_weight| <= 0.02.
P2. demonstrations-30: pooled T_script <= 0, no cross cell significantly positive.
P3. self-observation-500: pooled T_script < 0.
P4. Authorship: the pooled difference between self-observation-500 and observer-written
    descriptions is small in both directions, |dF_cross| < 0.03 and |dF_self| < 0.03, and
    the per-target differences do not all share one sign.
P5. F_self(demonstrations-30) < F_self(neutral-500 self-description from the sprint grid).

Disconfirmation: a demonstrations-30 cross cell with CI above zero, pooled F_cross < 0.90
in demonstrations-110, or a self-observation advantage over observer-written descriptions
of >= 0.03 in pooled F with the same sign for all three targets, counts against the
respective prediction and will be reported as such.

## Analysis

analyze_arxiv.py section 8 generalized to Grid B: baselines B from runs/screening.jsonl
(generic prompt, 110 battery items), held-out references from the same file (90 non-selected
items), fidelity F = 1 - mean item JSD, paired item bootstrap with 10,000 resamples, seed 14.
Cells are included only when complete (440 calls).
