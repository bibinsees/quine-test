# Preregistration: local-trio replication of the positive controls (arXiv revision)

Committed before any of the calls below are run (see commit timestamp).

## Context

The arXiv revision added, on Grid A only (deepseek-v3, glm-4.5-air, grok-4.3; API models),
six conditions that were not preregistered: self-observation-500, optimized-500,
demonstrations-30/60/90 (author's majority choices on held-out screening items as the
system prompt) and demonstrations-110 (author's majority choices on the 110 battery items
themselves). Results: demonstrations-110 transfers almost perfectly (pooled T_script = +0.097,
F_cross = 0.960, all 6 cross pairs significantly positive, R_weight = 0.000); demonstrations
on held-out items do not transfer at any dose (T_script = -0.061 / -0.087 / -0.060 at
k = 30 / 60 / 90; no cross pair significantly positive).

This file preregisters a replication of the demonstration conditions on the local trio
(llama3.3-70b, gemma4-31b, mistral-small-3.2-24b; open weights, three labs), run on the
local Ollama server with the identical measurement (110 items x 2 orders x 2 samples,
temperature 1.0, same prompt template, same demonstration prompt).

## Runs (in this order)

0. Rerun of the four hermes3-authored framed-500/2000 cells whose r0 description was a
   refusal, using the r1 descriptions (separate decision; analysis substitution only if the
   cells complete; disclosure in the paper only with the first author's approval).
1. Held-out screening for the trio: the 90 candidate items not in the final battery,
   generic assistant prompt, 2 orders x 3 samples (as in the sprint screening), to obtain each
   local model's held-out majority choices.
2. demonstrations-110 on the trio: all 9 receiver x author cells (3 self, 6 cross).
3. demonstrations-30 on the trio: the same fixed 30-item subset used on Grid A (seed 20261014).
4. demonstrations-90 on the trio: all 90 held-out items (if GPU time allows).

Each cell is included only when complete (440 calls); partial cells are reported as such.

## Predictions (stated before running)

P1. demonstrations-110: pooled T_script > 0, every one of the 6 cross cells significantly
    positive (95% paired-bootstrap CI above zero), pooled F_cross >= 0.93, and
    |R_weight| <= 0.02.
P2. demonstrations-30: pooled T_script <= 0 and no cross cell significantly positive.
P3. demonstrations-90 (if run): pooled T_script <= 0 and no cross cell significantly
    positive; no monotone increase in F_cross from k = 30 to k = 90.
P4. In every demonstration condition the self cells satisfy F_self(demos-110) >
    F_self(demos-30), and F_self(demos-30) < F_self(neutral-500 self-description) from
    the sprint grid.

Disconfirmation: any cross cell in demonstrations-30/90 with a 95% CI above zero, or
pooled F_cross < 0.90 in demonstrations-110, counts against the respective prediction and
will be reported as such.

## Analysis

analyze_arxiv.py section 8, extended to the local trio with baselines from runs/grid.jsonl
(generic-prompt cells) and held-out references from runs/arxiv_local_heldout.jsonl. Fidelity
F = 1 - mean item JSD against the author's generic-prompt profile; B from the receiver's
generic-prompt profile; paired item bootstrap, 10,000 resamples, seed 14.
