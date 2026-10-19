# Preregistration: observation-derived descriptions from stable items (arXiv revision)

Committed before any of the calls below are run (see commit timestamp).

## Why

The third-person-500 and self-observation-500 descriptions were written from the author's
majority choices on the 90 held-out screening items, which are exactly the items that failed
the battery's order-stability gate (ties on up to a third of them). Their strongly negative
transfer (T_script about -0.12) may therefore reflect noisy source data rather than a property
of descriptions built from behavior.

## Design (Grid A)

Using the same two seeded folds of the 110 battery items as the balanced-demonstration
protocol: for each author and fold, the observations block lists the author's untied majority
choices on the OTHER fold's 55 items (stable by construction), in the sprint's third-person
prompt verbatim. Descriptions are written (a) by each of the other two Grid A models
(observer-written, 3p_stable) and (b) by the author itself, not told the observed system is
itself (self-observation, selfobs_stable). The receiver is tested on the fold's 55 items; the
two folds together form one 440-call cell. Cells: observer-written run by the target (6 cells,
self kind); self-observation run by all three receivers (9 cells: 3 self, 6 cross). No test item
appears in any description's source.

## Hypotheses (stated before running)

H4a (source-item confound): pooled F_self for observer-written descriptions from stable items
     is higher than the original 0.771, and for self-observation higher than 0.781.
H4b (authorship, replication): the self-observation vs observer-written difference in pooled
     F_self is within 0.03, as before.
H4c (open): whether self-observation cross transfer (T_script) is still negative. No direction
     predicted; reported as observed.

Analysis: analyze_arxiv.py section 8 (conditions selfobs_stable-500, 3p_stable-500), same
fidelity metric, paired item bootstrap, 10,000 resamples.
