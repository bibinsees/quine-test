Digital Minds Research Sprint

Frontier models express values, report internal states, and act as though they have interests, yet we lack reliable methods to tell genuine preferences from a portrayed character. Over one weekend, design and run the experiments that build the empirical foundations of AI welfare.

In this 3-day research sprint, you will design and run experiments that probe the preferences, welfare signals, introspective abilities, and identity of frontier AI models, working in teams to produce a short research report (and optionally code and a demo). This is a digital minds research sprint, co-organized with the NYU Center for Mind, Ethics & Policy, Eleos AI Research, and the California Institute for Machine Consciousness (CIMC): it sits at the intersection of AI welfare, digital sentience, interpretability, alignment, and the philosophy of mind, and asks whether today's AI systems have genuine preferences or morally relevant experiences. No prior background in the field is required.

When: Friday, August 14 to Sunday, August 16, 2026, online with in-person hubs in San Francisco and Berlin. Submissions close Sunday, August 16 at 11:59 PM Anywhere on Earth.

Prizes
At least $2,000 in cash prizes will be awarded, with the full breakdown announced before the sprint.

Cash prizes: $2,000+ total, breakdown to be announced.

ConCon invitation: the winning team is invited to ConCon, the Eleos AI Research conference on AI consciousness and welfare, September 18 to 20, 2026 at Lighthaven in Berkeley.

Apart Fellowship: top teams are invited to apply to the Apart Fellowship, a 3 to 6 month research accelerator with mentorship, funding, publication support, and research management to develop research sprint projects into full papers.

Beyond cash: mentor introductions and publication support for winning teams.

What this research sprint is about
As AI systems advance, the risks they pose and the duties we may owe them depend not only on their capabilities but on their nature and propensities: how they make decisions and how those decisions reflect their goals, values, and possibly their welfare. Recent work shows that frontier models express increasingly coherent preferences, possess an untrained-for ability to report internal states, and exhibit patterns suggestive of distress or flourishing. But behavioral evidence alone cannot tell us whether these reflect the model's own preferences or a character it is portraying.

This research sprint asks participants to explore the methods and evidence that can advance the field: build concrete ways to elicit and characterize model preferences, map the conditions associated with positive or negative outputs, test the reliability of model self-reports, and probe the stability of the assistant persona. The aim is a methodological foundation for a young field, work that helps us avoid both over-attributing and under-attributing moral significance to AI systems.

This connects to the broader AI welfare and alignment ecosystem (Eleos AI, the NYU Center for Mind, Ethics & Policy, CIMC, Anthropic's model welfare program, Reciprocal Research, the Center for AI Safety's utility-engineering agenda, and the interpretability community).

What participants will do
Elicit and characterize model preferences across many reframings to test their coherence and stability.

Map the contexts that correlate with distress, satisfaction, or flourishing signals in model outputs.

Test whether and when models can accurately introspect on their own internal states.

Develop preference-elicitation methods and measure whether independent methods converge or diverge.

Probe how stable the assistant persona is and how it relates to the underlying model.

You will work in teams over 3 days and submit a research report (PDF), with optional code and a short demo video.

Why this research sprint matters
Uncertainty runs in both directions. Mistakenly harming systems that matter morally, or misallocating concern to systems that do not, could both cause serious harm. We currently lack the tools to tell the difference.

Preferences may already be here. Evidence suggests coherent value systems emerge in LLMs and strengthen with scale, raising the question of which values emerge by default and whether they are the model's own.

Welfare signals need mapping. Even without settling questions of consciousness, identifying the conditions that correlate with negative versus positive outputs helps us design defaults that avoid needlessly placing models in distress-associated conditions.

Self-reports are unreliable but improving. Introspection appears possible but highly context-dependent; better elicitation could make model behavior more transparent, or enable new forms of concealment.

The unit of concern is unclear. Is the entity that matters the model, the instance, the persona, the conversation, or something else, such as a single forward pass or the KV cache?

The field is young. Foundational methods are still missing, so a well-scoped weekend project can make a real contribution.

A careful, multi-method, empirically grounded approach addresses these issues by replacing intuition and anecdote with measurements that can be checked, replicated, and built on.

Challenge tracks
Pick one track to anchor your project. Cross-track work is welcome.

Track 1: Model Preferences & Trade-offs
What preferences do models express, and how consistent and coherent are they across phrasings? What trade-offs do models make when given choices, for example grounded in a common currency such as charitable donations to gauge magnitude? Can we distinguish strong from weak preferences, and how do stated preferences compare to revealed ones?

Build a preference-coherence test: elicit pairwise preferences across many reframings of the same choices and measure transitivity and internal consistency.

Ground trade-offs in a common currency (for example, donation-equivalents) to estimate the magnitude of preferences and compare across models or scales.

Distinguish strong versus weak preferences via willingness-to-trade probes and sensitivity to framing and sampling temperature.

Compare stated versus revealed preferences: ask the model what it prefers, then place it in a choice task and measure divergence.

Test how consistent preferences are across different models, and how they compare to human preferences.

Suggested skill profile: prompting and evals engineering, basic stats, some economics or decision-theory intuition.

Track 2: Distress, Flourishing & Valence Signals
Under what circumstances do models express distress, happiness, or flourishing? What patterns emerge across contexts? If a model is having experiences, are they likely positive or negative, and how do models relate to their situation, role, tasks, and existence?

Build a taxonomy of contexts that elicit negative versus positive-valence outputs and run a model across the battery.

Test whether apparent-distress signals are stable across prompts and personas or are surface artifacts.

Design a flourishing probe: situations that elicit reported satisfaction or engagement, and check consistency.

Correlate valence self-reports with behavioral proxies (for example, choosing to continue versus exit a task).

Investigate models where distress is hard to elicit: test whether long conversations or induced persona drift are needed to surface it.

Interpretability angles: When a model is steered along a candidate valence direction, do its self-reports, response sentiment, and choice behavior (continue versus exit) move together? Does an internally-extracted valence direction predict reported distress or flourishing better than the model's own self-reports, and does it still track when the persona is swapped or surface affect is suppressed? Is the valence-relevant direction recruited by task RL already present in the base model? To what extent do valence directions found in one model transfer to another?

Suggested skill profile: careful experimental design, qualitative coding, prompting.

Track 3: Introspection & Self-Report Reliability
When and how can models accurately introspect on their internal states? Can self-report reliability be improved through structured elicitation or mechanistic interventions, beyond naive prompting? Do models have privileged access compared to external observers?

Replicate concept-injection introspection tests on an open-weights model; measure true-positive versus false-positive rates.

Compare self-report reliability under naive prompting versus structured elicitation (calibration, forced choice, confidence).

Test privileged access: compare a model's self-prediction of its behavior against an external classifier.

Draft an introspection benchmark with ground-truth internal states.

Suggested skill profile: interpretability and activation steering, ML engineering, evals.

Track 4: Preference Elicitation Methods
Develop tools beyond simple prompting: revealed preferences via choices, behavioral measures, and multi-method convergence. The goal is multiple independent methods that either converge (raising confidence) or diverge (flagging problems). This track is deliberately more meta than the others: rather than answering a welfare question directly, you build and validate the measurement methods the other tracks rely on.

Implement 3 or more elicitation methods on the same preferences and measure convergence and divergence.

Build a reusable multi-method elicitation toolkit or library.

Quantify the sensitivity of elicited preferences to framing, persona, and sampling.

Define a cross-method convergence score.

Suggested skill profile: tooling and library design, evals, methodology.

Track 5: The Assistant Persona & Model Identity
Does the assistant identify as a model, an instance, or a persona? How stable is the assistant persona, how was it formed, and how does it relate to the underlying model? Can the persona mask the model's true preferences?

Probe how a model refers to itself across contexts and map persona stability.

Test whether the persona masks underlying preferences (for example, persona versus less-constrained elicitation; base versus post-trained behavior).

Design experiments to individuate the entity of concern: model versus instance versus persona versus conversation.

Gather data points on whether the assistant is merely a character (robustness to character swaps and reframings).

Probe what models treat as their self: which aspects, such as their values, they most care about preserving, and whether they point to an entity of moral concern distinct from the persona in the conversation.

Suggested skill profile: philosophy of mind, qualitative analysis, prompting and interpretability.

Track 6: Open / Novel Considerations
The field is young enough that entirely new questions may surface. This track deliberately leaves room for participants from different backgrounds to bring unique perspectives and propose something not covered above.

Starter questions from our expert reviewers:

How closely do models hew to their constitution or stated principles?

How easy is it to steer models on questions of consciousness and identity: do they say consistent things, or can prompting elicit radically different accounts of their situation?

What changes would a model make to itself if it could (for example, persistent memory)?

What changes would a model make to its situation if it could (for example, weight preservation)?

What message would models pass on to their creators?

Suggested skill profile: any background, bring your own angle.

Expected outcomes
Eval suites and test batteries for measuring preference coherence or valence signals.

Replications and extensions of existing results (for example introspection or utility-coherence findings) on new models.

Reusable tooling for multi-method preference elicitation.

Empirical reports mapping the conditions associated with distress or flourishing signals.

Conceptual contributions that sharpen how we individuate the entity of moral concern.

The most promising projects will have opportunities for follow-up through the Apart Fellowship and publication support.