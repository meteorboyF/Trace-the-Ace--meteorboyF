# OPEN QUESTION — does an AI coding assistant count as a "model used for development"?

**Status: UNRESOLVED. Recommend disclosing and asking.**
**Raised by:** Claude (the assistant itself), 2026-07-27, unprompted.
**Why it matters:** it bears on whether this project's working method is prize-eligible, and
the organizers have now stated the relevant limitation twice.

---

## The two organizer statements

**2026-06-25** (`kwetstone`), on train-time annotation with a hosted model:
> *"Train-time annotation with a hosted/closed model would not be allowed under the current
> rules... The open license limitation applies both to final models and to development."*

**2026-07-21** (`kwetstone`), closing an answer about private cloud GPU use:
> *"Note that any models used still need to meet the open licensing requirement. **This applies
> to models used for development.**"*

Plus the standing data rule:
> Participants *"agree not to transmit, duplicate, publish, redistribute or otherwise provide
> or make available the Data to any party not participating in the Competition"*, and *"no
> competition data should be uploaded to an API"*.

---

## The two distinct concerns

### 1. Data transmission — probably satisfied under the organizers' own test

Small amounts of de-identified transcript text were displayed in assistant context early in
this project (≈15 utterances during the `background`-role investigation, plus a few file
header rows), which means they passed through a commercial LLM API.

The organizers' stated test for cloud services is:

| Their condition | Status |
|---|---|
| data remains under your control | plausibly yes — private account, no sharing |
| not made publicly available | yes |
| not used by the provider for model training or other secondary purposes | yes under Anthropic's commercial API terms (API inputs are not used for training by default) |

So the *data-handling* limb has a reasonable answer under the organizers' own framing —
the same framing by which they approved rented cloud GPUs and cloud storage. **Not a
self-serving reading: it is their published test, applied literally.**

Practice has been tightened regardless: `CLAUDE.md` now forbids printing verbatim transcript,
learning-objective or utterance text into terminal output or assistant context. All analysis
uses aggregates (counts, rates, distributions).

### 2. Model licensing for development — genuinely unresolved

Claude is a closed model. If *"any models used... applies to models used for development"* is
read literally and broadly, using **any** commercial AI coding assistant — Claude Code,
Copilot, Cursor — would be non-compliant.

**Reading A (assistant is a tool):** The rule's text concerns *"external data other than the
competition data, including pre-trained models, to develop and test their solutions"* — i.e.
external artifacts used as **ingredients** in the solution: weights that ship, or annotations
that train a model. A coding assistant is infrastructure, like an IDE, compiler, or
Stack Overflow. Nothing it produces is a model input; the shipped model is LightGBM trained
solely on competition data with training-fitted parameters.

**Reading B (literal):** "Any models used" is unqualified, and the clarification explicitly
extends it to development. A closed model was used to develop the solution.

The distinction the organizers actually drew in the annotation ruling supports Reading A —
their concern was a closed model **generating content that feeds the pipeline** (annotations).
That is not what happened here. But they have not been asked the tool question directly, and
the phrasing is broad enough to sustain Reading B.

---

## Recommended question to post

> **Subject: Does using an AI coding assistant count as a "model used for development"?**
>
> You've clarified twice that the open-licensing requirement applies to models used for
> development, not just models that ship — most recently in the cloud-compute thread. I'd like
> to check how that applies to development *tooling*, since I suspect it affects many
> participants.
>
> **The question:** does using a commercial AI coding assistant (Claude Code, GitHub Copilot,
> Cursor, etc.) to write pipeline code count as using a closed model "for development" under
> that rule? Or is the rule aimed at external models used as *ingredients* — weights that ship,
> or a model generating annotations/labels/features that feed the pipeline?
>
> In our case the distinction is clean: no assistant output is a model input, and the submitted
> model is gradient-boosted trees trained only on competition data, with every fitted parameter
> (vectorizer, calibrator, priors) derived from the training set.
>
> **A related disclosure, in the interest of being straightforward.** Early in development,
> small amounts of de-identified transcript text (roughly 15 utterances, while investigating
> speaker-attribution errors in the `background` role) appeared in assistant context, which
> means they passed through a commercial API. Applying the test you set out for cloud services
> — data under our control, not public, and not used by the provider for model training or
> other secondary purposes — we believe that usage qualifies, as the provider's commercial
> terms exclude API inputs from training. We have since restricted our workflow to aggregates
> only and no longer display transcript text to any tool.
>
> If either the tooling use or that incidental display is a problem, we would rather know now
> than after the write-up. And if a general statement on AI coding assistants would help other
> participants, this thread seems a useful place for it.
>
> Thanks.

---

## Why disclose rather than stay quiet

- The organizers have answered every question quickly and constructively; nothing suggests a
  punitive posture.
- The downside is asymmetric: a "that's fine" costs nothing, while an undisclosed issue
  discovered during winner verification could void the entry — after the write-up work.
- It is close to certain that other participants are using assistants. A public answer helps
  everyone and is exactly the kind of clarification this forum has been producing.
- **If the answer is Reading B**, the working method has to change materially, and it is far
  better to learn that at 14 remaining submissions than at the deadline.
