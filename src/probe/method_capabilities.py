"""The evidence-producing methods this codebase has or could have —
declared as DATA, not code, so a future router reads a table instead
of accumulating if-statements per method. No router exists yet; this
module has no runtime effect today. It exists so the next two
instrument primitives (built one at a time, see instruments.py) have a
place to state what they're FOR before the router that picks between
them gets written, and so that router starts from a declared contract
about each method's ceiling rather than rediscovering it by trial.

SEVEN ROWS: the six `InstrumentPrimitive` values, plus `option_pair` —
the EXISTING, already-in-production disambiguation click mechanism
(disambiguate.py / reconcile_candidate, `source=click`). `option_pair`
is deliberately not folded into `choose`: `choose` is the instrument-
layer generalization of the same idea (a dedicated, purpose-built
exercise, not restricted to two options, not embedded mid-conversation)
and does not exist yet — conflating the two would make the router think
a mechanism running in production today is still unimplemented.

TWO AXES OF CLASSIFICATION PER METHOD, per the spec this table exists
to answer:

- `measures`: PREFERENCE (which framing/style the learner wants) or
  PERFORMANCE (whether they can do the thing). These are genuinely
  different questions with different evidentiary weight and, since the
  fix below, different STORAGE — conflating them used to be a silent
  mistake this table only made visible; it is now a shape mismatch
  `InteractionContract`'s own validator (models.py) refuses to
  construct.

- `claim_types` / `axes` (PREFERENCE rows only) / `skills`
  (PERFORMANCE rows only): which `StatedPreferenceLabel`+`ApproachAxis`
  values, or which `CapabilityLabel` values, a hand-written contract
  for this method can plausibly target. Empty means genuinely
  undetermined (see `order`/`construct`'s `OTHER` placeholder), not
  "any" — "any" is spelled out explicitly (`choose`/`option_pair`)
  precisely so a reader never has to guess which empty tuple means
  what.

A GAP THIS TABLE FOUND, AND IS NOW FIXED: an earlier version of this
table declared `locate`/`predict` — both PERFORMANCE — with
`claim_types=(StatedPreferenceLabel...,)`, because `StatedPreferenceLabel`
was the only claim vocabulary this codebase had. Checking what real
predict evidence had actually landed on confirmed the mechanical
category error that shape implied: performance observations
("correctly traced the swap forward") were accumulating directly into
a preference claim's confidence (RULE_BEFORE_EXAMPLE), which can
diverge from the true preference in either direction — someone can
prefer worked examples and still be perfectly able to derive forward
without them. `CapabilityClaim`/`CapabilityLabel` (models.py) and
`capability.py`'s own store now give PERFORMANCE evidence a separate
home — its own vocabulary, no axis (capability isn't bidirectional:
there's no opposite pole to a skill), matched by exact skill
(`find_by_skill`) rather than `find_by_axis`, so a capability
observation can never merge into a preference claim by construction,
not by a filter someone has to remember to apply.

WHAT EACH ROW CANNOT ANSWER matters as much as what it can: the
router's job is not just picking the right method for a question, it's
refusing the wrong one. `locate` says nothing about framing preference;
`option_pair`/`choose` say nothing about capability. A router that only
knows what a method IS GOOD FOR, not what it structurally cannot tell
you, will eventually reach for `choose` to answer a capability question
because nothing told it not to.
"""

from __future__ import annotations

from dataclasses import dataclass

from probe.models import ApproachAxis, CapabilityLabel, MeasurementKind, StatedPreferenceLabel

__all__ = ["METHOD_CAPABILITIES", "MeasurementKind", "MethodCapability"]


@dataclass(frozen=True)
class MethodCapability:
    """One row. `method` is an `InstrumentPrimitive` value for the six
    instrument primitives, or the literal string `"option_pair"` for
    the existing click mechanism (which predates, and is not part of,
    the instrument layer).

    Exactly one of (`claim_types`+`axes`) / `skills` is ever non-empty,
    matching `measures` — a PREFERENCE row has claim_types/axes and an
    empty `skills`; a PERFORMANCE row has `skills` and empty claim_types/
    axes. This mirrors `InteractionContract`'s own PREFERENCE-vs-
    PERFORMANCE shape split (models.py) — the fix for the incident
    this table's own earlier version helped surface: `locate`/`predict`
    used to declare `claim_types=(StatedPreferenceLabel...,)` despite
    measuring performance, which was the same category error the
    contract shape itself had. Empty means genuinely undetermined, not
    "any" (see this module's own docstring for `option_pair`/`choose`,
    the two rows where "any" is spelled out explicitly instead)."""

    method: str
    implemented: bool
    measures: MeasurementKind
    cannot_answer: str
    claim_types: tuple[StatedPreferenceLabel, ...] = ()
    axes: tuple[ApproachAxis, ...] = ()
    skills: tuple[CapabilityLabel, ...] = ()
    notes: str = ""


_ANY_CLAIM_TYPE = tuple(StatedPreferenceLabel)
_ANY_AXIS = tuple(ApproachAxis)


METHOD_CAPABILITIES: tuple[MethodCapability, ...] = (
    MethodCapability(
        method="option_pair",
        implemented=True,
        measures=MeasurementKind.PREFERENCE,
        claim_types=_ANY_CLAIM_TYPE,
        axes=_ANY_AXIS,
        cannot_answer=(
            "Capability/performance. A click between two tutor-authored "
            "framings, embedded in ordinary conversation, never tests whether "
            "the learner could produce or execute either framing themselves — "
            "it only reveals which one they preferred, given both were already "
            "fully worked for them."
        ),
        notes=(
            "The existing production mechanism (disambiguate.py / "
            "reconcile_candidate, source=click). Not an instrument — axis and "
            "value come from whatever option set AssessAndBranch/"
            "DisambiguationOptions generated for that turn, not fixed by this "
            "row; 'any' reflects that generality, not an unfilled placeholder."
        ),
    ),
    MethodCapability(
        method="choose",
        implemented=False,
        measures=MeasurementKind.PREFERENCE,
        claim_types=_ANY_CLAIM_TYPE,
        axes=_ANY_AXIS,
        cannot_answer=(
            "Capability/performance, for the identical reason option_pair "
            "can't answer it: picking a favorite among several pre-worked "
            "framings says nothing about whether the learner can produce or "
            "execute any of them."
        ),
        notes=(
            "UNIMPLEMENTED. The instrument-layer generalization of option_pair "
            "— a dedicated, purpose-built choice exercise, not embedded mid-"
            "conversation, not restricted to exactly two options."
        ),
    ),
    MethodCapability(
        method="order",
        implemented=False,
        measures=MeasurementKind.PERFORMANCE,
        skills=(CapabilityLabel.OTHER,),
        cannot_answer=(
            "Framing preference in general — correctly sequencing the steps "
            "of a process demonstrates procedural understanding, not which "
            "STYLE of explanation the learner wants. Says nothing about the "
            "analogy/formality/brevity axes at all, and (like every "
            "PERFORMANCE row here) cannot answer into the StatedPreferenceLabel "
            "vocabulary — it targets a CapabilityClaim, not a Claim."
        ),
        notes=(
            "UNIMPLEMENTED. Best-fit contract shape: arrange N steps into the "
            "correct order; a specific known wrong ordering is the "
            "contradicts_when case, the same trivially-writable shape as "
            "locate's own contract. `skills` left at the OTHER placeholder "
            "until a real contract forces a specific CapabilityLabel — same "
            "discipline `predict`'s row followed before its contract existed."
        ),
    ),
    MethodCapability(
        method="locate",
        implemented=True,
        measures=MeasurementKind.PERFORMANCE,
        skills=(CapabilityLabel.TRACES_WORKED_STEPS,),
        cannot_answer=(
            "Framing preference. Locate never asks which framing the learner "
            "would prefer — only whether they can catch an error given the "
            "one already on screen. Says nothing about analogy_formal, "
            "brevity_depth, or any axis at all — this row's evidence targets "
            "a CapabilityClaim (skill=traces_worked_steps), never a "
            "preference Claim. (An earlier version of this table had this "
            "row declaring claim_types=(WANTS_STEPS_SHOWN,) — that was the "
            "same category error the contract itself made before the fix;"
            " see MeasurementKind's own docstring.)"
        ),
        notes=(
            "IMPLEMENTED: instruments.LOCATE_DEMO_SPEC / "
            "build_locate_demo_contract. One worked example, one seeded error."
        ),
    ),
    MethodCapability(
        method="adjust",
        implemented=False,
        measures=MeasurementKind.PREFERENCE,
        claim_types=(StatedPreferenceLabel.PREFERS_BREVITY,),
        axes=(ApproachAxis.BREVITY_DEPTH, ApproachAxis.SCOPE_NARROW_BROAD),
        cannot_answer=(
            "Capability/performance. Moving a dial to a preferred setting "
            "(brief <-> deep, narrow <-> broad) says nothing about whether the "
            "learner could actually produce or follow an explanation at that "
            "setting."
        ),
        notes=(
            "UNIMPLEMENTED. Natural fit for axes with a graded feel, unlike "
            "the binary axes choose/locate suit."
        ),
    ),
    MethodCapability(
        method="predict",
        implemented=True,
        measures=MeasurementKind.PERFORMANCE,
        skills=(CapabilityLabel.DERIVES_FORWARD,),
        cannot_answer=(
            "Framing preference. Predicting an outcome before it's revealed "
            "tests mechanistic understanding, not which teaching style the "
            "learner wants — the closest of the six to a pure capability "
            "probe, which is why it's the second one built. Targets a "
            "CapabilityClaim (skill=derives_forward), never a preference "
            "Claim — this row originally declared "
            "claim_types=(RULE_BEFORE_EXAMPLE,), which was live confirmation "
            "of the exact category error MeasurementKind's own docstring "
            "describes: real predict evidence checked against the actual DB "
            "had landed entirely on RULE_BEFORE_EXAMPLE/WANTS_STEPS_SHOWN "
            "preference claims, built solely from capability observations."
        ),
        notes=(
            "IMPLEMENTED: instruments.PREDICT_DEMO_SPEC / "
            "build_predict_demo_contract. A two-variable swap traced step by "
            "step; the learner predicts the result before it's revealed."
        ),
    ),
    MethodCapability(
        method="construct",
        implemented=False,
        measures=MeasurementKind.PERFORMANCE,
        skills=(CapabilityLabel.OTHER,),
        cannot_answer=(
            "Framing preference, and — unlike locate/predict — its own skill "
            "isn't populated yet either: assembling a free-form artifact from "
            "parts is the least-specified primitive, genuinely open until a "
            "first contract forces the question. Targets a CapabilityClaim, "
            "never a preference Claim, once built."
        ),
        notes=(
            "UNIMPLEMENTED. The richest primitive — payload shape needs its "
            "own sub-schema once built (see instruments.ConstructEventPayload)."
        ),
    ),
)
