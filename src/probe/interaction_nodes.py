"""The three LLM-touching pieces of the interaction pipeline — all off
the critical path, all fast-tier, all optional in the sense that a
failure here degrades to "nothing new learned this turn," never a lost
turn. None of these run on the critical path a student is waiting on;
`SessionLoop` fires them as detached background tasks after the turn's
own response has already been returned (see loop.py's own comments on
where and why).

`ClassifyTurnOutcome` and `GenerateAbstractForm` are genuine LLM nodes
in this codebase's usual sense and are expected to be routed through
`SessionLoop._call_node` when fired from the loop, same as any other
node (CLAUDE.md invariant 2) — the fact that they run detached rather
than awaited inline doesn't exempt them from the audit trail.

`SelectionPredictor` is a `Protocol`, not a class with one obvious
shape, precisely so a learned selection policy can later replace
`LLMSelectionPredictor` without touching `predictions`' schema or
anything that calls `predict()` — the event/storage contract
(`Prediction.predicted_scores`/`retrieved_candidate_ids`/
`retrieval_provenance`) is what's fixed; the function that produces
`predicted_scores` is exactly the part meant to be swapped out.
"""

from __future__ import annotations

import json
import logging
from typing import Protocol
from uuid import UUID

from pydantic import BaseModel

from probe.domain_config import DomainConfig
from probe.llm import LLMClient
from probe.models import (
    AbstractionResult,
    InteractionOption,
    OutcomeClassification,
    ReferenceResolutionClassification,
    RetrievalCandidate,
    StatedPreferenceClassification,
    StatedPreferenceLabel,
    TurnOutcomeLabel,
)

logger = logging.getLogger(__name__)

# Bumped whenever the classification/abstraction prompt changes in a
# way that would make old and new outputs not directly comparable —
# written onto every row these nodes produce (turn_outcomes.
# classifier_version / interaction_abstracts.generator_version).
CLASSIFIER_VERSION = "classify-turn-outcome-v1"
ABSTRACTOR_VERSION = "generate-abstract-form-v1"
PREDICTOR_MODEL_VERSION = "llm-selection-predictor-v1"
STATED_PREFERENCE_CLASSIFIER_VERSION = "classify-stated-preference-v1"
REFERENCE_BINDING_CLASSIFIER_VERSION = "classify-reference-resolution-v1"


class ClassifierConfig(BaseModel):
    # Below this, treat the call as an abstention regardless of what
    # the model itself reported for `abstains` -- belt and suspenders,
    # since a model can under-report its own uncertainty.
    min_confidence: float = 0.6


def _classify_prompt(
    prior_question: str,
    prior_selected_option: str | None,
    prior_response: str,
    next_question: str,
    domain: DomainConfig | None = None,
) -> str:
    d = domain or DomainConfig.education()
    option_block = (
        f"\nThe {d.actor_noun} resolved that by selecting this option: "
        f'"{prior_selected_option}"\n'
        if prior_selected_option
        else ""
    )
    return (
        "CLASSIFY:TURN_OUTCOME\n"
        f'A {d.actor_noun} previously sent: "{prior_question}"\n'
        f"{option_block}"
        f'A {d.assistant_noun} responded: "{prior_response}"\n'
        f"The {d.actor_noun}'s VERY NEXT {d.next_item_noun} was: "
        f'"{next_question}"\n\n'
        f"Judge {d.outcome_judgment_phrase}:\n"
        f"- matched: the next {d.next_item_noun} builds naturally on the "
        "response, confirming it was understood and accepted as relevant\n"
        f"- contradicted_intent: the next {d.next_item_noun} reveals the "
        f"response answered the wrong thing, or the {d.actor_noun}'s actual "
        f"need was different from what the {d.assistant_noun} assumed\n"
        f"- moved_on: the {d.actor_noun} has moved to a genuinely different "
        "topic, whether satisfied or not -- do not try to guess which; "
        "that distinction is not answerable from this window alone\n\n"
        "If the evidence is genuinely ambiguous, set abstains=true rather "
        "than guessing -- a wrong label is worse than no label.\n"
        'Respond with JSON: {"outcome": "...", "confidence": 0.0-1.0, '
        '"abstains": true or false}'
    )


class ClassifyTurnOutcome:
    """Step run at turn N+1 (or on a cross-session catch-up — see
    interactions.InteractionRecorder), classifying turn N from its own
    question/selected-option/response and turn N+1's new question.

    Never called for a row still missing a response (interactions.py
    itself short-circuits an options-offered row to DEFERRED
    immediately, no LLM call needed for that case at all — see
    `InteractionRecorder.record`).
    """

    def __init__(
        self,
        llm: LLMClient,
        config: ClassifierConfig | None = None,
        domain_config: DomainConfig | None = None,
    ) -> None:
        self._llm = llm
        self._config = config or ClassifierConfig()
        self._domain = domain_config or DomainConfig.education()
        self.last_call_count: int = 0

    async def run(
        self,
        prior_question: str,
        prior_response: str,
        next_question: str,
        prior_selected_option: str | None = None,
    ) -> OutcomeClassification:
        self.last_call_count = 0
        raw = await self._llm.complete(
            _classify_prompt(
                prior_question, prior_selected_option, prior_response, next_question,
                self._domain,
            )
        )
        self.last_call_count = 1
        parsed = _parse_classification(raw, self._config.min_confidence)
        if parsed is None:
            logger.warning(
                "ClassifyTurnOutcome: unparseable response, treating as "
                "abstention: %r",
                raw[:200],
            )
            return OutcomeClassification(
                outcome=TurnOutcomeLabel.DEFERRED, confidence=0.0, abstains=True
            )
        return parsed


def _parse_classification(raw: str, min_confidence: float) -> OutcomeClassification | None:
    try:
        data = json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        return None
    if not isinstance(data, dict):
        return None
    outcome_raw = data.get("outcome")
    confidence = data.get("confidence")
    if not isinstance(outcome_raw, str) or not isinstance(confidence, (int, float)):
        return None
    try:
        outcome = TurnOutcomeLabel(outcome_raw)
    except ValueError:
        return None
    abstains = bool(data.get("abstains", False)) or confidence < min_confidence
    return OutcomeClassification(outcome=outcome, confidence=float(confidence), abstains=abstains)


def _abstract_prompt(question_text: str, response_text: str) -> str:
    return (
        "ABSTRACT:FORM\n"
        "Rewrite the exchange below, stripping every domain-specific "
        "noun, name, and technical term, while keeping the STRUCTURE of "
        "what happened -- what kind of thing was asked for, what kind "
        "of thing was given in return. The result should read the same "
        "way regardless of whether this was calculus, a programming "
        "question, or a history question.\n\n"
        f'Question: "{question_text}"\n'
        f'Response: "{response_text}"\n\n'
        'Example rewrite style: "chose the worked example over the '
        'stated rule", "asked for a definition before an application", '
        '"pushed back on the first explanation and asked for a '
        'simpler one".\n'
        'Respond with JSON: {"abstract_form": "..."}'
    )


class GenerateAbstractForm:
    """Runs off the critical path, alongside the outcome classifier —
    see `InteractionAbstract`'s own docstring for why this is a
    separate append-only table rather than a later UPDATE to
    `interactions`."""

    def __init__(self, llm: LLMClient) -> None:
        self._llm = llm
        self.last_call_count: int = 0

    async def run(self, question_text: str, response_text: str) -> AbstractionResult:
        self.last_call_count = 0
        raw = await self._llm.complete(_abstract_prompt(question_text, response_text))
        self.last_call_count = 1
        try:
            data = json.loads(raw)
            form = data.get("abstract_form") if isinstance(data, dict) else None
        except (json.JSONDecodeError, TypeError):
            form = None
        if not form or not isinstance(form, str):
            logger.warning(
                "GenerateAbstractForm: unparseable response, falling back "
                "to a generic form: %r",
                raw[:200],
            )
            form = "resolved a question about an unfamiliar topic"
        return AbstractionResult(abstract_form=form)


_LABEL_DESCRIPTIONS = (
    "- concrete_before_abstract: wants a concrete example or specific "
    "numbers before the formal rule or definition\n"
    "- rule_before_example: wants the rule or definition stated before "
    "an example\n"
    "- wants_steps_shown: wants every intermediate step shown "
    "explicitly, never skipped or compressed\n"
    "- prefers_brevity: wants short, minimal-elaboration answers\n"
    "- wants_analogies: wants real-world analogies or everyday "
    "comparisons\n"
    "- no_analogies: wants direct technical explanation, no analogies\n"
    "- other: any other explicit standing preference not covered above"
)


def _stated_preference_prompt(question_text: str, domain: DomainConfig | None = None) -> str:
    d = domain or DomainConfig.education()
    return (
        "CLASSIFY:STATED_PREFERENCE\n"
        f'A {d.actor_noun} sent this message to a {d.assistant_noun}: '
        f'"{question_text}"\n\n'
        f"Did the {d.actor_noun} EXPLICITLY state a standing preference "
        f"about HOW they want to be {d.stated_preference_scope_phrase} -- "
        "not what they want explained this turn, but an instruction "
        f"about {d.stated_preference_style_phrase}, format, or ordering "
        f"that should carry forward to future turns too "
        f"({d.stated_preference_examples})?\n\n"
        "Only count an EXPLICIT statement. Do NOT count: a one-off "
        "request specific to this question alone, a click on an "
        "option, or a pattern you might infer from repeated behavior -- "
        "an inferred pattern is a guess, not a stated fact, and belongs "
        "in a different layer entirely.\n\n"
        "If a preference was stated, classify it into EXACTLY ONE of "
        f"these labels:\n{_LABEL_DESCRIPTIONS}\n\n"
        'Respond with JSON: {"has_preference": true/false, '
        f'"stated_preference": "<the preference in the {d.actor_noun}\'s '
        'own terms, or null>", "label": "<one of the labels above, or '
        'null if has_preference is false>"}'
    )


class ClassifyStatedPreference:
    """Fix B's write side (see StatedPreference's own docstring for the
    full record of why this exists and why it is a separate table).
    Runs off the critical path, one fast-tier call per LEARNER-authored
    turn only -- loop.py never calls this for a system_option/click
    turn, since there is no free text from the student there to judge.

    Extracts the preference in the student's OWN words (`stated_preference`)
    AND classifies it into one of `StatedPreferenceLabel`'s closed
    vocabulary -- kept as two separate outputs deliberately: the raw
    text is the faithful record of what was said, while `label` is what
    lets `render_structural_requirement` map to a hand-written
    imperative instead of turning free text into a requirement on the
    fly. A `has_preference=True` result with an unparseable/missing
    label falls back to OTHER rather than being discarded -- the raw
    text is still real signal even when the model's own categorization
    attempt failed."""

    def __init__(self, llm: LLMClient, domain_config: DomainConfig | None = None) -> None:
        self._llm = llm
        self._domain = domain_config or DomainConfig.education()
        self.last_call_count: int = 0

    async def run(self, question_text: str) -> StatedPreferenceClassification:
        self.last_call_count = 0
        raw = await self._llm.complete(
            _stated_preference_prompt(question_text, self._domain)
        )
        self.last_call_count = 1
        try:
            data = json.loads(raw)
        except (json.JSONDecodeError, TypeError):
            data = None
        if not isinstance(data, dict) or not isinstance(data.get("has_preference"), bool):
            logger.warning(
                "ClassifyStatedPreference: unparseable response, "
                "treating as no preference stated: %r",
                raw[:200],
            )
            return StatedPreferenceClassification(has_preference=False)
        has_preference = data["has_preference"]
        stated = data.get("stated_preference")
        if not has_preference or not isinstance(stated, str) or not stated.strip():
            return StatedPreferenceClassification(has_preference=False)
        try:
            label = StatedPreferenceLabel(data.get("label"))
        except ValueError:
            logger.warning(
                "ClassifyStatedPreference: has_preference=True but label "
                "%r is invalid/missing, falling back to OTHER: %r",
                data.get("label"),
                raw[:200],
            )
            label = StatedPreferenceLabel.OTHER
        return StatedPreferenceClassification(
            has_preference=True, stated_preference=stated.strip(), label=label
        )


# render_structural_requirement's per-label imperatives -- hand-written,
# not derived from the classifier's raw text, and editable here without
# touching the classifier at all. Each one does three things, per the
# review that produced this design: states the required output shape
# in the imperative mood, names what to avoid EXPLICITLY (the negative
# clause is not optional -- it was present in the phrasing that
# actually flipped an answer in testing and absent from the one that
# didn't), and asserts priority over the model's default format.
# Tuned by reading real answers; expect these to be edited over time.
_STRUCTURAL_REQUIREMENT_IMPERATIVES: dict[StatedPreferenceLabel, str] = {
    StatedPreferenceLabel.CONCRETE_BEFORE_ABSTRACT: (
        "Open with a concrete worked example using specific numbers or "
        "a real scenario, and state the general rule or formal "
        "definition only afterward. Do not begin with the abstract "
        "rule, definition, or general statement. This ordering takes "
        "priority over your default explanation format."
    ),
    StatedPreferenceLabel.RULE_BEFORE_EXAMPLE: (
        "State the general rule or formal definition first, then "
        "illustrate it with an example. Do not open with a worked "
        "example or scenario before naming the rule. This ordering "
        "takes priority over your default explanation format."
    ),
    StatedPreferenceLabel.WANTS_STEPS_SHOWN: (
        "Show every intermediate step explicitly, in order. Do not "
        "skip steps, jump straight to a final result, or compress "
        "multiple steps into one. This level of detail takes priority "
        "over your default level of brevity."
    ),
    StatedPreferenceLabel.PREFERS_BREVITY: (
        "Keep the answer to a few sentences covering only the "
        "essential point. Do not add extended explanation, multiple "
        "examples, or optional elaboration. This brevity takes "
        "priority over your default level of detail."
    ),
    StatedPreferenceLabel.WANTS_ANALOGIES: (
        "Include a real-world analogy or everyday comparison to "
        "illustrate the idea. Do not give a purely technical or "
        "symbolic explanation with no everyday comparison. This takes "
        "priority over your default explanation style."
    ),
    StatedPreferenceLabel.NO_ANALOGIES: (
        "Explain using direct technical terms and the subject's own "
        "vocabulary only. Do not use an everyday analogy or real-world "
        "comparison as a stand-in for the technical explanation. This "
        "takes priority over your default explanation style."
    ),
}


def render_structural_requirement(
    label: StatedPreferenceLabel | None, stated_preference: str
) -> str:
    """Fix B's read side -- pure string templating, no LLM call, same
    determinism discipline as history_block.py's own formatter.

    Converts a STATED PREFERENCE into a REQUIREMENT on output shape via
    a deterministic label -> imperative lookup, rather than wrapping
    the raw text directly. That distinction is load-bearing, not
    stylistic: a live re-run found that rendering the raw text as a
    background fact -- even placed prominently, even phrased as
    "follow this" -- reads as a DESCRIPTION a model can agree with
    while still generating its default answer shape. The hand-written
    imperatives in `_STRUCTURAL_REQUIREMENT_IMPERATIVES` are what
    measurably did better in testing.

    `OTHER` and any label the lookup doesn't cover fall back to a
    generic wrapper around the raw text -- weaker than a hand-written
    imperative (it's still closer to a description than a requirement),
    but stronger than shipping nothing for a real, explicitly stated
    preference this classifier just doesn't have a specific imperative
    for yet."""
    imperative = _STRUCTURAL_REQUIREMENT_IMPERATIVES.get(label)
    if imperative is None:
        imperative = (
            f"Structure your answer so that: {stated_preference}. This "
            "takes priority over your default explanation format."
        )
    return f"\nStructural requirement: {imperative}\n"


class ReferenceResolutionClassifierConfig(BaseModel):
    # Stricter than ClassifierConfig.min_confidence (0.6, ClassifyTurnOutcome's
    # own bar): a wrong turn_outcome label costs a rerank bonus; a wrong
    # reference binding gets fed BACK into a future turn's prompt as an
    # assumed fact and can quietly steer an answer toward the wrong
    # meaning entirely. The feature's own review is explicit that "a
    # noisy table is worse than a sparse one here" -- that calls for a
    # higher bar, not the same one every other classifier here uses.
    min_confidence: float = 0.75


def _reference_resolution_prompt(
    recent_history: str,
    turn_question: str,
    turn_response: str | None,
    originating_question: str | None = None,
    domain: DomainConfig | None = None,
) -> str:
    d = domain or DomainConfig.education()
    history_block = f"\nRecent conversation:\n{recent_history}\n" if recent_history else ""
    # A click-resolution turn carries the strongest possible signal:
    # `originating_question` is the earlier ambiguous message verbatim,
    # `turn_question` is the specific reading the actor confirmed --
    # framed as a resolution event, not just another message, whenever
    # it's available.
    if originating_question:
        resolution_block = (
            f'\nThis turn resolved an earlier ambiguous message '
            f'("{originating_question}") by confirming: "{turn_question}"\n'
        )
    else:
        resolution_block = f"\nThe {d.actor_noun}'s message this turn: \"{turn_question}\"\n"
    response_block = (
        f'\nThe {d.assistant_noun} then responded: "{turn_response}"\n'
        if turn_response
        else ""
    )
    return (
        "CLASSIFY:REFERENCE_RESOLUTION\n"
        f"{history_block}"
        f"{resolution_block}"
        f"{response_block}"
        "\nDid this exchange settle the meaning of a genuinely AMBIGUOUS, "
        f'RECURRING reference the {d.actor_noun} uses -- a short standing '
        'phrase like "the usual", "my project", "that thing we did" -- '
        "whose meaning needed the broader conversation, a clarification, "
        f"or a chosen option to pin down, and which the {d.actor_noun} is "
        "likely to reuse VERBATIM in a later, otherwise unrelated turn "
        "or session?\n\n"
        "Do NOT count an ordinary pronoun or reference resolvable from "
        'just the immediately preceding turn ("it", "that", "she" '
        "referring to the last sentence) -- only a phrase that is its "
        "own standing shorthand for something specific to this "
        f"{d.actor_noun}, worth remembering across turns, counts.\n\n"
        "If you are not confident this happened, or the reference is "
        "ordinary and immediately resolvable, set resolved=false -- a "
        "missed one is fine, a wrong one is not.\n\n"
        'Respond with JSON: {"resolved": true or false, "reference_text": '
        '"<the exact recurring phrase, verbatim, or null>", "resolved_to": '
        '"<plain-English statement of what it means, or null>", '
        '"confidence": 0.0-1.0}'
    )


def _parse_reference_resolution(
    raw: str, min_confidence: float
) -> ReferenceResolutionClassification | None:
    try:
        data = json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        return None
    if not isinstance(data, dict):
        return None
    resolved = data.get("resolved")
    if not isinstance(resolved, bool):
        return None
    if not resolved:
        return ReferenceResolutionClassification(resolved=False)
    reference_text = data.get("reference_text")
    resolved_to = data.get("resolved_to")
    confidence = data.get("confidence")
    if (
        not isinstance(reference_text, str)
        or not reference_text.strip()
        or not isinstance(resolved_to, str)
        or not resolved_to.strip()
        or not isinstance(confidence, (int, float))
    ):
        return None
    if confidence < min_confidence:
        # Below the bar -- treated the same as "didn't happen," not a
        # weaker write. See ReferenceResolutionClassifierConfig's own
        # docstring for why this bar is stricter than the other
        # classifiers in this module.
        return ReferenceResolutionClassification(resolved=False)
    return ReferenceResolutionClassification(
        resolved=True,
        reference_text=reference_text.strip(),
        resolved_to=resolved_to.strip(),
        confidence=float(confidence),
    )


class ClassifyReferenceResolution:
    """The reference-resolution memory's write side (see
    `ReferenceBinding`'s own docstring for the full record). Runs off
    the critical path, fired whenever a turn produces a real response
    (same gate as `GenerateAbstractForm` -- see loop.py's
    `_record_interaction`), regardless of `question_author`: a branch
    selection (`question_author=SYSTEM_OPTION`) is explicitly one of
    the three ways a reference gets settled, unlike
    `ClassifyStatedPreference`, which is learner-turns-only because it
    needs the student's own free text.

    `resolved=False` (the expected outcome most turns) is a terminal
    result here -- unlike `ClassifyStatedPreference`/`ClassifyTurnOutcome`,
    the caller writes NOTHING to the store in that case. See
    `ReferenceBinding`'s docstring for why: this feature explicitly
    trades classifier coverage-auditability (every other classifier
    here writes a row every time) for a sparse, high-precision table.
    """

    def __init__(
        self,
        llm: LLMClient,
        config: ReferenceResolutionClassifierConfig | None = None,
        domain_config: DomainConfig | None = None,
    ) -> None:
        self._llm = llm
        self._config = config or ReferenceResolutionClassifierConfig()
        self._domain = domain_config or DomainConfig.education()
        self.last_call_count: int = 0

    async def run(
        self,
        recent_history: str,
        turn_question: str,
        turn_response: str | None = None,
        originating_question: str | None = None,
    ) -> ReferenceResolutionClassification:
        self.last_call_count = 0
        raw = await self._llm.complete(
            _reference_resolution_prompt(
                recent_history, turn_question, turn_response, originating_question,
                self._domain,
            )
        )
        self.last_call_count = 1
        parsed = _parse_reference_resolution(raw, self._config.min_confidence)
        if parsed is None:
            logger.warning(
                "ClassifyReferenceResolution: unparseable response, "
                "treating as unresolved: %r",
                raw[:200],
            )
            return ReferenceResolutionClassification(resolved=False)
        return parsed


class SelectionPredictor(Protocol):
    """Swappable by design: the storage contract (`Prediction`'s
    fields) never changes regardless of which implementation of this
    Protocol computes the scores. `context` is whatever
    `retrieval.retrieve()` surfaced for this learner/situation —
    deterministic, no LLM, computed by the caller before `predict()`
    is invoked."""

    async def predict(
        self, options: list[InteractionOption], context: list[RetrievalCandidate]
    ) -> dict[UUID, float]: ...


def _predict_prompt(options: list[InteractionOption], context: list[RetrievalCandidate]) -> str:
    option_lines = "\n".join(
        f"- id={o.option_id}: {o.option_text}" for o in options
    )
    if context:
        context_lines = "\n".join(
            f"- [{c.scope}] {c.text!r} (similarity={c.similarity:.2f}"
            + (f", outcome={c.outcome.value}" if c.outcome else "")
            + (f", support={c.support_count} learners" if c.distinct_learner_count else "")
            + ")"
            for c in context
        )
        context_block = (
            "\nRelevant history for this learner and similar learners, "
            f"most relevant first:\n{context_lines}\n"
        )
    else:
        context_block = ""
    return (
        "PREDICT:SELECTION\n"
        "A student was just offered these distinct readings of their "
        f"message, as clickable options:\n{option_lines}\n"
        f"{context_block}"
        "\nEstimate the probability the student selects each option, "
        "as a number between 0 and 1. The probabilities do not need to "
        "sum to exactly 1.\n"
        'Respond with JSON: {"<option id>": 0.0-1.0, ...}'
    )


class LLMSelectionPredictor:
    """The current, replaceable implementation of `SelectionPredictor`.
    One fast-tier call, informed by whatever `retrieve()` already
    found — never does its own retrieval, never blocks the turn that
    triggered it (see loop.py's fire-and-forget wiring)."""

    model_version = PREDICTOR_MODEL_VERSION

    def __init__(self, llm: LLMClient) -> None:
        self._llm = llm
        self.last_call_count: int = 0

    async def predict(
        self, options: list[InteractionOption], context: list[RetrievalCandidate]
    ) -> dict[UUID, float]:
        self.last_call_count = 0
        if not options:
            return {}
        raw = await self._llm.complete(_predict_prompt(options, context))
        self.last_call_count = 1
        valid_ids = {o.option_id for o in options}
        try:
            data = json.loads(raw)
        except (json.JSONDecodeError, TypeError):
            data = None
        scores: dict[UUID, float] = {}
        if isinstance(data, dict):
            for key, value in data.items():
                try:
                    option_id = UUID(str(key))
                except ValueError:
                    continue
                if option_id in valid_ids and isinstance(value, (int, float)):
                    scores[option_id] = float(value)
        if not scores:
            logger.warning(
                "LLMSelectionPredictor: unparseable or empty response, "
                "falling back to a uniform distribution: %r",
                raw[:200],
            )
            uniform = 1.0 / len(options)
            scores = {o.option_id: uniform for o in options}
        return scores
