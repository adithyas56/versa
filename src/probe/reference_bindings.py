"""Deterministic, exact-match reference-resolution memory -- the
counterpart to history_block.py's similarity-ranked memory, for a
different kind of past: a learner-specific recurring PHRASE ("that",
"the usual", "my project") whose meaning was pinned down once
(interaction_nodes.ClassifyReferenceResolution, off the critical path)
and should not need re-asking, in this session or a later one.

Two functions, same split as history_block.py:

`match_reference_bindings` is pure, no I/O: given a message and this
learner's known bindings (already fetched, latest per phrase), decide
which apply. An EXACT (case-insensitive) substring match on
`reference_text` -- no embedding, no ANN search, no LLM call anywhere
in this module. "A phrase either appears or it doesn't," per this
feature's own spec. Matches are filtered by a confidence computed from
`confirmation_count` and staleness (`last_confirmed_at`) -- a binding
heard once, months ago, must not be injected with the same weight as
one reconfirmed yesterday. Below `min_confidence`, a binding is not
injected at all, never injected weakly: a stale binding confidently
applied produces a fluent answer about the wrong thing, the feature's
own named worst failure mode.

`assemble_reference_bindings_block` does the one I/O call
(`ReferenceBindingStore.list_latest_for_learner`) plus the pure match/
render above.

Read by TWO different call sites each turn (loop.py):
`AssessAndBranch`, so a high-confidence match can suppress raising a
branch to disambiguate something already known, and `FinalAnswer`, so
the answer itself can use the known meaning. Both read the SAME
rendered block -- loop.py wraps it differently for each (an
instruction not to branch on it, versus a background-context framing)
rather than this module rendering two different strings.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from uuid import UUID

from pydantic import BaseModel

from probe.interactions import ReferenceBindingStore
from probe.models import ReferenceBinding


class ReferenceBindingConfig(BaseModel):
    """`half_life_days`/`target_confirmations`/`base_strength` set the
    decay rate; `min_confidence` is the one hard gate below which a
    binding is dropped entirely rather than injected weakly -- see this
    module's own docstring for why that asymmetry (drop, don't weaken)
    is the point, not an oversight.

    `base_strength` matters for the feature's own verification
    scenario: establishing a reference ONCE in one session and reusing
    it in a later one must already work (a single confirmation is not
    "provisional" -- it's a real classifier decision, evidenced by one
    real interaction). Strength ramps from `base_strength` toward 1.0
    as `confirmation_count` climbs toward `target_confirmations`,
    rather than starting at 0 -- so one confirmation is already
    confidently injectable, and repeated reuse only pushes it higher
    and makes it survive decay for longer, it never has to "earn" a
    first use.
    """

    enabled: bool = True
    half_life_days: float = 30.0
    base_strength: float = 0.6
    target_confirmations: int = 3
    min_confidence: float = 0.4


@dataclass
class ReferenceBindingMatch:
    binding: ReferenceBinding
    confidence: float


def _confidence(
    confirmation_count: int,
    last_confirmed_at: datetime,
    now: datetime,
    config: ReferenceBindingConfig,
) -> float:
    ramp = min(1.0, confirmation_count / config.target_confirmations)
    strength = config.base_strength + (1.0 - config.base_strength) * ramp
    age_days = max(0.0, (now - last_confirmed_at).total_seconds() / 86400)
    decay = 0.5 ** (age_days / config.half_life_days)
    return strength * decay


def match_reference_bindings(
    message_text: str,
    bindings: list[ReferenceBinding],
    config: ReferenceBindingConfig | None = None,
    now: datetime | None = None,
) -> list[ReferenceBindingMatch]:
    """Pure, deterministic: an EXACT (case-insensitive) substring match
    of `reference_text` in `message_text`, filtered by confidence.
    `bindings` is expected to already be latest-per-phrase (see
    `ReferenceBindingStore.list_latest_for_learner`) -- this function
    does no further deduplication."""
    if not message_text:
        return []
    cfg = config or ReferenceBindingConfig()
    now = now or datetime.now(UTC)
    lowered = message_text.lower()
    matches: list[ReferenceBindingMatch] = []
    for binding in bindings:
        if binding.reference_text.lower() not in lowered:
            continue
        confidence = _confidence(
            binding.confirmation_count, binding.last_confirmed_at, now, cfg
        )
        if confidence < cfg.min_confidence:
            continue
        matches.append(ReferenceBindingMatch(binding=binding, confidence=confidence))
    return matches


def render_reference_bindings_block(matches: list[ReferenceBindingMatch]) -> str:
    """Pure string templating, no I/O, no LLM call. Framed explicitly
    as PRIOR meaning, not current fact: the model is told to override
    it when the current message clearly means something else, and
    never to rewrite the learner's own question around it -- this is
    context, not a substitution."""
    if not matches:
        return ""
    lines = "\n".join(
        f'- "{m.binding.reference_text}" has previously meant: {m.binding.resolved_to}'
        for m in matches
    )
    return (
        "\nKNOWN REFERENCES FOR THIS LEARNER\n"
        f"{lines}\n"
        "These are prior meanings, not current facts -- use one only if "
        "it fits the current message, and disregard it if the current "
        "context clearly points to something else. Never rewrite or "
        "restate the learner's own question around it; this is "
        "background, not a substitution.\n"
    )


async def assemble_reference_bindings_block(
    store: ReferenceBindingStore,
    learner_id: UUID,
    message_text: str,
    config: ReferenceBindingConfig | None = None,
) -> tuple[str, list[UUID]]:
    """The one I/O call (`list_latest_for_learner`) plus the pure
    match/render above. Returns the rendered block (possibly "") and
    the matched bindings' ids, for the caller to log alongside the
    existing history-block logging (see loop.py)."""
    cfg = config or ReferenceBindingConfig()
    if not cfg.enabled or not message_text:
        return "", []
    bindings = await store.list_latest_for_learner(learner_id)
    if not bindings:
        return "", []
    matches = match_reference_bindings(message_text, bindings, cfg)
    if not matches:
        return "", []
    return render_reference_bindings_block(matches), [m.binding.id for m in matches]
