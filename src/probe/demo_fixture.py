"""A reproducible demo fixture: two learners with fixed, hand-authored
claim portraits that are deliberately OPPOSITE on the two axes whose
values have a defined opposite in this codebase's closed vocabulary
(`_OPPOSITE_LABELS` in claims.py — `concrete_before_abstract` <->
`rule_before_example`, `wants_analogies` <-> `no_analogies`), plus a
small fixed question set. Exists so a wrong-portrait comparison ("does
learner A's claims predict learner B's picks") has the same known
starting point every time it's run, instead of depending on a fresh
synthetic run's random draw through an LLM.

HAND-AUTHORED, NOT GENERATED: no LLM call, no embedding call
(`statement_embedding` is a zero vector — the same scaffolding choice
`instruments.py`'s own demo helpers make; nothing here depends on
embedding similarity, since these claims are looked up by learner
label, not by axis/similarity matching). `write_policy=LOCKED` and
`status=PROMOTED` are set directly at creation, which the real
inference pipeline never does on its own (a claim normally earns
PROMOTED through `ClaimStore.refresh`) — deliberate here: a fixture
portrait is a DECLARED ground truth for demo purposes, not something
meant to decay or need re-earning.

IDEMPOTENT: `seed_demo_fixture` finds the two learners by label
(`LearnerStore.get_by_label`) and skips any claim value already
present, so it's safe to call at the start of every demo session
rather than accumulating duplicates.

WHAT THIS DOES NOT BUILD: a comparison VIEW that reads these two
portraits and checks whether one predicts the other's picks. That
mechanism doesn't exist yet (see synthetic_v2.py's own note on the
"wrong-portrait control" it was laying groundwork for) — this module
is the reproducible DATA that comparison would need, not the
comparison itself.
"""

from __future__ import annotations

from uuid import UUID

import asyncpg

from probe.claims import ClaimStore
from probe.embeddings import EMBEDDING_DIM
from probe.learner import LearnerStore
from probe.models import Claim, ClaimSource, ClaimStatus, ClaimWritePolicy, Learner, StatedPreferenceLabel

CONCRETE_PORTRAIT_LABEL = "demo-portrait-concrete"
ABSTRACT_PORTRAIT_LABEL = "demo-portrait-abstract"

# Bounded, single-mechanism topics -- the shape this session's own
# synthetic runs found reliably forks on APPROACH (style), not aspect
# (subject). Small and fixed so a demo session asks the same questions
# every time, rather than a fresh random draw.
DEMO_QUESTION_SET: list[str] = [
    "Explain how binary search works.",
    "Explain how quicksort works.",
    "How do I calculate a percentage of a number?",
]

FIXTURE_CONFIDENCE = 0.85  # deliberately high and fixed: a clear, unambiguous portrait, not a borderline one

_CONCRETE_CLAIMS: list[tuple[str, str, StatedPreferenceLabel]] = [
    (
        "The learner prefers concrete, worked examples over abstract general rules.",
        "When given a choice between a concrete example and the general rule, the learner will choose the example.",
        StatedPreferenceLabel.CONCRETE_BEFORE_ABSTRACT,
    ),
    (
        "The learner prefers understanding new ideas through everyday analogies rather than formal technical language.",
        "When offered an everyday analogy versus a formal technical explanation, the learner will choose the analogy.",
        StatedPreferenceLabel.WANTS_ANALOGIES,
    ),
]
_ABSTRACT_CLAIMS: list[tuple[str, str, StatedPreferenceLabel]] = [
    (
        "The learner prefers the general rule stated up front, before any concrete example.",
        "When given a choice between the general rule and a concrete example, the learner will choose the rule.",
        StatedPreferenceLabel.RULE_BEFORE_EXAMPLE,
    ),
    (
        "The learner prefers formal technical language over everyday analogies.",
        "When offered a formal technical explanation versus an everyday analogy, the learner will choose the formal one.",
        StatedPreferenceLabel.NO_ANALOGIES,
    ),
]


async def _seed_portrait(
    claim_store: ClaimStore, learner_id: UUID, claims: list[tuple[str, str, StatedPreferenceLabel]]
) -> list[Claim]:
    existing_values = {c.value for c in await claim_store.list_for_learner(learner_id)}
    created = []
    for statement, test, value in claims:
        if value in existing_values:
            continue  # idempotent -- already seeded on a prior call
        claim = Claim(
            learner_id=learner_id, statement=statement, test=test, value=value,
            confidence=FIXTURE_CONFIDENCE, source=ClaimSource.STATED,
            write_policy=ClaimWritePolicy.LOCKED, status=ClaimStatus.PROMOTED,
            statement_embedding=[0.0] * EMBEDDING_DIM,
        )
        await claim_store.create(claim)
        created.append(claim)
    return created


async def seed_demo_fixture(pool: asyncpg.Pool) -> dict[str, Learner]:
    """Finds-or-creates the two demo learners and their fixed claim
    portraits. Safe to call repeatedly. Returns
    `{"concrete": Learner, "abstract": Learner}`."""
    learners = LearnerStore(pool)
    claim_store = ClaimStore(pool)

    concrete = await learners.get_by_label(CONCRETE_PORTRAIT_LABEL)
    if concrete is None:
        concrete = await learners.create(label=CONCRETE_PORTRAIT_LABEL)
    abstract = await learners.get_by_label(ABSTRACT_PORTRAIT_LABEL)
    if abstract is None:
        abstract = await learners.create(label=ABSTRACT_PORTRAIT_LABEL)

    await _seed_portrait(claim_store, concrete.id, _CONCRETE_CLAIMS)
    await _seed_portrait(claim_store, abstract.id, _ABSTRACT_CLAIMS)
    return {"concrete": concrete, "abstract": abstract}
