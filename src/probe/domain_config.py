"""The domain switch — lets the full pipeline run outside tutoring, to
test whether the architecture (storage, retrieval, ranking, the
history-block formatter) is genuinely domain-independent or only
happens to work because everything downstream was written assuming a
student and a tutor.

Scope, deliberately narrow: PROMPTS ONLY. `DomainConfig` holds nothing
but strings a node's own prompt-builder substitutes in — no behavior
branches on domain anywhere outside the six node classes that accept
one (AssessAndBranch, DisambiguationOptions, FinalAnswer,
ClassifyTurnOutcome, ClassifyStatedPreference, ClassifyReferenceResolution,
in disambiguate.py / interaction_nodes.py). If exercising `general`
mode for real ever turns out to need a schema change or a retrieval
change beyond the one explicitly carved out below, the answer is to
stop and report that finding, not to quietly patch it in here — that
leak IS the result this switch exists to surface.

The one explicit, requested exception: `interactions.domain` (migration
040) plus the domain filter in `retrieval.RetrievalContext`/
`stage1_filter` and `InteractionStore.get_recent_for_learner` — added
not because prompt changes needed it to function, but because mixing
an `education` learner's stored turns into a `general` run's retrieval
(or vice versa) would make every downstream measurement uninterpretable
regardless of how good the prompts are. That is a storage/retrieval
touch, made deliberately and reported as such — not a silent expansion
of this feature's own "prompts only" scope.

Deliberately excludes `GenerateAbstractForm` (interaction_nodes.py):
its whole purpose is staying domain-agnostic, so it is not given a
DomainConfig at all — it is instead the target of this feature's own
audit (does its current, single, unchanged prompt leak education-
flavored language into a general-domain abstract?). Also excludes
`render_structural_requirement`'s per-label imperatives
(interaction_nodes.py) and memory.py's thinking-style path-summary
language — both are genuinely pedagogy-flavored ("worked example",
"formal definition", "how they move through material") and neither
was named in this feature's own per-node list; left untouched and
called out as a known, separate leak point in this feature's
verification report rather than silently patched here.

Read ONCE, at startup (`load_domain_config`), never inside a node: a
node's constructor takes the resolved `DomainConfig` object, never the
env var or the `Domain` enum's own name — so a test constructs
`DomainConfig.general()` (or `.education()`) directly, with no
environment mutation required, per this feature's own instruction.
"""

from __future__ import annotations

import os
from enum import Enum

from pydantic import BaseModel

_ENV_VAR = "PROBE_DOMAIN"


class Domain(str, Enum):
    EDUCATION = "education"
    GENERAL = "general"


class DomainConfig(BaseModel):
    """One field per domain-facing string a node needs — deliberately
    NOT a single boolean/mode flag threaded into every prompt function,
    since that would just move an if/else into each node instead of
    removing it. Each field's comment names the one prompt-builder that
    reads it.
    """

    domain: Domain

    # Shared across AssessAndBranch / DisambiguationOptions / FinalAnswer /
    # ClassifyTurnOutcome / ClassifyStatedPreference / ClassifyReferenceResolution:
    # what to call the person on the asking side and, where relevant,
    # the one on the answering side.
    actor_noun: str  # "student" | "person"
    assistant_noun: str  # "tutor" | "assistant"

    # ClassifyTurnOutcome (_classify_prompt) — what to call the actor's
    # follow-up turn; "question" reads oddly for a general request that
    # was never phrased as a question.
    next_item_noun: str  # "question" | "message"

    # AssessAndBranch (_assess_prompt) — the ambiguity judgment's own
    # framing, and the noun phrase for "an unambiguous, concrete ask."
    ambiguity_scope_phrase: str  # "across subjects or approaches" | "across interpretations or intents"
    concrete_noun_phrase: str  # "A concrete question, a named problem" | "A concrete request, a named task"

    # DisambiguationOptions (_options_prompt) — how each option should
    # be phrased.
    options_style_phrase: str

    # FinalAnswer (FinalAnswer.run) — the register-setting opening line
    # and the closing constraint against ending on a checking-in
    # question. Full sentences, not sub-phrases: tutor/assistant register
    # differs enough in shape (not just word choice) that composing it
    # from smaller shared fragments would understate the difference the
    # spec asks for.
    final_answer_role_line: str
    final_answer_closing_line: str

    # ClassifyTurnOutcome (_classify_prompt) — the one judgment the
    # spec gives verbatim per domain; stored whole rather than
    # decomposed further.
    outcome_judgment_phrase: str  # "whether the next question suggests the answer landed" | "whether the next message suggests the response was useful"

    # ClassifyStatedPreference (_stated_preference_prompt) — "taught"/
    # "teaching style" both assume a classroom; the general variant
    # asks about being helped/responded to instead. `examples` is the
    # parenthetical illustrative list, swapped rather than reused
    # verbatim since one of the education examples ("a number example
    # before the formal rule") is itself classroom-flavored.
    stated_preference_scope_phrase: str  # "taught in general" | "helped in general"
    stated_preference_style_phrase: str  # "teaching style" | "response style"
    stated_preference_examples: str

    @classmethod
    def education(cls) -> "DomainConfig":
        return cls(
            domain=Domain.EDUCATION,
            actor_noun="student",
            assistant_noun="tutor",
            next_item_noun="question",
            ambiguity_scope_phrase="across subjects or approaches",
            concrete_noun_phrase="A concrete question, a named problem",
            options_style_phrase=(
                "the natural next thing a tutor would say to confirm that "
                "specific reading, worded as a subject-matter question"
            ),
            final_answer_role_line=(
                "You are a tutor having a conversation with a student. "
                "Respond directly and helpfully to their latest message.\n"
            ),
            final_answer_closing_line=(
                "Never end by asking how the student feels, what they "
                "prefer, or what kind of learner they are.\n"
            ),
            outcome_judgment_phrase="whether the next question suggests the answer landed",
            stated_preference_scope_phrase="taught in general",
            stated_preference_style_phrase="teaching style",
            stated_preference_examples=(
                'e.g. "always show me a number example before the formal '
                'rule", "keep your answers short", "use real-world examples"'
            ),
        )

    @classmethod
    def general(cls) -> "DomainConfig":
        return cls(
            domain=Domain.GENERAL,
            actor_noun="person",
            assistant_noun="assistant",
            next_item_noun="message",
            ambiguity_scope_phrase="across interpretations or intents",
            concrete_noun_phrase="A concrete request, a named task",
            options_style_phrase=(
                "a plain statement of that specific interpretation of "
                "what the person is asking for"
            ),
            final_answer_role_line=(
                "You are an assistant having a conversation with a "
                "person. Respond directly and helpfully to their latest "
                "message.\n"
            ),
            final_answer_closing_line=(
                "Never end by asking how they feel, what they prefer, or "
                "what kind of person they are.\n"
            ),
            outcome_judgment_phrase="whether the next message suggests the response was useful",
            stated_preference_scope_phrase="helped in general",
            stated_preference_style_phrase="response style",
            stated_preference_examples=(
                'e.g. "always give me the bottom line before the details", '
                '"keep your answers short", "use concrete examples"'
            ),
        )


def load_domain_config(env: dict[str, str] | None = None) -> DomainConfig:
    """The ONE place `PROBE_DOMAIN` is read — called once at startup
    (cli.py). `env` defaults to `os.environ`; a test wanting to check
    startup resolution itself (rather than just constructing
    `DomainConfig.general()` directly, the normal way to get a non-
    default config in a test) can pass a plain dict instead of mutating
    the real environment.

    An unrecognized value fails loudly rather than silently falling
    back to `education` -- a typo'd PROBE_DOMAIN should not run
    silently in the wrong mode.
    """
    source = env if env is not None else os.environ
    raw = source.get(_ENV_VAR, Domain.EDUCATION.value)
    try:
        domain = Domain(raw)
    except ValueError as exc:
        valid = ", ".join(d.value for d in Domain)
        raise ValueError(
            f"{_ENV_VAR}={raw!r} is not a recognized domain (expected one of: {valid})"
        ) from exc
    return DomainConfig.general() if domain is Domain.GENERAL else DomainConfig.education()
