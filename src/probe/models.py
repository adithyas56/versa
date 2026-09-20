from datetime import datetime, timezone
from enum import Enum
from uuid import UUID, uuid4

from pydantic import BaseModel, Field, field_validator, model_validator

from probe.domain_config import Domain


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class Learner(BaseModel):
    """Identity-level record only.

    Deliberately thin: label/display_name/preferred_register/timezone
    are the only fields, all optional, none behavioral. What a learner
    knows or is working toward is never tracked here.
    """

    id: UUID = Field(default_factory=uuid4)
    label: str | None = None
    display_name: str | None = None
    preferred_register: str | None = None
    timezone: str | None = None
    created_at: datetime = Field(default_factory=_utcnow)


class NodeCall(BaseModel):
    """One row from node_calls — for read paths (the web UI) that need
    a specific past call's input/output rather than just writing new
    ones (NodeCallStore.record is the only writer, per CLAUDE.md
    invariant 2)."""

    id: UUID
    node_name: str
    session_id: UUID
    turn_index: int
    input_json: dict
    output_json: object
    timestamp: datetime


class TurnRecord(BaseModel):
    """One student turn, as persisted by TranscriptStore.record_turn."""

    id: UUID
    session_id: UUID
    turn_index: int
    text: str
    created_at: datetime


class BranchStatus(str, Enum):
    OPEN = "open"
    MATCHED = "matched"
    UNMATCHED = "unmatched"
    SUPERSEDED = "superseded"


class OptionStatus(str, Enum):
    OPEN = "open"
    SELECTED = "selected"
    SUPERSEDED = "superseded"


class Option(BaseModel):
    """One clickable button — an interpretation-free evidence channel:
    a click is an unambiguous fact about which reading the student
    meant, with no guess about what they said. `branch_id` is a hard
    1:1 mapping onto the branch it confirms.

    Shared verbatim by the disambiguation flow (disambiguate.py /
    `disambiguation_options`); it was the tree-based system's model
    first, hence the name.
    """

    id: UUID = Field(default_factory=uuid4)
    branch_id: UUID
    generation_id: UUID
    session_id: UUID
    turn_index: int
    text: str
    status: OptionStatus = OptionStatus.OPEN
    created_at: datetime = Field(default_factory=_utcnow)


class OptionProposal(BaseModel):
    """DisambiguationOptions' raw per-item output before persistence —
    not itself DB-backed (`Option` is, once IDs/session/turn context
    are attached in loop.py).

    `side` ("first" or "second") is which pole of the chosen `axis`
    (`OptionSet.axis`) this option represents, matching the axis
    enum's own name order verbatim (e.g. for `concrete_general`,
    "first" is the concrete pole, "second" is the general one) — a
    live diagnostic found downstream consumers inferring this from
    option TEXT via keyword lists, which drifted silently every time
    an axis showed up in unanticipated phrasing (three separate
    documented incidents). `side` is decided by the same call that
    already decides `axis`, at generation time, and persisted
    (`InteractionOption.side`, migration 045) exactly like `kind`/
    `axis` already are — a downstream consumer reads which side an
    option was, it never re-derives it. Always `None` for a
    SUBJECT-kind option (there is no axis, hence no side)."""

    branch_id: UUID
    text: str
    side: str | None = None


class AmbiguityKind(str, Enum):
    """DisambiguationOptions' own first decision, made before it writes
    a single option — see that class's docstring for the audit finding
    (pairs mixing a subject option into an otherwise-settled approach
    set) that made this decision explicit instead of implicit in the
    option text.

    SUBJECT: the candidate readings genuinely disagree about WHAT TOPIC
    the message concerns, and that is not already settled by context.
    APPROACH: the topic is already established (by session history, a
    reference binding, or the readings themselves not naming different
    topics) and the real choice is about HOW to address it, not what
    it's about.
    """

    SUBJECT = "subject"
    APPROACH = "approach"


class ApproachAxis(str, Enum):
    """The closed vocabulary an APPROACH-kind option set must pick
    exactly one member of — see disambiguate.py's `_AXIS_DESCRIPTIONS`
    for what each one means. A single axis admits exactly two sides by
    construction, which is also what rules out a confounded three-way
    approach set: there is no third side to a single dimension.
    """

    CONCRETE_GENERAL = "concrete_general"
    SCOPE_NARROW_BROAD = "scope_narrow_broad"
    RIGOR_INTUITION = "rigor_intuition"
    MECHANISM_PROCEDURE = "mechanism_procedure"
    WORKED_STEPS_RESULT = "worked_steps_result"
    ANALOGY_FORMAL = "analogy_formal"
    SINGLE_EXAMPLE_PATTERN = "single_example_pattern"
    FORWARD_DERIVATION_BACKWARD_VERIFICATION = "forward_derivation_backward_verification"
    BREVITY_DEPTH = "brevity_depth"
    STRUCTURED_NARRATIVE = "structured_narrative"


class OptionSet(BaseModel):
    """DisambiguationOptions' full output — not itself DB-backed.
    `kind`/`axis` are the audit trail this feature's own review asked
    for ("log the chosen axis on the option set"): captured verbatim in
    `node_calls.output_json` (CLAUDE.md invariant 2) the moment this
    node runs, so a later analysis of what axis a set actually varied
    on reads a recorded decision, never a guess reverse-engineered from
    option text. Both `None` only on the exhausted-retries fallback
    (see DisambiguationOptions.run), where `proposals` is also empty —
    the same "nothing usable, degrade to a direct answer" shape as
    before this feature existed.

    `axis` is always `None` when `kind` is SUBJECT (there is no axis to
    report — the choice was about topic, not method) and always set
    when `kind` is APPROACH."""

    kind: AmbiguityKind | None = None
    axis: ApproachAxis | None = None
    proposals: list[OptionProposal] = Field(default_factory=list)


class DisambiguationTurn(BaseModel):
    """One AssessAndBranch call's identity — the "how did we think
    about this" log entry disambiguate.py's module docstring describes.
    Written unconditionally, whether or not `needs_branches` fires, so
    a turn where the message was judged unambiguous is still a
    queryable row (`needs_branches=False`), not a gap in the record.

    `turn_had_direct_answer` is True exactly when FinalAnswer ran
    against this same turn with no branch context. A click-resolution
    turn never creates one of these rows at all: it re-uses the
    branches already generated by an earlier DisambiguationTurn rather
    than running AssessAndBranch again.
    """

    id: UUID = Field(default_factory=uuid4)
    session_id: UUID
    turn_index: int
    needs_branches: bool
    turn_had_direct_answer: bool
    created_at: datetime = Field(default_factory=_utcnow)


class DisambiguationBranch(BaseModel):
    """One distinct plausible reading of an ambiguous message — flat by
    construction: no `parent_id` (no expansion, ever), no depth, no
    evidence gating, no predicted-next-turn forecast (a click or a
    typed message resolves it directly).

    `status` reuses `BranchStatus`, but only ever visits OPEN, MATCHED
    (via an option click) and SUPERSEDED (every sibling once one is
    clicked, or every branch in a generation the student typed past
    instead of clicking). UNMATCHED is never used: there is no
    prediction to fail to match.
    """

    id: UUID = Field(default_factory=uuid4)
    disambiguation_turn_id: UUID
    session_id: UUID
    turn_index: int
    statement: str
    status: BranchStatus = BranchStatus.OPEN
    created_at: datetime = Field(default_factory=_utcnow)


class DisambiguationAssessment(BaseModel):
    """AssessAndBranch's raw output before persistence — not itself
    DB-backed. Empty `branch_statements` is the normal, expected shape
    whenever `needs_branches` is False, not a parse failure."""

    needs_branches: bool
    branch_statements: list[str] = Field(default_factory=list)


class TurnDiagnostics(BaseModel):
    """One row per turn — the persisted form of what the loop already
    computes each turn (call counts, the MAX_CALLS_PER_TURN guardrail,
    wall-clock duration, warnings, whether the response call failed),
    so the web UI's Diagnostics strip can read it directly instead of
    re-deriving it.

    `retry_count` is this turn's total across every LLMClient retry
    (snapshotted before/after the turn — see loop.py's
    `_total_retry_count`); always 0 against StubLLMClient. `entropy_bits`
    is a nullable legacy column, always None now that the entropy-scaled
    reasoning budget is gone.

    The `memory_*` / `branching_skipped_by_memory` / `matched_fact_id`
    fields are the memory layer's own visibility fields (memory.py) —
    `branching_skipped_by_memory` is the literal behavioral consequence
    a reviewer searches for: AssessAndBranch never ran this turn
    because a past fact was confirmed to resolve the message.
    """

    id: UUID = Field(default_factory=uuid4)
    session_id: UUID
    turn_index: int
    node_call_counts: dict[str, int] = Field(default_factory=dict)
    total_call_count: int = 0
    guardrail_fired: bool = False
    entropy_bits: float | None = None
    duration_ms: float
    warnings: list[str] = Field(default_factory=list)
    teach_failed: bool = False
    retry_count: int = 0
    memory_match_found: bool = False
    memory_match_confirmed_resolution: bool = False
    branching_skipped_by_memory: bool = False
    matched_fact_id: UUID | None = None
    # history_block.py's own visibility fields -- whether a learner-
    # history block was actually threaded into FinalAnswer this turn,
    # which interactions/patterns it drew from (mixed interactions.id
    # and population_patterns.id, as strings), and which template
    # rendered it. The rendered TEXT itself is already in node_calls
    # (FinalAnswer's own input_json); these are the structured half.
    history_block_used: bool = False
    history_block_source_ids: list[str] = Field(default_factory=list)
    history_block_template_version: str | None = None
    # reference_bindings.py's own visibility field -- which
    # reference_bindings rows (as strings) were actually injected into
    # FinalAnswer's prompt this turn. AssessAndBranch's own use of the
    # same lookup is already visible for free in node_calls' input_json
    # (the `reference_binding_hint` kwarg) -- this column is
    # specifically about what the learner's actual answer saw.
    reference_bindings_injected: list[str] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=_utcnow)


class SessionSummary(BaseModel):
    """One row for the Setup page's resume view: a learner's prior
    session and how many turns it has. `topic` is a legacy optional
    kept only so existing read/render sites don't have to change — it
    is always None now that sessions have no concept graph attached.
    """

    session_id: UUID
    topic: str | None = None
    turn_count: int
    created_at: datetime


class LearnerSummary(BaseModel):
    """One row for the Setup page's existing-learner picker: a learner
    plus their session count and most recent session's timestamp
    (None if they have no sessions yet)."""

    learner: Learner
    session_count: int
    last_session_at: datetime | None


class LearnerFactType(str, Enum):
    BRANCH_RESOLUTION = "branch_resolution"
    DIRECT_ANSWER = "direct_answer"


class LearnerFact(BaseModel):
    """One row of the memory layer's durable, per-learner record (see
    memory.py's module docstring) — plain English, written every turn
    a real resolution happened (a click resolved an ambiguity, or
    FinalAnswer answered directly), never on a turn that only raised
    options with nothing yet decided.

    `situation`/`resolution` are always in "the student's own terms,"
    not restated jargon. `embedding` is over the combined
    situation+resolution text (see memory.WriteLearnerFact), so a
    search on either half of a past fact can still surface it.

    Append-only (CLAUDE.md invariant 10): a fact is never edited or
    superseded once written.
    """

    id: UUID = Field(default_factory=uuid4)
    learner_id: UUID
    session_id: UUID
    turn_index: int
    fact_type: LearnerFactType
    situation: str
    resolution: str
    embedding: list[float]
    source_turn_id: UUID
    created_at: datetime = Field(default_factory=_utcnow)


class ExtractedFact(BaseModel):
    """WriteLearnerFact's own node output (not DB-backed) — deliberately
    excludes the embedding vector and every id: a node's `output_json`
    in node_calls should read as a human-checkable fact, not carry a
    768-float array nobody will ever read there."""

    situation: str
    resolution: str


class FactSearchResult(BaseModel):
    """EmbedAndSearchFacts' own node output — the nearest learner_facts
    match (if any) and its cosine similarity, without the matched
    fact's embedding for the same node_calls-readability reason as
    ExtractedFact. `matched_fact_id` is None when the learner simply
    has no facts yet (a new learner, or turn 0) — not an error."""

    matched_fact_id: UUID | None = None
    situation: str | None = None
    resolution: str | None = None
    similarity: float | None = None


class FactMatchConfirmation(BaseModel):
    """ConfirmFactMatch's output — a strong vector-similarity hit is
    not itself proof the matched fact resolves THIS message; this is
    the structured judgment call that decides whether the memory
    pre-check is actually allowed to skip branching."""

    resolves: bool


class PathSummary(BaseModel):
    """SummarizeSessionPath's output — a label for the STRUCTURE of one
    session's ordered facts (e.g. "concrete example requested before
    abstract definition, repeatedly"), deliberately abstract enough to
    apply regardless of topic. Not itself a claim about the learner."""

    summary: str


class ThinkingStyleConfirmation(BaseModel):
    """ConfirmThinkingStyleMatch's output — does this session's labeled
    path genuinely share the same order-structure as an existing
    candidate, or only a superficial resemblance? Only a True here
    advances ThinkingStyleCandidate.confirmation_count/session_ids."""

    confirms: bool


class ThinkingStyleStatus(str, Enum):
    CANDIDATE = "candidate"
    CONFIRMED = "confirmed"
    RETIRED = "retired"


class ThinkingStyleCandidate(BaseModel):
    """A hypothesized, cross-session order-structure for one learner —
    e.g. "wants a concrete example before an abstract rule, regardless
    of topic." Starts life as `candidate` (confirmation_count=1) and
    stays there, accumulating independent confirmations, until
    `confirmation_count` crosses
    `MemoryConfig.thinking_style_promotion_threshold` — only then does
    it become `confirmed` and only then does it get fed into any live
    session's prompts.

    `confirmation_count`/`session_ids` only ever grow via an explicit
    `ConfirmThinkingStyleMatch` call saying yes.

    Append-only (CLAUDE.md invariant 10): `retired` is a status
    transition via UPDATE, same resurrection-over-deletion principle as
    every other store in this project.
    """

    id: UUID = Field(default_factory=uuid4)
    learner_id: UUID
    session_ids: list[UUID]
    path_summary: str
    path_summary_embedding: list[float]
    confirmation_count: int = 1
    status: ThinkingStyleStatus = ThinkingStyleStatus.CANDIDATE
    created_at: datetime = Field(default_factory=_utcnow)
    updated_at: datetime = Field(default_factory=_utcnow)


class EvidenceSourceType(str, Enum):
    # A deliberate scripted run to exercise a code path — proves a
    # mechanism functions, never that the system adapted to a real
    # student.
    STAGED_VERIFICATION = "staged_verification"
    # A finding lifted from a real, organic student session.
    ORGANIC_SESSION = "organic_session"


class EvidenceRecord(BaseModel):
    """One recorded verification finding (evidence_records, migration
    033). `source_type` is load-bearing: it keeps a staged mechanism
    test visibly separate from evidence produced by real usage, so the
    Evidence page can never present the former as the latter.

    `body` is the structured verbatim evidence for this finding
    (branches offered, facts written, `turn_diagnostics` fields,
    responses, consolidation output, …). `summary` is the one-line read
    — for `STAGED_VERIFICATION` it must use "mechanism verified"
    language, never "adapted to the student" (enforced by the writer,
    not the schema).

    Append-only (CLAUDE.md invariant 11): a finding is never edited or
    removed once written.
    """

    id: UUID = Field(default_factory=uuid4)
    source_type: EvidenceSourceType
    part: str
    title: str
    summary: str
    body: dict = Field(default_factory=dict)
    learner_id: UUID | None = None
    session_id: UUID | None = None
    created_at: datetime = Field(default_factory=_utcnow)


# ─────────────────────────── interactions ───────────────────────────
#
# The personalization/retrieval pipeline (migration 034,
# interactions.py/topics.py/retrieval.py) — a new, separate store from
# `learner_facts` above, built alongside it. NOT the "story"/learner-
# memory layer: that name is already taken (the web UI's `story` panel
# and prior verification transcripts both use "story" to mean a
# `LearnerFact` row). See migration 034's own header comment.


class QuestionAuthor(str, Enum):
    LEARNER = "learner"
    SYSTEM_OPTION = "system_option"


class EntryState(str, Enum):
    COLD_OPEN = "cold_open"
    CONTINUING = "continuing"
    RETURNING_AFTER_GAP = "returning_after_gap"
    STUCK_REPEAT = "stuck_repeat"
    # No similar match anywhere in the learner's recent window, on a
    # turn that isn't cold_open -- the learner abandoned or completed
    # one thing and moved to another. Deliberately its own state, not
    # folded into CONTINUING: it is one of the more informative signals
    # this table carries, and merging it into the single most common
    # state would erase it.
    TOPIC_SWITCH = "topic_switch"
    # Checked FIRST, before any similarity comparison: set whenever the
    # previous turn in the session had did_branch = True. A
    # click-resolution turn's own question_text is the option's copy,
    # generated FROM the immediately preceding question (see
    # QuestionAuthor.SYSTEM_OPTION) -- comparing it against that same
    # question would always read as maximally "similar," which is true
    # but analytically useless, so resolution pre-empts the comparison
    # entirely rather than let it produce a technically-correct,
    # meaningless CONTINUING.
    RESOLUTION = "resolution"


class InteractionPriorOutcome(str, Enum):
    """The SMALL, best-effort enum written onto `Interaction` itself at
    creation time — never backfilled. Distinct from `TurnOutcomeLabel`
    below (the classifier's own, richer 5-value vocabulary); retrieval
    reads the authoritative value through a view joining `turn_outcomes`,
    not this column.

    `DEFERRED` and `UNKNOWN` are NOT the same thing and must not be
    collapsed into each other: `UNKNOWN` means "we don't know what
    happened to the prior turn" (no classification exists yet, but one
    could in principle exist), while `DEFERRED` means "there is nothing
    to know yet, by construction" (the prior turn was an options-offered
    row with no response, or the last turn of a session — see
    `TurnOutcomeLabel.DEFERRED`). Merging them erases exactly the
    distinction `entry_state`'s STUCK_REPEAT-style rules (and any future
    consumer of this column) need to tell "genuinely unclassified" apart
    from "structurally has no classification to give."
    """

    RESOLVED = "resolved"
    CONTRADICTED = "contradicted"
    # Mirrors TurnOutcomeLabel.MOVED_ON's own collapse (see that enum's
    # docstring): replaces the old ABANDONED, which nothing produces
    # any more now that TurnOutcomeLabel no longer distinguishes
    # "abandoned" from "understood, moved on." Mapping MOVED_ON onto
    # either RESOLVED or ABANDONED here would just reintroduce the
    # same unwarranted-precision problem one enum downstream.
    MOVED_ON = "moved_on"
    DEFERRED = "deferred"
    UNKNOWN = "unknown"


class HelpLevel(str, Enum):
    NONE = "none"
    HINT = "hint"
    WORKED_EXAMPLE = "worked_example"
    DIRECT_ANSWER = "direct_answer"


class Interaction(BaseModel):
    """One row per `handle_turn` call — same granularity `turns`/
    `disambiguation_turns` already use, not a different unit. A
    branching exchange is TWO rows: Turn A (the ambiguous message —
    `did_branch=True`, `response_text=None`, options attach here via
    `InteractionOption`) and Turn A+1 (the click — `did_branch=False`,
    `question_author=SYSTEM_OPTION` since `question_text` is the
    clicked option's own button copy, `response_text` set once
    FinalAnswer's output lands).

    `originating_question` is set only when `question_author =
    SYSTEM_OPTION`: Turn A's own `question_text`. `question_embedding`
    is computed from `originating_question` in that case, never from
    `question_text` — embedding the option generator's own phrasing
    would make retrieval match on its stylistic tics rather than on
    anything about the learner (see migration 034's header comment).

    `response_text` is None both on an options-offered turn (nothing
    generated yet) and on a turn where FinalAnswer failed (nothing
    genuine was produced — treated as no-response, not fabricated
    content). Either way this is also what makes outcome
    classification resolve to DEFERRED for the row: no response, no
    N+1 evidence to classify yet.

    `abstract_form`/`abstract_embedding` are always None here by
    construction — see `InteractionAbstract`, appended separately so
    this row is never mutated after insert.

    `entry_state` is computed from `prev_question_sim`/
    `recent_similar_count`/`last_similar_turn_gap` — no topic cluster
    identity exists anywhere (see this class's own docstring history:
    a per-learner running-centroid topic mechanism was built, tuned
    across two rounds, replayed systematically against real data, and
    removed after confirming the mechanism itself — not just its
    threshold — cascades under a single fixed similarity threshold).
    The three fields are stored as raw numbers, not just the derived
    label, specifically so a wrong `same_subject_threshold`
    (retrieval_config.py) is survivable: labels recompute from these
    stored values, with no re-embedding and no touching an
    already-written row.

    Append-only, enforced at the database (a real trigger blocking
    UPDATE/DELETE), not just by convention — see migration 034.
    """

    id: UUID = Field(default_factory=uuid4)
    learner_id: UUID
    session_id: UUID
    turn_number: int
    question_text: str
    question_author: QuestionAuthor
    originating_question: str | None = None
    did_branch: bool
    response_text: str | None = None
    entry_state: EntryState
    # Count of this learner's recent questions (see InteractionConfig's
    # window size) that clear same_subject_threshold against this one.
    # Replaces the old turns_on_topic -- same meaning, honestly
    # re-derived from a pairwise comparison instead of a cluster
    # membership count.
    recent_similar_count: int = 0
    # Raw cosine similarity to this learner's immediately preceding
    # interaction, regardless of session boundary. None only when
    # there is no prior interaction at all for this learner.
    prev_question_sim: float | None = None
    # How many interactions back (most-recent-first) the closest
    # above-threshold match was found, searched over a wider window
    # than recent_similar_count's own. None when nothing in that wider
    # window matched -- the topic_switch case.
    last_similar_turn_gap: int | None = None
    prior_turn_outcome: InteractionPriorOutcome = InteractionPriorOutcome.UNKNOWN
    help_level: HelpLevel = HelpLevel.NONE
    elapsed_ms: int | None = None
    # Renamed from topic_embedding when topic clustering was removed --
    # the embedding itself is unchanged and still exactly what
    # recent_similar_count/prev_question_sim/last_similar_turn_gap (and
    # retrieval's own stage2 ANN search) compare against; only the name
    # tied it to a concept that no longer exists.
    question_embedding: list[float]
    abstract_form: str | None = None
    abstract_embedding: list[float] | None = None
    # The domain switch's one storage touch (domain_config.py, migration
    # 040) -- set once per interaction from whichever DomainConfig the
    # SessionLoop that wrote it was constructed with, never mixed within
    # one row. Default EDUCATION: every interaction ever written before
    # this field existed was, in fact, education-domain.
    domain: Domain = Domain.EDUCATION
    created_at: datetime = Field(default_factory=_utcnow)


class InteractionOption(BaseModel):
    """One option actually shown on an options-offered turn (Turn A).
    `shown_position` is the position AFTER shuffling — fixed ordering
    would make selection partly a function of the UI rather than the
    learner. `was_selected`/`selection_timestamp` are populated on a
    LATER turn than creation (the click, a separate `handle_turn`
    call) — a decision record, not an interaction record, so this
    table is deliberately excluded from the immutability trigger.

    `kind`/`axis` (migration 041) are DisambiguationOptions' own
    kind/axis decision (see that class's docstring), denormalized onto
    every option in the set it produced — the same value repeats
    across every row from one option set, the way `shown_position`'s
    sibling fields already do per-row. Persisted here, not just in
    `node_calls.output_json`, specifically so claim extraction can read
    "what axis was this option evidence for" as a normal query against
    this learner-scoped table, never by reverse-engineering it from
    option text or cross-referencing node_calls by session/turn.
    `axis` is always `None` when `kind` is SUBJECT or `None`.

    `side` (migration 045) is which pole of `axis` THIS option (not
    the whole set) represents — "first" or "second", matching the
    axis name's own word order (see `OptionProposal`'s own docstring
    for the incident this closes: text-based side inference drifting
    silently). Per-option, unlike `kind`/`axis` which repeat across
    the whole set — the two options in one approach-kind set always
    carry opposite sides. `None` for a SUBJECT-kind option."""

    id: UUID = Field(default_factory=uuid4)
    interaction_id: UUID
    learner_id: UUID
    option_id: UUID
    branch_id: UUID
    option_text: str
    shown_position: int
    was_selected: bool = False
    selection_timestamp: datetime | None = None
    kind: AmbiguityKind | None = None
    axis: ApproachAxis | None = None
    side: str | None = None
    created_at: datetime = Field(default_factory=_utcnow)


class InteractionAbstract(BaseModel):
    """The resolved turn, rewritten with domain nouns stripped (e.g.
    "chose the worked example over the stated rule") — generated off
    the critical path, alongside the outcome classifier. Append-only:
    re-running the generator writes a new row under a new
    `generator_version`; readers resolve to the latest version per
    `interaction_id`, same convention as `TurnOutcome`."""

    id: UUID = Field(default_factory=uuid4)
    interaction_id: UUID
    learner_id: UUID
    abstract_form: str
    abstract_embedding: list[float]
    generator_version: str
    created_at: datetime = Field(default_factory=_utcnow)


class TurnOutcomeLabel(str, Enum):
    MATCHED = "matched"
    CONTRADICTED_INTENT = "contradicted_intent"
    # Collapsed from two separate values (UNDERSTOOD_MOVED_ON,
    # ABANDONED_TOPIC) after a real 12-turn evaluation session: the
    # distinction between "left satisfied" and "left frustrated" is not
    # recoverable from a classifier reading only the prior question,
    # response, and next question -- there is no signal in that window
    # that distinguishes them, and a real run showed exactly this
    # ambiguity on 3 of 7 classifications. Guessing it per-turn and
    # storing the guess as a typed enum value is worse than not having
    # the distinction: it would make every downstream reader (rerank,
    # a future trained policy) treat a coin flip as fact. If this
    # distinction is ever needed, it needs its own explicit signal (a
    # later return to the topic, an affect read) -- not a finer label
    # squeezed out of the same three fields that already can't support
    # the ones kept here.
    MOVED_ON = "moved_on"
    # Emitted (never omitted) whenever no N+1 evidence exists yet:
    # response_text IS NULL on the interaction being classified, OR it
    # is the last interaction in its session. Both are the same
    # underlying condition — stated once, not as two special cases.
    DEFERRED = "deferred"


class TurnOutcome(BaseModel):
    """One classification of one interaction row ("turn N"), made at
    turn N+1 from N's own question/selected option/response and N+1's
    new question. Off the critical path. Append-only: re-running the
    classifier writes a new row under a new `classifier_version`;
    readers resolve to the latest version per `interaction_id` — never
    an update in place."""

    id: UUID = Field(default_factory=uuid4)
    interaction_id: UUID
    learner_id: UUID
    next_question_text: str | None = None
    outcome: TurnOutcomeLabel
    confidence: float
    classifier_version: str
    created_at: datetime = Field(default_factory=_utcnow)


class StatedPreferenceLabel(str, Enum):
    """A closed vocabulary the classifier picks ONE of, alongside the
    raw extracted text -- see StatedPreference's own docstring for why
    both exist side by side. `OTHER` covers any explicit standing
    preference that doesn't fit one of the specific shapes below; it is
    not a failure mode, it's a real, if less mappable, category.

    interaction_nodes.render_structural_requirement maps each label to
    a hand-written imperative -- the label is what makes that mapping
    possible without needing the classifier to also invent good
    imperative phrasing on the fly, which would reopen exactly the
    "description, not a requirement" problem raw-text-only rendering
    had (see disambiguate.FinalAnswer's own docstring for that record)."""

    CONCRETE_BEFORE_ABSTRACT = "concrete_before_abstract"
    RULE_BEFORE_EXAMPLE = "rule_before_example"
    WANTS_STEPS_SHOWN = "wants_steps_shown"
    PREFERS_BREVITY = "prefers_brevity"
    WANTS_ANALOGIES = "wants_analogies"
    NO_ANALOGIES = "no_analogies"
    OTHER = "other"


class StatedPreference(BaseModel):
    """Fix B's write side (history_block.py's module docstring / the
    feature's own review has the full record): did this learner
    EXPLICITLY state a standing preference about how they want to be
    taught, in this turn's own words? Only an explicit statement counts
    -- a repeated click pattern is an inference and belongs in a claim
    layer, never conflated with a fact the learner actually said.

    Classified off the critical path, one fast-tier call per LEARNER-
    authored turn (never a system_option/click turn -- there is no
    free text from the student there to classify). Written every time,
    `has_preference=False` included, so classifier coverage is
    auditable the same way `TurnOutcome`'s is. `stated_preference` is
    the preference in the student's OWN terms (classifier-extracted,
    not paraphrased into a template) -- the faithful record of what was
    actually said, kept verbatim regardless of `label`. `label` is the
    classifier's own closed-vocabulary categorization of that same
    statement, always set when `has_preference` is true (falls back to
    `OTHER` rather than being left null if the model's own label choice
    doesn't parse) -- it exists so the render step
    (interaction_nodes.render_structural_requirement) can map to a
    hand-written imperative instead of converting the raw text into a
    requirement on the fly, which measurably failed to reliably beat
    the model's own prior (see disambiguate.FinalAnswer's docstring).

    A separate table from `interactions` despite conceptually being
    "about" one interaction: `interactions` is immutable, and this
    classifier -- like the abstractor and outcome classifier -- runs
    after that row already exists."""

    id: UUID = Field(default_factory=uuid4)
    interaction_id: UUID
    learner_id: UUID
    has_preference: bool
    stated_preference: str | None = None
    label: StatedPreferenceLabel | None = None
    classifier_version: str
    created_at: datetime = Field(default_factory=_utcnow)


class ReferenceBinding(BaseModel):
    """The reference-resolution memory's write side (migration 038):
    what a learner-specific recurring phrase (`reference_text` — "that",
    "the usual", "my project") has meant, so a LATER turn — this
    session or a future one — doesn't need to re-ask.

    Classified off the critical path, by `interaction_nodes.
    ClassifyReferenceResolution`, only on a turn that actually SETTLED
    a genuinely ambiguous reference (a branch selection, a typed
    clarification, or the answer itself establishing the meaning) —
    never on an ordinary pronoun resolvable from the immediately
    preceding turn. Unlike `StatedPreference`/`TurnOutcome`, a
    non-resolution is never written at all: this feature's own review
    is explicit that a noisy table is worse than a sparse one here,
    since every spurious row is a chance to substitute the wrong
    meaning into a future answer.

    Append-only (same family as CLAUDE.md invariants 1, 4, 6-11): a
    re-confirmation of the SAME `resolved_to` for a
    (learner_id, reference_text) pair, or a change to a DIFFERENT one,
    both write a NEW row via `ReferenceBindingStore.record_resolution`
    rather than updating the existing one — readers resolve to the
    latest row per pair. `confirmation_count` is computed there (prior
    row's count + 1 on a genuine re-confirmation, reset to 1 on a
    changed meaning) and never incremented via UPDATE.

    `confirmation_count`/`last_confirmed_at` feed
    `reference_bindings.py`'s confidence decay on the read side: a
    binding heard once, long ago, must not be injected with the same
    weight as one reconfirmed recently — a stale binding confidently
    applied produces a fluent answer about the wrong thing, the
    feature's own named worst failure mode."""

    id: UUID = Field(default_factory=uuid4)
    learner_id: UUID
    reference_text: str
    resolved_to: str
    evidence_interaction_id: UUID
    confirmation_count: int = 1
    last_confirmed_at: datetime = Field(default_factory=_utcnow)
    classifier_version: str
    created_at: datetime = Field(default_factory=_utcnow)


class ReferenceResolutionClassification(BaseModel):
    """ClassifyReferenceResolution's output — not itself DB-backed.
    `resolved=False` (with every other field left at its default) is
    the expected, normal shape for most turns; only a confident,
    genuinely-settled recurring reference produces `resolved=True`
    with both `reference_text`/`resolved_to` set. See
    `ReferenceBinding`'s own docstring for why an unresolved turn
    writes nothing downstream rather than a `resolved=False` row."""

    resolved: bool
    reference_text: str | None = None
    resolved_to: str | None = None
    confidence: float = 0.0


class ClaimSource(str, Enum):
    STATED = "stated"
    INFERRED = "inferred"
    CORRECTED = "corrected"


class ClaimWritePolicy(str, Enum):
    """How fast this claim's confidence should decay with the age of
    its supporting evidence (claims.compute_confidence) — see that
    module's own docstring for the actual decay rates. LOCKED never
    decays (an explicitly `stated` preference, promoted to a claim
    verbatim, has no reason to fade just because time passed);
    SLOW_DRIFT is the default for an inferred standing trait;
    FAST_DECAY is for a claim whose own `test` is about something
    plausibly short-lived rather than a stable trait."""

    LOCKED = "locked"
    SLOW_DRIFT = "slow_drift"
    FAST_DECAY = "fast_decay"


class ClaimStatus(str, Enum):
    CANDIDATE = "candidate"
    PROMOTED = "promoted"
    CONTRADICTED = "contradicted"
    # Terminal, like CONTRADICTED, but for a different reason: this
    # claim's FOUNDING evidence row was itself ineligible (a subject-
    # kind pick with no genuine two-sided contest) -- the claim was
    # never legitimately established in the first place, as opposed to
    # having been established and later disproven. A live run found
    # such a claim (sitting honestly at the 0.5 prior) absorb a real
    # eligible contradicting row and close terminally -- evidence that
    # belonged to a real claim instead killed a premise that should
    # never have existed. RETRACTED claims are excluded from matching
    # (search_similar/find_by_axis) exactly like CONTRADICTED ones, so
    # they stop competing for evidence that belongs elsewhere.
    RETRACTED = "retracted"
    # Terminal, like the other two, but for a third reason: this claim
    # turned out to be a duplicate of another LIVE claim -- same
    # learner, same axis, same value -- not disproven, not illegitimate,
    # just redundant. `claims.merge_duplicate_claims` sets this on every
    # loser after copying its evidence onto the survivor (see that
    # function's own docstring for why the evidence is COPIED, not
    # repointed: `ClaimEvidence` has "no mutable field at all, not even
    # a status" -- mutating an existing row's `claim_id` would break
    # that). `superseded_by` names the survivor. Excluded from matching
    # (search_similar/find_by_axis) exactly like CONTRADICTED/RETRACTED,
    # so the loser stops competing for evidence that now belongs, in
    # full duplicate, to the survivor too.
    SUPERSEDED = "superseded"


class Claim(BaseModel):
    """The claim layer's write side (migration 042) — see claims.py's
    own module docstring for the full design record (extraction
    anchored on prediction error, the promotion gate, the confidence
    clamp). One row per distinct inferred-or-stated standing trait
    about a learner, created once by `claims.reconcile_candidate` and
    never edited after that except for `confidence`/`status`/
    `updated_at` (see this table's own migration comment for why that
    is still "append-only" in this codebase's established sense —
    the same status-transition-via-UPDATE convention
    ThinkingStyleStore/DisambiguationStore already use).

    `value` is one of `StatedPreferenceLabel`'s closed vocabulary —
    deliberately the SAME one `stated_preferences` uses, not a new one,
    specifically so `claims.render_claim_constraint` can reuse
    `interaction_nodes.render_structural_requirement` verbatim: "the
    same form that worked 5/5 for stated preferences," per this
    feature's own spec, rather than a second hand-written imperative
    set to keep in sync with the first.

    `test` is the falsifiable prediction this claim licenses, phrased
    as a situation plus a predicted choice over a FUTURE option set —
    extraction rejects any candidate without one (see
    `claims.ClaimExtractor`).
    """

    id: UUID = Field(default_factory=uuid4)
    learner_id: UUID
    statement: str
    test: str
    value: StatedPreferenceLabel
    confidence: float
    source: ClaimSource
    write_policy: ClaimWritePolicy
    context_scope: dict = Field(default_factory=dict)
    status: ClaimStatus = ClaimStatus.CANDIDATE
    statement_embedding: list[float]
    created_at: datetime = Field(default_factory=_utcnow)
    updated_at: datetime = Field(default_factory=_utcnow)
    # Set only when status is SUPERSEDED -- the survivor claim this one
    # was merged into. None otherwise.
    superseded_by: UUID | None = None


class EvidenceDirection(str, Enum):
    SUPPORTS = "supports"
    CONTRADICTS = "contradicts"


class EvidenceSource(str, Enum):
    """How this row's direction was decided. `CLICK` is every existing
    path (a disambiguation pick, a stated preference) — the default,
    so every row written before this field existed reads correctly
    with no backfill. `INSTRUMENT` is a purpose-built interaction
    (instruments.py) whose contract deterministically interpreted the
    event stream. Kept separate from `ClaimSource` (which describes
    how the CLAIM was created — stated vs inferred) — this is a
    per-evidence-row fact, and a single claim can accumulate both
    kinds of evidence over time. Needed before instrument-derived
    evidence can be trusted alongside click evidence: score_predictions.py
    checks the two populations separately."""

    CLICK = "click"
    INSTRUMENT = "instrument"


class ClaimEvidence(BaseModel):
    """One episode's bearing on one claim — points at `interactions` by
    reference only, never copies question/response content (migration
    042's own header). Fully append-only with no mutable field at all,
    not even a status: a claim's own status can change; the evidence
    that justified the change never does.

    `test_fired`/`contradiction_was_possible` are both set explicitly
    at write time, never inferred later — see `claims.py`'s own
    docstring for why a turn where the claim's test couldn't have
    failed must never count the same as one where it genuinely could
    have and didn't. `compute_confidence` only counts a row toward s/f
    when BOTH are true; a row failing that is still stored (kept as
    provenance — the review surface shows everything that was
    considered) but contributes nothing to the number.

    `axis` (the `ApproachAxis`, if any, that was live when this episode
    happened) is set alongside `topic` specifically so
    `compute_confidence`'s own cell-collapsing can group by (session,
    axis) rather than (session, topic) — two confirmations of the same
    axis in the same session are one observation regardless of what
    topic label each carried. `None` for evidence with no live axis
    (a stated-preference or contradicted_intent trigger with no option
    set behind it), in which case cell-collapsing falls back to topic.
    """

    id: UUID = Field(default_factory=uuid4)
    claim_id: UUID
    learner_id: UUID
    interaction_id: UUID
    direction: EvidenceDirection
    topic: str
    axis: ApproachAxis | None = None
    session_id: UUID
    test_fired: bool
    contradiction_was_possible: bool
    created_at: datetime = Field(default_factory=_utcnow)
    # Nullable, free-text, set by human diagnosis after the fact (never
    # automatically -- "if you could detect contamination automatically
    # you'd have prevented it") when this row is known to have come
    # from a broken test harness rather than genuine learner behavior.
    # Does NOT exclude the row from production confidence/status -- it
    # still counts, because it records what actually happened. Lets
    # `score_predictions.py`'s --exclude-contaminated flag and the
    # review surface separate the two pictures. Free text, not an enum,
    # since future harness bugs can't be enumerated in advance.
    provenance_note: str | None = None
    source: EvidenceSource = EvidenceSource.CLICK


class ClaimStatementRecord(BaseModel):
    """One rendering of a claim's (axis, value) against its evidence at
    some point in time — append-only (migration 048), same pattern as
    `turn_outcomes`. `Claim.statement` is set once at creation and
    never edited (see that model's own docstring); THIS table is what
    accumulates as evidence grows and generalizes past the founding
    episode's domain (see `claims.maybe_restate_claims`) — axis and
    value are the claim's immutable identity, evidence attaches by
    them, but the prose describing them is derivable from accumulated
    evidence and can be regenerated without touching what the claim IS.

    The first row for any claim is written at claim-creation time with
    the same text as `Claim.statement` — readers resolve to the LATEST
    row per `claim_id`, never to `Claim.statement` directly, so a claim
    with no restatement yet and one with several both resolve
    correctly. Keeping every prior row (rather than overwriting) means
    the wording's drift as evidence accumulated stays visible on
    request, the same audit-trail principle as every other append-only
    store in this codebase."""

    id: UUID = Field(default_factory=uuid4)
    claim_id: UUID
    statement: str
    derived_from_evidence_count: int
    generator_version: str
    created_at: datetime = Field(default_factory=_utcnow)


class ClaimCandidate(BaseModel):
    """One `ClaimExtractor`-proposed candidate for a single episode —
    not itself DB-backed; `claims.reconcile_candidate` turns it into
    either a fresh `Claim` row or a new `ClaimEvidence` row against an
    existing one. Multiple candidates from the SAME episode are kept
    side by side, deliberately, when the extractor judges the episode
    underdetermined — see `ClaimExtractor`'s own docstring for why that
    is a finding about the evidence, not something to resolve by
    picking the most plausible one."""

    statement: str
    test: str
    value: StatedPreferenceLabel
    topic: str


class ClaimExtractionResult(BaseModel):
    """`ClaimExtractor`'s raw output for one episode — zero, one, or
    several competing `ClaimCandidate`s. Empty is the expected, normal
    result for most extraction-eligible turns: being ranked into the
    extraction set (by prediction error, `contradicted_intent`, or a
    stated preference) does not by itself guarantee the episode
    actually licenses a testable claim."""

    candidates: list[ClaimCandidate] = Field(default_factory=list)


class Prediction(BaseModel):
    """One asynchronous selection-prediction run over the options shown
    on an options-offered turn. Fired after options are persisted and
    returned to the UI — NEVER awaited before that response goes out.
    May land after the learner has already clicked; persisted
    regardless, since `prediction_created_at` is what makes a
    late-arriving prediction interpretable and cannot be recovered if
    the row is skipped.

    `predicted_scores` is keyed by `option_id` (str) -> probability.
    `retrieved_candidate_ids`/`retrieval_provenance` are what
    `retrieval.retrieve()` surfaced as context for this prediction —
    fixed shape regardless of what actually computes
    `predicted_scores`, so a learned selection policy can later replace
    the LLM predictor without changing this schema.
    """

    id: UUID = Field(default_factory=uuid4)
    interaction_id: UUID
    learner_id: UUID
    predicted_scores: dict
    prediction_created_at: datetime = Field(default_factory=_utcnow)
    model_version: str
    retrieved_candidate_ids: list = Field(default_factory=list)
    retrieval_provenance: list = Field(default_factory=list)


class PopulationPattern(BaseModel):
    """One cross-learner pattern, derived from abstract forms only —
    never raw transcripts. Readable by retrieval only at
    `distinct_learner_count >= 20` AND `max_per_learner_share <=
    0.25`: the second gate matters as much as the first, since without
    it one heavy user can supply most of a pattern's support and the
    learner-count threshold passes on what is effectively one person's
    behavior. Append-only: each aggregation run inserts fresh rows; a
    stale pattern is superseded by a newer row from the next run, never
    edited in place."""

    id: UUID = Field(default_factory=uuid4)
    abstract_form: str
    embedding: list[float]
    support_count: int
    distinct_learner_count: int
    max_per_learner_share: float
    representative_features: dict = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=_utcnow)


# ── node outputs (not DB-backed) ──────────────────────────────────────


class AbstractionResult(BaseModel):
    """GenerateAbstractForm's output — the domain-noun-stripped
    rewrite. Not itself persisted; InteractionAbstractStore.append
    attaches the embedding and ids around it."""

    abstract_form: str


class OutcomeClassification(BaseModel):
    """ClassifyTurnOutcome's output. `abstains` is True on a
    low-confidence call: the caller writes nothing rather than guessing
    a label — an abstention leaves the interaction's outcome at
    whatever the previous classifier_version (if any) already
    established, not DEFERRED and not a forced guess."""

    outcome: TurnOutcomeLabel
    confidence: float
    abstains: bool = False


class StatedPreferenceClassification(BaseModel):
    """ClassifyStatedPreference's output — see StatedPreference's own
    docstring for what counts (explicit statement only, never an
    inferred pattern) and for why `label` exists alongside the raw
    `stated_preference` text. `label` is always set when
    `has_preference` is true (the node itself falls back to `OTHER`
    rather than leaving it None on a malformed label choice)."""

    has_preference: bool
    stated_preference: str | None = None
    label: StatedPreferenceLabel | None = None


class RetrievalCandidate(BaseModel):
    """One candidate surfaced by `retrieval.retrieve()` — the unified
    shape both personal and population scopes return, so the 4+1
    quota assembly never has to special-case either. Every candidate
    keeps enough provenance to justify why it was surfaced, per scope:

    - PERSONAL: `source_id` is an `interactions.id`, `learner_id` is
      always this learner's own id, `outcome`/`support_count` are
      usually None (an individual interaction, not an aggregate).
    - POPULATION: `source_id` is a `population_patterns.id`,
      `learner_id` is None (a population pattern is not attributed to
      any one learner), `support_count`/`distinct_learner_count` are
      always set.
    """

    scope: str  # "personal" | "population"
    source_id: UUID
    retrieval_key: str  # "topic" | "abstract"
    learner_id: UUID | None = None
    similarity: float
    recency_days: float | None = None
    outcome: TurnOutcomeLabel | None = None
    support_count: int | None = None
    distinct_learner_count: int | None = None
    text: str = ""
    score: float = 0.0


# ─────────────────────────── instrument layer (migrations 050-052) ────
#
# See instruments.py's own module docstring for the full design record
# (the calibration gate this foundation exists to be checked against
# before a second instrument gets built). These are the DB row shapes
# only — stores, the pure interpretation function, and the one
# hand-authored contract live in instruments.py, the same split
# claims.py/models.py already use.


class InstrumentPrimitive(str, Enum):
    """The closed set of instrument shapes. Only LOCATE is implemented
    in this build — the other five are defined now so the schema
    (InteractionContract/Instrument/instrument_events' event_type) is
    proven against all six from the start rather than reshaped per
    instrument later, which is how event schemas rot."""

    CHOOSE = "choose"
    ORDER = "order"
    LOCATE = "locate"
    ADJUST = "adjust"
    PREDICT = "predict"
    CONSTRUCT = "construct"


class InstrumentEventType(str, Enum):
    """Shared across every primitive. `start`/`submit`/`abandon` are
    what let ANY primitive distinguish completion from abandonment and
    carry timing (`elapsed_ms`, on every row); `move`/`revise` are what
    let any primitive with a revision trail record one — not every
    primitive uses every event type, but the enum itself is fixed
    across all six so a future primitive never needs a schema change
    to fit."""

    START = "start"
    MOVE = "move"
    REVISE = "revise"
    CLICK = "click"
    VALUE_CHANGE = "value_change"
    SUBMIT = "submit"
    ABANDON = "abandon"
    HINT_REQUEST = "hint_request"


class MeasurementKind(str, Enum):
    """Whether a method's evidence answers "which framing they want"
    (PREFERENCE) or "whether they can do the thing" (PERFORMANCE) —
    see method_capabilities.py's own module docstring for the category
    error conflating the two produces. A PERFORMANCE contract must
    target a `CapabilityClaim` (by `skill`), never a preference `Claim`
    (by `axis`+`value`) — `InteractionContract`'s own validator below
    enforces this shape."""

    PREFERENCE = "preference"
    PERFORMANCE = "performance"


class CapabilityLabel(str, Enum):
    """Closed vocabulary for capability/mastery claims — what a
    learner CAN DO, observed directly, never a preference-vocabulary
    label standing in for it. This is the fix for a real incident:
    `locate`/`predict` originally targeted `StatedPreferenceLabel`
    values (WANTS_STEPS_SHOWN, RULE_BEFORE_EXAMPLE) despite measuring
    performance, which meant capability observations were silently
    accumulating into a preference claim's confidence — "can trace a
    swap correctly" is not a preference, and the two can diverge
    (someone can prefer worked examples and still be perfectly able to
    derive forward without them)."""

    TRACES_WORKED_STEPS = "traces_worked_steps"  # locate: catches a seeded error by following each step
    DERIVES_FORWARD = "derives_forward"  # predict: predicts a mechanism's outcome before it's revealed
    OTHER = "other"


class CapabilityStatus(str, Enum):
    CANDIDATE = "candidate"
    PROMOTED = "promoted"


class CapabilityClaim(BaseModel):
    """The capability-side counterpart to `Claim` — same append-only
    discipline (statement/test/skill fixed at creation; confidence/
    status/updated_at the only mutable fields, via `capability.
    CapabilityClaimStore.refresh`) but a SEPARATE table (migration 054)
    and a separate closed vocabulary (`CapabilityLabel`, never
    `StatedPreferenceLabel`).

    No axis: capability isn't bidirectional — there's no "opposite"
    pole to a skill the way each `ApproachAxis` has two named poles,
    so nothing here plays the role axis/`find_by_axis` play for
    preference claims. Matching is exact-skill
    (`CapabilityClaimStore.find_by_skill`), not axis-sharing-then-
    similarity: at most one live claim per (learner, skill) should
    exist. No `statement_embedding` either — with no similarity-based
    fallback matching path (skill is always known, never None the way
    axis can be), there is nothing for an embedding to disambiguate."""

    id: UUID = Field(default_factory=uuid4)
    learner_id: UUID
    statement: str
    test: str
    skill: CapabilityLabel
    confidence: float
    write_policy: ClaimWritePolicy
    status: CapabilityStatus = CapabilityStatus.CANDIDATE
    created_at: datetime = Field(default_factory=_utcnow)
    updated_at: datetime = Field(default_factory=_utcnow)


class CapabilityEvidence(BaseModel):
    """The capability-side counterpart to `ClaimEvidence` — same
    fields, same eligibility gate (`test_fired AND
    contradiction_was_possible`), `skill` where `ClaimEvidence` has
    `axis`+`topic` (capability evidence is always exactly one skill,
    no topic-fallback branch needed since there's no axis-less case).
    Fully append-only, no mutable field except `provenance_note` (same
    one sanctioned exception `ClaimEvidence` has, for the identical
    reason)."""

    id: UUID = Field(default_factory=uuid4)
    claim_id: UUID
    learner_id: UUID
    interaction_id: UUID
    direction: EvidenceDirection
    skill: CapabilityLabel
    session_id: UUID
    test_fired: bool
    contradiction_was_possible: bool
    created_at: datetime = Field(default_factory=_utcnow)
    provenance_note: str | None = None
    source: EvidenceSource = EvidenceSource.INSTRUMENT


class InteractionContract(BaseModel):
    """One deterministic recipe for turning an instrument's event
    stream into supports/contradicts/uninformative — see
    `instruments.interpret_instrument`'s own docstring for how the
    three predicate lists are read. Append-only: no field here is ever
    edited after creation (migration 050's own header).

    `measures` decides which of two shapes this contract has: a
    PREFERENCE contract sets `target_axis`+`target_value` (and, when
    `target_claim_id` is set, that id names a `claims` row); a
    PERFORMANCE contract sets `target_skill` instead (`target_claim_id`,
    when set, names a `capability_claims` row instead). Never both —
    the validator below enforces this. `target_claim_id` carries no
    foreign key at the database level for exactly this reason: it is
    POLYMORPHIC across two tables, and which one it points into is the
    caller's responsibility to get right from `measures` — see
    `MeasurementKind`'s own docstring for the incident this shape
    change closes.

    `target_claim_id` is nullable — a contract may target an existing
    claim (evidence attaches directly, no matching needed: the path
    this build implements) or, in a future not built here, a candidate
    trait with no claim row yet ("no instrument generation yet").

    `uninformative_when` must be non-empty — enforced here (fail fast,
    before ever reaching the database's own CHECK constraint) AND at
    the database (the authoritative backstop): every contract must
    name at least one outcome it cannot read, or post-hoc
    interpretation fills the gap, which is where confabulation lives.
    """

    id: UUID = Field(default_factory=uuid4)
    target_claim_id: UUID | None = None
    measures: MeasurementKind
    target_axis: ApproachAxis | None = None
    target_value: StatedPreferenceLabel | None = None
    target_skill: CapabilityLabel | None = None
    primitive: InstrumentPrimitive
    supports_when: list[dict]
    contradicts_when: list[dict]
    uninformative_when: list[dict]
    generator_version: str
    created_at: datetime = Field(default_factory=_utcnow)

    @field_validator("uninformative_when")
    @classmethod
    def _uninformative_when_must_be_nonempty(cls, v: list[dict]) -> list[dict]:
        if not v:
            raise ValueError(
                "uninformative_when must name at least one outcome the contract "
                "cannot read (abandonment, timeout, an unanticipated response) — "
                "an empty list lets post-hoc interpretation fill the gap"
            )
        return v

    @model_validator(mode="after")
    def _target_shape_matches_measures(self) -> "InteractionContract":
        if self.measures is MeasurementKind.PREFERENCE:
            if self.target_axis is None or self.target_value is None:
                raise ValueError("a PREFERENCE contract must set target_axis and target_value")
            if self.target_skill is not None:
                raise ValueError("a PREFERENCE contract must not set target_skill")
        else:
            if self.target_skill is None:
                raise ValueError("a PERFORMANCE contract must set target_skill")
            if self.target_axis is not None or self.target_value is not None:
                raise ValueError("a PERFORMANCE contract must not set target_axis/target_value")
        return self


class Instrument(BaseModel):
    """One PRESENTATION of a contract to a learner. Append-only in this
    codebase's established sense: `completed_at`/`abandoned` resolve
    via UPDATE once (same status-transition-via-UPDATE convention as
    claims.status), every other field is fixed at creation.

    `interaction_id`/`session_id` correlate this presentation with the
    rest of the learner's history, but `interaction_id` is NOT foreign-
    keyed to `interactions` (migration 051's own header) — an
    instrument is deliberately outside the turn-based chat flow this
    build leaves unchanged."""

    id: UUID = Field(default_factory=uuid4)
    interaction_id: UUID
    learner_id: UUID
    session_id: UUID
    contract_id: UUID
    primitive: InstrumentPrimitive
    spec: dict
    presented_at: datetime = Field(default_factory=_utcnow)
    completed_at: datetime | None = None
    abandoned: bool = False


class InstrumentEvent(BaseModel):
    """One raw event from an instrument's presentation — fully append-
    only, no mutable field at all, not even a status (migration 052's
    own header): nothing ever writes an interpretation back onto a row
    here. `interpret_instrument` reads this table; nothing writes to it
    except the original capture.

    `seq` is caller-supplied (app.js assigns it as events fire), not a
    DB identity column — interpretation needs a deterministic,
    client-known ordering, and the database rejects two events
    claiming the same position rather than silently reordering them.
    `elapsed_ms` is measured from the instrument's `presented_at`."""

    id: UUID = Field(default_factory=uuid4)
    instrument_id: UUID
    seq: int
    event_type: InstrumentEventType
    payload: dict
    elapsed_ms: int
    created_at: datetime = Field(default_factory=_utcnow)
