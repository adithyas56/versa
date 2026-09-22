"""The session loop, after the full Diagnose/Infer/Update/Replan/Plan
reasoning path and the tree-based branch system were removed. Two
architectures remain, chosen once per session by
`ablation_config.mode` (ablation.py):

- `SessionMode.MINIMAL_BRANCH` (the default) -> `_handle_disambiguation_turn`:
  the minimal three-call flow (disambiguate.py) — AssessAndBranch ->
  [DisambiguationOptions] -> FinalAnswer, at most three LLM calls per
  exchange — wrapped by the memory layer (memory.py): a semantic
  pre-check that can skip branching entirely when a past fact for this
  learner already resolves the message, and a per-turn fact write.
- `SessionMode.BASELINE` (`ablation_config.is_full_bypass`) ->
  `_handle_bypass_turn`: one plain-LLM call per turn, no scaffolding at
  all. The floor MINIMAL_BRANCH is measured against.

All node invocations flow through `_call_node`, which records inputs
and outputs to `node_calls` per CLAUDE.md invariant 2. `turn_diagnostics`
(diagnostics.py) is written once per turn, opt-in via `diagnostics_store`.

`consolidate_session` (memory.py steps 6-8) is background-only — see its
docstring — driven by `probe consolidate-session`, `run_interactive`'s
turn-count-gated auto-trigger on exit, and the web UI's explicit
"End session & consolidate" button, never by `handle_turn` itself.
"""

from __future__ import annotations

import asyncio
import logging
import random
import time
from collections.abc import Callable
from typing import Any
from uuid import UUID

import asyncpg

from probe import claims as _claims
from probe import embeddings as _embeddings
from probe import history_block as _history_block
from probe import reference_bindings as _reference_bindings
from probe import retrieval as _retrieval
from probe.ablation import AblationConfig
from probe.audit import NodeCallStore, TranscriptStore
from probe.baseline import MAX_CALLS_PER_TURN, BaselineTeach
from probe.claims import (
    ClaimConfidenceConfig,
    ClaimExtractor,
    ClaimStore,
    ExtractionConfig,
)
from probe.diagnostics import TurnDiagnosticsStore
from probe.disambiguate import (
    AssessAndBranch,
    DisambiguationOptions,
    DisambiguationStore,
    FinalAnswer,
    build_typed_past_note,
)
from probe.domain_config import DomainConfig
from probe.embeddings import EmbeddingClient
from probe.history_block import HistoryBlockConfig
from probe.interaction_nodes import (
    ABSTRACTOR_VERSION,
    CLASSIFIER_VERSION,
    REFERENCE_BINDING_CLASSIFIER_VERSION,
    STATED_PREFERENCE_CLASSIFIER_VERSION,
    ClassifierConfig,
    ClassifyReferenceResolution,
    ClassifyStatedPreference,
    ClassifyTurnOutcome,
    GenerateAbstractForm,
    LLMSelectionPredictor,
    SelectionPredictor,
    render_structural_requirement,
)
from probe.interactions import (
    InteractionAbstractStore,
    InteractionOptionStore,
    InteractionRecorder,
    InteractionStore,
    PredictionStore,
    ReferenceBindingStore,
    StatedPreferenceStore,
    TurnOutcomeStore,
)
from probe.llm import LLMClient, ModelTierClients
from probe.memory import (
    ConfirmFactMatch,
    ConfirmThinkingStyleMatch,
    EmbedAndSearchFacts,
    LearnerFactStore,
    MemoryConfig,
    SummarizeSessionPath,
    ThinkingStyleStore,
    WriteLearnerFact,
)
from probe.memory_index import project_fact
from probe.memory_retrieval import (
    SessionMemoryCache,
    build_learner_context_block,
    retrieve_learner_memory,
)
from probe.models import (
    BranchStatus,
    DisambiguationAssessment,
    DisambiguationBranch,
    ExtractedFact,
    FactMatchConfirmation,
    FactSearchResult,
    Interaction,
    InteractionAbstract,
    InteractionOption,
    LearnerFactType,
    Option,
    OptionSet,
    OptionStatus,
    Prediction,
    QuestionAuthor,
    StatedPreference,
    TurnDiagnostics,
    TurnOutcome,
)
from probe.moss_client import MossService
from probe.reference_bindings import ReferenceBindingConfig
from probe.retrieval import RetrievalContext
from probe.retrieval_metrics import RetrievalMetrics, RetrievedMemory

logger = logging.getLogger(__name__)

_TEACH_FAILURE_MESSAGE = (
    "the tutor failed to respond this turn — see Diagnostics for the "
    "error; try sending your message again"
)

# The compact recent-history window AssessAndBranch / FinalAnswer share
# (see _build_disambiguation_history) — deliberately small: it exists
# for conversation continuity (resolving a bare "that"/"it"), not to
# re-derive the whole session.
_HISTORY_TURNS = 3

# Greetings / small-talk / filler that must NEVER trigger a "which of these
# did you mean?" clarification — asking a student to disambiguate "hi" is the
# opposite of conversational. These route straight to a normal FinalAnswer
# reply (and skip the AssessAndBranch LLM call entirely). Deliberately an
# exact-match set after light normalization, so real short questions
# ("explain trees") still go through ambiguity detection as intended.
_TRIVIAL_TURNS = frozenset(
    {
        "hi", "hii", "hey", "hey there", "hi there", "hiya", "hello", "hello there",
        "yo", "sup", "wassup", "whats up", "what's up", "howdy",
        "good morning", "good afternoon", "good evening", "good night", "gn",
        "how are you", "how are you doing", "how r u", "hru", "how's it going",
        "thanks", "thank you", "thankyou", "thx", "ty", "cheers",
        "ok", "okay", "k", "kk", "cool", "nice", "great", "awesome", "got it",
        "yes", "yep", "yeah", "yup", "no", "nope", "nah",
        "bye", "goodbye", "see you", "see ya", "later",
        "test", "testing", "hmm", "hm",
    }
)


def _is_trivial_turn(text: str) -> bool:
    """True for a greeting / acknowledgement / filler that should get a plain
    friendly reply, not an ambiguity clarification. Conservative: only exact
    matches after lowercasing and stripping surrounding punctuation."""
    t = " ".join(text.strip().lower().strip("!?.,").split())
    return t in _TRIVIAL_TURNS


def _total_retry_count(tiers: ModelTierClients) -> int:
    """Sum of GeminiLLMClient.retry_count across every distinct client
    in `tiers`, deduplicated by identity. Dedup matters because the
    default (no explicit model_tier_clients) is the *same* LLMClient
    instance for fast/capable/best. getattr(..., 0) makes this 0 for
    StubLLMClient without either client type needing to know about the
    other. Snapshotted before/after each turn; the delta is that turn's
    retry_count."""
    seen: set[int] = set()
    total = 0
    for client in (tiers.fast, tiers.capable, tiers.best):
        if id(client) in seen:
            continue
        seen.add(id(client))
        total += getattr(client, "retry_count", 0)
    return total


class SessionLoop:
    def __init__(
        self,
        transcript: TranscriptStore,
        node_calls: NodeCallStore,
        llm: LLMClient,
        model_tier_clients: ModelTierClients | None = None,
        diagnostics_store: TurnDiagnosticsStore | None = None,
        on_node_start: Callable[[str], None] | None = None,
        ablation_config: AblationConfig | None = None,
        disambiguation_store: DisambiguationStore | None = None,
        learner_fact_store: LearnerFactStore | None = None,
        thinking_style_store: ThinkingStyleStore | None = None,
        embedding_client: EmbeddingClient | None = None,
        memory_config: MemoryConfig | None = None,
        interaction_recorder: InteractionRecorder | None = None,
        interaction_option_store: InteractionOptionStore | None = None,
        interaction_abstract_store: InteractionAbstractStore | None = None,
        turn_outcome_store: TurnOutcomeStore | None = None,
        prediction_store: PredictionStore | None = None,
        retrieval_pool: asyncpg.Pool | None = None,
        selection_predictor: SelectionPredictor | None = None,
        history_block_config: HistoryBlockConfig | None = None,
        stated_preference_store: StatedPreferenceStore | None = None,
        reference_binding_store: ReferenceBindingStore | None = None,
        reference_binding_config: ReferenceBindingConfig | None = None,
        domain_config: DomainConfig | None = None,
        claim_store: ClaimStore | None = None,
        claim_extraction_config: ExtractionConfig | None = None,
        claim_confidence_config: ClaimConfidenceConfig | None = None,
        moss_service: MossService | None = None,
        moss_max_inject: int = 4,
    ) -> None:
        # Tiering (fast/capable/best -> real Gemini models, see
        # model_config.py) is opt-in via model_tier_clients. Omitting it
        # reproduces the pre-tiering behavior exactly: every node gets
        # the single `llm` argument.
        tiers = model_tier_clients or ModelTierClients(fast=llm, capable=llm, best=llm)
        self._tiers = tiers
        self._transcript = transcript
        self._node_calls = node_calls
        self._diagnostics = diagnostics_store
        self._on_node_start = on_node_start
        self._ablation = ablation_config or AblationConfig()
        # The domain switch (domain_config.py) -- fixed for this loop's
        # lifetime, same as ablation_config above: every node it
        # constructs below gets the identical DomainConfig, and every
        # interaction this loop records carries the same domain (see
        # `_record_interaction`).
        self._domain_config = domain_config or DomainConfig.education()

        # The plain-LLM BASELINE — same tier as FinalAnswer, since
        # that's what it's compared against. Cheap to construct
        # unconditionally (no store dependency at all).
        self.baseline_teach = BaselineTeach(tiers.best)

        # disambiguate.py's minimal three-call flow. Fast tier for the
        # two judgment calls (a narrow structural judgment, not the
        # final response); best tier for FinalAnswer, since it is the
        # response the student actually sees.
        self._disambiguation = disambiguation_store
        self.assess_and_branch = AssessAndBranch(tiers.fast, domain_config=self._domain_config)
        self.disambiguation_options = DisambiguationOptions(
            tiers.fast, domain_config=self._domain_config
        )
        self.final_answer = FinalAnswer(tiers.best, domain_config=self._domain_config)
        # The most recent AssessAndBranch-generating turn's id, if its
        # branches are still unresolved. None whenever the last turn was
        # a click resolution, a direct answer, or has already been
        # superseded by a later typed-past turn.
        self._prior_disambiguation_turn_id: UUID | None = None

        # The memory layer (memory.py) — additive on top of
        # minimal_branch, never required by it. `_memory_enabled` gates
        # the per-turn semantic pre-check / fact-writing (needs both a
        # place to search/write facts and something to embed with);
        # `_thinking_styles` (the cross-session layer) is independently
        # optional on top of that — consolidate_session needs it, but
        # nothing per-turn does except reading already-`confirmed`
        # candidates for AssessAndBranch's prompt (see
        # `_build_thinking_style_hint`).
        self._learner_facts = learner_fact_store
        self._thinking_styles = thinking_style_store
        self._embedding_client = embedding_client
        self._memory_config = memory_config or MemoryConfig()
        self._memory_enabled = (
            learner_fact_store is not None and embedding_client is not None
        )
        if self._memory_enabled:
            self.embed_and_search_facts = EmbedAndSearchFacts(
                embedding_client, learner_fact_store, self._memory_config
            )
            self.confirm_fact_match = ConfirmFactMatch(tiers.fast)
            self.write_learner_fact = WriteLearnerFact(
                tiers.fast, embedding_client, learner_fact_store
            )
        else:
            self.embed_and_search_facts = None
            self.confirm_fact_match = None
            self.write_learner_fact = None
        # Background-only (see consolidate_session) — needs the fact
        # store, the thinking-style store, and an embedding client.
        if (
            learner_fact_store is not None
            and thinking_style_store is not None
            and embedding_client is not None
        ):
            self.summarize_session_path = SummarizeSessionPath(tiers.fast)
            self.confirm_thinking_style_match = ConfirmThinkingStyleMatch(tiers.fast)
        else:
            self.summarize_session_path = None
            self.confirm_thinking_style_match = None

        # The interaction/retrieval pipeline (interactions.py,
        # retrieval.py) — additive on top of minimal_branch, never
        # required by it, same two-part gating discipline as the memory
        # layer above: needs both a place to persist (interaction_
        # recorder) and its supporting stores. Absent any of them, every
        # write/fire site below is a no-op and a deployment without
        # this configured behaves exactly as it did before this
        # existed. Nodes are cheap to construct unconditionally (fast
        # tier, no store dependency) -- only their USE is gated.
        self._interaction_recorder = interaction_recorder
        self._interaction_options = interaction_option_store
        self._interaction_abstracts = interaction_abstract_store
        self._turn_outcomes = turn_outcome_store
        self._predictions = prediction_store
        self._retrieval_pool = retrieval_pool
        self._interactions_enabled = interaction_recorder is not None
        # history_block.py — additive on top of the interaction pipeline
        # above (needs the same retrieval_pool/embedding_client), gated
        # by its own config flag (default on) rather than by whether
        # it's configured at all, since it has no dedicated store of
        # its own to be absent.
        self._history_block_config = history_block_config or HistoryBlockConfig()
        # Fix B's write side (see StatedPreference's own docstring):
        # gated on the store the same way the abstractor/outcome
        # classifier are gated on theirs -- absent it, this is a no-op.
        self._stated_preferences = stated_preference_store
        # reference_bindings.py -- additive on top of the interaction
        # pipeline, same two-part gating discipline (a store to persist
        # into, plus its own config flag, default on) as history_block
        # above. Absent the store, every fire/read site below is a
        # no-op.
        self._reference_bindings = reference_binding_store
        self._reference_binding_config = reference_binding_config or ReferenceBindingConfig()
        self.classify_turn_outcome = ClassifyTurnOutcome(
            tiers.fast, ClassifierConfig(), domain_config=self._domain_config
        )
        # GenerateAbstractForm deliberately does NOT take a domain_config
        # -- its whole purpose is staying domain-agnostic (see
        # domain_config.py's own module docstring); it is the target of
        # this feature's own audit, not a fourth prompt variant.
        self.generate_abstract_form = GenerateAbstractForm(tiers.fast)
        self.classify_stated_preference = ClassifyStatedPreference(
            tiers.fast, domain_config=self._domain_config
        )
        self.classify_reference_resolution = ClassifyReferenceResolution(
            tiers.fast, domain_config=self._domain_config
        )
        # The claim layer (claims.py) -- session-end only, never per
        # turn (see claims.py's own module docstring for why: extraction
        # is anchored on prediction error, not "read every session").
        # Gated on the store the same way the other optional layers
        # above are; the extractor node is cheap to construct
        # unconditionally.
        self._claim_store = claim_store
        self._claim_extraction_config = claim_extraction_config or ExtractionConfig()
        self._claim_confidence_config = claim_confidence_config or ClaimConfidenceConfig()
        self.claim_extractor = ClaimExtractor(tiers.fast)
        # Held separately (not read off claim_extractor, a private
        # detail of that class) for maybe_restate_claims's one fast-tier
        # call per qualifying claim -- see consolidate_session.
        self._claim_statement_llm = tiers.fast
        self.selection_predictor = selection_predictor or LLMSelectionPredictor(tiers.fast)

        # The Moss learner-memory retrieval layer (moss_client.py /
        # memory_retrieval.py) -- additive, separately gated, same
        # discipline as every optional layer above. When a MossService is
        # wired, `_run_final_answer` retrieves this learner's memories on
        # the live path and injects them into FinalAnswer's prompt (VERSA
        # build context §0.3/§13), and a newly written LearnerFact is
        # synced into the index off the response path (§16). None ->
        # the feature is simply absent; minimal_branch is unchanged.
        self._moss_service = moss_service
        self._moss_max_inject = moss_max_inject
        # One in-process short-term cache per loop == per session in the
        # webserver's one-loop-per-session model (§17): guarantees a fact
        # created this turn is retrievable next turn without waiting on the
        # async index sync. Not a Moss replacement -- very recent local
        # state only.
        self._session_cache: SessionMemoryCache | None = (
            SessionMemoryCache() if moss_service is not None else None
        )
        # Read by the webserver after a turn to emit the retrieval
        # telemetry event/panel (§21.3/§30). Reset every turn that runs
        # FinalAnswer; None means no Moss retrieval happened this turn.
        self.last_moss_metrics: RetrievalMetrics | None = None
        self.last_moss_memories: list[RetrievedMemory] = []

        # Strong references to fire-and-forget background tasks
        # (classification, abstraction, prediction) so they cannot be
        # garbage-collected mid-flight -- a known asyncio gotcha for a
        # detached asyncio.create_task with nothing else holding it.
        self._background_tasks: set[asyncio.Task] = set()

    def _fire_background(self, coro) -> None:
        task = asyncio.create_task(coro)
        self._background_tasks.add(task)
        task.add_done_callback(self._background_tasks.discard)

    async def wait_for_background_tasks(self) -> None:
        """Test-support only: production code never awaits the tasks
        `_fire_background` creates (that would defeat the entire point
        of firing them detached). Exists so a test can assert on a
        classification/abstraction/prediction's result deterministically
        instead of guessing at a sleep duration."""
        if self._background_tasks:
            await asyncio.gather(*self._background_tasks, return_exceptions=True)

    async def run_interactive(self, learner_id: UUID) -> UUID:
        # ablation_config is fixed for a session's lifetime (set-once,
        # see TranscriptStore.set_ablation_config) — pass this loop's
        # own config through explicitly so the persisted row always
        # matches what actually ran.
        session_id = await self._transcript.create_session(
            learner_id, ablation_config=self._ablation
        )
        print(f"probe: new session {session_id}")
        print("probe: type your message. ctrl-D or empty line + ctrl-C to exit.")
        turn_index = 0
        loop = asyncio.get_running_loop()
        while True:
            try:
                turn_text = await loop.run_in_executor(None, input, "you: ")
            except (EOFError, KeyboardInterrupt):
                print()
                break
            turn_text = turn_text.strip()
            if not turn_text:
                continue
            message = await self.handle_turn(session_id, turn_index, turn_text)
            print(f"probe: {message}")
            turn_index += 1

        # Auto-consolidation on interactive exit (memory.py steps 6-8),
        # gated by turn count: a session below
        # MemoryConfig.min_turns_for_cli_auto_consolidation is not
        # eligible at all. No-ops harmlessly (via consolidate_session's
        # own guards) for a BASELINE session (nothing in learner_facts
        # to consolidate) or when the memory layer isn't configured.
        if turn_index >= self._memory_config.min_turns_for_cli_auto_consolidation:
            result = await self.consolidate_session(session_id)
            if result is not None:
                print(
                    f"probe: consolidated this session's path — "
                    f"thinking-style candidate {result.id} now has "
                    f"confirmation_count={result.confirmation_count} "
                    f"(status={result.status.value})"
                )
        return session_id

    async def handle_turn(
        self,
        session_id: UUID,
        turn_index: int,
        turn_text: str,
        selected_option_id: UUID | None = None,
    ) -> str:
        if self._ablation.is_full_bypass:
            # SessionMode.BASELINE — plain LLM, one call, no scaffolding.
            return await self._handle_bypass_turn(session_id, turn_index, turn_text)
        return await self._handle_disambiguation_turn(
            session_id, turn_index, turn_text, selected_option_id
        )

    async def _handle_bypass_turn(
        self, session_id: UUID, turn_index: int, turn_text: str
    ) -> str:
        """SessionMode.BASELINE's turn: student message -> one LLM call
        (with prior turns in this session as context) -> response. Every
        store touched here is the transcript (turns + the one node_calls
        row for the call itself, plus turn_diagnostics if configured)
        and nothing else. This is what makes the comparison honest —
        MINIMAL_BRANCH's cost is measured against this floor, so this
        floor must not itself carry any of the loop's overhead.
        """
        start = time.monotonic()
        retry_count_start = _total_retry_count(self._tiers)

        await self._transcript.record_turn(session_id, turn_index, turn_text)
        prior_turns = [
            t.text
            for t in await self._transcript.list_turns(session_id)
            if t.turn_index < turn_index
        ]

        teach_failed = False
        warnings: list[str] = []
        try:
            message = await self._call_node(
                self.baseline_teach,
                session_id,
                turn_index,
                turn_text=turn_text,
                prior_turns=prior_turns,
            )
            call_count = self.baseline_teach.last_call_count
        except Exception as exc:
            teach_failed = True
            message = _TEACH_FAILURE_MESSAGE
            call_count = 0
            warnings.append(f"BaselineTeach failed: {exc}")
            logger.warning(
                "BaselineTeach failed on turn %d for session %s: %s",
                turn_index,
                session_id,
                exc,
                exc_info=True,
            )

        if self._diagnostics is not None:
            await self._diagnostics.record(
                TurnDiagnostics(
                    session_id=session_id,
                    turn_index=turn_index,
                    node_call_counts={"BaselineTeach": call_count},
                    total_call_count=call_count,
                    guardrail_fired=False,
                    entropy_bits=None,
                    duration_ms=(time.monotonic() - start) * 1000,
                    warnings=warnings,
                    teach_failed=teach_failed,
                    retry_count=_total_retry_count(self._tiers) - retry_count_start,
                )
            )
        return message

    async def _get_turn_text(self, session_id: UUID, turn_index: int) -> str | None:
        """Turn A's own question_text, looked up from the plain `turns`
        table (recorded unconditionally every turn) rather than from
        the interactions pipeline itself -- this must work even when
        that pipeline is disabled, and it must never depend on the very
        row it is being used to construct."""
        for t in await self._transcript.list_turns(session_id):
            if t.turn_index == turn_index:
                return t.text
        return None

    async def _record_interaction(
        self,
        *,
        learner_id: UUID,
        session_id: UUID,
        turn_index: int,
        turn_text: str,
        question_author: QuestionAuthor,
        originating_question: str | None,
        did_branch: bool,
        response_text: str | None,
        recent_history: str = "",
    ) -> Interaction | None:
        """The single write path every one of `_handle_disambiguation_
        turn`'s finish-points calls into, right after it already has
        `message`/`teach_failed` in hand -- never through `_finish_
        turn_with_fact` / `_run_final_answer` / `FinalAnswer.run`,
        which stay exactly as they were: that chain renders an answer
        and must not grow persistence responsibilities.

        A no-op returning None when the pipeline isn't configured.
        `response_text=None` covers both an options-offered turn AND a
        turn where FinalAnswer failed (nothing genuine was produced --
        see Interaction's own docstring).

        `recent_history` is the same compact window `AssessAndBranch`/
        `FinalAnswer` were given this turn -- threaded through only for
        `ClassifyReferenceResolution` (see below), which needs it to
        recognize a reference phrase used in an EARLIER turn that this
        turn's exchange just settled the meaning of.
        """
        if self._interaction_recorder is None:
            return None
        interaction = await self._interaction_recorder.record(
            learner_id=learner_id,
            session_id=session_id,
            turn_number=turn_index,
            question_text=turn_text,
            question_author=question_author,
            originating_question=originating_question,
            did_branch=did_branch,
            response_text=response_text,
            domain=self._domain_config.domain,
        )
        if response_text is not None:
            self._fire_background(
                self._run_abstraction(interaction, session_id, turn_index)
            )
            # Fired on every response-bearing turn regardless of
            # question_author -- unlike stated-preference classification
            # below, a branch selection (question_author=SYSTEM_OPTION)
            # is explicitly one of the three ways a reference gets
            # settled (see ReferenceBinding's own docstring).
            self._fire_background(
                self._run_reference_binding_classification(
                    interaction, recent_history, session_id, turn_index
                )
            )
        target = self._interaction_recorder.last_classification_target
        if target is not None:
            self._fire_background(
                self._run_classification(target, turn_text, session_id, turn_index)
            )
        # Fix B's write side, learner-authored turns only -- a click
        # carries no free text from the student to classify (see
        # StatedPreference's own docstring).
        if question_author is QuestionAuthor.LEARNER:
            self._fire_background(
                self._run_stated_preference_classification(interaction, session_id, turn_index)
            )
        return interaction

    async def _run_stated_preference_classification(
        self, interaction: Interaction, session_id: UUID, turn_index: int
    ) -> None:
        """Off the critical path, same discipline as _run_abstraction/
        _run_classification: a failure here is logged and swallowed,
        never a lost turn. Written every time (has_preference=False
        included), so coverage is auditable rather than silently
        sparse."""
        if self._stated_preferences is None:
            return
        try:
            result = await self._call_node(
                self.classify_stated_preference,
                session_id,
                turn_index,
                question_text=interaction.question_text,
            )
            await self._stated_preferences.append(
                StatedPreference(
                    interaction_id=interaction.id,
                    learner_id=interaction.learner_id,
                    has_preference=result.has_preference,
                    stated_preference=result.stated_preference,
                    label=result.label,
                    classifier_version=STATED_PREFERENCE_CLASSIFIER_VERSION,
                )
            )
        except Exception as exc:
            logger.warning(
                "ClassifyStatedPreference background task failed for "
                "interaction %s: %s",
                interaction.id,
                exc,
                exc_info=True,
            )

    async def _run_reference_binding_classification(
        self, interaction: Interaction, recent_history: str, session_id: UUID, turn_index: int
    ) -> None:
        """Off the critical path, same discipline as the other
        background classifiers here: a failure is logged and
        swallowed, never a lost turn.

        Deliberately NOT written every time, unlike
        StatedPreference/TurnOutcome (which write a `has_preference=
        False`/DEFERRED row for coverage auditability) -- see
        ReferenceBinding's own docstring: this feature's own review is
        explicit that a noisy table is worse than a sparse one here,
        so a `resolved=False` classification writes nothing at all.
        """
        if self._reference_bindings is None:
            return
        try:
            result = await self._call_node(
                self.classify_reference_resolution,
                session_id,
                turn_index,
                recent_history=recent_history,
                turn_question=interaction.question_text,
                turn_response=interaction.response_text,
                originating_question=interaction.originating_question,
            )
            if not result.resolved:
                return
            await self._reference_bindings.record_resolution(
                learner_id=interaction.learner_id,
                reference_text=result.reference_text,
                resolved_to=result.resolved_to,
                evidence_interaction_id=interaction.id,
                classifier_version=REFERENCE_BINDING_CLASSIFIER_VERSION,
            )
        except Exception as exc:
            logger.warning(
                "ClassifyReferenceResolution background task failed for "
                "interaction %s: %s",
                interaction.id,
                exc,
                exc_info=True,
            )

    async def _build_reference_binding_hint(self, learner_id: UUID, message_text: str) -> str:
        """AssessAndBranch's own read of reference_bindings.py, computed
        against `turn_text` directly (always the learner's own raw
        words at this call site -- see `_handle_disambiguation_turn`'s
        3b path, the only one that calls AssessAndBranch at all).
        `FinalAnswer` computes its own copy separately, inside
        `_run_final_answer`, against `embed_text` -- not reused from
        here, since `_run_final_answer` also serves the click-resolution
        path, where the correct text to match against is
        `originating_question`, never the option's own generated copy.
        A read failure degrades to no hint, never a failed turn."""
        if self._reference_bindings is None or not self._reference_binding_config.enabled:
            return ""
        try:
            block, _ = await _reference_bindings.assemble_reference_bindings_block(
                self._reference_bindings, learner_id, message_text,
                config=self._reference_binding_config,
            )
            return block
        except Exception as exc:
            logger.warning(
                "Reference binding hint lookup failed for learner %s: %s",
                learner_id,
                exc,
                exc_info=True,
            )
            return ""

    async def _run_abstraction(
        self, interaction: Interaction, session_id: UUID, turn_index: int
    ) -> None:
        """Off the critical path -- fired via `_fire_background`, never
        awaited by the turn that triggered it. A failure here is
        logged and otherwise swallowed: a missing abstract just means
        this interaction stays unrecallable by abstract-key retrieval,
        never a lost turn."""
        if self._interaction_abstracts is None or self._embedding_client is None:
            return
        try:
            result = await self._call_node(
                self.generate_abstract_form,
                session_id,
                turn_index,
                question_text=interaction.question_text,
                response_text=interaction.response_text,
            )
            embedding = await self._embedding_client.embed(
                result.abstract_form, task_type=_embeddings.TASK_DOCUMENT
            )
            await self._interaction_abstracts.append(
                InteractionAbstract(
                    interaction_id=interaction.id,
                    learner_id=interaction.learner_id,
                    abstract_form=result.abstract_form,
                    abstract_embedding=embedding,
                    generator_version=ABSTRACTOR_VERSION,
                )
            )
        except Exception as exc:
            logger.warning(
                "GenerateAbstractForm background task failed for "
                "interaction %s: %s",
                interaction.id,
                exc,
                exc_info=True,
            )

    async def _run_classification(
        self, target: Interaction, next_question_text: str, session_id: UUID, turn_index: int
    ) -> None:
        """Classifies `target` (turn N) using its own question/selected
        option/response and `next_question_text` (turn N+1's new
        question, possibly from a different, later session -- see
        InteractionRecorder's cross-session catch-up). Off the critical
        path; a failure or an abstention leaves `target`'s outcome at
        whatever the previous classifier_version already established,
        never a forced guess."""
        if self._turn_outcomes is None:
            return
        try:
            prior_question = target.originating_question or target.question_text
            prior_selected_option = (
                target.question_text
                if target.question_author is QuestionAuthor.SYSTEM_OPTION
                else None
            )
            result = await self._call_node(
                self.classify_turn_outcome,
                session_id,
                turn_index,
                prior_question=prior_question,
                prior_response=target.response_text,
                next_question=next_question_text,
                prior_selected_option=prior_selected_option,
            )
            if result.abstains:
                return
            await self._turn_outcomes.append(
                TurnOutcome(
                    interaction_id=target.id,
                    learner_id=target.learner_id,
                    next_question_text=next_question_text,
                    outcome=result.outcome,
                    confidence=result.confidence,
                    classifier_version=CLASSIFIER_VERSION,
                )
            )
        except Exception as exc:
            logger.warning(
                "ClassifyTurnOutcome background task failed for "
                "interaction %s: %s",
                target.id,
                exc,
                exc_info=True,
            )

    async def _fire_prediction(
        self,
        interaction: Interaction,
        options: list[InteractionOption],
        session_id: UUID,
        turn_index: int,
    ) -> None:
        """Fired after options are persisted and returned to the UI --
        this coroutine itself is never awaited before that response
        goes out (see `_fire_background`'s caller). May complete after
        the learner has already clicked; persisted regardless, per
        Prediction's own docstring."""
        if self._predictions is None or self._retrieval_pool is None:
            return
        try:
            result = await _retrieval.retrieve(
                self._retrieval_pool, interaction.learner_id, interaction.question_embedding,
                ctx=RetrievalContext(domain=self._domain_config.domain),
            )
            scores = await self.selection_predictor.predict(options, result.candidates)
            await self._predictions.append(
                Prediction(
                    interaction_id=interaction.id,
                    learner_id=interaction.learner_id,
                    predicted_scores={str(k): v for k, v in scores.items()},
                    model_version=getattr(
                        self.selection_predictor, "model_version", "unknown"
                    ),
                    retrieved_candidate_ids=[str(c.source_id) for c in result.candidates],
                    retrieval_provenance=[
                        c.model_dump(mode="json") for c in result.candidates
                    ],
                )
            )
        except Exception as exc:
            logger.warning(
                "Prediction background task failed for interaction %s: %s",
                interaction.id,
                exc,
                exc_info=True,
            )

    async def _handle_disambiguation_turn(
        self,
        session_id: UUID,
        turn_index: int,
        turn_text: str,
        selected_option_id: UUID | None = None,
    ) -> str:
        """SessionMode.MINIMAL_BRANCH's turn — see disambiguate.py's
        module docstring for the full flow."""
        assert self._disambiguation is not None, (
            "SessionMode.MINIMAL_BRANCH requires disambiguation_store"
        )
        start = time.monotonic()
        retry_count_start = _total_retry_count(self._tiers)
        warnings: list[str] = []
        node_call_counts: dict[str, int] = {}

        turn_id = await self._transcript.record_turn(session_id, turn_index, turn_text)
        learner_id = await self._transcript.get_learner_id(session_id)

        # Same compact recent-history window AssessAndBranch gets --
        # computed once, up front, so every call site into FinalAnswer
        # below threads the identical context AssessAndBranch itself
        # judged against.
        recent_history = await self._build_disambiguation_history(session_id, turn_index)

        # 3a: a click resolves immediately -- no AssessAndBranch this
        # turn, the reading is already known, and no memory pre-check
        # applies either. An unrecognized/stale option id falls through
        # to the normal flow below.
        if selected_option_id is not None:
            option = await self._disambiguation.get_option(selected_option_id)
            if option is not None:
                await self._disambiguation.set_option_status(
                    option.id, OptionStatus.SELECTED
                )
                branch = await self._disambiguation.mark_matched(option.branch_id)
                sibling_branches = await self._disambiguation.list_branches_for_turn(
                    branch.disambiguation_turn_id
                )
                await self._disambiguation.supersede_open_branches(
                    branch.disambiguation_turn_id, exclude_ids=[branch.id]
                )
                await self._disambiguation.supersede_open_options(option.generation_id)
                self._prior_disambiguation_turn_id = None
                # Fetched up front (not after _finish_turn_with_fact, as
                # before) -- the history-block retrieval query inside it
                # needs this same value to embed the student's own
                # original words, never the clicked option's copy. See
                # _finish_turn_with_fact's own docstring.
                originating_question = await self._get_turn_text(
                    session_id, branch.turn_index
                )
                message, teach_failed, history_source_ids, reference_binding_ids = (
                    await self._finish_turn_with_fact(
                        session_id, turn_index, turn_text, turn_id, learner_id,
                        branch_context=branch.statement,
                        memory_context=None,
                        recent_history=recent_history,
                        fact_type=LearnerFactType.BRANCH_RESOLUTION,
                        branch_statements=[b.statement for b in sibling_branches],
                        node_call_counts=node_call_counts,
                        warnings=warnings,
                        question_author=QuestionAuthor.SYSTEM_OPTION,
                        originating_question=originating_question,
                    )
                )
                if self._interactions_enabled:
                    await self._record_interaction(
                        learner_id=learner_id,
                        session_id=session_id,
                        turn_index=turn_index,
                        turn_text=turn_text,
                        question_author=QuestionAuthor.SYSTEM_OPTION,
                        originating_question=originating_question,
                        did_branch=False,
                        response_text=None if teach_failed else message,
                        recent_history=recent_history,
                    )
                    await self._interaction_options.mark_selected(option.id)
                await self._record_disambiguation_diagnostics(
                    session_id, turn_index, node_call_counts, warnings,
                    teach_failed, start, retry_count_start,
                    history_source_ids=history_source_ids,
                    reference_binding_ids=reference_binding_ids,
                )
                return message

        # 3b: nothing was clicked. If the immediately preceding
        # AssessAndBranch turn still has unresolved (open) branches, the
        # student typed past them -- no attempt is made to match the
        # typed text against that old set; it is superseded outright and
        # threaded into THIS turn's own fresh AssessAndBranch call as
        # context instead.
        typed_past_note = ""
        prior_turn_id = self._prior_disambiguation_turn_id
        self._prior_disambiguation_turn_id = None
        if prior_turn_id is not None:
            prior_branches = await self._disambiguation.list_branches_for_turn(
                prior_turn_id
            )
            open_branches = [b for b in prior_branches if b.status is BranchStatus.OPEN]
            if open_branches:
                typed_past_note = build_typed_past_note(open_branches)
                await self._disambiguation.supersede_open_branches(prior_turn_id)
                await self._disambiguation.supersede_open_options(prior_turn_id)
                warnings.append(
                    "disambiguation_typed_past: student typed past the "
                    "prior turn's options -- superseded, threaded as "
                    "context into this turn's AssessAndBranch instead of "
                    "being matched against"
                )

        # Semantic pre-check (memory.py steps 3-4): does a past fact for
        # THIS learner -- possibly from an earlier session -- already
        # resolve this exact message? Vector similarity alone never
        # decides this; only a confirmed "yes" is allowed to skip
        # AssessAndBranch entirely.
        memory_match_found = False
        memory_match_confirmed = False
        matched_fact_id: UUID | None = None
        memory_context: str | None = None
        if self._memory_enabled:
            search_result = await self._call_node_or_warn(
                self.embed_and_search_facts,
                session_id,
                turn_index,
                "EmbedAndSearchFacts",
                FactSearchResult(),
                warnings,
                learner_id=learner_id,
                message=turn_text,
            )
            node_call_counts["EmbedAndSearchFacts"] = (
                self.embed_and_search_facts.last_call_count
            )
            if search_result.matched_fact_id is not None:
                memory_match_found = True
                matched_fact_id = search_result.matched_fact_id
                confirmation = await self._call_node_or_warn(
                    self.confirm_fact_match,
                    session_id,
                    turn_index,
                    "ConfirmFactMatch",
                    FactMatchConfirmation(resolves=False),
                    warnings,
                    matched_situation=search_result.situation,
                    matched_resolution=search_result.resolution,
                    current_message=turn_text,
                )
                node_call_counts["ConfirmFactMatch"] = (
                    self.confirm_fact_match.last_call_count
                )
                if confirmation.resolves:
                    memory_match_confirmed = True
                    memory_context = (
                        f"{search_result.situation} -- {search_result.resolution}"
                    )

        if memory_match_confirmed:
            # AssessAndBranch never runs this turn -- no DisambiguationTurn
            # row is created for it either (there is nothing it would
            # record beyond what turn_diagnostics already makes visible).
            message, teach_failed, history_source_ids, reference_binding_ids = (
                await self._finish_turn_with_fact(
                    session_id, turn_index, turn_text, turn_id, learner_id,
                    branch_context=None,
                    memory_context=memory_context,
                    recent_history=recent_history,
                    fact_type=LearnerFactType.DIRECT_ANSWER,
                    branch_statements=None,
                    node_call_counts=node_call_counts,
                    warnings=warnings,
                )
            )
            if self._interactions_enabled:
                await self._record_interaction(
                    learner_id=learner_id,
                    session_id=session_id,
                    turn_index=turn_index,
                    turn_text=turn_text,
                    question_author=QuestionAuthor.LEARNER,
                    originating_question=None,
                    did_branch=False,
                    response_text=None if teach_failed else message,
                    recent_history=recent_history,
                )
            await self._record_disambiguation_diagnostics(
                session_id, turn_index, node_call_counts, warnings,
                teach_failed, start, retry_count_start,
                memory_match_found=memory_match_found,
                memory_match_confirmed_resolution=memory_match_confirmed,
                branching_skipped_by_memory=True,
                matched_fact_id=matched_fact_id,
                history_source_ids=history_source_ids,
                reference_binding_ids=reference_binding_ids,
            )
            return message

        # Greetings / small-talk never get a clarification prompt — answer
        # them naturally and skip the AssessAndBranch call entirely.
        if _is_trivial_turn(turn_text):
            assessment = DisambiguationAssessment(needs_branches=False, branch_statements=[])
            reference_binding_hint = None
            node_call_counts["AssessAndBranch"] = 0
        else:
            thinking_style_hint = await self._build_thinking_style_hint(learner_id)
            reference_binding_hint = await self._build_reference_binding_hint(learner_id, turn_text)
            assessment = await self._call_node_or_warn(
                self.assess_and_branch,
                session_id,
                turn_index,
                "AssessAndBranch",
                DisambiguationAssessment(needs_branches=False, branch_statements=[]),
                warnings,
                message=turn_text,
                recent_history=recent_history,
                typed_past_note=typed_past_note,
                thinking_style_hint=thinking_style_hint,
                reference_binding_hint=reference_binding_hint,
            )
            node_call_counts["AssessAndBranch"] = self.assess_and_branch.last_call_count

        if not assessment.needs_branches:
            # Persisted unconditionally -- a turn judged unambiguous is a
            # queryable row, not a gap.
            await self._disambiguation.create_turn(
                session_id, turn_index, needs_branches=False, turn_had_direct_answer=True
            )
            message, teach_failed, history_source_ids, reference_binding_ids = (
                await self._finish_turn_with_fact(
                    session_id, turn_index, turn_text, turn_id, learner_id,
                    branch_context=None,
                    memory_context=None,
                    recent_history=recent_history,
                    fact_type=LearnerFactType.DIRECT_ANSWER,
                    branch_statements=None,
                    node_call_counts=node_call_counts,
                    warnings=warnings,
                )
            )
            if self._interactions_enabled:
                await self._record_interaction(
                    learner_id=learner_id,
                    session_id=session_id,
                    turn_index=turn_index,
                    turn_text=turn_text,
                    question_author=QuestionAuthor.LEARNER,
                    originating_question=None,
                    did_branch=False,
                    response_text=None if teach_failed else message,
                    recent_history=recent_history,
                )
            await self._record_disambiguation_diagnostics(
                session_id, turn_index, node_call_counts, warnings,
                teach_failed, start, retry_count_start,
                memory_match_found=memory_match_found,
                matched_fact_id=matched_fact_id,
                history_source_ids=history_source_ids,
                reference_binding_ids=reference_binding_ids,
            )
            return message

        disamb_turn = await self._disambiguation.create_turn(
            session_id, turn_index, needs_branches=True, turn_had_direct_answer=False
        )
        branches = [
            DisambiguationBranch(
                disambiguation_turn_id=disamb_turn.id,
                session_id=session_id,
                turn_index=turn_index,
                statement=statement,
            )
            for statement in assessment.branch_statements
        ]
        await self._disambiguation.add_branches(branches)

        option_set = await self._call_node_or_warn(
            self.disambiguation_options,
            session_id,
            turn_index,
            "DisambiguationOptions",
            OptionSet(),
            warnings,
            branches=branches,
            message=turn_text,
            recent_history=recent_history,
            reference_binding_hint=reference_binding_hint,
        )
        node_call_counts["DisambiguationOptions"] = (
            self.disambiguation_options.last_call_count
        )
        proposals = option_set.proposals
        if proposals and len(proposals) < len(branches):
            # DisambiguationOptions dropped a branch (an off-topic
            # subject reading the session's own context already ruled
            # out, or -- for an approach-kind decision -- a branch
            # beyond the two the chosen axis was built from). Explicit,
            # not left dangling: an un-offered branch would otherwise
            # sit in 'open' status forever, since only a click can move
            # it to matched/superseded and it was never shown to click.
            used_branch_ids = [p.branch_id for p in proposals]
            await self._disambiguation.supersede_open_branches(
                disamb_turn.id, exclude_ids=used_branch_ids
            )

        if not proposals:
            # DisambiguationOptions produced nothing usable -- graceful
            # degrade: answer directly rather than show broken/empty
            # buttons or drop the turn.
            warnings.append(
                "disambiguation_options_empty: the message was judged "
                "ambiguous but no usable options were generated -- "
                "answering directly instead of showing nothing"
            )
            message, teach_failed, history_source_ids, reference_binding_ids = (
                await self._finish_turn_with_fact(
                    session_id, turn_index, turn_text, turn_id, learner_id,
                    branch_context=None,
                    memory_context=None,
                    recent_history=recent_history,
                    fact_type=LearnerFactType.DIRECT_ANSWER,
                    branch_statements=None,
                    node_call_counts=node_call_counts,
                    warnings=warnings,
                )
            )
            if self._interactions_enabled:
                # did_branch=True: AssessAndBranch DID judge ambiguity
                # genuine and emit branches (persisted just above) --
                # it never "went straight to FinalAnswer." That this
                # turn ultimately answered directly is
                # DisambiguationOptions' own failure to produce usable
                # options, a separate fact from whether branching itself
                # occurred.
                await self._record_interaction(
                    learner_id=learner_id,
                    session_id=session_id,
                    turn_index=turn_index,
                    turn_text=turn_text,
                    question_author=QuestionAuthor.LEARNER,
                    originating_question=None,
                    did_branch=True,
                    response_text=None if teach_failed else message,
                    recent_history=recent_history,
                )
            await self._record_disambiguation_diagnostics(
                session_id, turn_index, node_call_counts, warnings,
                teach_failed, start, retry_count_start,
                memory_match_found=memory_match_found,
                matched_fact_id=matched_fact_id,
                history_source_ids=history_source_ids,
                reference_binding_ids=reference_binding_ids,
            )
            return message

        # Shuffled once, here -- BEFORE persisting either table. Fixed
        # ordering would make selection partly a function of the UI
        # rather than the learner, so both what disambiguation_options
        # stores (what the UI actually renders, via
        # DisambiguationStore.list_options_for_turn's created_at order)
        # and interaction_options.shown_position below reflect the
        # identical shuffled order -- one shuffle, two consistent
        # records of it, not two independently-ordered ones.
        shuffled_proposals = proposals.copy()
        random.shuffle(shuffled_proposals)
        new_options = [
            Option(
                branch_id=p.branch_id,
                generation_id=disamb_turn.id,
                session_id=session_id,
                turn_index=turn_index,
                text=p.text,
            )
            for p in shuffled_proposals
        ]
        await self._disambiguation.create_options(new_options)
        self._prior_disambiguation_turn_id = disamb_turn.id

        # No FinalAnswer this turn -- options are shown INSTEAD of an
        # answer. Nothing was resolved yet, so no fact is written
        # either. The option texts are read by the caller (web UI/CLI)
        # from DisambiguationStore.list_options_for_turn.
        message = "Which of these did you mean?"

        # Order, per spec: persist interaction -> shuffle and persist
        # options (above) -> return to UI (below, this method's return)
        # -> fire prediction -> persist on completion (inside
        # _fire_prediction, never awaited here).
        if self._interactions_enabled:
            interaction = await self._record_interaction(
                learner_id=learner_id,
                session_id=session_id,
                turn_index=turn_index,
                turn_text=turn_text,
                question_author=QuestionAuthor.LEARNER,
                originating_question=None,
                did_branch=True,
                response_text=None,
                recent_history=recent_history,
            )
            interaction_options = [
                InteractionOption(
                    interaction_id=interaction.id,
                    learner_id=learner_id,
                    option_id=new_options[i].id,
                    branch_id=new_options[i].branch_id,
                    option_text=new_options[i].text,
                    shown_position=i,
                    # Persisted here, not just in node_calls -- see
                    # InteractionOption's own docstring for why claim
                    # extraction needs this as a normal column, not
                    # something to reverse-engineer from option text.
                    kind=option_set.kind,
                    axis=option_set.axis,
                    # Per-option (unlike kind/axis, which repeat across
                    # the set) -- shuffled_proposals[i] is the SAME
                    # proposal new_options[i] was built from, so the
                    # side lines up with this exact option, not just
                    # the set as a whole.
                    side=shuffled_proposals[i].side,
                )
                for i in range(len(new_options))
            ]
            await self._interaction_options.create_many(interaction_options)
            self._fire_background(
                self._fire_prediction(interaction, interaction_options, session_id, turn_index)
            )

        await self._record_disambiguation_diagnostics(
            session_id, turn_index, node_call_counts, warnings,
            teach_failed=False, start=start, retry_count_start=retry_count_start,
            memory_match_found=memory_match_found,
            matched_fact_id=matched_fact_id,
        )
        return message

    async def _finish_turn_with_fact(
        self,
        session_id: UUID,
        turn_index: int,
        turn_text: str,
        turn_id: UUID,
        learner_id: UUID,
        branch_context: str | None,
        memory_context: str | None,
        recent_history: str,
        fact_type: LearnerFactType,
        branch_statements: list[str] | None,
        node_call_counts: dict[str, int],
        warnings: list[str],
        question_author: QuestionAuthor = QuestionAuthor.LEARNER,
        originating_question: str | None = None,
    ) -> tuple[str, bool, list[UUID], list[UUID]]:
        """Every path through `_handle_disambiguation_turn` that
        actually resolves something (a click, a memory-confirmed
        shortcut, an unambiguous direct answer, or the empty-options
        fallback) ends here: run FinalAnswer, then — memory.py step 5 —
        write exactly one fact recording what just happened, unless
        FinalAnswer itself failed (nothing real to record then).

        `question_author`/`originating_question` mirror the same split
        `InteractionRecorder` uses: on a click-resolution turn,
        `turn_text` is the clicked option's own system-authored copy,
        never the student's words -- the history-block retrieval query
        (see `_run_final_answer`) must embed `originating_question`
        instead, for exactly the reason storage does."""
        message, teach_failed, history_source_ids, reference_binding_ids = (
            await self._run_final_answer(
                session_id, turn_index, turn_text, branch_context, recent_history,
                node_call_counts, warnings, learner_id=learner_id,
                question_author=question_author, originating_question=originating_question,
                memory_context=memory_context,
            )
        )
        if not teach_failed and self._memory_enabled:
            await self._call_node_or_warn(
                self.write_learner_fact,
                session_id,
                turn_index,
                "WriteLearnerFact",
                ExtractedFact(situation="", resolution=""),
                warnings,
                fact_type=fact_type,
                learner_id=learner_id,
                session_id=session_id,
                turn_index=turn_index,
                source_turn_id=turn_id,
                student_message=turn_text,
                tutor_message=message,
                branch_statements=branch_statements,
            )
            node_call_counts["WriteLearnerFact"] = self.write_learner_fact.last_call_count
            # Project the just-written fact into the memory index (§16):
            # session cache synchronously (immediate continuity, §17), the
            # long-term Moss index off the response path (§16.1).
            await self._sync_new_fact_to_memory_index(session_id, turn_index, warnings)
        return message, teach_failed, history_source_ids, reference_binding_ids

    async def _sync_new_fact_to_memory_index(
        self, session_id: UUID, turn_index: int, warnings: list[str]
    ) -> None:
        """Read back the one fact WriteLearnerFact just wrote for this
        (session, turn), project it to a Moss document, add it to the
        session cache immediately, and fire the Moss index upsert in the
        background so it never blocks the turn (VERSA build context §16.1).

        Failure-isolated: memory-index sync is best-effort; a failure here
        must never turn a successful answer into a failed turn."""
        if self._moss_service is None or self._learner_facts is None:
            return
        try:
            facts = await self._learner_facts.list_by_session(session_id)
            fact = next(
                (f for f in reversed(facts) if f.turn_index == turn_index), None
            )
            if fact is None:
                return
            doc = project_fact(fact)
            if self._session_cache is not None:
                self._session_cache.add(doc)
            self._fire_background(self._moss_service.sync_documents([doc]))
        except Exception as exc:
            warnings.append(f"moss memory sync failed: {exc}")
            logger.warning(
                "Moss memory sync failed on turn %d for session %s: %s",
                turn_index,
                session_id,
                exc,
                exc_info=True,
            )

    async def _run_final_answer(
        self,
        session_id: UUID,
        turn_index: int,
        turn_text: str,
        branch_context: str | None,
        recent_history: str,
        node_call_counts: dict[str, int],
        warnings: list[str],
        learner_id: UUID,
        question_author: QuestionAuthor = QuestionAuthor.LEARNER,
        originating_question: str | None = None,
        memory_context: str | None = None,
    ) -> tuple[str, bool, list[UUID], list[UUID]]:
        """FinalAnswer has no fallback -- its output IS the turn. A
        failure here degrades to a fixed in-band message and is the
        caller's concern, not this method's.

        `recent_history` is the same window `AssessAndBranch` was given
        this turn — FinalAnswer must not judge a poorer context than
        AssessAndBranch already reasoned against.

        The learner-history block (history_block.py) is assembled here,
        on the critical path, since FinalAnswer's own prompt needs it —
        a separate embedding call from the one `InteractionRecorder`
        makes later when this turn's row is written (that one isn't
        computed yet; the response doesn't exist until FinalAnswer
        returns). A failure here degrades to no history block, not a
        failed turn -- retrieval quality is not FinalAnswer's contract.
        """
        # Computed unconditionally (pure, no I/O): both the history
        # block below AND reference_bindings.py's own read need the
        # same "the student's own words, never a system-authored
        # option's copy" text -- see Interaction's own docstring for
        # why originating_question is the right choice on a
        # click-resolution turn.
        embed_text = (
            originating_question
            if question_author is QuestionAuthor.SYSTEM_OPTION
            else turn_text
        )

        learner_history_block = ""
        history_source_ids: list[UUID] = []
        if (
            self._interactions_enabled
            and self._retrieval_pool is not None
            and self._embedding_client is not None
            and self._history_block_config.enabled
        ):
            try:
                query_vec = await self._embedding_client.embed(
                    embed_text, task_type=_embeddings.TASK_QUERY
                )
                learner_history_block, history_source_ids = (
                    await _history_block.assemble_history_block(
                        self._retrieval_pool, learner_id, session_id, query_vec,
                        history_config=self._history_block_config,
                        domain=self._domain_config.domain,
                    )
                )
            except Exception as exc:
                warnings.append(f"history block assembly failed: {exc}")
                logger.warning(
                    "History block assembly failed on turn %d for session %s: %s",
                    turn_index,
                    session_id,
                    exc,
                    exc_info=True,
                )
                learner_history_block, history_source_ids = "", []

        # reference_bindings.py's read side: independent of the history
        # block's own config flag (a separate feature, gated on its own
        # store/config) -- exact-match, no embedding, no LLM call.
        reference_bindings_block = ""
        reference_binding_ids: list[UUID] = []
        if self._reference_bindings is not None and self._reference_binding_config.enabled:
            try:
                reference_bindings_block, reference_binding_ids = (
                    await _reference_bindings.assemble_reference_bindings_block(
                        self._reference_bindings, learner_id, embed_text,
                        config=self._reference_binding_config,
                    )
                )
            except Exception as exc:
                warnings.append(f"reference binding lookup failed: {exc}")
                logger.warning(
                    "Reference binding lookup failed on turn %d for session %s: %s",
                    turn_index,
                    session_id,
                    exc,
                    exc_info=True,
                )
                reference_bindings_block, reference_binding_ids = "", []

        # Fix B's read side: the single most recent explicitly-stated
        # preference for this learner, rendered as a requirement on the
        # answer's shape, not folded into learner_history_block above
        # -- see disambiguate.FinalAnswer's own docstring for why
        # position and phrasing both matter here.
        structural_requirement = ""
        if self._stated_preferences is not None:
            try:
                latest_pref = await self._stated_preferences.get_latest_for_learner(learner_id)
                if latest_pref is not None and latest_pref.stated_preference:
                    structural_requirement = render_structural_requirement(
                        latest_pref.label, latest_pref.stated_preference
                    )
            except Exception as exc:
                warnings.append(f"stated preference lookup failed: {exc}")
                logger.warning(
                    "Stated preference lookup failed on turn %d for session %s: %s",
                    turn_index,
                    session_id,
                    exc,
                    exc_info=True,
                )

        # The claim layer's read side (claims.py): promoted claims only
        # -- a candidate never renders (see render_claim_constraint's
        # own docstring). Independent of stated_preferences' own
        # mechanism above, which stays untouched; this is an additive,
        # separately-gated read, same two-part discipline (a store
        # to read plus its own failure isolation) as every other
        # optional layer here. Under the current confidence clamp
        # (claims.clamp_confidence_for_decisions) no claim can be
        # promoted yet, so this reads as an empty list in practice --
        # wired in now so nothing else has to change once calibration
        # lifts that clamp.
        claim_constraints_block = ""
        if self._claim_store is not None:
            try:
                promoted = await self._claim_store.list_promoted_for_learner(learner_id)
                if promoted:
                    claim_constraints_block = "".join(
                        _claims.render_claim_constraint(c) for c in promoted
                    )
            except Exception as exc:
                warnings.append(f"claim constraint lookup failed: {exc}")
                logger.warning(
                    "Claim constraint lookup failed on turn %d for session %s: %s",
                    turn_index,
                    session_id,
                    exc,
                    exc_info=True,
                )

        # The Moss learner-memory read side (memory_retrieval.py): the
        # low-latency semantic index queried on the live path and injected
        # into FinalAnswer's prompt (VERSA build context §0.3/§13). Same
        # failure isolation as every block above -- a retrieval failure
        # degrades to no Moss block and a logged warning, never a failed
        # turn (§20.2). `embed_text` is the student's own words (already
        # click-resolution-corrected above), the right query per §12.1.
        moss_memory_block = ""
        self.last_moss_metrics = None
        self.last_moss_memories = []
        if self._moss_service is not None:
            try:
                memories, moss_metrics = await retrieve_learner_memory(
                    service=self._moss_service,
                    learner_id=str(learner_id),
                    user_text=embed_text,
                    session_cache=self._session_cache,
                    max_inject=self._moss_max_inject,
                )
                moss_memory_block = build_learner_context_block(memories)
                self.last_moss_metrics = moss_metrics
                self.last_moss_memories = memories
            except Exception as exc:
                warnings.append(f"moss retrieval failed: {exc}")
                logger.warning(
                    "Moss retrieval failed on turn %d for session %s: %s",
                    turn_index,
                    session_id,
                    exc,
                    exc_info=True,
                )
                moss_memory_block = ""

        try:
            message = await self._call_node(
                self.final_answer,
                session_id,
                turn_index,
                student_message=turn_text,
                branch_context=branch_context,
                recent_history=recent_history,
                memory_context=memory_context,
                learner_history_block=learner_history_block,
                structural_requirement=structural_requirement,
                reference_bindings_block=reference_bindings_block,
                claim_constraints_block=claim_constraints_block,
                moss_memory_block=moss_memory_block,
            )
            node_call_counts["FinalAnswer"] = self.final_answer.last_call_count
            return message, False, history_source_ids, reference_binding_ids
        except Exception as exc:
            warnings.append(f"FinalAnswer failed: {exc}")
            logger.warning(
                "FinalAnswer failed on turn %d for session %s: %s",
                turn_index,
                session_id,
                exc,
                exc_info=True,
            )
            node_call_counts["FinalAnswer"] = 0
            return _TEACH_FAILURE_MESSAGE, True, history_source_ids, reference_binding_ids

    async def _build_thinking_style_hint(self, learner_id: UUID) -> str:
        """Step 8's "only once promoted does it get fed into future
        sessions' prompts" — the one place `AssessAndBranch`'s prompt
        input is built from `thinking_style_candidates`, and the only
        read it is allowed to use (`list_confirmed_for_prompt`
        structurally excludes anything not yet `confirmed`). Empty
        string whenever this layer is off or nothing has been promoted
        yet for this learner."""
        if self._thinking_styles is None:
            return ""
        confirmed = await self._thinking_styles.list_confirmed_for_prompt(learner_id)
        if not confirmed:
            return ""
        return "; ".join(c.path_summary for c in confirmed)

    async def consolidate_session(self, session_id: UUID):
        """Steps 6-8 of memory.py's flow — background-only, by design
        never called from `handle_turn`: labeling and comparing an
        ENTIRE session's order-structure only means something once the
        session actually has one, and doing it mid-turn would mean
        re-doing the same expensive comparison every single turn for no
        new information most of the time. Callers: cli.py's standalone
        `probe consolidate-session` command, `run_interactive`'s own
        turn-count-gated auto-trigger on exit, and the web UI's explicit
        "End session & consolidate" button — never SessionLoop itself.

        Returns None when there is nothing to consolidate (the
        thinking-style layer isn't configured, or this session wrote no
        facts at all). Otherwise returns the `ThinkingStyleCandidate`
        this session ended up confirming (which may have just been
        promoted to `confirmed`) or newly created.

        Also the session-end hook for the (independent, separately
        gated) interaction pipeline: marks this session's trailing
        resolved-but-never-classified interaction DEFERRED, per the
        generalized "emit deferred whenever no N+1 evidence exists yet"
        rule -- run unconditionally, before the thinking-style-layer
        early return below, since this has nothing to do with whether
        that layer is configured. A no-op when the pipeline isn't
        configured, when there's no trailing interaction, or when one
        is already classified (idempotent — safe to call from every one
        of this method's existing callers without new gating).
        """
        if self._interaction_recorder is not None and self._turn_outcomes is not None:
            await self._interaction_recorder.mark_trailing_interaction_deferred(session_id)

        learner_id = await self._transcript.get_learner_id(session_id)

        # The claim layer's session-end hook (claims.py) -- run
        # unconditionally here too, before the thinking-style-layer
        # early return below, for the identical reason the trailing-
        # deferred marking above is: this has nothing to do with
        # whether that layer is configured. A no-op when the claim
        # store isn't configured.
        if (
            self._claim_store is not None
            and self._embedding_client is not None
            and self._retrieval_pool is not None
        ):
            turns_for_claims = await self._transcript.list_turns(session_id)
            claim_turn_index = max((t.turn_index for t in turns_for_claims), default=0)

            async def _record_claim_node_call(node_name, input_json, output_json):
                await self._node_calls.record(
                    node_name=node_name, session_id=session_id, turn_index=claim_turn_index,
                    input_json=input_json, output_json=output_json,
                )

            try:
                await _claims.extract_claims_for_session(
                    self._retrieval_pool, self._claim_store, self.claim_extractor,
                    self._embedding_client, session_id, learner_id,
                    extraction_config=self._claim_extraction_config,
                    confidence_config=self._claim_confidence_config,
                    on_node_call=_record_claim_node_call,
                )
            except Exception as exc:
                logger.warning(
                    "Claim extraction failed for session %s: %s", session_id, exc, exc_info=True,
                )

            # The repair half of duplicate-claim merging. reconcile_candidate
            # already merges same-axis, value-compatible live claims the
            # moment a NEW episode's candidate would otherwise have to pick
            # between them -- but that's pull-based: a duplicate created by
            # any OTHER path (ClaimStore.reopen putting a previously-
            # terminal claim back among the live set is the one actually
            # observed; a future admin/repair script is another) sits
            # unmerged until an episode happens to land on that exact axis
            # again, which may be never. Running the full learner-scoped
            # sweep here -- session-end, already the batch checkpoint this
            # layer uses -- closes that gap without new scheduling
            # infrastructure (this codebase has no periodic job runner, and
            # session volume per learner is far too low to justify building
            # one just for this). Idempotent and cheap when there is
            # nothing to merge (see merge_duplicate_claims's own docstring).
            try:
                await _claims.merge_duplicate_claims(
                    self._retrieval_pool, self._claim_store, learner_id,
                    confidence_config=self._claim_confidence_config,
                )
            except Exception as exc:
                logger.warning(
                    "Duplicate-claim merge failed for learner %s: %s", learner_id, exc, exc_info=True,
                )

            # Restatement check -- deliberately AFTER the merge above, so
            # a just-consolidated survivor's full evidence history (not
            # a pre-merge fragment's partial one) is what gets evaluated
            # for whether its wording has gone stale. See
            # maybe_restate_claims's own docstring for the threshold
            # gate; most sessions change nothing here.
            try:
                await _claims.maybe_restate_claims(
                    self._claim_store, self._claim_statement_llm, learner_id,
                    on_node_call=_record_claim_node_call,
                )
            except Exception as exc:
                logger.warning(
                    "Claim restatement failed for learner %s: %s", learner_id, exc, exc_info=True,
                )

        if (
            self.summarize_session_path is None
            or self.confirm_thinking_style_match is None
            or self._learner_facts is None
            or self._thinking_styles is None
            or self._embedding_client is None
        ):
            return None

        facts = await self._learner_facts.list_by_session(session_id)
        if not facts:
            return None

        turns = await self._transcript.list_turns(session_id)
        last_turn_index = max((t.turn_index for t in turns), default=0)

        path_summary = await self._call_node(
            self.summarize_session_path, session_id, last_turn_index, facts=facts,
        )
        # Symmetric compare: this session's path summary against other
        # sessions' path summaries — same kind of text on both sides, so
        # SEMANTIC_SIMILARITY, not the QUERY/DOCUMENT asymmetry the
        # fact search uses.
        embedding = await self._embedding_client.embed(
            path_summary.summary, task_type=_embeddings.TASK_SIMILARITY
        )

        nearest = await self._thinking_styles.search_similar(learner_id, embedding, limit=1)
        if nearest:
            candidate, similarity = nearest[0]
            if similarity >= self._memory_config.thinking_style_similarity_threshold:
                confirmation = await self._call_node(
                    self.confirm_thinking_style_match,
                    session_id,
                    last_turn_index,
                    existing_path_summary=candidate.path_summary,
                    new_path_summary=path_summary.summary,
                )
                if confirmation.confirms:
                    return await self._thinking_styles.confirm(
                        candidate.id,
                        session_id,
                        promotion_threshold=self._memory_config.thinking_style_promotion_threshold,
                    )

        # No existing candidate was even worth asking about, or the
        # confirmation call said the resemblance was only superficial --
        # either way, this session's own labeled path becomes a brand
        # new candidate (confirmation_count=1).
        return await self._thinking_styles.create_candidate(
            learner_id, session_id, path_summary.summary, embedding,
        )

    # ─────────────────────────── instrument turns (manual trigger) ─────
    #
    # An instrument presented mid-session, at the point where the
    # system is uncertain, tells you about a learner learning; the
    # same instrument on its own standalone page (webserver.py's
    # /instrument route) only tells you about someone doing an
    # exercise. These three methods are the whole of that wiring: no
    # router exists yet, nothing here decides WHEN to present one --
    # `handle_turn` never calls any of them. A caller (the CLI, the
    # web UI, or eventually a router) picks the primitive and calls
    # `present_instrument_turn` directly.

    async def present_instrument_turn(
        self, session_id: UUID, primitive: "_instruments.InstrumentPrimitive"
    ) -> "_instruments.Instrument":
        """Presents a hand-authored instrument (instruments.py) as the
        NEXT turn in this session -- same session_id, next turn_number,
        so it shows up in this learner's real history rather than a
        side channel. Requires the claim layer to be configured
        (`claim_store`/`retrieval_pool` -- the same dependency
        `extract_claims_for_session` already has): an instrument's
        evidence has nowhere to go without it."""
        from probe import instruments as _instruments

        if self._claim_store is None or self._retrieval_pool is None:
            raise RuntimeError(
                "present_instrument_turn requires claim_store and retrieval_pool "
                "to be configured on this SessionLoop"
            )
        learner_id = await self._transcript.get_learner_id(session_id)
        turns = await self._transcript.list_turns(session_id)
        turn_number = max((t.turn_index for t in turns), default=-1) + 1

        from probe.capability import CapabilityClaimStore

        interaction_store = InteractionStore(self._retrieval_pool)
        instrument_store = _instruments.InstrumentStore(self._retrieval_pool)
        contract_store = _instruments.InteractionContractStore(self._retrieval_pool)
        capability_store = CapabilityClaimStore(self._retrieval_pool)
        return await _instruments.present_demo_instrument_in_session(
            interaction_store, instrument_store, contract_store, self._claim_store,
            capability_store, primitive, learner_id, session_id, turn_number,
        )

    async def record_instrument_event(self, event: "_instruments.InstrumentEvent") -> None:
        """Appends one raw event through the normal path -- no
        interpretation happens here (instrument_events' own "nothing
        writes an interpretation back onto a row here")."""
        from probe import instruments as _instruments

        await _instruments.InstrumentEventStore(self._retrieval_pool).append(event)

    async def finalize_instrument_turn(
        self, instrument_id: UUID
    ) -> "_instruments.InstrumentOutcome":
        """Reads the full event stream, interprets it deterministically
        (no LLM), writes evidence through the existing claim-evidence
        path if the outcome is informative, and resolves the
        instrument's `completed_at`/`abandoned` -- the same three steps
        webserver.py's standalone `/finalize` route performs, now
        available on a session that's actually being tutored rather
        than only on the isolated demo page."""
        from probe import instruments as _instruments
        from probe.capability import CapabilityClaimStore

        instrument_store = _instruments.InstrumentStore(self._retrieval_pool)
        contract_store = _instruments.InteractionContractStore(self._retrieval_pool)
        event_store = _instruments.InstrumentEventStore(self._retrieval_pool)
        capability_store = CapabilityClaimStore(self._retrieval_pool)

        instrument = await instrument_store.get(instrument_id)
        if instrument is None:
            raise KeyError(f"instrument {instrument_id} not found")
        contract = await contract_store.get(instrument.contract_id)
        events = await event_store.list_for_instrument(instrument_id)

        outcome = _instruments.interpret_instrument(contract, events)
        await _instruments.write_instrument_evidence(
            self._claim_store, capability_store, contract, instrument, outcome,
            self._claim_confidence_config,
        )
        if any(e.event_type is _instruments.InstrumentEventType.ABANDON for e in events):
            await instrument_store.mark_abandoned(instrument_id)
        else:
            await instrument_store.mark_completed(instrument_id)
        return outcome

    async def _build_disambiguation_history(
        self, session_id: UUID, turn_index: int
    ) -> str:
        """Compact recent-history input shared by `AssessAndBranch` and
        `FinalAnswer` (computed once per turn and threaded into both) —
        the last few student turns paired with FinalAnswer's own
        responses to them, read back out of node_calls (invariant 2
        already guarantees FinalAnswer's rendered output is durable)."""
        if turn_index == 0:
            return ""
        turns = [
            t
            for t in await self._transcript.list_turns(session_id)
            if t.turn_index < turn_index
        ][-_HISTORY_TURNS:]
        answer_calls = await self._node_calls.get_recent_calls(
            session_id, "FinalAnswer", turn_index, _HISTORY_TURNS
        )
        answers_by_turn = {c.turn_index: c.output_json for c in answer_calls}
        lines: list[str] = []
        for t in turns:
            lines.append(f"turn {t.turn_index} student: {t.text}")
            answer = answers_by_turn.get(t.turn_index)
            if answer:
                lines.append(f"turn {t.turn_index} tutor: {answer}")
        return "\n".join(lines)

    async def _record_disambiguation_diagnostics(
        self,
        session_id: UUID,
        turn_index: int,
        node_call_counts: dict[str, int],
        warnings: list[str],
        teach_failed: bool,
        start: float,
        retry_count_start: int,
        memory_match_found: bool = False,
        memory_match_confirmed_resolution: bool = False,
        branching_skipped_by_memory: bool = False,
        matched_fact_id: UUID | None = None,
        history_source_ids: list[UUID] | None = None,
        reference_binding_ids: list[UUID] | None = None,
    ) -> None:
        if self._diagnostics is None:
            return
        history_source_ids = history_source_ids or []
        reference_binding_ids = reference_binding_ids or []
        total_call_count = sum(node_call_counts.values())
        guardrail_fired = total_call_count > MAX_CALLS_PER_TURN
        if guardrail_fired:
            warnings.append(
                f"MAX_CALLS_PER_TURN exceeded: {total_call_count} calls "
                f"(limit {MAX_CALLS_PER_TURN})"
            )
        if branching_skipped_by_memory:
            # Must be visible and auditable per turn, never a silent
            # shortcut (see memory.py's module docstring).
            warnings.append(
                f"branching_skipped_by_memory: fact {matched_fact_id} was "
                "confirmed to resolve this message -- AssessAndBranch was "
                "never called this turn"
            )
        await self._diagnostics.record(
            TurnDiagnostics(
                session_id=session_id,
                turn_index=turn_index,
                node_call_counts=node_call_counts,
                total_call_count=total_call_count,
                guardrail_fired=guardrail_fired,
                entropy_bits=None,
                duration_ms=(time.monotonic() - start) * 1000,
                warnings=warnings,
                teach_failed=teach_failed,
                retry_count=_total_retry_count(self._tiers) - retry_count_start,
                memory_match_found=memory_match_found,
                memory_match_confirmed_resolution=memory_match_confirmed_resolution,
                branching_skipped_by_memory=branching_skipped_by_memory,
                matched_fact_id=matched_fact_id,
                history_block_used=bool(history_source_ids),
                history_block_source_ids=[str(i) for i in history_source_ids],
                history_block_template_version=(
                    _history_block.TEMPLATE_VERSION if history_source_ids else None
                ),
                reference_bindings_injected=[str(i) for i in reference_binding_ids],
            )
        )

    async def _call_node_or_warn(
        self,
        node: Any,
        session_id: UUID,
        turn_index: int,
        label: str,
        fallback: Any,
        warnings: list[str],
        /,
        **kwargs: Any,
    ) -> Any:
        """Same as `_call_node`, except a raised exception is caught,
        recorded into `warnings`, and swallowed in favor of `fallback`
        — so one node's transient failure doesn't lose the whole turn.
        Not used for FinalAnswer, which has no safe fallback (see
        `_run_final_answer`'s dedicated try/except)."""
        try:
            return await self._call_node(node, session_id, turn_index, **kwargs)
        except Exception as exc:
            warnings.append(f"{label} failed: {exc}")
            logger.warning(
                "%s failed on turn %d for session %s: %s",
                label,
                turn_index,
                session_id,
                exc,
                exc_info=True,
            )
            return fallback

    async def _call_node(
        self,
        node: Any,
        session_id: UUID,
        turn_index: int,
        /,
        **kwargs: Any,
    ) -> Any:
        # session_id/turn_index/node are positional-only so a node's own
        # run() kwargs can share a name with them without colliding.
        if self._on_node_start is not None:
            self._on_node_start(type(node).__name__)
        output = await node.run(**kwargs)
        await self._node_calls.record(
            node_name=type(node).__name__,
            session_id=session_id,
            turn_index=turn_index,
            input_json=kwargs,
            output_json=output,
        )
        return output
