-- probe: interactions — append-only per-turn interaction log for the
-- personalization/retrieval subsystem (see src/probe/interactions.py,
-- src/probe/retrieval.py).
--
-- No topic table, no clustering, no centroid: an earlier version of
-- this migration had a `topics` table (per-learner running-centroid
-- clusters, cosine-attach above a fixed threshold) that
-- entry_state/turns_on_topic were computed from. It was removed after
-- two live-tuning rounds plus a systematic replay against real
-- embeddings confirmed the mechanism itself is broken, not just its
-- threshold: averaging same-subject embeddings into a centroid
-- genericizes it rather than sharpening it, which makes the centroid
-- MORE attractive to unrelated content, which blends it further --
-- a cascade with no fixed threshold that escapes it. A live run
-- collapsed 12 turns across 3 unrelated subjects into ONE topic. What
-- replaced it: everything that mechanism was FOR (continuing,
-- stuck_repeat, returning_after_gap, turns_on_topic) is answerable as
-- a direct pairwise similarity comparison against a learner's recent
-- questions -- no cluster identity needs to exist for any of them.
-- See interactions.py's module docstring and retrieval_config.py's
-- `same_subject_threshold` for the replacement.
--
-- NOT the "story"/learner-memory layer. `learner_facts` (migration
-- 030, memory.py) and the "story" panel in the web UI keep their
-- existing name, schema, and behavior completely unchanged by this
-- migration — this is a new, separate pipeline built alongside it,
-- reading nothing from learner_facts and writing nothing to it. The
-- table is deliberately named `interactions`, not `stories`: that
-- word already means "a learner_facts row" everywhere else in this
-- codebase (the UI's `story` panel, `storyHtml()`, `.story-card`, and
-- prior verification transcripts all use it that way) — reusing it
-- here would silently collide two different concepts under one name.
--
-- `CREATE EXTENSION IF NOT EXISTS vector` first, same reasoning as
-- migration 030: a from-empty-schema replay must self-document.
--
-- halfvec(768), not vector(768) (migration 030's choice): half the
-- per-dimension storage, deliberately chosen here given this table's
-- expected volume (one row per turn, 16-way hash-partitioned) versus
-- learner_facts' much sparser one-row-per-resolution rate.
--
-- PARTITION BY HASH (learner_id), 16 partitions: personal-scope
-- retrieval (stage1_filter) always carries a learner_id predicate, so
-- partition pruning keeps each query scoped to one partition rather
-- than scanning the whole table.
--
-- A hash-partitioned table's PRIMARY KEY must include the partition
-- column (Postgres requires this for any unique index on a
-- partitioned table) — hence PRIMARY KEY (learner_id, id), not id
-- alone. This has one further consequence, validated live before
-- writing this migration: `id` alone can never be made unique across
-- a hash-partitioned parent, so nothing can do a plain
-- `REFERENCES interactions (id)`. Every child table below therefore
-- carries its own `learner_id` column and a COMPOSITE
-- FOREIGN KEY (learner_id, interaction_id) REFERENCES
-- interactions (learner_id, id) — the exact same pattern this
-- codebase already used for `concept_nodes` in migration 010, for the
-- identical reason.
--
-- Immutability is enforced at the database (a real trigger), not just
-- by the AST-based no-delete-method/no-DELETE-SQL scan every other
-- append-only store in this project relies on (CLAUDE.md invariants
-- 1/4/6-11) — explicitly requested for this table specifically,
-- because the retrieval/prediction pipeline downstream of it depends
-- on every row being a fixed, replayable fact once written.
--
-- `interaction_options` is deliberately EXCLUDED from the
-- immutability trigger: `was_selected`/`selection_timestamp` are
-- populated on a LATER turn than the row's creation (the turn where
-- the option was shown and the turn where it was clicked are two
-- separate `handle_turn` calls — see interactions.py's module
-- docstring). It is a decision record, not an interaction record; the
-- immutability invariant is about `interactions` specifically.
--
-- `interaction_abstracts` and `turn_outcomes` are append-only by the
-- same convention as every other versioned store in this codebase
-- (no UPDATE/DELETE code path, latest `generator_version`/
-- `classifier_version` per interaction_id wins on read) rather than a
-- hard DB trigger — only `interactions` itself was asked to be
-- enforced at that level.

CREATE EXTENSION IF NOT EXISTS vector;

-- --- enum types ------------------------------------------------------

CREATE TYPE interaction_question_author AS ENUM ('learner', 'system_option');

-- All five non-resolution values are now direct comparisons over
-- prev_question_sim / recent_similar_count / last_similar_turn_gap
-- (see interactions.py) -- no topic identity is computed or stored.
--
-- 'resolution' added deliberately: without it, the turn immediately
-- following a branch offer (Turn A+1, the click) always computes to
-- 'continuing' -- Turn A asked something similar seconds earlier by
-- construction (the click embeds Turn A's own originating_question),
-- which is technically true and analytically useless. 'resolution' is
-- checked FIRST, before any similarity comparison, whenever the
-- previous turn in the session had did_branch = true, so a
-- click-resolution turn is never mislabeled by similarity to the
-- very question it was generated from.
--
-- 'topic_switch' added deliberately too: a question with no similar
-- match anywhere in the learner's recent window, while the session
-- itself isn't cold, is the learner abandoning or completing one
-- thing and moving to another -- one of the more informative states
-- here. Folding it into 'continuing' (the single most common state in
-- this table) would erase it, not merely under-label it.
CREATE TYPE interaction_entry_state AS ENUM (
    'cold_open', 'continuing', 'returning_after_gap', 'stuck_repeat',
    'topic_switch', 'resolution'
);

-- 'deferred' is NOT 'unknown': 'unknown' is "we don't know what
-- happened to the prior turn" (no classification exists yet, but one
-- could); 'deferred' is "there is nothing to know yet, by
-- construction" (the prior turn was an options-offered row, or the
-- last turn of a session -- see interaction_turn_outcome's own
-- 'deferred' value). Collapsing them erases exactly the distinction
-- entry_state's stuck_repeat-style rules, and any future consumer of
-- this column, need.
-- 'abandoned' replaced by 'moved_on', mirroring interaction_turn_
-- outcome's own collapse -- nothing produces 'abandoned' any more now
-- that the classifier no longer distinguishes it from understood-and-
-- moved-on.
CREATE TYPE interaction_prior_outcome AS ENUM (
    'resolved', 'contradicted', 'moved_on', 'deferred', 'unknown'
);

CREATE TYPE interaction_help_level AS ENUM (
    'none', 'hint', 'worked_example', 'direct_answer'
);

-- 'understood_moved_on' and 'abandoned_topic' collapsed into a single
-- 'moved_on' after a real evaluation session: a classifier reading
-- only the prior question/response/next question cannot tell "left
-- satisfied" from "left frustrated" -- there is no signal in that
-- window for it, confirmed live (3 of 7 real classifications sat
-- exactly on this ambiguity). Guessing it per-turn and storing the
-- guess as a typed value would be worse than not distinguishing it.
CREATE TYPE interaction_turn_outcome AS ENUM (
    'matched', 'contradicted_intent', 'moved_on', 'deferred'
);

-- --- interactions --------------------------------------------------------

CREATE TABLE interactions (
    id                   UUID NOT NULL,
    learner_id           UUID NOT NULL REFERENCES learners (id) ON DELETE RESTRICT,
    session_id           UUID NOT NULL REFERENCES sessions (id) ON DELETE RESTRICT,
    turn_number          INT NOT NULL,
    -- Verbatim, whatever turns.text holds for this turn. On a
    -- click-resolution turn (question_author = system_option) this is
    -- the CLICKED OPTION'S OWN button copy, not the student's words --
    -- app.js's submit(o.text, o.id) sends the option text as this
    -- turn's message. Nothing here is fabricated; question_author is
    -- what makes that fact legible instead of silently misleading.
    question_text        TEXT NOT NULL,
    question_author      interaction_question_author NOT NULL,
    -- Set only when question_author = system_option: Turn A's own
    -- question_text (the original ambiguous message), looked up by
    -- session_id + the branch's turn_number. NULL whenever
    -- question_author = learner. question_embedding is computed from
    -- THIS column when present, never from question_text in that
    -- case -- see this migration's own header comment and
    -- interactions.py for why (embedding the option generator's own
    -- phrasing would make retrieval match on its stylistic tics
    -- rather than on anything about the learner).
    originating_question TEXT,
    did_branch           BOOLEAN NOT NULL,
    -- NULL on an options-offered turn (no FinalAnswer ran) and on a
    -- turn where FinalAnswer failed (nothing genuine was produced --
    -- treated as no-response, not as fabricated content). This is
    -- also what makes prior_turn_outcome/turn_outcomes naturally
    -- resolve to 'deferred' for such rows: no response, no N+1
    -- evidence to classify yet.
    response_text        TEXT,
    entry_state          interaction_entry_state NOT NULL,
    -- The three columns entry_state is actually computed from, over
    -- one similarity query against this learner's recent questions
    -- (see interactions.py) -- no cluster identity anywhere. Storing
    -- the raw numbers, not just the derived label, is what makes this
    -- survivable if same_subject_threshold is wrong: labels recompute
    -- from these stored values with no re-embedding and no touching
    -- an already-written (immutable) row.
    --
    -- Count of this learner's recent questions (last N, see
    -- InteractionConfig) that clear same_subject_threshold against
    -- this one. Replaces the old turns_on_topic -- same meaning
    -- ("how much recent history backs this"), honestly re-derived
    -- from a pairwise comparison instead of a cluster membership
    -- count.
    recent_similar_count  INT NOT NULL DEFAULT 0,
    -- Raw cosine similarity to the immediately preceding interaction
    -- this learner had (regardless of session boundary -- comparing
    -- across a session gap is exactly what makes returning_after_gap
    -- possible to compute from the same field on a cold_open turn).
    -- NULL only when there is no prior interaction at all for this
    -- learner (their very first turn, ever).
    prev_question_sim     DOUBLE PRECISION,
    -- How many interactions back, within the SAME last-N window
    -- recent_similar_count is counted over (see
    -- InteractionConfig.similarity_window), the nearest above-threshold
    -- match was found -- one query produces all three of these
    -- columns, not a separate wider lookup. NULL when nothing in that
    -- window matched at all (the topic_switch case). recent_similar_
    -- count > 0 with prev_question_sim below threshold means "matched
    -- something in the window, just not the immediately preceding
    -- question" -- returning_after_gap.
    last_similar_turn_gap INT,
    -- Best-effort at creation from what is already known; NEVER
    -- backfilled. The authoritative value lives in turn_outcomes;
    -- retrieval reads a view that joins it, not this column.
    prior_turn_outcome   interaction_prior_outcome NOT NULL DEFAULT 'unknown',
    -- Always 'none' for now -- nothing in the current flow produces a
    -- real value, and a guessed one is worse than an honest constant.
    -- Becomes real once FinalAnswer starts carrying an explicit move
    -- label.
    help_level           interaction_help_level NOT NULL DEFAULT 'none',
    -- Measured client-side, options-rendered to selection-received.
    -- NULL until app.js is wired to stamp it -- a nice signal, not a
    -- load-bearing one.
    elapsed_ms           INT,
    -- Embeds question_text when question_author = learner, else
    -- originating_question. Never response_text (dominated by the
    -- model's own prose) and never the system-authored option text.
    -- Renamed from topic_embedding when topic clustering was removed
    -- -- a column named for a deleted concept is worse than no rename,
    -- even though the embedding itself is unchanged and still exactly
    -- what recent_similar_count/prev_question_sim/last_similar_turn_gap
    -- (and retrieval's own stage2 ANN search) compare against.
    question_embedding    halfvec(768) NOT NULL,
    -- Always NULL at insert time -- see interaction_abstracts below,
    -- which is the actual source retrieval reads. Kept as a column
    -- here (rather than dropped) because the abstraction step must
    -- never mutate this row once written; these two columns and their
    -- per-partition HNSW indexes exist to satisfy that constraint and
    -- the "HNSW per partition on both embedding columns" requirement,
    -- not because anything ever reads them here.
    abstract_form        TEXT,
    abstract_embedding   halfvec(768),
    created_at           TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    PRIMARY KEY (learner_id, id)
) PARTITION BY HASH (learner_id);

DO $$
BEGIN
    FOR i IN 0..15 LOOP
        EXECUTE format(
            'CREATE TABLE interactions_p%1$s PARTITION OF interactions '
            'FOR VALUES WITH (MODULUS 16, REMAINDER %1$s)',
            i
        );
    END LOOP;
END $$;

-- Both created on the partitioned parent -- Postgres propagates a
-- matching index to every existing (and future) partition
-- automatically; validated live before writing this migration.
CREATE INDEX idx_interactions_question_embedding_hnsw
    ON interactions USING hnsw (question_embedding halfvec_cosine_ops)
    WITH (m = 16, ef_construction = 64);
CREATE INDEX idx_interactions_abstract_embedding_hnsw
    ON interactions USING hnsw (abstract_embedding halfvec_cosine_ops)
    WITH (m = 16, ef_construction = 64);
-- ^ always empty: abstract_embedding on this table is permanently
-- NULL by construction. Retrieval's abstract-scope recall queries
-- idx_interaction_abstracts_embedding_hnsw (below) instead, joined
-- back to interactions -- see retrieval.py.

-- Supports both InteractionRecorder's own "last N questions for this
-- learner" lookup (the similarity comparison entry_state is now
-- computed from) and retrieval's learner-scoped, recency-ordered
-- reads -- no topic_id predicate any more; stage-1 filtering never
-- needed the speed (p95 16ms on a 100k-row benchmark), only the
-- clustering it was built to filter by, which is gone.
CREATE INDEX idx_interactions_learner_recency
    ON interactions (learner_id, created_at DESC);
CREATE INDEX idx_interactions_session_turn
    ON interactions (session_id, turn_number);
CREATE INDEX idx_interactions_learner_entry_state
    ON interactions (learner_id, entry_state);
CREATE INDEX idx_interactions_learner_help_level
    ON interactions (learner_id, help_level);

CREATE OR REPLACE FUNCTION interactions_block_mutation() RETURNS trigger AS $$
BEGIN
    RAISE EXCEPTION 'interactions rows are immutable: % is not allowed (learner_id=%, id=%)',
        TG_OP, OLD.learner_id, OLD.id;
END;
$$ LANGUAGE plpgsql;

-- Created on the partitioned parent -- propagates to every partition
-- automatically (validated live); blocks UPDATE/DELETE regardless of
-- which partition a given row lands in.
CREATE TRIGGER interactions_no_mutate
    BEFORE UPDATE OR DELETE ON interactions
    FOR EACH ROW EXECUTE FUNCTION interactions_block_mutation();

-- --- interaction_options -------------------------------------------------
-- One row per option actually shown on an options-offered turn (Turn
-- A). Attaches to that turn's interaction row, never to the later
-- click-resolution row. option_id/branch_id reference the EXISTING
-- disambiguation_options/disambiguation_branches tables unchanged --
-- this is a read-oriented denormalization for retrieval, not a new
-- system of record for the click-resolve mechanism itself.

CREATE TABLE interaction_options (
    id                   UUID PRIMARY KEY,
    interaction_id       UUID NOT NULL,
    learner_id           UUID NOT NULL,
    option_id            UUID NOT NULL REFERENCES disambiguation_options (id) ON DELETE RESTRICT,
    branch_id            UUID NOT NULL REFERENCES disambiguation_branches (id) ON DELETE RESTRICT,
    option_text          TEXT NOT NULL,
    -- Position actually shown after shuffling, 0-indexed. Fixed
    -- ordering would make selection partly a function of the UI
    -- rather than the learner.
    shown_position       INT NOT NULL,
    was_selected         BOOLEAN NOT NULL DEFAULT FALSE,
    selection_timestamp  TIMESTAMPTZ,
    created_at           TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    FOREIGN KEY (learner_id, interaction_id) REFERENCES interactions (learner_id, id)
);

CREATE INDEX idx_interaction_options_interaction ON interaction_options (interaction_id);
CREATE INDEX idx_interaction_options_option ON interaction_options (option_id);

-- --- interaction_abstracts ------------------------------------------------
-- Appended off the critical path, alongside the outcome classifier.
-- The interactions row is written immediately with abstract_form/
-- abstract_embedding NULL; this table is the actual source retrieval
-- reads for abstract-scope candidates.

CREATE TABLE interaction_abstracts (
    id                  UUID PRIMARY KEY,
    interaction_id      UUID NOT NULL,
    learner_id          UUID NOT NULL,
    abstract_form       TEXT NOT NULL,
    abstract_embedding  halfvec(768) NOT NULL,
    generator_version   TEXT NOT NULL,
    created_at          TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    -- DB-assigned, strictly monotonic -- see node_calls.seq (migration
    -- 026) for why created_at alone is not a reliable "latest wins"
    -- ordering key (ties under fast/back-to-back writes).
    seq                 BIGSERIAL,
    FOREIGN KEY (learner_id, interaction_id) REFERENCES interactions (learner_id, id)
);

CREATE INDEX idx_interaction_abstracts_interaction
    ON interaction_abstracts (interaction_id, seq DESC);
CREATE INDEX idx_interaction_abstracts_embedding_hnsw
    ON interaction_abstracts USING hnsw (abstract_embedding halfvec_cosine_ops)
    WITH (m = 16, ef_construction = 64);

-- --- turn_outcomes ---------------------------------------------------------
-- At turn N+1, one fast-tier call classifies turn N from N's own
-- question/selected option/response and N+1's new question. Runs off
-- the critical path. Re-running the classifier appends a new row
-- under a new classifier_version; readers resolve to the latest
-- version per interaction_id.
--
-- 'deferred' is emitted (never omitted) whenever no N+1 evidence
-- exists yet -- response_text IS NULL (an options-offered turn with
-- nothing to classify) or this interaction is the last in its
-- session. Both are the same underlying condition stated once, not
-- two special cases. When the learner returns, their first new
-- question IS the missing evidence, and classification runs then.

CREATE TABLE turn_outcomes (
    id                  UUID PRIMARY KEY,
    interaction_id      UUID NOT NULL,
    learner_id          UUID NOT NULL,
    next_question_text  TEXT,
    outcome             interaction_turn_outcome NOT NULL,
    confidence          DOUBLE PRECISION NOT NULL,
    classifier_version  TEXT NOT NULL,
    created_at          TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    seq                 BIGSERIAL,
    FOREIGN KEY (learner_id, interaction_id) REFERENCES interactions (learner_id, id)
);

CREATE INDEX idx_turn_outcomes_interaction
    ON turn_outcomes (interaction_id, seq DESC);

-- --- predictions -----------------------------------------------------------
-- Fired after options are persisted and returned to the UI -- never
-- awaited before that response goes out. May land after the learner
-- has already clicked; persisted regardless, since
-- prediction_created_at is what makes a late-arriving prediction
-- interpretable and cannot be recovered if the row is skipped.

CREATE TABLE predictions (
    id                        UUID PRIMARY KEY,
    interaction_id            UUID NOT NULL,
    learner_id                UUID NOT NULL,
    predicted_scores          JSONB NOT NULL,
    prediction_created_at     TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    model_version             TEXT NOT NULL,
    retrieved_candidate_ids   JSONB NOT NULL,
    retrieval_provenance      JSONB NOT NULL,
    seq                       BIGSERIAL,
    FOREIGN KEY (learner_id, interaction_id) REFERENCES interactions (learner_id, id)
);

CREATE INDEX idx_predictions_interaction
    ON predictions (interaction_id, seq DESC);

-- --- population_patterns ----------------------------------------------------
-- Derived from abstract forms only, never raw transcripts. Readable
-- (by retrieval) only at distinct_learner_count >= 20 AND
-- max_per_learner_share <= 0.25 -- the second gate matters as much as
-- the first: without it, one heavy user can supply most of a
-- pattern's support and the learner-count threshold passes on what is
-- effectively one person's behavior. Append-only: each aggregation
-- run inserts fresh rows; a stale pattern is superseded by a newer
-- row from the next run, never edited in place.

CREATE TABLE population_patterns (
    id                        UUID PRIMARY KEY,
    abstract_form             TEXT NOT NULL,
    embedding                 halfvec(768) NOT NULL,
    support_count             INT NOT NULL,
    distinct_learner_count    INT NOT NULL,
    max_per_learner_share     DOUBLE PRECISION NOT NULL,
    representative_features   JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at                TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX idx_population_patterns_embedding_hnsw
    ON population_patterns USING hnsw (embedding halfvec_cosine_ops)
    WITH (m = 16, ef_construction = 64);
CREATE INDEX idx_population_patterns_readable
    ON population_patterns (distinct_learner_count, max_per_learner_share);

-- --- interactions_current: the joined view retrieval.py reads -------------
-- "retrieval reads a view that joins it" (turn_outcomes), extended to
-- interaction_abstracts too -- one non-materialized view is both the
-- authoritative prior_turn_outcome source AND the source of a live
-- abstract_embedding to recall against (interactions.abstract_embedding
-- itself is always NULL -- see that column's own comment above).
-- A plain (non-materialized) view over a partitioned table with LEFT
-- JOIN LATERAL still lets the planner push WHERE/ORDER BY/LIMIT down
-- into the base table scan -- partition pruning and HNSW index usage
-- are unaffected; verified by EXPLAIN ANALYZE as part of this
-- feature's own benchmark, not assumed.
CREATE VIEW interactions_current AS
SELECT
    i.*,
    resolved.outcome             AS resolved_outcome,
    resolved.confidence          AS resolved_outcome_confidence,
    resolved.classifier_version  AS resolved_outcome_version,
    current_abstract.abstract_form       AS current_abstract_form,
    current_abstract.abstract_embedding  AS current_abstract_embedding,
    current_abstract.generator_version   AS current_abstract_version
FROM interactions i
LEFT JOIN LATERAL (
    SELECT outcome, confidence, classifier_version
    FROM turn_outcomes t
    WHERE t.interaction_id = i.id
    ORDER BY t.seq DESC
    LIMIT 1
) resolved ON TRUE
LEFT JOIN LATERAL (
    SELECT abstract_form, abstract_embedding, generator_version
    FROM interaction_abstracts a
    WHERE a.interaction_id = i.id
    ORDER BY a.seq DESC
    LIMIT 1
) current_abstract ON TRUE;
