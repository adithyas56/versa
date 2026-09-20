-- probe: turn_diagnostics gains the history-block feature's own
-- visibility columns (CLAUDE.md invariant 7 already covers this
-- table's append-only-ness; this just adds more nullable-with-default
-- columns to it, same pattern as migrations 021/023/027/028/031).
--
-- Required by history_block.py's own design constraint: a bad answer
-- must be traceable to bad retrieval versus a bad generation, which
-- requires knowing, per turn, whether a history block was actually
-- used and which interactions/patterns it was built from -- guessing
-- at this after the fact is exactly what this column exists to avoid.
-- The rendered TEXT itself is already captured for free in node_calls
-- (CLAUDE.md invariant 2: FinalAnswer's own input_json includes
-- whatever `learner_history_block` text was passed to it that turn) --
-- these columns add the structured, directly-queryable half of that
-- record rather than duplicating the text a second time.

ALTER TABLE turn_diagnostics
    ADD COLUMN history_block_used BOOLEAN NOT NULL DEFAULT FALSE,
    ADD COLUMN history_block_source_ids JSONB NOT NULL DEFAULT '[]',
    ADD COLUMN history_block_template_version TEXT;
