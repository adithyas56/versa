-- probe: turn_diagnostics gains a visibility field for the reference-
-- binding feature (reference_bindings.py / migration 038), same
-- discipline as migration 035's history_block_* fields: which known
-- bindings, if any, were actually injected into this turn's prompt(s)
-- -- so a wrong answer traceable to a wrong substituted meaning can be
-- found by reading this column, not guessed at.
--
-- Deliberately a single list, not split by which call site used it
-- (AssessAndBranch vs FinalAnswer): AssessAndBranch's own use is
-- already captured for free in node_calls' input_json (CLAUDE.md
-- invariant 2 -- `reference_binding_hint` is a kwarg on that call).
-- This column records what FinalAnswer's own answer actually saw --
-- the one that produced what the learner reads.

ALTER TABLE turn_diagnostics
    ADD COLUMN reference_bindings_injected JSONB NOT NULL DEFAULT '[]';
