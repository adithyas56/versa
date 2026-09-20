-- probe: one row per instrument PRESENTATION (instruments.py). Append-
-- only in this codebase's established sense: no DELETE, resolution of
-- `completed_at`/`abandoned` happens via UPDATE of those two columns
-- only (same status-transition-via-UPDATE convention as
-- claims.status/confidence) -- identity fields (contract_id, primitive,
-- spec, presented_at) are never touched again after insert.
--
-- `interaction_id`/`learner_id` are NOT foreign-keyed to `interactions`
-- here deliberately: an instrument presentation is explicitly OUTSIDE
-- the turn-based chat flow this build was told to leave unchanged (no
-- mode selector), so it does not ride on that table's heavier
-- invariants (embeddings, entry_state, etc.). `claim_evidence.
-- interaction_id` still requires a real interactions row when evidence
-- is written (see instruments.py's present_instrument, which creates a
-- minimal placeholder one) -- but instruments itself only needs the id
-- as a correlation key, not a foreign key.
CREATE TABLE instruments (
    id              UUID PRIMARY KEY,
    interaction_id  UUID NOT NULL,
    learner_id      UUID NOT NULL REFERENCES learners (id) ON DELETE RESTRICT,
    session_id      UUID NOT NULL REFERENCES sessions (id) ON DELETE RESTRICT,
    contract_id     UUID NOT NULL REFERENCES interaction_contracts (id) ON DELETE RESTRICT,
    primitive       TEXT NOT NULL,
    spec            JSONB NOT NULL,
    presented_at    TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    completed_at    TIMESTAMPTZ,
    abandoned       BOOLEAN NOT NULL DEFAULT FALSE,
    CONSTRAINT instruments_primitive_check
        CHECK (primitive IN ('choose', 'order', 'locate', 'adjust', 'predict', 'construct'))
);

CREATE INDEX idx_instruments_learner ON instruments (learner_id, presented_at DESC);
