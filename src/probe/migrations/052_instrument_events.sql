-- probe: raw instrument events (instruments.py). Fully append-only,
-- no mutable field at all, not even a status -- same guarantee
-- claim_evidence's own docstring makes: nothing ever writes an
-- interpretation back onto a row here. `interpret_instrument` reads
-- this table but never writes to it.
--
-- `seq` is CALLER-supplied (app.js assigns it per instrument as events
-- fire), not a DB-generated identity column -- interpretation needs a
-- deterministic, client-known ordering it can reason about, and a
-- UNIQUE constraint below rejects two events claiming the same
-- position rather than silently reordering them.
CREATE TABLE instrument_events (
    id             UUID PRIMARY KEY,
    instrument_id  UUID NOT NULL REFERENCES instruments (id) ON DELETE RESTRICT,
    seq            INTEGER NOT NULL,
    event_type     TEXT NOT NULL,
    payload        JSONB NOT NULL,
    elapsed_ms     INTEGER NOT NULL,
    created_at     TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    CONSTRAINT instrument_events_type_check
        CHECK (
            event_type IN (
                'start', 'move', 'revise', 'click', 'value_change',
                'submit', 'abandon', 'hint_request'
            )
        ),
    CONSTRAINT instrument_events_unique_seq UNIQUE (instrument_id, seq)
);

CREATE INDEX idx_instrument_events_instrument ON instrument_events (instrument_id, seq);
