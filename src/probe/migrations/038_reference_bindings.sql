-- probe: reference_bindings -- the reference-resolution memory: exact-
-- match, deterministic recall of what a learner-specific shorthand
-- phrase ("the usual", "my project", "that thing we did") has meant,
-- so a later turn doesn't need to re-ask. See interaction_nodes.
-- ClassifyReferenceResolution (write side, off the critical path) and
-- reference_bindings.py (read side: exact-match + confidence decay,
-- no embedding, no LLM call).
--
-- Append-only per the feature's own spec: a re-confirmation of the
-- SAME meaning, or a change to a DIFFERENT one, both write a NEW row
-- rather than updating an existing one -- readers resolve to the
-- latest row per (learner_id, reference_text) via `seq DESC`. A
-- phrase that used to mean one thing and now means another stays
-- fully on record as two facts (what it meant before, what it means
-- now), never one row silently overwritten -- same principle as every
-- other append-only store in this project (CLAUDE.md invariants 1, 4,
-- 6-11; this is the newest member of that family).
--
-- FK mirrors stated_preferences' own (learner_id, interaction_id)
-- composite and for the identical reason: `interactions` is immutable
-- (migration 034's trigger) and this classifier runs OFF the critical
-- path, evidenced by whichever turn's exchange actually settled the
-- reference's meaning (a branch selection, a typed clarification, or
-- the answer itself establishing it).

CREATE TABLE reference_bindings (
    id                       UUID PRIMARY KEY,
    learner_id               UUID NOT NULL,
    -- The learner's own recurring shorthand, e.g. "the usual" -- what
    -- the read side exact-matches (case-insensitive substring, no
    -- embedding) against a later message's text.
    reference_text           TEXT NOT NULL,
    -- Plain-English statement of what the phrase means, e.g. "worked
    -- examples with real numbers before the general form."
    resolved_to              TEXT NOT NULL,
    evidence_interaction_id  UUID NOT NULL,
    -- How many times (across rows sharing this reference_text, going
    -- back through the append-only chain) this exact resolved_to has
    -- been reconfirmed -- feeds the read side's confidence decay.
    -- Computed and written by ReferenceBindingStore.record_resolution,
    -- never incremented via UPDATE.
    confirmation_count       INT NOT NULL DEFAULT 1,
    last_confirmed_at        TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    classifier_version       TEXT NOT NULL,
    created_at               TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    seq                      BIGSERIAL,
    FOREIGN KEY (learner_id, evidence_interaction_id) REFERENCES interactions (learner_id, id)
);

-- The read path this table exists for: "every known binding for this
-- learner, latest per phrase." The exact-match against message text
-- happens in Python (reference_bindings.py), never in SQL -- a phrase
-- either appears in the message or it doesn't, which needs no index
-- beyond fetching this learner's own rows.
CREATE INDEX idx_reference_bindings_learner_latest
    ON reference_bindings (learner_id, seq DESC);
