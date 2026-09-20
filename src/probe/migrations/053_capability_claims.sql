-- probe: the capability/mastery claim type (probe/capability.py) --
-- the fix for a real incident: locate/predict originally targeted
-- StatedPreferenceLabel claims (target_axis/target_value on
-- interaction_contracts) despite measuring performance, so capability
-- observations ("can trace a swap correctly") were silently
-- accumulating into a preference claim's confidence. Checking real
-- predict evidence confirmed it: every claim it landed on was a
-- genuine wants_steps_shown/rule_before_example preference claim,
-- built entirely from capability-observation rows.
--
-- Same append-only discipline as claims/claim_evidence: no DELETE,
-- capability_claims resolves confidence/status/updated_at via UPDATE
-- only (CapabilityClaimStore.refresh), capability_evidence has no
-- mutable field at all except provenance_note (same one sanctioned
-- exception claim_evidence has).
--
-- Deliberately NO axis column: capability isn't bidirectional -- a
-- skill has no "opposite" pole the way each ApproachAxis has two
-- named poles, so there is nothing for find_by_axis's role to do
-- here. Matching is exact-skill (find_by_skill), never axis-sharing-
-- then-similarity -- which is also why there is no
-- statement_embedding column: with no similarity fallback path,
-- there is nothing for an embedding to disambiguate.
CREATE TABLE capability_claims (
    id           UUID PRIMARY KEY,
    learner_id   UUID NOT NULL REFERENCES learners (id) ON DELETE RESTRICT,
    statement    TEXT NOT NULL,
    test         TEXT NOT NULL,
    skill        TEXT NOT NULL,
    confidence   DOUBLE PRECISION NOT NULL,
    write_policy TEXT NOT NULL,
    status       TEXT NOT NULL DEFAULT 'candidate',
    created_at   TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at   TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    CONSTRAINT capability_claims_skill_check
        CHECK (skill IN ('traces_worked_steps', 'derives_forward', 'other')),
    CONSTRAINT capability_claims_write_policy_check
        CHECK (write_policy IN ('locked', 'slow_drift', 'fast_decay')),
    CONSTRAINT capability_claims_status_check
        CHECK (status IN ('candidate', 'promoted'))
);

CREATE INDEX idx_capability_claims_learner_skill ON capability_claims (learner_id, skill);

CREATE TABLE capability_evidence (
    id                        UUID PRIMARY KEY,
    claim_id                  UUID NOT NULL REFERENCES capability_claims (id) ON DELETE RESTRICT,
    learner_id                UUID NOT NULL,
    interaction_id            UUID NOT NULL,
    direction                 TEXT NOT NULL,
    skill                     TEXT NOT NULL,
    session_id                UUID NOT NULL REFERENCES sessions (id) ON DELETE RESTRICT,
    test_fired                BOOLEAN NOT NULL,
    contradiction_was_possible BOOLEAN NOT NULL,
    created_at                TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    provenance_note           TEXT,
    source                    TEXT NOT NULL DEFAULT 'instrument',
    CONSTRAINT capability_evidence_direction_check CHECK (direction IN ('supports', 'contradicts')),
    CONSTRAINT capability_evidence_skill_check
        CHECK (skill IN ('traces_worked_steps', 'derives_forward', 'other')),
    CONSTRAINT capability_evidence_source_check CHECK (source IN ('click', 'instrument')),
    FOREIGN KEY (learner_id, interaction_id) REFERENCES interactions (learner_id, id)
);

CREATE INDEX idx_capability_evidence_claim ON capability_evidence (claim_id);
