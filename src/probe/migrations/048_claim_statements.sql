-- Append-only rendering history for a claim's statement, same pattern
-- as turn_outcomes (034_interactions.sql): claims.statement is set
-- once at creation and never edited (Claim's own docstring). This
-- table is what accumulates as evidence grows and generalizes past
-- the founding episode's domain -- axis and value stay the claim's
-- immutable identity; the prose describing them against accumulated
-- evidence is what regenerates. Readers resolve to the latest row per
-- claim_id; the first row (written at claim-creation time) preserves
-- the original wording so the drift is itself visible on request.
CREATE TABLE claim_statements (
    id                          UUID PRIMARY KEY,
    claim_id                    UUID NOT NULL REFERENCES claims (id) ON DELETE RESTRICT,
    statement                   TEXT NOT NULL,
    derived_from_evidence_count INTEGER NOT NULL,
    generator_version           TEXT NOT NULL,
    created_at                  TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    seq                         BIGSERIAL
);

CREATE INDEX idx_claim_statements_claim ON claim_statements (claim_id, seq DESC);
