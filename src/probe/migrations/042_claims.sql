-- probe: claims / claim_evidence -- the claim layer, gated on the
-- disambiguation-options audit (migrations 038-041) actually passing:
-- extraction is anchored on prediction error, not "read every
-- session," specifically so this table stays sparse and each row is
-- attributable to a real surprise, never confabulation that reads
-- like a finding. See interaction_nodes.ClaimExtractor /
-- claims.py for the write side and claims.compute_confidence for the
-- read side.
--
-- Both append-only in the same sense as every other table in this
-- family (CLAUDE.md invariants 1, 4, 6-11): no delete method, no
-- DELETE SQL anywhere. `claims.statement`/`test`/`value` are set once
-- at creation and never edited -- the ONLY fields that ever change via
-- UPDATE are `confidence` (a periodically refreshed snapshot of
-- claims.compute_confidence's own pure-function output, recomputed
-- whenever new evidence lands) and `status` (candidate -> promoted or
-- candidate/promoted -> contradicted), the identical "resurrection
-- over deletion, status transition via UPDATE" pattern
-- ThinkingStyleStore.confirm()/DisambiguationStore.mark_matched()
-- already use elsewhere in this codebase. A contradicted claim is
-- never deleted or hidden -- it stays fully visible, with its full
-- evidence history, on the review surface; nothing here can be
-- re-derived correctly if a step is silently discarded.
--
-- `claim_evidence` is fully append-only with no mutable field at all
-- (not even a status) -- a claim's status changes, the evidence that
-- justified the change does not.

CREATE TABLE claims (
    id                  UUID PRIMARY KEY,
    learner_id          UUID NOT NULL REFERENCES learners (id) ON DELETE RESTRICT,
    -- Human-readable prose ("this learner wants a concrete case before
    -- the general rule"). Never edited after creation -- a claim that
    -- turns out to be mis-stated is superseded by a NEW claims row
    -- (matched against by reconciliation's own semantic-similarity
    -- step next time), not rewritten in place.
    statement           TEXT NOT NULL,
    -- The falsifiable prediction this claim licenses, over a FUTURE
    -- option set: a situation plus a predicted choice -- e.g. "given a
    -- concrete_general axis choice, they will pick the concrete
    -- option." Extraction rejects any candidate without one (see
    -- claims.py) rather than storing an untestable claim.
    test                TEXT NOT NULL,
    -- The specific StatedPreferenceLabel value (interaction_nodes.py)
    -- this claim asserts -- what makes rendering a deterministic
    -- label -> imperative lookup, reusing
    -- interaction_nodes.render_structural_requirement verbatim,
    -- exactly the form that worked 5/5 for stated preferences (see
    -- claims.render_claim_constraint).
    value               TEXT NOT NULL,
    -- A cached SNAPSHOT of claims.compute_confidence's own output,
    -- refreshed via UPDATE every time new evidence is reconciled onto
    -- this claim -- never the sole authority (that's the pure
    -- function, unit-tested independently), but fast to read on the
    -- review surface without recomputing from every claim_evidence row
    -- on every page view.
    confidence          DOUBLE PRECISION NOT NULL,
    source              TEXT NOT NULL CHECK (source IN ('stated', 'inferred', 'corrected')),
    write_policy        TEXT NOT NULL CHECK (write_policy IN ('locked', 'slow_drift', 'fast_decay')),
    -- Free-form scoping context (e.g. which domain/topic family this
    -- claim's test is meaningful within) -- jsonb since its shape
    -- isn't fixed across claims the way the other columns are.
    context_scope       JSONB NOT NULL DEFAULT '{}',
    status              TEXT NOT NULL DEFAULT 'candidate'
                            CHECK (status IN ('candidate', 'promoted', 'contradicted')),
    -- Symmetric-comparison embedding of `statement` -- what
    -- reconciliation's semantic-similarity match runs against for this
    -- learner's existing claims (see ThinkingStyleStore.search_similar
    -- for the identical TASK_SIMILARITY precedent this reuses).
    statement_embedding halfvec(768) NOT NULL,
    created_at          TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at          TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX idx_claims_learner ON claims (learner_id);
CREATE INDEX idx_claims_learner_status ON claims (learner_id, status);
CREATE INDEX idx_claims_embedding_hnsw
    ON claims USING hnsw (statement_embedding halfvec_cosine_ops)
    WITH (m = 16, ef_construction = 64);

CREATE TABLE claim_evidence (
    id                          UUID PRIMARY KEY,
    claim_id                    UUID NOT NULL REFERENCES claims (id) ON DELETE RESTRICT,
    learner_id                  UUID NOT NULL,
    interaction_id              UUID NOT NULL,
    direction                   TEXT NOT NULL CHECK (direction IN ('supports', 'contradicts')),
    -- The domain-neutral subject-area this episode was about (not the
    -- literal question text) -- what the >=2-distinct-topics
    -- promotion gate counts over, so a claim confirmed five times in
    -- one subject never promotes on that alone (see claims.py's own
    -- module docstring for why that gate is "the important half").
    topic                       TEXT NOT NULL,
    session_id                  UUID NOT NULL REFERENCES sessions (id) ON DELETE RESTRICT,
    -- Did the claim's own `test` actually get a chance to fire on this
    -- turn (a genuinely relevant option set was shown), and could the
    -- outcome have gone the other way (a real contradicting option was
    -- actually on offer)? Both set explicitly, never inferred later --
    -- a turn where the claim's test couldn't have failed must never
    -- inflate confidence the way a turn where it genuinely could have
    -- failed, and didn't, does.
    test_fired                  BOOLEAN NOT NULL,
    contradiction_was_possible  BOOLEAN NOT NULL,
    created_at                  TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    seq                         BIGSERIAL,
    -- Points at interactions by reference only -- never copies
    -- question/response content (see this migration's own header).
    FOREIGN KEY (learner_id, interaction_id) REFERENCES interactions (learner_id, id)
);

CREATE INDEX idx_claim_evidence_claim ON claim_evidence (claim_id, seq);
CREATE INDEX idx_claim_evidence_interaction ON claim_evidence (interaction_id);
