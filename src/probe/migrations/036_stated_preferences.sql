-- probe: stated_preferences — Fix B's write side (the diagnostic in
-- this feature's own review confirmed the chain-rule prior beats a
-- preference stated as background prose; the fix is stating it as a
-- structural requirement on the answer, not a better place to put the
-- same fact -- see history_block.py and disambiguate.py's FinalAnswer
-- for the read side).
--
-- A SEPARATE table, not two new columns on `interactions` (despite
-- reading naturally as "on the interaction"): `interactions` is
-- immutable (migration 034's hard trigger) and this classification
-- runs OFF the critical path, same as the abstractor/outcome
-- classifier -- both of which are separate tables for the identical
-- reason (interaction_abstracts, turn_outcomes). A column literally on
-- `interactions` would require classifying synchronously, before the
-- row is even written, which is exactly the critical-path cost this
-- was asked to avoid.
--
-- One row per LEARNER-authored turn (never a system_option/click turn
-- -- there is no free text from the student to classify there), always
-- written regardless of outcome (has_preference = false is still a
-- recorded fact, not a gap), so the classifier's own coverage is
-- auditable the same way turn_outcomes' is.
--
-- Append-only: no delete method, no DELETE SQL, latest-per-interaction-id
-- resolved by `seq DESC` on read, same pattern as turn_outcomes /
-- interaction_abstracts (CLAUDE.md invariants 1/4/6-11's own family).

CREATE TABLE stated_preferences (
    id                  UUID PRIMARY KEY,
    interaction_id      UUID NOT NULL,
    learner_id          UUID NOT NULL,
    has_preference      BOOLEAN NOT NULL,
    -- The preference in the student's own terms (classifier-extracted,
    -- not paraphrased into a template) -- NULL whenever has_preference
    -- is false. Rendered into FinalAnswer's prompt as an imperative
    -- structural requirement, never as a background fact -- see
    -- history_block.py's own module docstring for why that distinction
    -- is load-bearing, not stylistic.
    stated_preference   TEXT,
    classifier_version  TEXT NOT NULL,
    created_at          TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    seq                 BIGSERIAL,
    FOREIGN KEY (learner_id, interaction_id) REFERENCES interactions (learner_id, id)
);

CREATE INDEX idx_stated_preferences_interaction
    ON stated_preferences (interaction_id, seq DESC);
-- The read path this whole table exists for: "the most recent TRUE
-- stated preference for this learner, across all sessions" -- a
-- partial index (has_preference only) since a false row is never the
-- target of this lookup.
CREATE INDEX idx_stated_preferences_learner_latest
    ON stated_preferences (learner_id, seq DESC)
    WHERE has_preference;
