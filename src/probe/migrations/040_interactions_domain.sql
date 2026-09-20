-- probe: interactions gains a `domain` column -- the ONE deliberate,
-- explicitly requested exception to the domain switch's otherwise
-- "prompts only" scope (see domain_config.py's own module docstring
-- for the full boundary). Not added because a general-domain run
-- failed to function without it -- added so an education run and a
-- general run for the same learner_id never blend together in
-- retrieval or in any later measurement, which would make either run
-- uninterpretable regardless of prompt quality.
--
-- Added at the END of the interactions column list (ALTER TABLE ADD
-- COLUMN always appends). Because of that, `interactions_current`
-- (migration 034) cannot be updated with CREATE OR REPLACE VIEW: its
-- `i.*` expansion would gain `domain` in the MIDDLE of the view's
-- overall output list (before the LEFT JOIN LATERAL columns), shifting
-- every later column's ordinal position -- Postgres rejects a REPLACE
-- that does that. A plain DROP + CREATE is safe here: the view carries
-- no data of its own, and nothing else (no rule, no other view)
-- depends on it.
--
-- Default 'education' for every existing row: this pipeline only ever
-- ran in tutoring mode before this migration, so nothing in the
-- historical data was ever actually general-domain.

ALTER TABLE interactions
    ADD COLUMN domain TEXT NOT NULL DEFAULT 'education';

ALTER TABLE interactions
    ADD CONSTRAINT interactions_domain_check CHECK (domain IN ('education', 'general'));

-- The predicate retrieval.stage1_filter and InteractionStore.
-- get_recent_for_learner add whenever a domain is given -- always
-- alongside the existing learner_id predicate, never alone (domain
-- alone is not selective enough to be worth an index by itself).
CREATE INDEX idx_interactions_learner_domain ON interactions (learner_id, domain);

DROP VIEW interactions_current;

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
