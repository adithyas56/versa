-- probe: interaction_contracts becomes POLYMORPHIC across the
-- preference/capability split (probe/capability.py, migration 053) --
-- the fix for the same incident 053's own header describes. A
-- PREFERENCE contract keeps the original shape (target_axis +
-- target_value, target_claim_id names a `claims` row); a PERFORMANCE
-- contract now sets target_skill instead (target_claim_id names a
-- `capability_claims` row instead). Never both -- enforced below at
-- the database (the authoritative backstop) and in
-- InteractionContract's own model validator (models.py, fail-fast).
--
-- target_claim_id's FK to claims(id) is dropped for exactly this
-- reason: it is polymorphic across two tables now, so a literal FK
-- can no longer describe it. Which table it points into is the
-- caller's responsibility to get right from `measures` -- there is no
-- database-level way to enforce a foreign key against "one of two
-- tables, chosen by another column" in Postgres without a much
-- heavier trigger-based scheme, which is more machinery than this
-- foundation-stage layer has earned yet.
--
-- Existing rows (all locate/predict demo/test data, pre-fix) keep
-- their literal current shape (target_axis/target_value set) under
-- the 'preference' default below -- disposable test data, not
-- repaired in place, since going forward the demo contracts declare
-- measures='performance' explicitly at creation. The default is
-- dropped afterward so every future INSERT must state `measures`
-- itself.
ALTER TABLE interaction_contracts DROP CONSTRAINT interaction_contracts_target_claim_id_fkey;

ALTER TABLE interaction_contracts ALTER COLUMN target_axis DROP NOT NULL;
ALTER TABLE interaction_contracts ALTER COLUMN target_value DROP NOT NULL;

ALTER TABLE interaction_contracts ADD COLUMN measures TEXT NOT NULL DEFAULT 'preference';
ALTER TABLE interaction_contracts ADD COLUMN target_skill TEXT;

ALTER TABLE interaction_contracts
    ADD CONSTRAINT interaction_contracts_measures_check
        CHECK (measures IN ('preference', 'performance'));

ALTER TABLE interaction_contracts
    ADD CONSTRAINT interaction_contracts_target_skill_check
        CHECK (target_skill IS NULL OR target_skill IN ('traces_worked_steps', 'derives_forward', 'other'));

ALTER TABLE interaction_contracts
    ADD CONSTRAINT interaction_contracts_measures_shape_check
        CHECK (
            (measures = 'preference' AND target_axis IS NOT NULL AND target_value IS NOT NULL AND target_skill IS NULL)
            OR
            (measures = 'performance' AND target_skill IS NOT NULL AND target_axis IS NULL AND target_value IS NULL)
        );

ALTER TABLE interaction_contracts ALTER COLUMN measures DROP DEFAULT;
