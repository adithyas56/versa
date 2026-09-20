-- probe: claim_evidence gains `axis` -- a live review of the claim
-- layer's first real run found `compute_confidence` collapsing
-- evidence cells by (session, topic) alone, when the actual repeat-
-- observation concern is (session, axis): two confirmations of the
-- SAME axis in the SAME session are one observation regardless of
-- what topic label each got, since a chatty session testing one axis
-- twice under two topic labels is not two independent samples of the
-- learner's behavior. `topic` stays -- it is still what the
-- >=2-distinct-topics promotion gate counts over -- this is an
-- ADDITIONAL column, not a replacement.
--
-- Nullable: evidence with no live option-set axis behind it (a
-- stated-preference or contradicted_intent trigger) has none to
-- record; `compute_confidence` falls back to (session, topic) for
-- those rows. Same TEXT + CHECK convention as migration 041's
-- interaction_options.axis, for the identical closed vocabulary.

ALTER TABLE claim_evidence
    ADD COLUMN axis TEXT;

ALTER TABLE claim_evidence
    ADD CONSTRAINT claim_evidence_axis_check
        CHECK (
            axis IS NULL OR axis IN (
                'concrete_general', 'scope_narrow_broad', 'rigor_intuition',
                'mechanism_procedure', 'worked_steps_result', 'analogy_formal',
                'single_example_pattern', 'forward_derivation_backward_verification',
                'brevity_depth', 'structured_narrative'
            )
        );
