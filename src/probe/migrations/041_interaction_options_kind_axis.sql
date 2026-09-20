-- probe: interaction_options gains `kind`/`axis` -- the disambiguation
-- gate review's own follow-up: DisambiguationOptions' kind/axis
-- decision (models.AmbiguityKind/ApproachAxis) was previously only
-- ever captured in node_calls.output_json (CLAUDE.md invariant 2's
-- automatic audit capture), which is enough for a one-off debugging
-- read but not for a extraction pipeline that needs to query "what
-- axis was live for this option, per learner, at scale" as a normal
-- read path. Claim extraction (migration 042) reads this column
-- directly -- without it, extraction would have to reverse-engineer
-- the axis from option text, which is exactly the guessing this
-- whole mechanism exists to remove.
--
-- Same value on every row in one option set (an approach-kind set's
-- two options share one axis; a subject-kind set's options all carry
-- axis = NULL) -- a property of the SET, denormalized onto each row
-- the same way `shown_position`/`option_text` already are, so a
-- single clicked option is self-sufficient without a join back to
-- node_calls.
--
-- interaction_options is not under the interactions immutability
-- trigger (see migration 034's own header) -- was_selected/
-- selection_timestamp are already populated on a later turn than
-- creation, so adding nullable columns here is unremarkable.
--
-- TEXT + CHECK, not a Postgres ENUM -- same choice migration 040 made
-- for interactions.domain, for the same reason (no CREATE TYPE
-- ceremony for a value this codebase already treats as a plain closed
-- vocabulary in Python).

ALTER TABLE interaction_options
    ADD COLUMN kind TEXT,
    ADD COLUMN axis TEXT;

ALTER TABLE interaction_options
    ADD CONSTRAINT interaction_options_kind_check
        CHECK (kind IS NULL OR kind IN ('subject', 'approach')),
    ADD CONSTRAINT interaction_options_axis_check
        CHECK (
            axis IS NULL OR axis IN (
                'concrete_general', 'scope_narrow_broad', 'rigor_intuition',
                'mechanism_procedure', 'worked_steps_result', 'analogy_formal',
                'single_example_pattern', 'forward_derivation_backward_verification',
                'brevity_depth', 'structured_narrative'
            )
        );
