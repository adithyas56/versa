-- probe: the instrument layer's contract table (instruments.py) --
-- append-only, same convention as every other store in this codebase:
-- no DELETE, no UPDATE of any column after insert (a contract is
-- fully immutable; there is no terminal-status transition to model
-- here at all, unlike claims/branches/options).
--
-- `target_claim_id` is nullable: a contract may target an EXISTING
-- claim (evidence attaches directly, no matching needed) or, in a
-- future not built here ("no instrument generation yet"), a
-- candidate trait that doesn't have a claims row yet. This build only
-- exercises the non-null path -- see write_instrument_evidence's own
-- docstring.
--
-- The three predicate columns are structured JSON arrays of
-- {"event": ..., "field": ..., "equals"|"not_equals": ...} objects,
-- not prose, so `interpret_instrument` can be a pure, deterministic
-- function with no LLM in the loop. `uninformative_when` is NOT NULL
-- and, per its own CHECK below, may never be an empty array: every
-- contract must name at least one outcome it cannot read (an
-- abandoned attempt, a timeout, a response outside what the contract
-- anticipated) -- the alternative is interpretation guessing a
-- direction for data it was never designed to read, which is where
-- confabulation lives.
CREATE TABLE interaction_contracts (
    id                  UUID PRIMARY KEY,
    target_claim_id     UUID REFERENCES claims (id) ON DELETE RESTRICT,
    target_axis         TEXT NOT NULL,
    target_value        TEXT NOT NULL,
    primitive           TEXT NOT NULL,
    supports_when       JSONB NOT NULL,
    contradicts_when    JSONB NOT NULL,
    uninformative_when  JSONB NOT NULL,
    generator_version   TEXT NOT NULL,
    created_at          TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    CONSTRAINT interaction_contracts_axis_check
        CHECK (
            target_axis IN (
                'concrete_general', 'scope_narrow_broad', 'rigor_intuition',
                'mechanism_procedure', 'worked_steps_result', 'analogy_formal',
                'single_example_pattern', 'forward_derivation_backward_verification',
                'brevity_depth', 'structured_narrative'
            )
        ),
    CONSTRAINT interaction_contracts_value_check
        CHECK (
            target_value IN (
                'concrete_before_abstract', 'rule_before_example', 'wants_steps_shown',
                'prefers_brevity', 'wants_analogies', 'no_analogies', 'other'
            )
        ),
    CONSTRAINT interaction_contracts_primitive_check
        CHECK (primitive IN ('choose', 'order', 'locate', 'adjust', 'predict', 'construct')),
    CONSTRAINT interaction_contracts_uninformative_nonempty
        CHECK (jsonb_typeof(uninformative_when) = 'array' AND jsonb_array_length(uninformative_when) > 0)
);
