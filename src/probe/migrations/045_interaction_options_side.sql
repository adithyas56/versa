-- probe: interaction_options gains `side` -- which pole of `axis`
-- (migration 041) THIS specific option represents, "first" or
-- "second", matching the axis enum's own name order (e.g. for
-- concrete_general, "first" is the concrete pole). Unlike kind/axis,
-- which repeat across every row in one option set, side is PER-OPTION
-- -- the two options in an approach-kind set always carry opposite
-- sides.
--
-- Closes a repeat incident: three separate times, a downstream reader
-- (claim extraction's own reconciliation test harness) inferred which
-- option was which side by matching keywords against option TEXT, and
-- each time a real generated phrasing didn't contain the expected
-- keyword (or, worse, contained a keyword from the OTHER side's list
-- in an unrelated sense -- "comparison" meaning an array-comparison
-- operation, not a rhetorical analogy) the inference silently picked
-- the wrong option. DisambiguationOptions already decides this at
-- generation time (it has to, to write two options that differ ONLY
-- along the chosen axis); this persists that decision as data instead
-- of asking every downstream reader to re-derive it from prose.

ALTER TABLE interaction_options
    ADD COLUMN side TEXT;

ALTER TABLE interaction_options
    ADD CONSTRAINT interaction_options_side_check
        CHECK (side IS NULL OR side IN ('first', 'second'));
