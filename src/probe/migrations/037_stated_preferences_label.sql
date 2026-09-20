-- probe: stated_preferences gains a closed-vocabulary `label` column
-- alongside the raw extracted text -- see StatedPreferenceLabel's own
-- docstring (models.py) for why both exist side by side.
--
-- The first version of this feature rendered FinalAnswer's structural
-- requirement straight off the raw extracted text ("I always want the
-- concrete case before the abstract rule"). A live re-run found this
-- reads as a DESCRIPTION of a preference, not a REQUIREMENT on output
-- shape -- a model can read a description, agree with it, and still
-- generate its own default answer shape. The version that actually
-- flipped the answer in testing was a hand-written IMPERATIVE naming
-- what to avoid explicitly. `label` is what makes that mapping
-- possible: the classifier still extracts and stores the raw text
-- verbatim (the faithful record of what the learner said), but ALSO
-- picks one of a fixed set of labels, which interaction_nodes.
-- render_structural_requirement maps to a hand-written imperative
-- instead of converting the raw text into a requirement on the fly.
--
-- Nullable: a `has_preference = false` row never gets a label (there
-- is nothing to categorize), same as `stated_preference` staying NULL
-- on those rows.

CREATE TYPE stated_preference_label AS ENUM (
    'concrete_before_abstract', 'rule_before_example', 'wants_steps_shown',
    'prefers_brevity', 'wants_analogies', 'no_analogies', 'other'
);

ALTER TABLE stated_preferences
    ADD COLUMN label stated_preference_label;
