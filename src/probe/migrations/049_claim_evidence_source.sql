-- Distinguishes click-derived evidence (the existing path: a learner's
-- pick among disambiguation options, or a stated preference) from
-- instrument-derived evidence (a purpose-built interaction whose
-- contract deterministically interprets what happened). Needed so
-- score_predictions.py can check the two populations separately before
-- instrument evidence is trusted alongside click evidence -- see
-- instruments.py's own module docstring. Defaults to 'click' so every
-- row written before this column existed reads correctly with no
-- backfill needed.
ALTER TABLE claim_evidence ADD COLUMN source TEXT NOT NULL DEFAULT 'click';
ALTER TABLE claim_evidence
    ADD CONSTRAINT claim_evidence_source_check CHECK (source IN ('click', 'instrument'));
