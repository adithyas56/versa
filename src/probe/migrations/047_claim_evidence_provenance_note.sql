-- Nullable, free-text annotation for a claim_evidence row known (by
-- human diagnosis, after the fact) to have come from a broken test
-- harness rather than genuine learner behavior. Does NOT change
-- confidence, status, or eligibility -- the row still counts in
-- production exactly as before; this only lets analysis queries
-- (score-predictions --exclude-contaminated) separate the two
-- pictures and lets the review surface show which rows are which.
ALTER TABLE claim_evidence ADD COLUMN provenance_note TEXT NULL;
