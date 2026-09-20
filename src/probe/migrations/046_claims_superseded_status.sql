ALTER TABLE claims DROP CONSTRAINT claims_status_check;
ALTER TABLE claims
    ADD CONSTRAINT claims_status_check
        CHECK (status IN ('candidate', 'promoted', 'contradicted', 'retracted', 'superseded'));

ALTER TABLE claims ADD COLUMN superseded_by UUID REFERENCES claims(id);
