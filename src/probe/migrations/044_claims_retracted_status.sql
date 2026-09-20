-- probe: claims gains a 'retracted' status -- a THIRD terminal state,
-- distinct from 'contradicted'. A live run found a claim founded on a
-- subject-kind (uncontestable) pick sitting honestly at the 0.5 prior,
-- then absorb a genuinely eligible contradicting row via search_similar
-- and close terminally -- real evidence spent killing a premise that
-- was never legitimately established, when it belonged to a real
-- claim instead. 'retracted' marks exactly that case (founding
-- evidence was ineligible) so reconcile_candidate can exclude these
-- from matching the same way it already excludes 'contradicted' ones,
-- stopping them from competing for evidence at all going forward.
--
-- Same resurrection-over-deletion precedent as every other status
-- transition in this codebase: a retracted claim is never deleted,
-- just marked via UPDATE and excluded from live matching, exactly
-- like a contradicted one.

ALTER TABLE claims DROP CONSTRAINT claims_status_check;

ALTER TABLE claims
    ADD CONSTRAINT claims_status_check
        CHECK (status IN ('candidate', 'promoted', 'contradicted', 'retracted'));
