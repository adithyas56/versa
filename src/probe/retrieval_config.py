"""Every tunable constant `retrieval.py` uses, named and justified in
one place — same "config value, not a magic number" discipline as
`MemoryConfig`/`ValueFunctionConfig` elsewhere in this codebase.
Weights are meant to be tuned by feel against real retrieval behavior,
not derived from anything; this module exists so that tuning is a
one-place edit, not a hunt through `retrieval.py`'s scoring code.
"""

from __future__ import annotations

from pydantic import BaseModel


class RetrievalWeights(BaseModel):
    """stage3_rerank's scoring function is:

        score = semantic_similarity_weight   * similarity
              + recency_weight               * exp(-age_days / recency_half_life_days)
              + contradicted_outcome_bonus    * (1 if outcome == CONTRADICTED_INTENT else 0)
              + support_weight                * normalized_support
              - topic_abstract_disagreement_penalty * max(0, question_sim - abstract_sim)

    `normalized_support` is `n_supported_claims` (population scope's
    `support_count`, 0 for an individual personal-scope interaction)
    divided by `support_normalization_cap` and clamped to [0, 1], so
    one heavily-aggregated population pattern can't dominate the score
    by raw magnitude alone.

    `contradicted_outcome_bonus` is deliberately positive and sizable:
    a turn where the system read the learner wrong carries more
    information about them than one where it read them right (see
    interactions.py's module docstring) — this is not a penalty term,
    it is a relevance signal.

    `topic_abstract_disagreement_penalty` targets a specific, observed
    failure: a real 12-turn session's personal-scope retrieval surfaced
    a "financial derivatives" turn for a calculus "integrals" query at
    question_sim=0.616 -- pure lexical overlap on the word
    "derivatives," structurally unrelated. The SAME run's top-4 also
    surfaced a biology turn via the abstract key ("asked for a
    comparison between two related concepts") on that same calculus
    query -- structurally similar, different subject, exactly the
    cross-topic transfer this whole abstract-key mechanism exists to
    produce. Both are "cross-topic," but only the first is a defect;
    collapsing them into one number was the wrong read on that result.
    Correct fix in principle: penalize a candidate that matches
    strongly on the question key but weakly on the abstract key (see
    stage3_rerank's own docstring for the one-directional max(0, ...)
    that leaves the abstract-transfer case alone).

    Checked, not assumed: re-running this exact pair on the real data
    once `RecallHit.counterpart_similarity` existed to measure it found
    question_sim=0.616/abstract_sim=0.601 for the financial-derivatives
    turn -- a gap of 0.015, not the large one this penalty is built to
    catch. The biology turn's own gap runs the other way
    (question_sim=0.486/abstract_sim=0.624) and is correctly untouched.
    At 0.4, the derivatives turn's score barely moves (1.416 -> 1.409)
    and stays ranked first -- this specific false positive is NOT fixed
    by this mechanism on this data. Root cause, not a threshold problem:
    this session's abstractor output is stylistically homogeneous
    (nearly every abstract opens "Asked ... received/responded ..."),
    which correlates question_sim and abstract_sim across unrelated
    turns and compresses exactly the gap this penalty depends on.
    Kept rather than reverted -- it is a sound, harmless mechanism
    (near-zero penalty whenever the gap is genuinely small, as
    confirmed here) that may still separate a future case with a real
    gap, but it did not solve the one case it was written for, and that
    is itself the finding: the fix belongs in abstraction diversity, if
    anywhere, not in another retrieval weight.
    """

    semantic_similarity_weight: float = 1.0
    recency_weight: float = 0.3
    recency_half_life_days: float = 14.0
    contradicted_outcome_bonus: float = 0.5
    support_weight: float = 0.2
    support_normalization_cap: float = 50.0
    topic_abstract_disagreement_penalty: float = 0.4


class RetrievalQuotas(BaseModel):
    """Fixed quotas, assembled from two independently-ranked pools —
    NOT one merged pool with the top 5 taken overall. Population
    aggregates carry higher support and would crowd out personal
    continuity if pooled, which is exactly what makes a session feel
    like it remembers a specific learner (see retrieval.py's module
    docstring)."""

    personal: int = 4
    population: int = 1


class RetrievalConfig(BaseModel):
    weights: RetrievalWeights = RetrievalWeights()
    quotas: RetrievalQuotas = RetrievalQuotas()
    # Candidates considered per stage1 filter pass, before stage2's
    # own top-50-per-key recall narrows it further.
    stage1_limit: int = 500
    stage2_top_n: int = 50
    # pgvector HNSW search-time recall/speed knob (build-time m/
    # ef_construction are fixed at index-creation, migration 034).
    hnsw_ef_search: int = 40
    # Population pattern readability gate (migration 034 / models.py's
    # PopulationPattern docstring) — duplicated here as the config
    # knob retrieval actually reads at query time, rather than only
    # documented where the table is defined.
    population_min_distinct_learners: int = 20
    population_max_learner_share: float = 0.25

    # Cosine similarity two questions' embeddings must clear to count
    # as "the same subject" -- used by interactions.py's per-turn
    # entry_state computation (continuing/stuck_repeat/returning_
    # after_gap/topic_switch), a direct pairwise comparison with no
    # clustering, no centroid, and no accumulation of any kind. Lives
    # here, not in interactions.py, because it is a retrieval-adjacent
    # tunable of the same kind as the weights above, not a structural
    # constant of the recorder.
    #
    # 0.545 is NOT carried over from the deleted topic-clustering
    # mechanism's own last attempt at this same number -- that number
    # was measured for the wrong algorithm (a static all-pairs matrix
    # calibrated against a RUNNING CENTROID's drifting similarity,
    # which is a different distribution and is exactly what caused
    # that mechanism's real collapse). It is the same VALUE because it
    # now measures the right thing: hand-labeled same-subject (n=14)
    # vs. different-subject (n=31) question pairs from a real 12-turn
    # session, raw pairwise cosine similarity, no averaging anywhere --
    # same-subject floor 0.548, different-subject ceiling 0.539, zero
    # errors anywhere in [0.540, 0.546]. This field's comparison IS
    # that static pairwise comparison (no centroid drift possible,
    # because nothing here is ever averaged), so the calibration
    # applies directly instead of by analogy. Biased to the high end of
    # that window for the same asymmetry reason as before: a false
    # "same subject" claim is worse than a missed one.
    #
    # Still n=12, one real session -- expected to be retuned as more
    # real data accumulates, same as every other threshold in this
    # codebase.
    same_subject_threshold: float = 0.545
