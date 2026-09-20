"""population_patterns.py — the aggregation step of "raw interaction ->
abstract form -> aggregation -> multi-learner support -> retrieval."
No LLM, deterministic greedy threshold-attach clustering built on
vector_math.py's cosine_similarity/running_mean.
"""

from uuid import uuid4

import pytest

from probe.models import InteractionAbstract
from probe.population_patterns import (
    PopulationAggregationConfig,
    PopulationPatternStore,
    _cluster_abstracts,
    aggregate_population_patterns,
)


def _abstract(embedding, learner_id=None, form="a form") -> InteractionAbstract:
    return InteractionAbstract(
        interaction_id=uuid4(), learner_id=learner_id or uuid4(),
        abstract_form=form, abstract_embedding=embedding, generator_version="test-v1",
    )


def _vec(x: float) -> list[float]:
    return [x] + [0.0] * 767


def test_similar_abstracts_from_different_learners_cluster_together():
    abstracts = [_abstract(_vec(1.0), learner_id=uuid4()) for _ in range(5)]
    clusters = _cluster_abstracts(abstracts, PopulationAggregationConfig())
    assert len(clusters) == 1
    assert clusters[0].n == 5
    assert clusters[0].distinct_learner_count == 5


def test_dissimilar_abstracts_form_separate_clusters():
    close = [_abstract(_vec(1.0)) for _ in range(3)]
    far = [_abstract([0.0, 1.0] + [0.0] * 766) for _ in range(3)]
    clusters = _cluster_abstracts(close + far, PopulationAggregationConfig())
    assert len(clusters) == 2


def test_readability_requires_min_distinct_learners():
    config = PopulationAggregationConfig(min_distinct_learners=20)
    # 5 abstracts, but only 2 distinct learners -- must not be readable
    # regardless of raw count.
    learner_a, learner_b = uuid4(), uuid4()
    abstracts = [_abstract(_vec(1.0), learner_id=learner_a) for _ in range(3)] + [
        _abstract(_vec(1.0), learner_id=learner_b) for _ in range(2)
    ]
    clusters = _cluster_abstracts(abstracts, config)
    assert len(clusters) == 1
    assert clusters[0].is_readable(config) is False


def test_readability_requires_no_single_learner_dominance():
    """20 distinct learners clears the count gate, but if one of them
    contributed the overwhelming majority of the support, the pattern
    is effectively that one person's behavior -- must still fail."""
    config = PopulationAggregationConfig(min_distinct_learners=20, max_per_learner_share=0.25)
    dominant = uuid4()
    abstracts = [_abstract(_vec(1.0), learner_id=dominant) for _ in range(100)]
    abstracts += [_abstract(_vec(1.0), learner_id=uuid4()) for _ in range(20)]
    clusters = _cluster_abstracts(abstracts, config)
    assert len(clusters) == 1
    cluster = clusters[0]
    assert cluster.distinct_learner_count == 21  # >= 20, passes that gate
    assert cluster.max_per_learner_share > 0.25  # dominant learner still fails this one
    assert cluster.is_readable(config) is False


def test_a_cluster_clearing_both_gates_is_readable():
    config = PopulationAggregationConfig(min_distinct_learners=20, max_per_learner_share=0.25)
    abstracts = [_abstract(_vec(1.0), learner_id=uuid4()) for _ in range(25)]
    clusters = _cluster_abstracts(abstracts, config)
    assert clusters[0].is_readable(config) is True


def test_clustering_is_pure_and_deterministic():
    abstracts = [_abstract(_vec(1.0), learner_id=uuid4()) for _ in range(5)]
    c1 = _cluster_abstracts(abstracts, PopulationAggregationConfig())
    c2 = _cluster_abstracts(abstracts, PopulationAggregationConfig())
    assert [c.n for c in c1] == [c.n for c in c2]


@pytest.mark.asyncio(loop_scope="session")
async def test_end_to_end_aggregation_writes_only_readable_clusters(
    interaction_recorder, interaction_abstract_store, transcript, learner_store, clean_pool
):
    from probe.models import QuestionAuthor

    pattern_store = PopulationPatternStore(clean_pool)
    config = PopulationAggregationConfig(min_distinct_learners=5, max_per_learner_share=0.5)

    # 6 distinct learners, all with a similar abstract -- should end up
    # readable at these thresholds.
    for i in range(6):
        learner = await learner_store.create()
        session_id = await transcript.create_session(learner.id)
        interaction = await interaction_recorder.record(
            learner_id=learner.id, session_id=session_id, turn_number=0,
            question_text="q", question_author=QuestionAuthor.LEARNER,
            originating_question=None, did_branch=False, response_text="a",
        )
        await interaction_abstract_store.append(
            InteractionAbstract(
                interaction_id=interaction.id, learner_id=learner.id,
                abstract_form="chose the worked example over the stated rule",
                abstract_embedding=_vec(1.0), generator_version="test-v1",
            )
        )

    # One learner, alone, with a totally different abstract -- must
    # never surface (only 1 distinct learner).
    lone_learner = await learner_store.create()
    lone_session = await transcript.create_session(lone_learner.id)
    lone_interaction = await interaction_recorder.record(
        learner_id=lone_learner.id, session_id=lone_session, turn_number=0,
        question_text="q", question_author=QuestionAuthor.LEARNER,
        originating_question=None, did_branch=False, response_text="a",
    )
    await interaction_abstract_store.append(
        InteractionAbstract(
            interaction_id=lone_interaction.id, learner_id=lone_learner.id,
            abstract_form="an isolated, unshared pattern",
            abstract_embedding=[0.0, 1.0] + [0.0] * 766, generator_version="test-v1",
        )
    )

    written = await aggregate_population_patterns(interaction_abstract_store, pattern_store, config)
    forms = [p.abstract_form for p in written]
    assert "chose the worked example over the stated rule" in forms
    assert "an isolated, unshared pattern" not in forms

    readable = next(p for p in written if p.abstract_form == "chose the worked example over the stated rule")
    assert readable.distinct_learner_count == 6
    assert readable.support_count == 6


@pytest.mark.asyncio(loop_scope="session")
async def test_aggregation_with_no_abstracts_writes_nothing(
    interaction_abstract_store, clean_pool
):
    pattern_store = PopulationPatternStore(clean_pool)
    written = await aggregate_population_patterns(interaction_abstract_store, pattern_store)
    assert written == []


@pytest.mark.asyncio(loop_scope="session")
async def test_population_pattern_store_append_and_count_readable(clean_pool):
    from probe.models import PopulationPattern

    store = PopulationPatternStore(clean_pool)
    assert await store.count_readable() == 0
    await store.append(
        PopulationPattern(
            abstract_form="x", embedding=_vec(1.0), support_count=30,
            distinct_learner_count=25, max_per_learner_share=0.1,
        )
    )
    await store.append(
        PopulationPattern(
            abstract_form="y", embedding=_vec(1.0), support_count=10,
            distinct_learner_count=3, max_per_learner_share=0.5,
        )
    )
    assert await store.count_readable() == 1
