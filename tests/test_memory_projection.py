"""Unit tests for the LearnerFact -> Moss document projection
(memory_index.py). Pure: no Postgres, no Moss SDK, no network — the
projection is a pure function (VERSA build context §37 unit tests).
"""

from uuid import uuid4

from probe.embeddings import EMBEDDING_DIM
from probe.memory_index import (
    MemoryDocument,
    project_fact,
    project_facts,
)
from probe.models import LearnerFact, LearnerFactType


def _fact(fact_type: LearnerFactType, learner_id=None) -> LearnerFact:
    return LearnerFact(
        learner_id=learner_id or uuid4(),
        session_id=uuid4(),
        turn_index=3,
        fact_type=fact_type,
        situation="how to explain recursion",
        resolution="a real-world example first",
        embedding=[0.1] * EMBEDDING_DIM,
        source_turn_id=uuid4(),
    )


def test_projection_has_required_filter_metadata():
    fact = _fact(LearnerFactType.BRANCH_RESOLUTION)
    doc = project_fact(fact)
    # §12.4: retrieval filters on learner_id + status; both always present.
    assert doc.metadata["learner_id"] == str(fact.learner_id)
    assert doc.metadata["status"] == "active"
    assert doc.metadata["source_interaction_id"] == str(fact.source_turn_id)


def test_projection_id_is_prefixed_and_derived_from_fact():
    fact = _fact(LearnerFactType.DIRECT_ANSWER)
    doc = project_fact(fact)
    assert doc.id == f"mem_{fact.id}"


def test_branch_resolution_maps_to_ambiguity_resolution_memory_type():
    doc = project_fact(_fact(LearnerFactType.BRANCH_RESOLUTION))
    assert doc.metadata["memory_type"] == "ambiguity_resolution"
    # An explicit branch click is strong evidence -> confidence 1.0 (§10.2).
    assert doc.metadata["confidence"] == "1.0"


def test_direct_answer_maps_to_recent_learning_memory_type():
    doc = project_fact(_fact(LearnerFactType.DIRECT_ANSWER))
    assert doc.metadata["memory_type"] == "recent_learning"
    assert doc.metadata["confidence"] == "0.7"


def test_projected_text_is_a_natural_language_statement_not_a_label():
    # §11.4: the indexed text must read as a sentence a semantic query can
    # match, containing the situation and resolution, not a bare label.
    doc = project_fact(_fact(LearnerFactType.BRANCH_RESOLUTION))
    assert "recursion" in doc.text
    assert "real-world example" in doc.text
    assert len(doc.text.split()) > 5


def test_all_metadata_values_are_strings():
    # Moss metadata is Dict[str, str]; confidence in particular must be
    # stringified, never a float.
    doc = project_fact(_fact(LearnerFactType.DIRECT_ANSWER))
    assert all(isinstance(v, str) for v in doc.metadata.values())


def test_projection_carries_the_durable_embedding():
    fact = _fact(LearnerFactType.DIRECT_ANSWER)
    doc = project_fact(fact)
    assert doc.embedding == list(fact.embedding)


def test_project_facts_batches():
    facts = [_fact(LearnerFactType.DIRECT_ANSWER) for _ in range(3)]
    docs = project_facts(facts)
    assert len(docs) == 3
    assert all(isinstance(d, MemoryDocument) for d in docs)
