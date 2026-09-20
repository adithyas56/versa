"""Unit tests for the centralized learner/status filter primitives
(moss_client.py). The filter DSL translation and the in-process matcher
are the one place cross-learner isolation is enforced (§11.3/§34.5), so
they are tested directly, apart from the full retrieval path.
"""

from probe.moss_client import MossConfig, _matches_filter, _to_moss_filter


def test_to_moss_filter_builds_and_eq_dsl():
    dsl = _to_moss_filter({"learner_id": "learner_a", "status": "active"})
    assert dsl == {
        "$and": [
            {"field": "learner_id", "condition": {"$eq": "learner_a"}},
            {"field": "status", "condition": {"$eq": "active"}},
        ]
    }


def test_to_moss_filter_none_for_empty():
    assert _to_moss_filter(None) is None
    assert _to_moss_filter({}) is None


def test_matches_filter_requires_all_fields():
    md = {"learner_id": "learner_a", "status": "active"}
    assert _matches_filter(md, {"learner_id": "learner_a", "status": "active"})
    assert not _matches_filter(md, {"learner_id": "learner_b"})
    assert not _matches_filter(md, {"status": "retired"})


def test_matches_filter_missing_key_is_a_non_match():
    assert not _matches_filter({"learner_id": "a"}, {"status": "active"})


def test_config_from_env_defaults_are_stub_first():
    cfg = MossConfig.from_env({})
    assert cfg.engine == "moss"
    assert cfg.required is False
    assert cfg.has_credentials is False
    assert cfg.index_name == "versa-learner-memory"
    assert cfg.top_k == 5


def test_config_from_env_reads_credentials_and_knobs():
    cfg = MossConfig.from_env(
        {
            "MOSS_PROJECT_ID": "pid",
            "MOSS_PROJECT_KEY": "pkey",
            "MOSS_TOP_K": "8",
            "MOSS_MIN_SCORE": "0.3",
            "MOSS_REQUIRED": "true",
            "VERSA_RETRIEVAL_ENGINE": "stub",
        }
    )
    assert cfg.has_credentials is True
    assert cfg.top_k == 8
    assert cfg.min_score == 0.3
    assert cfg.required is True
    assert cfg.engine == "stub"
