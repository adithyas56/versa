"""method_capabilities.py -- declared data, not behavior. These tests
check the table's own internal consistency and completeness (every
primitive present exactly once, every declared claim_type/axis/skill
is a real enum member, every row states what it cannot answer, and
PREFERENCE/PERFORMANCE rows never populate the other kind's fields),
not any router logic -- no router exists yet."""
from probe.method_capabilities import METHOD_CAPABILITIES, MeasurementKind
from probe.models import ApproachAxis, CapabilityLabel, InstrumentPrimitive, StatedPreferenceLabel


def test_every_primitive_appears_exactly_once_plus_option_pair():
    methods = [row.method for row in METHOD_CAPABILITIES]
    assert len(methods) == len(set(methods)), "duplicate method row"
    primitive_values = {p.value for p in InstrumentPrimitive}
    table_values = set(methods) - {"option_pair"}
    assert table_values == primitive_values
    assert "option_pair" in methods


def test_option_pair_is_marked_implemented_and_choose_is_not():
    by_method = {row.method: row for row in METHOD_CAPABILITIES}
    assert by_method["option_pair"].implemented is True
    assert by_method["choose"].implemented is False


def test_locate_and_predict_are_the_implemented_instrument_primitives():
    """Matches the actual state of instruments.py as of this table --
    if this fails after building a third primitive, update the row,
    don't just widen the assertion."""
    by_method = {row.method: row for row in METHOD_CAPABILITIES}
    implemented_primitives = {
        row.method for row in METHOD_CAPABILITIES
        if row.implemented and row.method != "option_pair"
    }
    assert implemented_primitives == {"locate", "predict"}
    assert by_method["locate"].measures is MeasurementKind.PERFORMANCE
    assert by_method["predict"].measures is MeasurementKind.PERFORMANCE
    assert by_method["locate"].skills == (CapabilityLabel.TRACES_WORKED_STEPS,)
    assert by_method["predict"].skills == (CapabilityLabel.DERIVES_FORWARD,)


def test_every_declared_claim_type_axis_and_skill_is_a_real_enum_member():
    for row in METHOD_CAPABILITIES:
        for ct in row.claim_types:
            assert isinstance(ct, StatedPreferenceLabel)
        for axis in row.axes:
            assert isinstance(axis, ApproachAxis)
        for skill in row.skills:
            assert isinstance(skill, CapabilityLabel)


def test_every_row_states_what_it_cannot_answer():
    for row in METHOD_CAPABILITIES:
        assert row.cannot_answer.strip(), f"{row.method} has no cannot_answer"


def test_general_purpose_rows_declare_the_full_vocabulary_not_a_subset():
    """option_pair/choose are explicitly "any axis, any claim type" --
    this pins that down to the literal full enum, not an incomplete
    hand-typed list that silently drifts as the vocabularies grow."""
    by_method = {row.method: row for row in METHOD_CAPABILITIES}
    for method in ("option_pair", "choose"):
        assert set(by_method[method].claim_types) == set(StatedPreferenceLabel)
        assert set(by_method[method].axes) == set(ApproachAxis)
        assert by_method[method].skills == ()


def test_performance_rows_never_populate_preference_fields_and_vice_versa():
    """Mirrors InteractionContract's own shape validator (models.py) --
    the table and the schema it describes must agree on this split."""
    for row in METHOD_CAPABILITIES:
        if row.measures is MeasurementKind.PERFORMANCE:
            assert row.claim_types == (), f"{row.method} is PERFORMANCE but declares claim_types"
            assert row.axes == (), f"{row.method} is PERFORMANCE but declares axes"
        else:
            assert row.skills == (), f"{row.method} is PREFERENCE but declares skills"
