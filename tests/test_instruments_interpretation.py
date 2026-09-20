"""interpret_instrument -- pure function, no I/O, no LLM. Tests build
event streams by hand against the real hand-authored locate contract
(build_locate_demo_contract) so these assertions exercise the exact
predicates shipped, not a synthetic stand-in."""
from uuid import uuid4

from probe.instruments import (
    LOCATE_DIAGNOSTIC_WRONG_ELEMENT,
    LOCATE_SEEDED_ERROR_ELEMENT,
    InstrumentOutcome,
    build_locate_demo_contract,
    interpret_instrument,
)
from probe.models import InstrumentEvent, InstrumentEventType

CONTRACT = build_locate_demo_contract(target_claim_id=uuid4())


def _event(event_type: InstrumentEventType, payload: dict | None = None, elapsed_ms: int = 0, seq: int = 0) -> InstrumentEvent:
    return InstrumentEvent(
        instrument_id=uuid4(), seq=seq, event_type=event_type,
        payload=payload or {}, elapsed_ms=elapsed_ms,
    )


def test_clicking_the_seeded_error_supports():
    events = [
        _event(InstrumentEventType.START, seq=0),
        _event(
            InstrumentEventType.CLICK,
            {"clicked_element": LOCATE_SEEDED_ERROR_ELEMENT}, elapsed_ms=1200, seq=1,
        ),
    ]
    assert interpret_instrument(CONTRACT, events) is InstrumentOutcome.SUPPORTS


def test_clicking_the_diagnostic_wrong_element_contradicts():
    events = [
        _event(InstrumentEventType.START, seq=0),
        _event(
            InstrumentEventType.CLICK,
            {"clicked_element": LOCATE_DIAGNOSTIC_WRONG_ELEMENT}, elapsed_ms=800, seq=1,
        ),
    ]
    assert interpret_instrument(CONTRACT, events) is InstrumentOutcome.CONTRADICTS


def test_clicking_a_different_step_is_uninformative():
    """Neither the seeded error nor the diagnostic wrong element --
    ambiguous, teaches nothing about the standing trait, must not be
    guessed as either direction."""
    events = [
        _event(InstrumentEventType.START, seq=0),
        _event(InstrumentEventType.CLICK, {"clicked_element": "step_1"}, elapsed_ms=500, seq=1),
    ]
    assert interpret_instrument(CONTRACT, events) is InstrumentOutcome.UNINFORMATIVE


def test_explicit_abandonment_is_uninformative():
    events = [
        _event(InstrumentEventType.START, seq=0),
        _event(InstrumentEventType.ABANDON, elapsed_ms=3000, seq=1),
    ]
    assert interpret_instrument(CONTRACT, events) is InstrumentOutcome.UNINFORMATIVE


def test_abandonment_wins_even_if_a_click_also_occurred():
    """uninformative_when is checked FIRST -- an explicit exit
    condition always wins over a predicate that happens to also
    match, since the contract declared that exit uninterpretable."""
    events = [
        _event(InstrumentEventType.START, seq=0),
        _event(
            InstrumentEventType.CLICK,
            {"clicked_element": LOCATE_SEEDED_ERROR_ELEMENT}, elapsed_ms=500, seq=1,
        ),
        _event(InstrumentEventType.ABANDON, elapsed_ms=4000, seq=2),
    ]
    assert interpret_instrument(CONTRACT, events) is InstrumentOutcome.UNINFORMATIVE


def test_timeout_with_no_terminal_event_is_uninformative():
    """The distinct timeout case: no submit, no explicit abandon --
    just a trail that stops. Falls through to the default, not an
    explicit uninformative_when match -- a genuinely different code
    path from the abandonment test above."""
    events = [
        _event(InstrumentEventType.START, seq=0),
        _event(InstrumentEventType.MOVE, {"x": 10, "y": 20}, elapsed_ms=1000, seq=1),
        _event(InstrumentEventType.MOVE, {"x": 15, "y": 22}, elapsed_ms=2000, seq=2),
    ]
    assert interpret_instrument(CONTRACT, events) is InstrumentOutcome.UNINFORMATIVE


def test_empty_event_stream_is_uninformative():
    assert interpret_instrument(CONTRACT, []) is InstrumentOutcome.UNINFORMATIVE
