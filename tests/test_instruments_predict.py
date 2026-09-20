"""interpret_instrument against the predict contract -- the check this
build asked for: the SAME pure function that handles locate's
`clicked_element` payloads handles predict's `chosen_option` ones with
zero code changes, because the predicate shape was already generic
over field names. If this file needed interpret_instrument to change,
that would mean the predicates were too primitive-specific."""
from uuid import uuid4

from probe.instruments import (
    PREDICT_CORRECT_OPTION,
    PREDICT_DIAGNOSTIC_WRONG_OPTION,
    InstrumentOutcome,
    build_predict_demo_contract,
    interpret_instrument,
)
from probe.models import InstrumentEvent, InstrumentEventType

CONTRACT = build_predict_demo_contract(target_claim_id=uuid4())


def _event(event_type: InstrumentEventType, payload: dict | None = None, elapsed_ms: int = 0, seq: int = 0) -> InstrumentEvent:
    return InstrumentEvent(
        instrument_id=uuid4(), seq=seq, event_type=event_type,
        payload=payload or {}, elapsed_ms=elapsed_ms,
    )


def test_correct_prediction_supports():
    events = [
        _event(InstrumentEventType.START, seq=0),
        _event(InstrumentEventType.CLICK, {"chosen_option": PREDICT_CORRECT_OPTION}, elapsed_ms=2000, seq=1),
    ]
    assert interpret_instrument(CONTRACT, events) is InstrumentOutcome.SUPPORTS


def test_diagnostic_wrong_prediction_contradicts():
    events = [
        _event(InstrumentEventType.START, seq=0),
        _event(InstrumentEventType.CLICK, {"chosen_option": PREDICT_DIAGNOSTIC_WRONG_OPTION}, elapsed_ms=1500, seq=1),
    ]
    assert interpret_instrument(CONTRACT, events) is InstrumentOutcome.CONTRADICTS


def test_other_wrong_prediction_is_uninformative():
    events = [
        _event(InstrumentEventType.START, seq=0),
        _event(InstrumentEventType.CLICK, {"chosen_option": "opt_14"}, elapsed_ms=1800, seq=1),
    ]
    assert interpret_instrument(CONTRACT, events) is InstrumentOutcome.UNINFORMATIVE


def test_abandonment_is_uninformative():
    events = [
        _event(InstrumentEventType.START, seq=0),
        _event(InstrumentEventType.ABANDON, elapsed_ms=6000, seq=1),
    ]
    assert interpret_instrument(CONTRACT, events) is InstrumentOutcome.UNINFORMATIVE


def test_timeout_with_no_terminal_event_is_uninformative():
    events = [
        _event(InstrumentEventType.START, seq=0),
        _event(InstrumentEventType.MOVE, {"x": 1, "y": 1}, elapsed_ms=1000, seq=1),
    ]
    assert interpret_instrument(CONTRACT, events) is InstrumentOutcome.UNINFORMATIVE


def test_revised_prediction_lands_on_the_final_click():
    """Same precedence as locate's own revision test: supports_when is
    checked before contradicts_when, so settling on the correct answer
    after an earlier wrong guess still supports."""
    events = [
        _event(InstrumentEventType.START, seq=0),
        _event(InstrumentEventType.CLICK, {"chosen_option": PREDICT_DIAGNOSTIC_WRONG_OPTION}, elapsed_ms=1000, seq=1),
        _event(InstrumentEventType.CLICK, {"chosen_option": PREDICT_CORRECT_OPTION}, elapsed_ms=2500, seq=2),
    ]
    assert interpret_instrument(CONTRACT, events) is InstrumentOutcome.SUPPORTS
