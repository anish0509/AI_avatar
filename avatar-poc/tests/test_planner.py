"""Unit tests for app/services/planner.py -- build_planner_prompt() and
parse_planner_output() are pure logic, no API calls. plan() itself (the
actual LLM call) is verified manually instead (scripts/check_planner.py),
same convention every other paid-API service in this codebase follows."""

from app.services.planner import CONVERSATIONAL, build_planner_prompt, parse_planner_output


def test_build_planner_prompt_includes_message_and_history():
    history = [{"role": "user", "content": "my name is Sam"}]
    prompt = build_planner_prompt(history, "what's my name?")

    assert "what's my name?" in prompt
    assert "my name is Sam" in prompt


def test_build_planner_prompt_handles_empty_history():
    prompt = build_planner_prompt([], "hi there")
    assert "(none)" in prompt
    assert "hi there" in prompt


def test_parse_planner_output_recognizes_conversational_sentinel():
    decision = parse_planner_output("CONVERSATIONAL")
    assert decision.is_conversational is True
    assert decision.query == CONVERSATIONAL


def test_parse_planner_output_strips_decoration_around_sentinel():
    for raw in ["CONVERSATIONAL", "conversational", "`CONVERSATIONAL`", '"CONVERSATIONAL"', "CONVERSATIONAL.", "  CONVERSATIONAL  "]:
        decision = parse_planner_output(raw)
        assert decision.is_conversational is True, f"failed for {raw!r}"


def test_parse_planner_output_treats_anything_else_as_search_query():
    decision = parse_planner_output("what is OLAP")
    assert decision.is_conversational is False
    assert decision.query == "what is OLAP"


def test_parse_planner_output_strips_wrapping_quotes_from_query():
    decision = parse_planner_output('"how does OLAP differ from data mining"')
    assert decision.is_conversational is False
    assert decision.query == "how does OLAP differ from data mining"
