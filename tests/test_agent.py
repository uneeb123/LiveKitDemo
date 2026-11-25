"""SDK plumbing tests: scripted model output, real tool execution and SQLite."""

import json
import sqlite3
from collections import deque

import pytest
from livekit.agents import AgentSession, llm
from livekit.agents.types import DEFAULT_API_CONNECT_OPTIONS

from max_agent.agent import IntakeAgent, MaxState
from max_agent.memory import MemoryStore


class ScriptedLLM(llm.LLM):
    def __init__(self, responses):
        super().__init__()
        self.responses = deque(responses)

    def chat(self, *, chat_ctx, tools=None, conn_options=DEFAULT_API_CONNECT_OPTIONS, **kwargs):
        response = self.responses.popleft()

        class Stream(llm.LLMStream):
            async def _run(self):
                self._event_ch.send_nowait(llm.ChatChunk(id="fixture", delta=response))

        return Stream(self, chat_ctx=chat_ctx, tools=tools or [], conn_options=conn_options)


def remember(value="Alice"):
    return llm.ChoiceDelta(
        role="assistant",
        tool_calls=[
            llm.FunctionToolCall(
                name="remember_details",
                arguments=json.dumps({"updates": [{"field": "name", "status": "known", "value": value}]}),
                call_id="save-1",
            )
        ],
    )


async def test_real_sdk_saves_with_user_message_provenance(tmp_path):
    store = MemoryStore(tmp_path / "memory.db")
    model = ScriptedLLM([remember(), llm.ChoiceDelta(role="assistant", content="What is your main goal?")])
    async with AgentSession(llm=model, vad=None, userdata=MaxState("alice-id", store)) as session:
        await session.start(IntakeAgent({}))
        result = await session.run(user_input="Call me Alice.")
        result.expect.next_event().is_function_call(
            name="remember_details",
            arguments={"updates": [{"field": "name", "status": "known", "value": "Alice"}]},
        )
        result.expect.next_event().is_function_call_output()
        result.expect.next_event().is_message(role="assistant")
        result.expect.no_more_events()
        fact = MemoryStore(store.path).load("alice-id")["name"]
        assert fact["value"] == "Alice"
        assert fact["source_message_id"] in {
            i.id for i in session.history.items if i.type == "message" and i.role == "user"
        }
        assert store.load("other-user") == {}
        returning = IntakeAgent(store.load("alice-id"))
        assert "Alice" in returning.instructions


async def test_real_sdk_failed_write_reports_error(tmp_path, monkeypatch):
    store = MemoryStore(tmp_path / "memory.db")

    def unavailable(*args):
        raise sqlite3.OperationalError("fixture failure")

    monkeypatch.setattr(store, "patch", unavailable)
    model = ScriptedLLM([remember(), llm.ChoiceDelta(role="assistant", content="I could not save that.")])
    async with AgentSession(llm=model, vad=None, userdata=MaxState("alice-id", store)) as session:
        await session.start(IntakeAgent({}))
        result = await session.run(user_input="Call me Alice.")
        result.expect.next_event().is_function_call(name="remember_details")
        result.expect.next_event().is_function_call_output(is_error=True)
        result.expect.next_event().is_message(role="assistant")
        result.expect.no_more_events()
    assert store.load("alice-id") == {}


@pytest.mark.parametrize("value", ["NaN", "Infinity", "-1", "12.345", "1e1000"])
def test_invalid_money_is_rejected(tmp_path, value):
    from max_agent.memory import Update

    store = MemoryStore(tmp_path / "memory.db")
    with pytest.raises(ValueError):
        store.patch("alice", [Update(field="income_monthly", value=value)], "message")
    assert store.load("alice") == {}


async def test_sdk_handoff_carries_saved_profile(tmp_path):
    from max_agent.agent import BudgetAgent
    from max_agent.memory import Update

    store = MemoryStore(tmp_path / "memory.db")
    facts = store.patch("alice", [Update(field="goals", value="Build an emergency fund")], "earlier")
    model = ScriptedLLM(
        [
            llm.ChoiceDelta(
                role="assistant",
                tool_calls=[
                    llm.FunctionToolCall(name="start_budgeting", arguments="{}", call_id="handoff-1")
                ],
            ),
            llm.ChoiceDelta(role="assistant", content="Let’s work on your emergency fund."),
        ]
    )
    async with AgentSession(llm=model, vad=None, userdata=MaxState("alice", store)) as session:
        await session.start(IntakeAgent(facts))
        result = await session.run(user_input="Let’s make a budget now.")
        result.expect.contains_function_call(name="start_budgeting")
        assert isinstance(session.current_agent, BudgetAgent)
        assert "Build an emergency fund" in session.current_agent.instructions
