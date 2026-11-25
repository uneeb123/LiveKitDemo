"""Handoff mechanics with the real SDK, using deterministic model fixtures."""

from unittest.mock import AsyncMock

import httpx
from livekit.agents import AgentSession, inference, llm
from test_agent import ScriptedLLM

from max_agent.agent import MaxState
from max_agent.memory import MemoryStore, Update
from max_agent.remote import BackendConfig
from max_agent.runtimes import PlannerAgent, RuntimeFactory, VoiceAgent


def call(name, arguments="{}"):
    return llm.ChoiceDelta(
        role="assistant",
        tool_calls=[llm.FunctionToolCall(name=name, arguments=arguments, call_id=f"call-{name}")],
    )


def reply(text):
    return llm.ChoiceDelta(role="assistant", content=text)


class TestFactory:
    __test__ = False

    def __init__(self):
        self.planner_model = ScriptedLLM(
            [
                call("calculate_budget"),
                reply("Your remaining monthly amount is 600 dollars."),
                call("return_to_voice", '{"summary":"Budget reviewed; 600 USD remaining monthly."}'),
            ]
        )
        self.planner_model.ready = AsyncMock()
        self.voice_models = iter(
            [
                ScriptedLLM(
                    [call("handoff_to_planner", '{"summary":"Review my budget and remaining cash."}')]
                ),
                ScriptedLLM([reply("We have a plan. What else would you like to discuss?")]),
            ]
        )

    def voice(self, facts, chat_ctx=None, summary=""):
        return VoiceAgent(facts, model=next(self.voice_models), chat_ctx=chat_ctx, summary=summary)

    def planner(self, facts, chat_ctx=None, summary=""):
        return PlannerAgent(
            facts, model=self.planner_model, stt=None, tts=None, chat_ctx=chat_ctx, summary=summary
        )


async def test_roundtrip_preserves_identity_profile_and_context(tmp_path):
    store = MemoryStore(tmp_path / "memory.db")
    facts = store.patch(
        "alice",
        [
            Update(field=k, value=v)
            for k, v in {
                "currency": "USD",
                "income_monthly": "4000",
                "fixed_expenses_monthly": "1800",
                "variable_expenses_monthly": "900",
                "debt_payments_monthly": "200",
                "savings_target_monthly": "500",
            }.items()
        ],
        "earlier-user-message",
    )
    state = MaxState("alice", store, runtimes=TestFactory())
    async with AgentSession(vad=None, userdata=state) as session:
        await session.start(state.runtimes.voice(facts))
        result = await session.run(user_input="Please review my budget and tell me what is left.")
        result.expect.contains_function_call(name="handoff_to_planner")
        result.expect.contains_function_call(name="calculate_budget")
        assert isinstance(session.current_agent, PlannerAgent)
        assert session.userdata is state
        assert "Review my budget" in session.current_agent.instructions
        result = await session.run(user_input="Thanks, let’s go back to chatting.")
        result.expect.contains_function_call(name="return_to_voice")
        assert isinstance(session.current_agent, VoiceAgent)
        assert "600 USD remaining" in session.current_agent.instructions
        assert session.userdata.user_id == "alice"
        assert any(
            "back to chatting" in (item.text_content or "")
            for item in session.current_agent.chat_ctx.items
            if item.type == "message"
        )
    assert store.load("alice") == facts
    assert store.load("bob") == {}


async def test_unavailable_planner_does_not_handoff(tmp_path):
    store = MemoryStore(tmp_path / "memory.db")
    factory = TestFactory()
    factory.planner_model.ready.side_effect = httpx.ConnectError("offline")
    voice = VoiceAgent(
        {},
        model=ScriptedLLM(
            [
                call("handoff_to_planner", '{"summary":"Build a detailed budget"}'),
                reply("The planner is unavailable right now."),
            ]
        ),
    )
    state = MaxState("alice", store, runtimes=factory)
    async with AgentSession(vad=None, userdata=state) as session:
        await session.start(voice)
        result = await session.run(user_input="Build a detailed budget for me.")
        result.expect.contains_function_call_output(is_error=True)
        assert session.current_agent is voice
        assert store.load("alice") == {}


async def test_factory_selects_real_duplex_and_separate_cascade(monkeypatch, tmp_path):
    monkeypatch.setenv("LIVEKIT_API_KEY", "fixture-key")
    monkeypatch.setenv("LIVEKIT_API_SECRET", "fixture-secret")
    state = MaxState("alice", MemoryStore(tmp_path / "memory.db"))
    factory = RuntimeFactory(BackendConfig("http://localhost:8080", "x" * 32), state)
    voice = factory.voice({})
    planner = factory.planner({})
    assert isinstance(voice.llm.duplex_model, inference.DuplexModel)
    assert voice.stt is None and voice.tts is None
    assert planner.llm is factory.planner_model
    assert isinstance(planner.stt, inference.STT)
    assert isinstance(planner.tts, inference.TTS)
    await voice.llm.aclose()
    await planner.stt.aclose()
    await planner.tts.aclose()
