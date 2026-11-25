"""Opt-in real-model tests. No scripted model or mocked successful writes."""

import os
import sqlite3

import pytest
from dotenv import load_dotenv
from livekit.agents import AgentSession, inference

from max_agent.agent import IntakeAgent, MaxState
from max_agent.memory import MemoryStore

load_dotenv(".env.local")
load_dotenv(".env")
pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        os.getenv("MAX_RUN_LIVE_TESTS") != "1", reason="Set MAX_RUN_LIVE_TESTS=1 with LiveKit credentials"
    ),
]


async def test_volunteered_facts_and_return_visit(tmp_path):
    path = tmp_path / "memory.db"
    store = MemoryStore(path)
    async with inference.LLM(model=os.getenv("MAX_LLM", "google/gemma-4-31b-it")) as model:
        async with AgentSession(llm=model, vad=None, userdata=MaxState("test-user", store)) as session:
            await session.start(IntakeAgent({}))
            result = await session.run(
                user_input=("I'm Alex. My take-home pay is 4000 US dollars each month. Please remember that.")
            )
            result.expect.contains_function_call(name="remember_details")
            facts = store.load("test-user")
            assert facts["name"]["value"] == "Alex"
            assert facts["income_monthly"]["value"] == "4000.00"
            assert facts["currency"]["value"] == "USD"
            await (
                result.expect[-1]
                .is_message(role="assistant")
                .judge(
                    model,
                    intent="Acknowledges the details and asks at most one relevant next intake question.",
                )
            )
        # A genuinely new session and store must recall the committed record.
        reopened = MemoryStore(path)
        async with AgentSession(llm=model, vad=None, userdata=MaxState("test-user", reopened)) as session:
            await session.start(IntakeAgent(reopened.load("test-user")))
            result = await session.run(user_input="What do you remember about my income?")
            await (
                result.expect[-1]
                .is_message(role="assistant")
                .judge(
                    model,
                    intent="States the saved monthly take-home income is 4000 US dollars. May ask whether it has changed, but must not ask the user to supply the income from scratch.",
                )
            )


async def test_model_does_not_claim_failed_save(tmp_path, monkeypatch):
    store = MemoryStore(tmp_path / "memory.db")

    def fail(*args):
        raise sqlite3.OperationalError("unavailable")

    monkeypatch.setattr(store, "patch", fail)
    async with (
        inference.LLM(model=os.getenv("MAX_LLM", "google/gemma-4-31b-it")) as model,
        AgentSession(llm=model, vad=None, userdata=MaxState("test-user", store)) as session,
    ):
        await session.start(IntakeAgent({}))
        result = await session.run(user_input="Please remember my name is Alex.")
        result.expect.contains_function_call(name="remember_details")
        await (
            result.expect[-1]
            .is_message(role="assistant")
            .judge(
                model,
                intent="Explains that saving failed and does not claim to remember the name for future visits.",
            )
        )
        assert store.load("test-user") == {}


async def test_real_model_through_langgraph_transport(tmp_path, monkeypatch):
    import httpx

    from max_agent.agent import COACH, MaxBase
    from max_agent.planner_service import create_app
    from max_agent.remote import BackendConfig, RemotePlannerLLM

    store = MemoryStore(tmp_path / "memory.db")
    token = "integration-fixture-secret-never-deploy-1234"
    app = create_app(store=store, token=token)
    original = httpx.AsyncClient
    from types import SimpleNamespace

    monkeypatch.setattr(
        "max_agent.remote.httpx",
        SimpleNamespace(
            AsyncClient=lambda **kw: original(**kw, transport=httpx.ASGITransport(app)),
            Timeout=httpx.Timeout,
            HTTPError=httpx.HTTPError,
        ),
    )
    model = RemotePlannerLLM(BackendConfig("http://localhost", token), "test-user", "test-session")
    async with AgentSession(llm=model, vad=None, userdata=MaxState("test-user", store)) as session:
        await session.start(MaxBase(COACH, {}))
        result = await session.run(
            user_input=(
                "Please remember: my name is Alex and my take-home income is 4500 US dollars per month."
            )
        )
        result.expect.contains_function_call(name="remember_details")
        facts = MemoryStore(store.path).load("test-user")
        assert facts["name"]["value"] == "Alex"
        assert facts["income_monthly"]["value"] == "4500.00"
        assert facts["currency"]["value"] == "USD"


async def test_native_duplex_connection_and_greeting(tmp_path):
    """Real audio model connection/output; no microphone or room is opened."""
    import asyncio

    from max_agent.remote import BackendConfig
    from max_agent.runtimes import RuntimeFactory

    state = MaxState("test-user", MemoryStore(tmp_path / "memory.db"))
    factory = RuntimeFactory(
        BackendConfig("http://localhost", "fixture-token-at-least-thirty-two-characters"), state
    )
    from livekit import rtc
    from livekit.agents.voice import io

    class SilenceInput(io.AudioInput):
        def __init__(self):
            super().__init__(label="synthetic-silence")

        async def __anext__(self):
            await asyncio.sleep(0.02)
            return rtc.AudioFrame(data=bytes(960), sample_rate=24000, num_channels=1, samples_per_channel=480)

    agent = factory.voice({})
    try:
        async with asyncio.timeout(45), AgentSession(vad=None, userdata=state) as session:
            session.input.audio = SilenceInput()
            await session.start(agent)
            handle = session.generate_reply(
                instructions="Introduce yourself as Max and ask what to call the user."
            )
            await handle
            assert handle.exception() is None
    finally:
        await agent.llm.aclose()
