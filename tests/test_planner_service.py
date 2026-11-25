import asyncio
import json

import httpx
import pytest
from livekit.agents import llm
from test_agent import ScriptedLLM

from max_agent.memory import MemoryStore
from max_agent.planner_service import create_app
from max_agent.remote import BackendConfig, RemotePlannerLLM

TOKEN = "test-only-secret-not-for-deployment-123456789"
HEADERS = {"Authorization": f"Bearer {TOKEN}"}


async def test_api_auth_shared_memory_and_atomic_rejection(tmp_path):
    path = tmp_path / "memory.db"
    app = create_app(store=MemoryStore(path), token=TOKEN)
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app), base_url="http://test") as client:
        assert (await client.post("/v1/profile/read", json={"user_id": "alice"})).status_code == 401
        response = await client.post(
            "/v1/profile/update",
            headers=HEADERS,
            json={
                "user_id": "alice",
                "source_message_id": "real-turn-1",
                "updates": [
                    {"field": "name", "value": "Alice"},
                    {"field": "income_monthly", "value": "4000"},
                ],
            },
        )
        assert response.status_code == 200
        assert MemoryStore(path).load("alice")["income_monthly"]["value"] == "4000.00"
        response = await client.post("/v1/profile/read", headers=HEADERS, json={"user_id": "bob"})
        assert response.json() == {"profile": {}}
        response = await client.post(
            "/v1/profile/update",
            headers=HEADERS,
            json={
                "user_id": "alice",
                "source_message_id": "real-turn-2",
                "updates": [{"field": "name", "value": "Wrong"}, {"field": "income_monthly", "value": "-10"}],
            },
        )
        assert response.status_code == 422
        assert MemoryStore(path).load("alice")["name"]["value"] == "Alice"


async def test_adapter_streams_real_langgraph_and_preserves_tool_arguments(tmp_path, monkeypatch):
    from max_agent.agent import BudgetAgent

    tool_call = llm.FunctionToolCall(name="calculate_budget", arguments="{}", call_id="budget-call")
    app = create_app(
        store=MemoryStore(tmp_path / "memory.db"),
        token=TOKEN,
        model_factory=lambda: ScriptedLLM([llm.ChoiceDelta(role="assistant", tool_calls=[tool_call])]),
    )
    original = httpx.AsyncClient
    monkeypatch.setattr(httpx, "AsyncClient", lambda **kw: original(**kw, transport=httpx.ASGITransport(app)))
    model = RemotePlannerLLM(BackendConfig("http://localhost", TOKEN), "alice", "session-1")
    chat = llm.ChatContext()
    chat.add_message(role="user", content="What is left after my expenses?")
    async with model.chat(chat_ctx=chat, tools=BudgetAgent({}).tools) as stream:
        chunks = [chunk async for chunk in stream]
    calls = [call for chunk in chunks if chunk.delta for call in chunk.delta.tool_calls]
    assert calls == [tool_call]


async def test_service_failure_is_sanitized(tmp_path):
    def unavailable():
        raise RuntimeError("private backend error details")

    app = create_app(store=MemoryStore(tmp_path / "memory.db"), token=TOKEN, model_factory=unavailable)
    chat = llm.ChatContext()
    chat.add_message(role="user", content="Help me budget.")
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app), base_url="http://test") as client:
        response = await client.post(
            "/v1/reason",
            headers=HEADERS,
            json={
                "user_id": "alice",
                "session_id": "s",
                "request_id": "r",
                "chat": chat.to_dict(),
                "tools": [],
            },
        )
    events = [json.loads(line) for line in response.text.splitlines()]
    assert events[-1] == {"type": "error", "message": "Reasoning unavailable"}
    assert "private backend" not in response.text


async def test_adapter_discards_tools_if_stream_fails(monkeypatch):
    call = llm.FunctionToolCall(name="remember_details", arguments="{}", call_id="unsafe")
    chunk = llm.ChatChunk(id="fixture", delta=llm.ChoiceDelta(role="assistant", tool_calls=[call]))
    body = "\n".join(
        [json.dumps({"type": "chunk", "chunk": chunk.model_dump()}), json.dumps({"type": "error"})]
    )
    original = httpx.AsyncClient
    transport = httpx.MockTransport(lambda request: httpx.Response(200, text=body))
    monkeypatch.setattr(httpx, "AsyncClient", lambda **kw: original(**kw, transport=transport))
    model = RemotePlannerLLM(BackendConfig("http://localhost", TOKEN), "alice", "s")
    async with model.chat(chat_ctx=llm.ChatContext()) as stream:
        chunks = [chunk async for chunk in stream]
    assert not any(chunk.delta.tool_calls for chunk in chunks if chunk.delta)
    assert "couldn't finish" in "".join(chunk.delta.content or "" for chunk in chunks if chunk.delta)


async def test_interruption_closes_pending_http_request(monkeypatch):
    started, closed = asyncio.Event(), asyncio.Event()

    class SlowStream(httpx.AsyncByteStream):
        async def __aiter__(self):
            started.set()
            await asyncio.Event().wait()
            yield b""

        async def aclose(self):
            closed.set()

    transport = httpx.MockTransport(lambda request: httpx.Response(200, stream=SlowStream()))
    original = httpx.AsyncClient
    monkeypatch.setattr(httpx, "AsyncClient", lambda **kw: original(**kw, transport=transport))
    model = RemotePlannerLLM(BackendConfig("http://localhost", TOKEN), "alice", "s")
    stream = model.chat(chat_ctx=llm.ChatContext())
    await asyncio.wait_for(started.wait(), 2)
    await stream.aclose()
    await asyncio.wait_for(closed.wait(), 2)


@pytest.mark.parametrize("url", ["http://example.com", "https://user:password@example.com"])
def test_backend_configuration_rejects_insecure_urls(url):
    with pytest.raises(ValueError):
        BackendConfig(url, TOKEN)
