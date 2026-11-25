"""Authenticated transport to Max's shared memory and LangGraph reasoning service."""

import json
import logging
import sqlite3
from dataclasses import dataclass, field
from urllib.parse import urlparse
from uuid import uuid4

import httpx
from livekit.agents import llm
from livekit.agents.llm.utils import build_legacy_openai_schema
from livekit.agents.types import DEFAULT_API_CONNECT_OPTIONS, APIConnectOptions

from max_agent.memory import Update

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class BackendConfig:
    url: str
    token: str = field(repr=False)
    exe_token: str = field(default="", repr=False)

    def __post_init__(self):
        parsed = urlparse(self.url)
        if parsed.scheme != "https" and not (
            parsed.scheme == "http" and parsed.hostname in {"localhost", "127.0.0.1", "::1"}
        ):
            raise ValueError("Backend URL requires HTTPS, except on localhost.")
        if not parsed.hostname or parsed.username or parsed.password or parsed.query or parsed.fragment:
            raise ValueError("Use a backend base URL without credentials, query or fragment.")
        if len(self.token) < 32:
            raise ValueError("MAX_BACKEND_TOKEN must contain at least 32 characters.")

    @property
    def headers(self):
        headers = {"Authorization": f"Bearer {self.token}"}
        if self.exe_token:
            headers["X-Exedev-Authorization"] = f"Bearer {self.exe_token}"
        return headers


class RemoteMemoryStore:
    """Sync interface called off-loop by the existing memory tools. No local fallback."""

    def __init__(self, config: BackendConfig):
        self.config = config

    def _post(self, path: str, body: dict):
        try:
            with httpx.Client(timeout=10) as client:
                response = client.post(
                    self.config.url.rstrip("/") + path, headers=self.config.headers, json=body
                )
                if response.status_code == 422:
                    raise ValueError("Memory update rejected; check fields, amounts and currency changes.")
                response.raise_for_status()
                return response.json()
        except (httpx.HTTPError, json.JSONDecodeError) as exc:
            raise sqlite3.OperationalError("Shared memory service unavailable") from exc

    def load(self, user_id: str) -> dict:
        return self._post("/v1/profile/read", {"user_id": user_id})["profile"]

    def patch(self, user_id: str, updates: list[Update], source_message_id: str) -> dict:
        return self._post(
            "/v1/profile/update",
            {
                "user_id": user_id,
                "updates": [u.model_dump() for u in updates],
                "source_message_id": source_message_id,
            },
        )["profile"]


class RemotePlannerLLM(llm.LLM):
    """LangGraph streams SDK chat chunks; LiveKit owns tool execution and audio.

    Per-request HTTP streams close on cancellation. Requests are never retried:
    replaying a generation after a partial tool call could repeat an effect.
    """

    def __init__(self, config: BackendConfig, user_id: str, session_id: str):
        super().__init__()
        self.config, self.user_id, self.session_id = config, user_id, session_id

    async def ready(self):
        async with httpx.AsyncClient(timeout=5) as client:
            response = await client.get(self.config.url.rstrip("/") + "/health", headers=self.config.headers)
            response.raise_for_status()

    def chat(self, *, chat_ctx, tools=None, conn_options=DEFAULT_API_CONNECT_OPTIONS, **kwargs):
        return PlannerStream(
            self, chat_ctx=chat_ctx, tools=tools or [], conn_options=APIConnectOptions(max_retry=0)
        )


class PlannerStream(llm.LLMStream):
    async def _run(self):
        model = self._llm
        schemas = []
        for tool in self._tools:
            if llm.is_function_tool(tool):
                schemas.append(build_legacy_openai_schema(tool)["function"])
            elif llm.is_raw_function_tool(tool):
                schemas.append(tool.info.raw_schema)
            else:
                raise ValueError("Remote planner supports function tools only.")
        body = {
            "user_id": model.user_id,
            "session_id": model.session_id,
            "request_id": str(uuid4()),
            "chat": self._chat_ctx.to_dict(exclude_metrics=True, exclude_config_update=True),
            "tools": schemas,
        }
        pending_calls = []
        try:
            async with httpx.AsyncClient(timeout=httpx.Timeout(45, connect=5)) as client:  # noqa: SIM117
                async with client.stream(
                    "POST",
                    model.config.url.rstrip("/") + "/v1/reason",
                    headers=model.config.headers,
                    json=body,
                ) as response:
                    response.raise_for_status()
                    async for line in response.aiter_lines():
                        if not line:
                            continue
                        event = json.loads(line)
                        if event["type"] == "error":
                            raise ValueError("Remote reasoning failed")
                        if event["type"] == "done":
                            if pending_calls:
                                self._event_ch.send_nowait(
                                    llm.ChatChunk(
                                        id="planner-tools",
                                        delta=llm.ChoiceDelta(role="assistant", tool_calls=pending_calls),
                                    )
                                )
                            return
                        chunk = llm.ChatChunk.model_validate(event["chunk"])
                        if chunk.delta and chunk.delta.tool_calls:
                            pending_calls.extend(chunk.delta.tool_calls)
                            chunk.delta.tool_calls = []
                        self._event_ch.send_nowait(chunk)
                    raise ValueError("Remote reasoning stream ended before completion")
        except (httpx.HTTPError, ValueError, KeyError) as exc:
            logger.warning("Planner stream unavailable (%s)", type(exc).__name__)
            self._event_ch.send_nowait(
                llm.ChatChunk(
                    id="planner-unavailable",
                    delta=llm.ChoiceDelta(
                        role="assistant",
                        content=" I couldn't finish that budget analysis. Please try again shortly.",
                    ),
                )
            )
