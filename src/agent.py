"""Run: uv run python src/agent.py console (local) or dev (LiveKit rooms)."""

import asyncio
import os
import sys

from dotenv import load_dotenv
from livekit import agents
from livekit.agents import AgentServer, AgentSession, inference, room_io

from max_agent.agent import IntakeAgent, MaxState
from max_agent.memory import MemoryStore
from max_agent.remote import BackendConfig, RemoteMemoryStore
from max_agent.runtimes import RuntimeFactory

load_dotenv(".env.local")
load_dotenv(".env")
# API checked against LiveKit docs and installed SDK; live voice path awaits credentials.
server = AgentServer()


@server.rtc_session(agent_name="max")
async def max_session(ctx: agents.JobContext):
    # Console identity is an explicit local-only setting. In rooms identity comes
    # from the participant's signed LiveKit token, never model/tool arguments.
    console = "console" in sys.argv[1:]
    if console:
        user_id = os.environ.get("MAX_CONSOLE_USER_ID", "local-demo")
        participant_identity = None
    else:
        await ctx.connect()
        participant = await ctx.wait_for_participant()
        user_id = participant.identity
        participant_identity = participant.identity
    mode = os.getenv("MAX_RUNTIME_MODE", "dual")
    if mode == "dual":
        config = BackendConfig(
            os.getenv("MAX_BACKEND_URL", "http://127.0.0.1:8080"),
            os.environ.get("MAX_BACKEND_TOKEN", ""),
            os.environ.get("MAX_EXE_VM_TOKEN", ""),
        )
        store = RemoteMemoryStore(config)
        state = MaxState(user_id, store)
        state.runtimes = RuntimeFactory(config, state)
        ctx.add_shutdown_callback(state.runtimes.aclose)
        await state.runtimes.planner_model.ready()
        facts = await asyncio.to_thread(store.load, user_id)
        initial_agent = state.runtimes.voice(facts)
        # Explicit VAD supports playback interruption and is also used by cascade.
        session = AgentSession(userdata=state, vad=inference.VAD(model="silero"))
    elif mode == "cascade":
        store = await asyncio.to_thread(MemoryStore, os.getenv("MAX_MEMORY_DB", "./data/max.sqlite3"))
        facts = await asyncio.to_thread(store.load, user_id)
        initial_agent = IntakeAgent(facts)
        session = AgentSession(
            userdata=MaxState(user_id, store),
            stt=inference.STT(model=os.getenv("MAX_STT", "assemblyai/assemblyai-stt-3-7-realtime-agent")),
            llm=inference.LLM(model=os.getenv("MAX_LLM", "google/gemma-4-31b-it")),
            tts=inference.TTS(
                model=os.getenv("MAX_TTS", "fishaudio/s2.1-pro"),
                voice=os.getenv("MAX_VOICE", "fa4c9eb3dccc4806b382b40d61c6b10a"),
            ),
        )
    else:
        raise ValueError("MAX_RUNTIME_MODE must be dual or cascade")
    await session.start(
        room=ctx.room,
        agent=initial_agent,
        room_options=(
            room_io.RoomOptions(participant_identity=participant_identity)
            if participant_identity
            else room_io.RoomOptions()
        ),
    )
    await session.generate_reply(
        instructions=(
            "Start the conversation now. Follow the first-time or returning-user greeting "
            "instructions based on the saved profile and ask just one question."
        )
    )


if __name__ == "__main__":
    agents.cli.run_app(server)
