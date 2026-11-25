# Max — two-runtime voice budgeting agent

Max starts with **native full-duplex GPT-Live** for conversation and intake. For
calculations, comparisons and detailed budget plans, a model-selected tool hands
off to **STT → remote LangGraph reasoning → TTS**. When the user is ready, the
planner can hand back to full-duplex conversation in the same LiveKit room.

The two active-agent types share a trusted user identity, session ID and persistent
profile. Each handoff carries a bounded recent conversation, a task summary and
fresh saved facts. Only the current agent owns audio output. Returning to duplex
creates a new model connection because GPT-Live's instructions are immutable.

GPT-Live uses its provider's Responses delegation for small memory operations and
the handoff decision. That is internal to the voice runtime, not the LangGraph
planner. Voice and backend prompts are separate, as required by GPT-Live. The
planner's LangGraph loads the authoritative profile and streams inference output;
LiveKit executes validated tools and returns their results for the next graph run.

## Run the two runtimes locally

Requires Python 3.12+ and [uv](https://docs.astral.sh/uv/).

```sh
uv sync --locked
```

Create `.env.local` from `.env.example` if it does not already exist. Set LiveKit
Cloud credentials and a random `MAX_BACKEND_TOKEN` of at least 32 characters.
Use the same backend token on the worker and reasoning service. Never put real
credentials in `.env.example`. Both runtimes use LiveKit Inference; no separate
provider keys are required.

Start the planner service in one terminal:

```sh
uv run uvicorn max_agent.planner_service:create_app --factory --app-dir src --host 127.0.0.1 --port 8080
```

Start the voice worker in another:

```sh
uv run python src/agent.py download-files
uv run python src/agent.py console
```

`MAX_RUNTIME_MODE=dual` is the default. The SDK's microphone console works without
the LiveKit CLI, although it is deprecated in favor of `lk agent console`. The dual
runtime requires audio: do not use the text-only console with GPT-Live.

Try an introduction, then ask for a detailed monthly budget. Once the planning
conversation is done, ask to return to chatting. Stop and restart with the same
`MAX_CONSOLE_USER_ID` to test memory recall. The console identity defaults to
`local-demo`; use different IDs to test separate users.

For a LiveKit web/mobile client:

```sh
uv run python src/agent.py dev
```

Dispatch agent name `max` to a private, single-user room. There is no custom web
frontend or token endpoint here. Your authenticated backend must issue stable,
unique participant identities; spoken names and client metadata never select a
memory profile. Only authorized users should receive room tokens, since room
members can hear its audio.

The previous local cascade implementation remains available:

```sh
MAX_RUNTIME_MODE=cascade uv run python src/agent.py console --text
```

That standalone mode uses the worker's local database. In dual mode, the service
is the sole database owner. When migrating existing memory to a remote host, copy
the database while the old writer is stopped and configure the service's path.
Do not run independent databases and expect them to synchronize.

## Host the LangGraph service on exe.dev

The worker can stay local or run elsewhere. Copy the repo to your chosen VM,
configure a private `.env.local` there with LiveKit credentials and the shared
backend token, and run:

```sh
docker compose -f deploy/compose.yaml up -d --build
```

The supplied container runs as a non-root user, exposes port 8080, restarts after
failure, and keeps SQLite in the `max-memory` named volume. Do not remove that
volume when redeploying. This is a single-host backend; use a shared database
before scaling the backend across hosts. Back up the volume separately.

Keep the exe.dev VM private. Configure its HTTPS proxy to target port 8080 and
create a VM-scoped HTTPS access token using exe.dev's account tooling. See the
[proxy documentation](https://exe.dev/docs/proxy.md) and
[VM tokens](https://exe.dev/docs/https-tokens-for-vms.md). The deployed VM is `max-planner.exe.xyz`; its private HTTPS proxy targets port 8080.

On the **worker**, set:

```dotenv
MAX_RUNTIME_MODE=dual
MAX_BACKEND_URL=https://YOUR-VM.exe.xyz
MAX_BACKEND_TOKEN=YOUR-SHARED-APPLICATION-SECRET
MAX_EXE_VM_TOKEN=YOUR-VM-HTTPS-TOKEN
```

`MAX_EXE_VM_TOKEN` travels in `X-Exedev-Authorization` for the private proxy;
`MAX_BACKEND_TOKEN` travels in `Authorization` for the application. Do not send
these secrets to a browser. The service trusts authenticated workers to supply
user identities; it is not a public end-user API. HTTPS is required off localhost.
Both backend and worker Docker images have been built successfully on the exe.dev VM.

## Deployment status (2026-11-25)

- **LangGraph backend deployed:** `https://max-planner.exe.xyz`, private exe.dev
  proxy plus application bearer authentication. VM app directory: `~/max`.
  Docker Compose service: `planner`; persistent volume: `deploy_max-memory`.
  Authenticated health, live reasoning, and profile persistence across a container
  restart were verified. Anonymous access is rejected.
- **LiveKit worker deployed:** agent `max` (`CA_ZksiDbnBPaew`), project
  `max-r8munulw` (`p_2g8kpmk589y`), region `us-east`, version `BLNpEH2kSTJ9`.
  Cloud worker startup and registration succeeded. The existing `assistant-ded`
  agent was not modified. Deployment binding is saved in `livekit.toml`.
- The predeployment intake audio simulation passed (`SR_qqrQEvxoTYRK`).
  Deployed handoff audio simulation `SR_QCEH2xzHodS3` passed its scenario:
  successful memory save, complete calculator result, voice → planner → voice
  handoffs, correct 1200/1400 USD comparison, and name recall after returning.
  The initial run exposed prompt issues that were corrected before this verification.
- Worker deployment secrets are prepared in ignored `.env.livekit-deploy` (mode
  0600). They contain backend credentials only; Cloud supplies LiveKit credentials.
  The VM proxy token expires in November 2027 and must be rotated before then.

For subsequent releases, load the credentials from `.env.local` into the shell
without printing them, then use the existing deployment binding:

```sh
LK_SOURCE=skills lk agent deploy --yes
LK_SOURCE=skills lk agent status
```

Do not run `agent create` again for this worker. Verify rollout status and an audio
conversation after deployment.

## Inbound telephone number

`+15103351777` is attached to Max through dispatch rule `SDR_oLNDYFjPQqom`.
LiveKit reports inbound status ACTIVE. The rule is scoped to phone number
`PN_PPN_wLhwRKo2ag7a`, creates a separate `max-call-` room for each call, and
explicitly dispatches agent `max`. Its configuration is in `deploy/telephony.json`;
do not create a duplicate rule when redeploying the worker.

Routing configuration has been verified; a real PSTN call has not yet been tested.

## Memory and failure behavior

Max asks one question at a time and remembers volunteered details immediately.
Users can correct or forget specific fields, say they are unsure, or skip them.
Money uses decimal arithmetic. Missing values are not zero, currencies cannot
silently relabel saved amounts, and negative cash flow stays negative.

The profile database defaults to `./data/max.sqlite3`, or `/data/max.sqlite3` in
the container. It stores financial facts in plaintext with owner-only permissions
and is excluded from git. Clearing a profile field is not secure erasure of backups,
current conversation context, or separately recorded Cloud transcripts.

A failed handoff leaves the voice agent active. A failed memory write is reported
as failure. A dropped reasoning stream closes its upstream request; tool calls
are buffered until a complete response so incomplete generations cannot mutate
memory. Disconnects do not undo tool writes that already committed. Reasoning
requests are not automatically replayed after partial output.

The LangGraph service does not persist a second chat history or checkpoint an
in-progress generation. LiveKit owns live session history; the shared profile
survives sessions and restarts. A worker restart ends its current call context.

Different duplex/TTS voice providers may sound different at the boundary. The
full-duplex model controls its own speech timing; it cannot guarantee a verbatim
script. LiveKit can cut audible playback while the duplex model continues internally.

## Verification

```sh
uv run python -m pytest -q
uv run ruff check src tests
```

Offline tests cover persistence, isolation, arithmetic, real SDK tool execution,
bidirectional handoff/context, backend authentication, streaming, cancellation and
failed-stream tool suppression. Scripted providers prove mechanics, not natural
language understanding.

Opt-in live checks incur inference usage:

```sh
MAX_RUN_LIVE_TESTS=1 uv run python -m pytest tests/test_live_behavior.py -v
```

Live tests cover natural-language saving/recall, write-failure behavior, the actual
LangGraph/model transport, and native GPT-Live greeting output with synthetic
silence input. No microphone or room is opened by the greeting test. These unit-level live checks do not verify spoken handoffs or voice quality.
The deployed audio scenario separately verified spoken handoffs and budget calculation;
broader interruption and noisy-audio testing remains outside this smoke check. A live audio simulation of intake with missing budget facts also passed.

The Max LiveKit project already has session recording enabled; this setting was inspected and left unchanged.

## Code

- `src/agent.py`: room identity, runtime selection and lifecycle.
- `src/max_agent/runtimes.py`: full-duplex ↔ cascade handoff and split prompts.
- `src/max_agent/remote.py`: streaming LLM adapter and shared memory client.
- `src/max_agent/planner_service.py`: authenticated API and LangGraph workflow.
- `src/max_agent/memory.py`: authoritative atomic facts and budget arithmetic.
- `src/max_agent/agent.py`: common tools and original standalone cascade agents.

References: [LiveKit GPT-Live](https://docs.livekit.io/agents/models/realtime/gpt-live/),
[agent handoffs](https://docs.livekit.io/agents/logic/agents-handoffs/),
[LangGraph streaming](https://docs.langchain.com/oss/python/langgraph/streaming).

## WhatsApp inbound calls (setup pending)

A separate public webhook service is deployed at
`https://max-whatsapp.exe.xyz/webhooks/whatsapp`. The private planner VM is unchanged.
Meta app `1532455218704175`, Messaging account `3232264240304596`, phone number ID
`1486324711226460`, test number `+15551514289`. Calling status is confirmed ENABLED.

The app dashboard still needs its callback URL, verification token, and `calls`
subscription configured. Use Graph API webhook version v26.0. Current temporary
access token expires 2026-11-25 19:00 UTC; replace with a suitable system-user token
before relying on this service. Credentials live only in ignored `.env.whatsapp`
and the VM's mode-0600 environment file. Do not commit them.

Deploy with `docker compose -f deploy/compose.whatsapp.yaml up -d --build` in
`~/max` on `max-whatsapp.exe.xyz`. One process handles signed events with persistent
SQLite deduplication in `deploy_whatsapp-events`. Calls explicitly dispatch `max`
and use stable `whatsapp:<caller>` identities, separate from telephone profiles.
Webhook query access logging is disabled to avoid recording the verify token.

HTTP health, verification challenge, unsigned rejection, and signed-event acceptance
passed against the deployed endpoint. Offline tests cover SDK request construction,
account scoping, retry handling, and duplicate/terminated events across restarts.
A real WhatsApp call is not yet verified; no text chat or outbound calls are implemented.
