# livekit-plugins-avarole

Avarole plugin for [LiveKit Agents](https://docs.livekit.io/agents/). It puts an **Avarole avatar** in your LiveKit room:
it listens to the people there, answers out loud with its face and voice, and can do everything an Avarole assistant
does on the platform (its knowledge base, tickets, scheduling, the tools you enable) plus **actions you give it in your
own app**.

Avarole runs the whole conversation. This package is the bridge: people in your room reach Avarole, and the avatar's
video and voice come back into your room as one participant.

```
people in your room ──(microphone, 16 kHz)──▶ Avarole
Avarole avatar ──(video + voice)──▶ published into your room
```

## Install

```bash
pip install livekit-plugins-avarole
```

You need an Avarole assistant: sign in at [app.avarole.com](https://app.avarole.com), build one, and copy its **agent id**
and **API key** from *Integrations*.

## Quick start (LiveKit Agents)

```python
from livekit.agents import JobContext, WorkerOptions, cli
from livekit.plugins import avarole


async def entrypoint(ctx: JobContext):
    await ctx.connect()
    session = avarole.AvatarSession(agent_id="YOUR_AGENT_ID", api_key="sk_YOUR_KEY")
    await session.start(ctx.room)  # the avatar joins and greets whoever is in the room


if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint))
```

The avatar listens to every person in the room (not to other agents), and the call ends when the room closes, you call
`await session.aclose()`, or the plan's session time runs out. Use `listen_to=lambda p: p.identity == "alice"` to narrow
who it hears.

Keep the key in an environment variable, not in code.

## Let the avatar act in your app

List what your app can do. The assistant can call these and nothing else. Anything that changes something should
set `confirm=True`: the user must allow it first.

```python
from livekit.plugins.avarole import Action


async def apply_coupon(args):
    return await cart.apply(str(args["code"]))  # validate arguments like any user input


session = avarole.AvatarSession(
    agent_id=...,
    api_key=...,
    confirm=ask_the_user,  # async (title, body) -> bool
    actions=[
        Action(
            "apply_coupon",
            "Apply a coupon code to the cart",
            apply_coupon,
            params={"type": "object", "properties": {"code": {"type": "string"}}, "required": ["code"]},
        ),
        Action("cancel_order", "Cancel the current order", cancel_order, confirm=True),
    ],
)
session.set_screen(
    avarole.ScreenInfo("Checkout", "Cart with 2 items", [{"id": "email", "label": "Email", "filled": True}])
)
```

Handlers return data, or `{"ok": False, "status": "...", "reason": "..."}` to report a problem; the assistant reads it
and tells the user. At most 3 actions run at once and 30 per minute. `set_screen` takes labels only, never what the
user typed.

## Captions, transcripts and the rest

```python
call = session.client
call.on("transcript", lambda text: print("user:", text))  # what the user said
call.on("caption", lambda text: print("avatar:", text))  # what the avatar is saying
call.on("agent_status", print)  # "looking that up"
call.on("emotion", print)
call.on("interrupt", lambda _: print("barge-in"))
call.on("limit", lambda seconds: ...)  # session length, then "ending" and "expired"
```

Events: `status`, `session`, `agent_status`, `transcript`, `caption`, `emotion`, `interrupt`, `limit`, `ending`,
`expired`, `error`, `end`. Listeners may be plain functions or coroutines; one that raises never breaks the call.

## Without LiveKit

`AvaroleClient` is the same call for any app that can supply microphone audio (16 kHz, mono, 16-bit PCM):

```python
from livekit.plugins.avarole import AvaroleClient

async with AvaroleClient(agent_id="...", api_key="sk_...") as call:
    call.on("caption", print)
    await call.send_audio(pcm_bytes)  # any chunk size; sent as 60 ms frames
    join = await call.livekit_join()  # url + token to watch the avatar yourself
```

It also reads account data: `await call.agent_config()` (the assistant's public settings) and
`await call.sessions(limit=50)` (past calls, newest first; needs the API key).

## Good to know

- Everyone in the room is mixed into one microphone signal, so several people can talk at once.
- The avatar's picture is published at its full size (1280x720). Re-encoding it costs about half a CPU core per call.
- Calls last at most the agent's session limit (set on the Build page); `limit` / `ending` / `expired` tell you.
- If the connection drops the client retries three times, then ends the call and emits `error`.
- When the call ends, the avatar's video and voice are removed from your room (they are never left frozen there).
- Page tools (clicking, scrolling) only exist in a browser; here the assistant is told they are unavailable.
- Use one avatar per room. Another agent that also speaks in the room is not heard by the avatar.
- Tested against a real LiveKit server (1.13.11) with real WebRTC: picture and voice reach the room, one or several
  people are heard at the right pitch and rate, reconnects, interrupts, leaving and rejoining, repeated calls
  without leaks, and a 30-second run with people joining, leaving and the connection dropping twice.

## Development

```bash
pip install -e ".[test]"
pytest tests --ignore=tests/e2e                   # fast, no server needed

# against a real LiveKit server (WebRTC, nothing faked)
git clone --branch v1.13.11 https://github.com/livekit/livekit && (cd livekit && go build -o ../livekit-server ./cmd/server)
LIVEKIT_SERVER_BIN=$PWD/livekit-server pytest tests/e2e

ruff check . && ruff format --check .
```

Apache-2.0. © Avarole.
