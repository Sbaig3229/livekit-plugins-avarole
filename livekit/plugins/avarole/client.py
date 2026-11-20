"""The Avarole call, with no LiveKit in it: microphone audio in, captions and actions out.

    microphone -> 60 ms PCM frames -> websocket /stream        (Avarole hears the user)
    websocket events (captions, status, limits, actions)       (what the assistant says and asks your app to do)
    LiveKit room named after the session                       (the avatar's video and voice, see ``AvatarSession``)

This is the same protocol the Avarole web widget and the React Native SDK use.
"""

from __future__ import annotations

import asyncio
import json
import re
import secrets
import time
from collections.abc import Awaitable
from dataclasses import dataclass, field
from typing import Any, Callable, Optional, Union
from urllib.parse import quote, urlencode

import aiohttp

from .log import logger

DEFAULT_APP_URL = "https://app.avarole.com"
DEFAULT_SERVICE_URL = "https://ai.avarole.com:8080"
FRAME_BYTES = 1920  # 60 ms of 16 kHz mono 16-bit audio: what the call service reads
SAMPLE_RATE = 16000
MAX_ACTIONS = 30
MAX_RECONNECTS = 3
ACTION_TIMEOUT_S = 25.0
MAX_INFLIGHT_ACTIONS = 3
MAX_ACTIONS_PER_MINUTE = 30
MAX_RESULT_CHARS = 2500
_ACTION_NAME = re.compile(r"^[a-z][a-z0-9_]{0,39}$")

EVENTS = (
    "status",
    "session",
    "agent_status",
    "transcript",
    "caption",
    "emotion",
    "interrupt",
    "limit",
    "ending",
    "expired",
    "error",
    "end",
)


class AvaroleError(Exception):
    """The call could not be made or was refused."""


@dataclass
class LiveKitJoin:
    url: str
    token: str
    room: str
    session_id: str


@dataclass
class ScreenInfo:
    """Where the user is in your app. Labels only: never put what they typed here."""

    name: str
    summary: str = ""
    fields: list = field(default_factory=list)  # [{"id", "label", "type", "filled"}]


@dataclass
class Action:
    """Something your app can do that the assistant may call (it sees it as ``app_<name>``).

    ``confirm=True`` asks the user first (via the ``confirm`` callback of the client); without one it is refused.
    ``handler`` receives the arguments the assistant chose, so validate them like any user input.
    """

    name: str
    description: str
    handler: Callable[[dict], Union[Any, Awaitable[Any]]]
    params: dict = field(default_factory=lambda: {"type": "object", "properties": {}})
    confirm: bool = False


def _result(ok: bool, status: str, **extra: Any) -> dict:
    return {"ok": ok, "status": status, **extra}


class AvaroleClient:
    """One call with an Avarole assistant.

    ``agent_id`` and ``api_key`` come from the Avarole dashboard (Integrations). Use it as an async context manager,
    or call ``start()`` / ``stop()`` yourself::

        async with AvaroleClient(agent_id="...", api_key="...") as call:
            call.on("caption", print)
            await call.send_audio(pcm_16khz_mono)
    """

    def __init__(
        self,
        agent_id: str,
        api_key: Optional[str] = None,
        *,
        app_url: str = DEFAULT_APP_URL,
        service_url: str = DEFAULT_SERVICE_URL,
        user_id: Optional[str] = None,
        contact: Optional[dict] = None,
        confirm: Optional[Callable[[str, str], Union[bool, Awaitable[bool]]]] = None,
        actions: Optional[list] = None,
        http_session: Optional[aiohttp.ClientSession] = None,
    ) -> None:
        if not agent_id:
            raise ValueError("agent_id is required")
        self.agent_id = agent_id
        self.api_key = api_key
        self.app_url = app_url.rstrip("/")
        self.service_url = service_url.rstrip("/")
        self.user_id = user_id
        self.contact = contact or {}
        self._confirm = confirm
        self.status = "idle"  # idle | connecting | live | reconnecting | ended | error
        self.session_id = ""
        self._http = http_session
        self._own_http = http_session is None
        self._ws: Optional[aiohttp.ClientWebSocketResponse] = None
        self._reader: Optional[asyncio.Task] = None
        self._tasks: set = set()
        self._listeners: dict = {}
        self._actions: dict = {}
        self._screen: Optional[dict] = None
        self._buf = bytearray()
        self._muted = False
        self._intentional = True
        self._attempts = 0
        self._inflight = 0
        self._recent: list = []
        self._live = asyncio.Event()
        if actions:
            self.register_actions(actions)

    # ---------------------------------------------------------------- events

    def on(self, event: str, fn: Callable[[Any], Any]) -> Callable[[], None]:
        """Listen for one of ``EVENTS``. Returns a function that stops listening.

        ``caption`` is what the assistant is saying, ``transcript`` what the user said, ``agent_status`` what it is
        doing ("looking that up"), ``limit``/``ending``/``expired`` the session clock, ``interrupt`` a barge-in."""
        if event not in EVENTS:
            raise ValueError("unknown event %r, expected one of %s" % (event, ", ".join(EVENTS)))
        self._listeners.setdefault(event, []).append(fn)
        return lambda: self._listeners.get(event, []).remove(fn) if fn in self._listeners.get(event, []) else None

    def _emit(self, event: str, value: Any = None) -> None:
        for fn in list(self._listeners.get(event, [])):
            try:
                out = fn(value)
                if asyncio.iscoroutine(out):
                    self._spawn(out)
            except Exception:  # a listener must never break the call
                logger.exception("avarole: a %s listener raised", event)

    def _set_status(self, status: str) -> None:
        if self.status != status:
            self.status = status
            self._emit("status", status)

    def _spawn(self, coro: Awaitable) -> None:
        task = asyncio.ensure_future(coro)
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)

    # ------------------------------------------------------------------ call

    async def __aenter__(self) -> AvaroleClient:
        await self.start()
        return self

    async def __aexit__(self, *exc: object) -> None:
        await self.stop()

    def _http_session(self) -> aiohttp.ClientSession:
        if self._http is None:
            self._http = aiohttp.ClientSession()
        return self._http

    def _ws_url(self) -> str:
        q = {"session_id": self.session_id, "agent_id": self.agent_id, "source": "sdk"}
        if self.api_key:
            q["api_key"] = self.api_key
        if self.user_id:
            q["user_id"] = self.user_id
        if self.contact.get("email"):
            q["contact_email"] = self.contact["email"]
        if self.contact.get("phone"):
            q["contact_phone"] = self.contact["phone"]
        return self.service_url.replace("http", "ws", 1) + "/stream?" + urlencode(q)

    async def start(self, timeout: float = 20.0) -> str:
        """Open the call and return its session id once Avarole has accepted it."""
        if self.status in ("connecting", "live", "reconnecting"):
            return self.session_id
        self.session_id = secrets.token_hex(12)
        self._intentional = False
        self._attempts = 0
        self._buf.clear()
        self._live.clear()
        self._set_status("connecting")
        try:
            await self._connect()
            await asyncio.wait_for(self._live.wait(), timeout)
        except Exception as exc:
            await self.stop()
            self._set_status("error")
            raise AvaroleError("could not start the call: %s" % (exc or type(exc).__name__)) from exc
        return self.session_id

    async def _connect(self) -> None:
        self._ws = await self._http_session().ws_connect(self._ws_url(), heartbeat=20, max_msg_size=0)
        self._reader = asyncio.ensure_future(self._read(self._ws))

    async def _read(self, ws: aiohttp.ClientWebSocketResponse) -> None:
        deliberate = False
        try:
            async for msg in ws:
                if msg.type == aiohttp.WSMsgType.TEXT and len(msg.data) <= 256 * 1024:
                    try:
                        await self._on_message(json.loads(msg.data))
                    except ValueError:
                        continue
                elif msg.type in (aiohttp.WSMsgType.CLOSE, aiohttp.WSMsgType.CLOSING, aiohttp.WSMsgType.CLOSED):
                    break
            deliberate = ws.close_code in (1000, 1001)
        except Exception:
            logger.debug("avarole: websocket read ended", exc_info=True)
        if self._intentional or ws is not self._ws:
            return
        if not deliberate and self._attempts < MAX_RECONNECTS and self._live.is_set():
            self._attempts += 1
            self._set_status("reconnecting")
            await asyncio.sleep(min(1.0 * self._attempts, 5.0))
            if not self._intentional:
                try:
                    await self._connect()
                    return
                except Exception:
                    logger.warning("avarole: reconnect failed", exc_info=True)
        self._set_status("ended" if deliberate else "error")
        if not deliberate:
            self._emit("error", "connection lost")
        self._finish()

    async def _on_message(self, m: dict) -> None:
        kind = m.get("type")
        if kind == "session":
            self.session_id = str(m.get("sessionId") or self.session_id)
            self._attempts = 0
            self._set_status("live")
            self._live.set()
            self._emit("session", self.session_id)
            await self._push_actions()
        elif kind == "status":
            self._emit("agent_status", str(m.get("msg") or ""))
        elif kind == "transcript":
            self._emit("transcript", str(m.get("text") or ""))
        elif kind == "response_chunk":
            self._emit("caption", str(m.get("text") or ""))
        elif kind == "emotion":
            self._emit("emotion", str(m.get("emotion") or ""))
        elif kind == "interrupt":
            self._emit("interrupt")
        elif kind == "session_limit":
            self._emit("limit", float(m.get("seconds") or 0))
        elif kind == "session_ending":
            self._emit("ending", float(m.get("seconds") or 0))
        elif kind == "session_expired":
            self._emit("expired", float(m.get("minutes") or 0))
            self._spawn(self.stop())
        elif kind == "error":
            self._emit("error", str(m.get("msg") or "error"))
        elif kind == "host_action":
            self._spawn(self._run_action(m))
        elif kind == "nav_action":
            # Page tools (clicking, scrolling) only exist in a browser. Answer so the assistant does not wait.
            self._spawn(
                self._post(
                    "/nav/result",
                    {
                        "id": m.get("id"),
                        "tool": m.get("tool"),
                        "result": _result(
                            False,
                            "not_supported",
                            reason="This is not a web page, there is nothing to click or scroll.",
                        ),
                    },
                )
            )

    async def send_audio(self, pcm: bytes) -> None:
        """Send microphone audio: 16 kHz, mono, 16-bit little-endian PCM, any chunk size."""
        if self._muted or self._ws is None or self._ws.closed or self.status not in ("live", "connecting"):
            return
        self._buf.extend(pcm)
        while len(self._buf) >= FRAME_BYTES:
            frame = bytes(self._buf[:FRAME_BYTES])
            del self._buf[:FRAME_BYTES]
            try:
                await self._ws.send_bytes(frame)
            except Exception:
                return  # the reader notices the drop and reconnects or ends the call

    def mute(self, on: bool = True) -> None:
        """Stop sending the microphone; the call stays up."""
        self._muted = on
        if on:
            self._buf.clear()

    @property
    def muted(self) -> bool:
        return self._muted

    async def stop(self) -> None:
        """End the call. Safe to call twice."""
        self._intentional = True
        ws, self._ws = self._ws, None
        if ws is not None and not ws.closed:
            try:
                await ws.close(code=1000)
            except Exception:
                pass
        if self._reader is not None and self._reader is not asyncio.current_task():
            self._reader.cancel()
        self._reader = None
        was_live = self.status not in ("idle", "ended", "error")
        if was_live:
            self._set_status("ended")
        self._finish()
        for task in list(self._tasks):
            if task is not asyncio.current_task():
                task.cancel()
        if self._own_http and self._http is not None:
            await self._http.close()
            self._http = None

    def _finish(self) -> None:
        self._buf.clear()
        self._emit("end")

    # --------------------------------------------------------------- avatar

    async def livekit_join(self) -> LiveKitJoin:
        """What is needed to see and hear the avatar: the LiveKit server, a token for this call's room."""
        url = "%s/livekit/token?room=%s&session_id=%s" % (
            self.service_url,
            quote(self.session_id),
            quote(self.session_id),
        )
        async with self._http_session().get(url) as r:
            if r.status != 200:
                raise AvaroleError("avatar token refused (%s)" % r.status)
            d = await r.json()
        if not d.get("token") or not d.get("url"):
            raise AvaroleError("avatar token response was incomplete")
        return LiveKitJoin(url=d["url"], token=d["token"], room=self.session_id, session_id=self.session_id)

    async def viewer_ready(self) -> None:
        """Tell Avarole the avatar's first video frame is on screen, so it greets someone who can see it."""
        try:
            async with self._http_session().post(
                "%s/livekit/viewer-ready?session_id=%s" % (self.service_url, quote(self.session_id))
            ):
                pass
        except Exception:
            logger.debug("avarole: viewer-ready not delivered", exc_info=True)

    # -------------------------------------------------------------- actions

    def register_actions(self, actions: list) -> dict:
        """Offer actions the assistant may call. Returns ``{"added": [...], "rejected": [...]}``."""
        added, rejected = [], []
        for a in actions or []:
            ok = (
                isinstance(a, Action)
                and _ACTION_NAME.match(a.name or "")
                and callable(a.handler)
                and (a.name in self._actions or len(self._actions) < MAX_ACTIONS)
            )
            if not ok:
                rejected.append(str(getattr(a, "name", a)))
                continue
            self._actions[a.name] = a
            added.append(a.name)
        if self.status == "live":
            self._spawn(self._push_actions())
        return {"added": added, "rejected": rejected}

    def set_screen(self, info: Optional[ScreenInfo]) -> None:
        """Tell the assistant where the user is. Labels only, never what they typed."""
        self._screen = (
            None
            if info is None
            else {
                "name": str(info.name or "")[:80],
                "summary": str(info.summary or "")[:300],
                "fields": [
                    {
                        "id": str(f.get("id"))[:40],
                        "label": str(f.get("label"))[:60],
                        "type": str(f.get("type") or "")[:20],
                        "filled": bool(f.get("filled")),
                    }
                    for f in (info.fields or [])[:40]
                ],
            }
        )
        if self.status == "live":
            self._spawn(self._push_screen())

    async def _push_actions(self) -> None:
        if self._actions:
            await self._post(
                "/host/actions",
                {
                    "actions": [
                        {
                            "name": a.name,
                            "description": str(a.description or "")[:300],
                            "params": a.params,
                            "confirm": bool(a.confirm),
                        }
                        for a in self._actions.values()
                    ]
                },
            )
            await self._push_screen()

    async def _push_screen(self) -> None:
        if self._screen:
            await self._post("/host/context", {"screen": self._screen})

    async def _post(self, path: str, body: dict) -> None:
        try:
            async with self._http_session().post(self.service_url + path, json={"session_id": self.session_id, **body}):
                pass
        except Exception:
            logger.debug("avarole: POST %s failed", path, exc_info=True)

    async def _run_action(self, m: dict) -> None:
        aid, name = m.get("id"), str(m.get("name") or "")

        async def reply(result: dict) -> None:
            await self._post("/host/result", {"id": aid, "name": name, "result": result})

        now = time.monotonic()
        self._recent = [t for t in self._recent if now - t < 60]
        if self._inflight >= MAX_INFLIGHT_ACTIONS:
            return await reply(_result(False, "busy"))
        if len(self._recent) >= MAX_ACTIONS_PER_MINUTE:
            return await reply(_result(False, "rate_limited", reason="Too many actions in a minute."))
        action = self._actions.get(name)
        if action is None:
            return await reply(_result(False, "unknown_action", reason=name))
        self._recent.append(now)
        self._inflight += 1
        try:
            if action.confirm:
                yes = False
                if self._confirm is not None:
                    yes = self._confirm(
                        "Allow this action?", "The assistant wants to: %s." % (action.description or action.name)
                    )
                    if asyncio.iscoroutine(yes):
                        yes = await yes
                if not yes:
                    return await reply(
                        _result(
                            False,
                            "declined",
                            reason=(
                                "The user chose not to allow it."
                                if self._confirm
                                else "There is no way to ask the user, so it was not run."
                            ),
                        )
                    )
            args = m.get("args") if isinstance(m.get("args"), dict) else {}
            try:
                out = action.handler(args)
                if asyncio.iscoroutine(out):
                    out = await asyncio.wait_for(out, ACTION_TIMEOUT_S)
                result = (
                    out if isinstance(out, dict) and isinstance(out.get("ok"), bool) else _result(True, "ok", data=out)
                )
            except asyncio.TimeoutError:
                result = _result(False, "timeout", reason="The app took too long to respond.")
            except Exception as exc:
                result = _result(False, "error", reason=str(exc)[:120])
        finally:
            self._inflight -= 1
        if len(json.dumps(result, default=str)) > MAX_RESULT_CHARS:
            result = _result(
                result["ok"],
                result.get("status", "ok"),
                reason="The result was too large and was cut.",
                data=json.dumps(result.get("data"), default=str)[: MAX_RESULT_CHARS - 200],
            )
        await reply(result)

    # ----------------------------------------------------------- the account

    async def agent_config(self) -> dict:
        """The assistant's public settings (name, avatar, language...) as configured in the dashboard."""
        async with self._http_session().get(
            "%s/assistant/agents/%s/config" % (self.app_url, quote(self.agent_id))
        ) as r:
            if r.status != 200:
                raise AvaroleError("could not read the agent's settings (%s)" % r.status)
            return await r.json()

    async def sessions(self, limit: int = 50, offset: int = 0) -> list:
        """Past calls on your account, newest first. Needs the API key."""
        if not self.api_key:
            raise AvaroleError("api_key is required to list sessions")
        async with self._http_session().get(
            "%s/assistant/sessions?limit=%d&offset=%d" % (self.app_url, max(1, min(limit, 200)), max(0, offset)),
            headers={"X-API-Key": self.api_key},
        ) as r:
            if r.status != 200:
                raise AvaroleError("could not list sessions (%s)" % r.status)
            return await r.json()
