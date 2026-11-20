"""A stand-in for Avarole's call service: the same endpoints and messages the real one speaks."""

import asyncio
import json
from types import SimpleNamespace

import aiohttp
import pytest
from aiohttp import web
from aiohttp.test_utils import TestServer


@pytest.fixture
async def svc():
    s = SimpleNamespace(
        queries=[], frames=[], posts=[], ws=None, sockets=0, drop_first=False, reject_ws=False, got_ws=asyncio.Event()
    )

    async def stream(request):
        s.sockets += 1
        if s.reject_ws:
            return web.Response(status=403)
        ws = web.WebSocketResponse()
        await ws.prepare(request)
        s.queries.append(dict(request.query))
        s.ws = ws
        await ws.send_str(json.dumps({"type": "session", "sessionId": request.query["session_id"]}))
        s.got_ws.set()
        if s.drop_first and s.sockets == 1:
            await asyncio.sleep(0.05)
            await ws.close(code=1011)  # the network blips
            return ws
        async for msg in ws:
            if msg.type == aiohttp.WSMsgType.BINARY:
                s.frames.append(msg.data)
        return ws

    async def post(request):
        s.posts.append((request.path, await request.json() if request.can_read_body else {}))
        return web.json_response({"status": "ok"})

    async def token(request):
        return web.json_response({"token": "tok-" + request.query["session_id"], "url": "wss://lk.example"})

    async def config(request):
        return web.json_response({"agent_id": request.match_info["agent"], "avatar_display_name": "Nancy"})

    async def sessions(request):
        if request.headers.get("X-API-Key") != "sk_good":
            return web.Response(status=401)
        return web.json_response([{"session_id": "a", "limit": request.query["limit"]}])

    app = web.Application()
    app.router.add_get("/stream", stream)
    app.router.add_get("/livekit/token", token)
    app.router.add_get("/assistant/agents/{agent}/config", config)
    app.router.add_get("/assistant/sessions", sessions)
    for path in ("/livekit/viewer-ready", "/host/actions", "/host/context", "/host/result", "/nav/result"):
        app.router.add_post(path, post)
    server = TestServer(app)
    await server.start_server()
    s.url = str(server.make_url("")).rstrip("/")
    yield s
    await server.close()


async def until(cond, timeout=3.0):
    end = asyncio.get_event_loop().time() + timeout
    while not cond():
        if asyncio.get_event_loop().time() > end:
            raise AssertionError("condition not met in time")
        await asyncio.sleep(0.01)
