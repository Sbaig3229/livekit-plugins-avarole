"""A real LiveKit server (WebRTC, no mocks) and a stand-in Avarole call service that publishes a real avatar into it.

Run with:  LIVEKIT_SERVER_BIN=/path/to/livekit-server pytest tests/e2e -q
Build the server with:  git clone --branch v1.13.11 https://github.com/livekit/livekit && cd livekit && go build ./cmd/server
"""

import asyncio
import json
import os
import shutil
import socket
import subprocess
import time
from types import SimpleNamespace

import aiohttp
import numpy as np
import pytest
from aiohttp import web
from aiohttp.test_utils import TestServer

from livekit import api, rtc

BIN = os.environ.get("LIVEKIT_SERVER_BIN") or shutil.which("livekit-server")
URL = "ws://127.0.0.1:7880"
KEY, SECRET = "devkey", "secret"
AVATAR_HZ, AVATAR_RATE = 440, 24000
VIDEO_W, VIDEO_H = 320, 240


def pytest_collection_modifyitems(config, items):
    if not BIN:
        skip = pytest.mark.skip(reason="set LIVEKIT_SERVER_BIN to run against a real LiveKit server")
        for item in items:
            if "e2e" in str(item.fspath):
                item.add_marker(skip)


def token(identity, room, **grants):
    g = api.VideoGrants(room_join=True, room=room, **grants)
    return api.AccessToken(KEY, SECRET).with_identity(identity).with_grants(g).to_jwt()


def dominant_hz(samples, rate):
    x = np.asarray(samples, dtype=np.float64)
    if len(x) < 256:
        return 0.0
    spec = np.abs(np.fft.rfft(x * np.hanning(len(x))))
    return float(np.argmax(spec[1:]) + 1) * rate / len(x)


def peaks_hz(samples, rate, count=2, floor=0.15):
    """Frequencies whose strength is within `floor` of the strongest: how many voices are in the signal."""
    x = np.asarray(samples, dtype=np.float64)
    spec = np.abs(np.fft.rfft(x * np.hanning(len(x))))[1:]
    top = spec.max()
    found = []
    for i in np.argsort(spec)[::-1]:
        hz = (i + 1) * rate / len(x)
        if spec[i] < floor * top:
            break
        if all(abs(hz - f) > 60 for f in found):
            found.append(hz)
        if len(found) == count:
            break
    return sorted(found)


@pytest.fixture(scope="session")
def lk_server():
    proc = subprocess.Popen(
        [BIN, "--dev", "--bind", "127.0.0.1", "--node-ip", "127.0.0.1"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    end = time.time() + 30
    while time.time() < end:
        try:
            socket.create_connection(("127.0.0.1", 7880), timeout=0.5).close()
            break
        except OSError:
            time.sleep(0.3)
    else:
        proc.kill()
        raise RuntimeError("livekit-server did not start")
    yield URL
    proc.terminate()
    try:
        proc.wait(timeout=10)
    except subprocess.TimeoutExpired:
        proc.kill()


async def start_tone(room, hz, rate=48000, source=rtc.TrackSource.SOURCE_MICROPHONE, name="tone"):
    """Publish a steady sine as a microphone track, in real time. Returns (task, publication)."""
    src = rtc.AudioSource(rate, 1)
    track = rtc.LocalAudioTrack.create_audio_track(name, src)
    pub = await room.local_participant.publish_track(track, rtc.TrackPublishOptions(source=source))

    async def pump():
        n, t = rate // 100, 0
        while True:
            s = (12000 * np.sin(2 * np.pi * hz * (t + np.arange(n)) / rate)).astype(np.int16)
            t += n
            await src.capture_frame(rtc.AudioFrame(s.tobytes(), rate, 1, n))

    return asyncio.ensure_future(pump()), pub


class Stub:
    """Avarole's call service, as far as the plugin can tell: /stream, /livekit/token, /livekit/viewer-ready,
    and an avatar that publishes a 440 Hz voice and a 320x240 picture into the room named after the session."""

    def __init__(self):
        self.frames, self.posts, self.queries, self.sockets = [], [], [], 0
        self.ws = None
        self.avatar_room = None
        self._pub_tasks = []
        self.ws_closed = asyncio.Event()
        self.reject = False
        self.size = (VIDEO_W, VIDEO_H)
        self.frame_times = []

    async def start(self):
        app = web.Application()
        app.router.add_get("/stream", self._stream)
        app.router.add_get("/livekit/token", self._token)
        for p in ("/livekit/viewer-ready", "/host/actions", "/host/context", "/host/result", "/nav/result"):
            app.router.add_post(p, self._post)
        self.server = TestServer(app)
        await self.server.start_server()
        self.url = str(self.server.make_url("")).rstrip("/")
        return self

    async def close(self):
        await self._stop_avatar()
        await self.server.close()

    async def _stream(self, request):
        self.sockets += 1
        if self.reject:
            return web.Response(status=403)
        ws = web.WebSocketResponse()
        await ws.prepare(request)
        self.ws, self.sid = ws, request.query["session_id"]
        self.queries.append(dict(request.query))
        self.ws_closed.clear()
        await ws.send_str(json.dumps({"type": "session", "sessionId": self.sid}))
        if self.avatar_room is None:
            self._pub_tasks = [asyncio.ensure_future(self._avatar(self.sid))]
        try:
            async for msg in ws:
                if msg.type == aiohttp.WSMsgType.BINARY:
                    self.frames.append(msg.data)
                    self.frame_times.append(time.monotonic())
        finally:
            self.ws_closed.set()
        return ws

    async def _avatar(self, sid):
        room = rtc.Room()
        await room.connect(URL, token("avatar", sid))
        self.avatar_room = room
        w, h = self.size
        a_src, v_src = rtc.AudioSource(AVATAR_RATE, 1), rtc.VideoSource(w, h)
        await room.local_participant.publish_track(
            rtc.LocalAudioTrack.create_audio_track("voice", a_src),
            rtc.TrackPublishOptions(source=rtc.TrackSource.SOURCE_MICROPHONE),
        )
        await room.local_participant.publish_track(
            rtc.LocalVideoTrack.create_video_track("face", v_src),
            rtc.TrackPublishOptions(source=rtc.TrackSource.SOURCE_CAMERA, simulcast=False),
        )

        async def audio():
            n, t = AVATAR_RATE // 100, 0
            while True:
                s = (12000 * np.sin(2 * np.pi * AVATAR_HZ * (t + np.arange(n)) / AVATAR_RATE)).astype(np.int16)
                t += n
                await a_src.capture_frame(rtc.AudioFrame(s.tobytes(), AVATAR_RATE, 1, n))

        async def video():
            i = 0
            while True:
                px = np.zeros((h, w, 4), dtype=np.uint8)
                px[..., 0], px[..., 1], px[..., 2], px[..., 3] = (i * 6) % 256, 100, 200, 255
                v_src.capture_frame(rtc.VideoFrame(w, h, rtc.VideoBufferType.RGBA, px.tobytes()))
                i += 1
                await asyncio.sleep(1 / 15)

        self._pub_tasks += [asyncio.ensure_future(audio()), asyncio.ensure_future(video())]

    async def _stop_avatar(self):
        for t in self._pub_tasks:
            t.cancel()
        self._pub_tasks = []
        if self.avatar_room is not None:
            await self.avatar_room.disconnect()
            self.avatar_room = None

    async def _token(self, request):
        sid = request.query["session_id"]
        return web.json_response({"token": token("viewer-" + sid[:6], sid), "url": URL})

    async def _post(self, request):
        self.posts.append((request.path, await request.json() if request.can_read_body else {}))
        return web.json_response({"status": "ok"})

    async def send(self, **msg):
        await self.ws.send_str(json.dumps(msg))

    async def drop(self):
        """The network blips: close the call socket abnormally."""
        await self.ws.close(code=1011)

    # what the stand-in heard, as 16 kHz mono samples
    def heard(self, start=0):
        data = b"".join(self.frames[start:])
        return np.frombuffer(data, dtype="<i2")


class Watcher:
    """A person in the developer's room, watching what the agent publishes there."""

    def __init__(self, room, identity="agent-worker"):
        self.identity = identity
        self.audio, self.video_px, self.video_sizes = [], [], set()
        self.sources, self.gone, self._tasks = set(), set(), []
        room.on("track_subscribed", self._sub)
        room.on("track_unsubscribed", lambda t, p, part: self.gone.add(t.kind) if part.identity == identity else None)

    def _sub(self, track, pub, participant):
        if participant.identity != self.identity:
            return
        self.sources.add(pub.source)
        if track.kind == rtc.TrackKind.KIND_AUDIO:
            self._tasks.append(asyncio.ensure_future(self._read_audio(track)))
        else:
            self._tasks.append(asyncio.ensure_future(self._read_video(track)))

    async def _read_audio(self, track):
        async for ev in rtc.AudioStream(track, sample_rate=48000, num_channels=1):
            self.audio.append(np.frombuffer(ev.frame.data, dtype=np.int16).copy())

    async def _read_video(self, track):
        async for ev in rtc.VideoStream(track):
            f = ev.frame.convert(rtc.VideoBufferType.RGBA)
            self.video_sizes.add((f.width, f.height))
            arr = np.frombuffer(f.data, np.uint8).reshape(f.height, f.width, 4)
            self.video_px.append(tuple(int(v) for v in arr[f.height // 2, f.width // 2]))

    def stop(self):
        for t in self._tasks:
            t.cancel()


async def until(cond, timeout=15.0, step=0.05, what="condition"):
    end = time.monotonic() + timeout
    while not cond():
        if time.monotonic() > end:
            raise AssertionError("timed out waiting for " + what)
        await asyncio.sleep(step)


@pytest.fixture
async def world(lk_server):
    """A stub Avarole, the developer's room with one person in it, and the agent worker's connection to it."""
    stub = await Stub().start()
    room_name = "room-%d" % int(time.time() * 1000)
    customer, worker = rtc.Room(), rtc.Room()
    await customer.connect(URL, token("customer", room_name))
    await worker.connect(URL, token("agent-worker", room_name))
    w = SimpleNamespace(
        stub=stub,
        customer=customer,
        worker=worker,
        room_name=room_name,
        watch=Watcher(customer),
        cleanup=[],
        rooms=[customer, worker],
    )

    def make_client():
        from livekit.plugins.avarole import AvaroleClient

        return AvaroleClient("agent-1", "sk_good", app_url=stub.url, service_url=stub.url)

    w.make_client = make_client
    yield w
    w.watch.stop()
    for t in w.cleanup:
        t.cancel()
    for r in w.rooms:
        await r.disconnect()
    await stub.close()
