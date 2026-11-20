"""The LiveKit bridge, with stand-ins for the rtc objects (no LiveKit server needed)."""

import asyncio
from types import SimpleNamespace

import numpy as np
import pytest

from livekit import rtc
from livekit.plugins.avarole import AvaroleClient, AvatarSession
from tests.conftest import until

KIND_AUDIO, KIND_VIDEO = rtc.TrackKind.KIND_AUDIO, rtc.TrackKind.KIND_VIDEO
USER, AGENT = rtc.ParticipantKind.PARTICIPANT_KIND_STANDARD, rtc.ParticipantKind.PARTICIPANT_KIND_AGENT


class FakeRoom:
    made = []

    def __init__(self):
        self.handlers, self.remote_participants, self.connected_to, self.disconnected = {}, {}, None, False
        self.published = []
        self.local_participant = SimpleNamespace(publish_track=self._publish)
        FakeRoom.made.append(self)

    def on(self, event, fn=None):
        self.handlers.setdefault(event, []).append(fn)

    def isconnected(self):
        return True

    async def connect(self, url, token, options=None):
        self.connected_to = (url, token)

    async def disconnect(self):
        self.disconnected = True

    async def _publish(self, track, options):
        self.published.append((track, options.source))

    def fire(self, event, *args):
        for fn in self.handlers.get(event, []):
            fn(*args)


class FakeStream:
    feeds = {}  # track id -> list of frames

    def __init__(self, track, **kw):
        self.track, self.kw = track, kw

    def __aiter__(self):
        return self._gen()

    async def _gen(self):
        for frame in FakeStream.feeds.get(self.track.sid, []):
            yield SimpleNamespace(frame=frame)
        await asyncio.sleep(3600)

    async def aclose(self):
        pass


class FakeSource:
    def __init__(self, *a):
        self.args, self.frames = a, []

    async def capture_frame(self, frame):
        self.frames.append(frame)


class FakeVideoSource(FakeSource):
    def capture_frame(self, frame):
        self.frames.append(frame)


@pytest.fixture
def lk(monkeypatch, svc):
    FakeRoom.made, FakeStream.feeds = [], {}
    monkeypatch.setattr(rtc, "Room", FakeRoom)
    monkeypatch.setattr(rtc, "AudioStream", FakeStream)
    monkeypatch.setattr(rtc, "VideoStream", FakeStream)
    monkeypatch.setattr(rtc, "AudioSource", FakeSource)
    monkeypatch.setattr(rtc, "VideoSource", FakeVideoSource)
    monkeypatch.setattr(rtc.LocalAudioTrack, "create_audio_track", staticmethod(lambda name, src: ("audio", name, src)))
    monkeypatch.setattr(rtc.LocalVideoTrack, "create_video_track", staticmethod(lambda name, src: ("video", name, src)))
    return svc


def track(sid, kind):
    return SimpleNamespace(sid=sid, kind=kind)


def person(identity, kind=USER):
    return SimpleNamespace(identity=identity, kind=kind, track_publications={})


def pcm(n):
    return SimpleNamespace(
        data=np.zeros(n // 2, dtype=np.int16), sample_rate=48000, num_channels=1, width=640, height=480
    )


async def test_people_in_the_room_are_heard_and_the_avatar_is_shown(lk):
    room = FakeRoom()
    client = AvaroleClient("agent-1", "sk_good", app_url=lk.url, service_url=lk.url)
    session = AvatarSession("agent-1", client=client)
    FakeStream.feeds = {"mic": [pcm(1920), pcm(1920)], "voice": [pcm(960)], "face": [pcm(0)]}
    await session.start(room)
    avatar_room = FakeRoom.made[-1]
    assert avatar_room.connected_to == ("wss://lk.example", "tok-" + client.session_id)  # joined Avarole's room

    room.fire("track_subscribed", track("mic", KIND_AUDIO), None, person("alice"))
    room.fire(
        "track_subscribed", track("agent-voice", KIND_AUDIO), None, person("bot", AGENT)
    )  # another agent: ignored
    await until(lambda: len(lk.frames) == 2)
    assert all(len(f) == 1920 for f in lk.frames)  # alice's microphone reached Avarole

    avatar_room.fire("track_subscribed", track("voice", KIND_AUDIO), None, person("avatar"))
    avatar_room.fire("track_subscribed", track("face", KIND_VIDEO), None, person("avatar"))
    await until(lambda: len(room.published) == 2 and any(p == "/livekit/viewer-ready" for p, _ in lk.posts))
    assert {s for _, s in room.published} == {rtc.TrackSource.SOURCE_MICROPHONE, rtc.TrackSource.SOURCE_CAMERA}
    voice = next(t for t, _ in room.published if t[0] == "audio")
    assert voice[2].args == (48000, 1) and len(voice[2].frames) == 1  # its voice, at the rate it arrived

    await session.aclose()
    assert avatar_room.disconnected and client.status == "ended"
    await session.aclose()  # twice is fine


async def test_a_track_already_live_when_we_join_is_picked_up(lk):
    room = FakeRoom()
    room.remote_participants = {
        "alice": SimpleNamespace(
            identity="alice", kind=USER, track_publications={"p": SimpleNamespace(track=track("mic", KIND_AUDIO))}
        )
    }
    FakeStream.feeds = {"mic": [pcm(1920)]}
    session = AvatarSession("agent-1", client=AvaroleClient("agent-1", "sk_good", app_url=lk.url, service_url=lk.url))
    await session.start(room)
    await until(lambda: len(lk.frames) == 1)
    await session.aclose()


async def test_listen_to_narrows_who_is_heard(lk):
    room = FakeRoom()
    FakeStream.feeds = {"a": [pcm(1920)], "b": [pcm(1920)]}
    session = AvatarSession(
        "agent-1",
        listen_to=lambda p: p.identity == "alice",
        client=AvaroleClient("agent-1", "sk_good", app_url=lk.url, service_url=lk.url),
    )
    await session.start(room)
    room.fire("track_subscribed", track("a", KIND_AUDIO), None, person("alice"))
    room.fire("track_subscribed", track("b", KIND_AUDIO), None, person("bob"))
    await until(lambda: len(lk.frames) == 1)
    await asyncio.sleep(0.1)
    assert len(lk.frames) == 1
    await session.aclose()


async def test_the_avatar_leaves_when_your_room_closes(lk):
    room = FakeRoom()
    client = AvaroleClient("agent-1", "sk_good", app_url=lk.url, service_url=lk.url)
    session = AvatarSession("agent-1", client=client)
    await session.start(room)
    room.fire("disconnected")
    await until(lambda: client.status == "ended" and FakeRoom.made[-1].disconnected)
