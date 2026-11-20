"""An Avarole avatar inside your LiveKit room.

Avarole runs the whole conversation (hearing, thinking, speaking, its knowledge base and tools) and renders the avatar
into its own LiveKit room. ``AvatarSession`` is the bridge between that room and yours:

    people in your room --(their microphone, 16 kHz PCM)--> Avarole
    Avarole's avatar    --(video + voice)-----------------> published into your room as one participant
"""

from __future__ import annotations

import asyncio
from typing import Callable, Optional

from livekit import rtc

from .client import SAMPLE_RATE, Action, AvaroleClient, ScreenInfo
from .log import logger


class AvatarSession:
    """Bring an Avarole avatar into a LiveKit room.

        session = AvatarSession(agent_id="...", api_key="...")
        await session.start(ctx.room)

    The avatar greets people, listens to every person in the room (``listen_to`` narrows who), speaks, shows captions through
    ``session.client.on("caption", ...)`` and can call the ``actions`` you give it. ``await session.aclose()`` ends
    the call; it also ends when the room closes or the plan's session time runs out.
    """

    def __init__(
        self,
        agent_id: str,
        api_key: Optional[str] = None,
        *,
        actions: Optional[list] = None,
        client: Optional[AvaroleClient] = None,
        listen_to: Optional[Callable[[rtc.RemoteParticipant], bool]] = None,
        **client_kwargs,
    ) -> None:
        self.client = client or AvaroleClient(agent_id, api_key, actions=actions, **client_kwargs)
        # By default Avarole hears every person in the room, and no other agent (so two agents never talk to each other).
        self._listen_to = listen_to or (lambda p: p.kind != rtc.ParticipantKind.PARTICIPANT_KIND_AGENT)
        self._room: Optional[rtc.Room] = None
        self._avatar_room: Optional[rtc.Room] = None
        self._tasks: set = set()
        self._audio_source: Optional[rtc.AudioSource] = None
        self._video_source: Optional[rtc.VideoSource] = None
        self._heard: set = set()
        self._closed = False
        self._first_video = asyncio.Event()

    # ------------------------------------------------------------- lifecycle

    async def start(self, room: rtc.Room) -> None:
        """Start the call and put the avatar in ``room`` (a connected ``rtc.Room``, e.g. ``ctx.room``)."""
        self._room = room
        self._closed = False
        self.client.on("expired", lambda _m: self._spawn(self.aclose()))
        self.client.on("end", lambda _v: self._spawn(self.aclose()))

        await self.client.start()
        join = await self.client.livekit_join()

        room.on("track_subscribed", self._on_user_track)
        for participant in room.remote_participants.values():
            for pub in participant.track_publications.values():
                if pub.track is not None:
                    self._on_user_track(pub.track, pub, participant)

        self._avatar_room = rtc.Room()
        self._avatar_room.on("track_subscribed", self._on_avatar_track)
        await self._avatar_room.connect(join.url, join.token, options=rtc.RoomOptions(auto_subscribe=True))
        # The track may already be live when we join: pick it up too.
        for participant in self._avatar_room.remote_participants.values():
            for pub in participant.track_publications.values():
                if pub.track is not None:
                    self._on_avatar_track(pub.track, pub, participant)
        room.on("disconnected", lambda *_: self._spawn(self.aclose()))

    async def aclose(self) -> None:
        """End the call and take the avatar out of the room. Safe to call twice."""
        if self._closed:
            return
        self._closed = True
        await self.client.stop()
        for task in list(self._tasks):
            if task is not asyncio.current_task():
                task.cancel()
        if self._avatar_room is not None:
            try:
                await self._avatar_room.disconnect()
            except Exception:
                logger.debug("avarole: avatar room disconnect failed", exc_info=True)
        self._avatar_room = None
        self._audio_source = self._video_source = None

    # -------------------------------------------------- what you can ask for

    def register_actions(self, actions: list[Action]) -> dict:
        """Offer actions the avatar may call. Same as ``client.register_actions``."""
        return self.client.register_actions(actions)

    def set_screen(self, info: Optional[ScreenInfo]) -> None:
        """Tell the avatar what the user is looking at. Same as ``client.set_screen``."""
        self.client.set_screen(info)

    # ----------------------------------------------------- user -> Avarole

    def _on_user_track(
        self, track: rtc.Track, pub: rtc.RemoteTrackPublication, participant: rtc.RemoteParticipant
    ) -> None:
        if track.kind != rtc.TrackKind.KIND_AUDIO or not self._listen_to(participant):
            return
        if (participant.identity, track.sid) in self._heard:
            return
        self._heard.add((participant.identity, track.sid))
        self._spawn(self._forward_user_audio(track))

    async def _forward_user_audio(self, track: rtc.Track) -> None:
        stream = rtc.AudioStream(track, sample_rate=SAMPLE_RATE, num_channels=1)
        try:
            async for ev in stream:
                await self.client.send_audio(ev.frame.data.tobytes())
        finally:
            await stream.aclose()

    # --------------------------------------------------- Avarole -> your room

    def _on_avatar_track(
        self, track: rtc.Track, pub: rtc.RemoteTrackPublication, participant: rtc.RemoteParticipant
    ) -> None:
        if track.kind == rtc.TrackKind.KIND_AUDIO:
            self._spawn(self._republish_audio(track))
        elif track.kind == rtc.TrackKind.KIND_VIDEO:
            self._spawn(self._republish_video(track))

    async def _publish(self, track: rtc.LocalTrack, source: rtc.TrackSource) -> None:
        if self._room is not None and self._room.isconnected():
            await self._room.local_participant.publish_track(track, rtc.TrackPublishOptions(source=source))

    async def _republish_audio(self, track: rtc.Track) -> None:
        stream = rtc.AudioStream(track)
        try:
            async for ev in stream:
                frame = ev.frame
                if self._audio_source is None:
                    self._audio_source = rtc.AudioSource(frame.sample_rate, frame.num_channels)
                    await self._publish(
                        rtc.LocalAudioTrack.create_audio_track("avarole-voice", self._audio_source),
                        rtc.TrackSource.SOURCE_MICROPHONE,
                    )
                await self._audio_source.capture_frame(frame)
        finally:
            await stream.aclose()

    async def _republish_video(self, track: rtc.Track) -> None:
        stream = rtc.VideoStream(track)
        try:
            async for ev in stream:
                frame = ev.frame
                if self._video_source is None:
                    self._video_source = rtc.VideoSource(frame.width, frame.height)
                    await self._publish(
                        rtc.LocalVideoTrack.create_video_track("avarole-avatar", self._video_source),
                        rtc.TrackSource.SOURCE_CAMERA,
                    )
                self._video_source.capture_frame(frame)
                if not self._first_video.is_set():
                    self._first_video.set()
                    await self.client.viewer_ready()  # Avarole greets once the avatar can be seen
        finally:
            await stream.aclose()

    def _spawn(self, coro) -> None:
        task = asyncio.ensure_future(coro)
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)
