"""The plugin against a real LiveKit server. Nothing about LiveKit is faked here."""

import asyncio
import logging
import time

import numpy as np

from livekit import rtc
from livekit.plugins.avarole import AvatarSession
from tests.e2e.conftest import AVATAR_HZ, VIDEO_H, VIDEO_W, dominant_hz, peaks_hz, start_tone, until

MIC = rtc.TrackSource.SOURCE_MICROPHONE
CAM = rtc.TrackSource.SOURCE_CAMERA


async def start(w, **kw):
    session = AvatarSession("agent-1", client=w.make_client(), **kw)
    await session.start(w.worker)
    return session


async def avatar_visible(w, timeout=20):
    await until(lambda: w.watch.audio and w.watch.video_px, timeout, what="the avatar's voice and picture in the room")


async def test_the_avatar_appears_in_the_room_and_hears_the_person(world):
    w = world
    task, _ = await start_tone(w.customer, 1000)  # the person is already talking when the avatar joins
    w.cleanup.append(task)
    session = await start(w)
    await avatar_visible(w)
    await until(lambda: len(w.stub.frames) >= 40, what="microphone audio reaching Avarole")
    await asyncio.sleep(1.5)

    assert w.watch.sources == {MIC, CAM}  # published as a normal microphone and camera
    assert w.watch.video_sizes == {(VIDEO_W, VIDEO_H)}  # the picture keeps its size
    r, g, b, _a = zip(*w.watch.video_px)
    assert abs(np.mean(g) - 100) < 25 and abs(np.mean(b) - 200) < 25  # and its colours
    assert len(set(r)) > 3  # and it is moving, not one frozen frame
    voice = np.concatenate(w.watch.audio)
    assert abs(dominant_hz(voice, 48000) - AVATAR_HZ) < 8  # the avatar's voice arrives at the right pitch
    assert np.abs(voice).max() > 3000  # at a normal volume

    heard = w.stub.heard()
    assert all(len(f) == 1920 for f in w.stub.frames)  # Avarole gets exact 60 ms frames
    assert abs(dominant_hz(heard[-16000:], 16000) - 1000) < 15  # of the person's voice, at the right pitch
    assert any(p == "/livekit/viewer-ready" for p, _ in w.stub.posts)  # and is told the avatar can be seen
    assert w.stub.queries[0]["source"] == "sdk" and w.stub.queries[0]["agent_id"] == "agent-1"
    await session.aclose()


async def test_a_person_who_starts_talking_after_the_avatar_is_heard_too(world):
    w = world
    session = await start(w)
    await avatar_visible(w)
    n = len(w.stub.frames)
    task, _ = await start_tone(w.customer, 700)
    w.cleanup.append(task)
    await until(lambda: len(w.stub.frames) >= n + 40, what="the late microphone")
    assert abs(dominant_hz(w.stub.heard()[-16000:], 16000) - 700) < 15
    await session.aclose()


async def test_two_people_talking_are_mixed_not_interleaved(world):
    w = world
    second = rtc.Room()
    await second.connect(
        "ws://127.0.0.1:7880", __import__("tests.e2e.conftest", fromlist=["token"]).token("second", w.room_name)
    )
    w.rooms.append(second)
    t1, _ = await start_tone(w.customer, 1000)
    t2, _ = await start_tone(second, 2000)
    w.cleanup += [t1, t2]
    session = await start(w)
    await avatar_visible(w)
    await asyncio.sleep(1.0)
    start_n = len(w.stub.frames)
    t0 = time.monotonic()
    await asyncio.sleep(4.0)
    seconds = time.monotonic() - t0
    got = len(w.stub.frames) - start_n
    assert 0.8 * seconds * 16.67 < got < 1.2 * seconds * 16.67, (
        "got %d frames in %.1fs; two voices must share one timeline" % (got, seconds)
    )
    pk = peaks_hz(w.stub.heard(start_n)[: 16000 * 3], 16000)
    assert len(pk) == 2 and abs(pk[0] - 1000) < 20 and abs(pk[1] - 2000) < 20, pk  # both voices are in the same signal
    await session.aclose()


async def test_ending_the_session_takes_the_avatar_out_of_the_room(world):
    w = world
    session = await start(w)
    await avatar_visible(w)
    await session.aclose()
    await until(
        lambda: {rtc.TrackKind.KIND_AUDIO, rtc.TrackKind.KIND_VIDEO} <= w.watch.gone, 10, what="the avatar leaving"
    )
    await until(lambda: w.stub.ws_closed.is_set(), 5, what="the call socket closing")
    assert session.client.status == "ended"
    await session.aclose()  # twice is fine


async def test_avarole_ending_the_call_ends_it_everywhere(world):
    w = world
    session = await start(w)
    await avatar_visible(w)
    await w.stub.send(type="session_expired", minutes=15)
    await until(
        lambda: rtc.TrackKind.KIND_VIDEO in w.watch.gone, 10, what="the avatar leaving after the session expired"
    )
    assert session.client.status == "ended"


async def test_the_avatar_leaves_when_the_agents_room_closes(world):
    w = world
    session = await start(w)
    await avatar_visible(w)
    await w.worker.disconnect()
    await until(lambda: session.client.status == "ended", 10, what="the call ending")
    await until(lambda: w.stub.ws_closed.is_set(), 5, what="the call socket closing")


async def test_a_dropped_call_connection_comes_back_and_the_avatar_stays(world):
    w = world
    task, _ = await start_tone(w.customer, 1000)
    w.cleanup.append(task)
    session = await start(w)
    await avatar_visible(w)
    await w.stub.drop()
    await until(lambda: w.stub.sockets == 2, 15, what="the reconnect")
    await until(lambda: session.client.status == "live", 10, what="the call being live again")
    n2 = len(w.stub.frames)
    await until(lambda: len(w.stub.frames) >= n2 + 40, 10, what="audio flowing again")
    assert abs(dominant_hz(w.stub.heard()[-16000:], 16000) - 1000) < 15
    assert not w.watch.gone  # the avatar never left the room
    await session.aclose()


async def test_a_person_muting_and_unmuting_and_leaving_does_not_break_it(world):
    w = world
    task, pub = await start_tone(w.customer, 1000)
    w.cleanup.append(task)
    session = await start(w)
    await avatar_visible(w)
    await until(lambda: len(w.stub.frames) >= 20, what="audio")
    await w.customer.local_participant.unpublish_track(pub.sid)  # they stop sharing their microphone
    await asyncio.sleep(1.0)
    t2, _ = await start_tone(w.customer, 500)  # and later share it again
    w.cleanup.append(t2)
    n = len(w.stub.frames)
    await until(lambda: len(w.stub.frames) >= n + 40, 10, what="the new microphone being heard")
    assert abs(dominant_hz(w.stub.heard()[-16000:], 16000) - 500) < 15
    await w.customer.disconnect()  # then they leave the room
    await asyncio.sleep(1.0)
    assert session.client.status == "live"  # the avatar waits for the next person
    await session.aclose()


async def test_listen_to_decides_who_is_heard(world):
    w = world
    from tests.e2e.conftest import token

    guest = rtc.Room()
    await guest.connect("ws://127.0.0.1:7880", token("guest", w.room_name))
    w.rooms.append(guest)
    t1, _ = await start_tone(w.customer, 1000)
    t2, _ = await start_tone(guest, 2000)
    w.cleanup += [t1, t2]
    session = await start(w, listen_to=lambda p: p.identity == "customer")
    await avatar_visible(w)
    await asyncio.sleep(2.0)
    pk = peaks_hz(w.stub.heard()[-16000 * 2 :], 16000)
    assert len(pk) == 1 and abs(pk[0] - 1000) < 20, pk  # only the customer, never the guest
    await session.aclose()


async def test_a_barge_in_silences_the_avatar_at_once(world):
    w = world
    session = await start(w)
    await avatar_visible(w)
    audio_before = len(w.watch.audio)
    await w.stub.send(type="interrupt")
    await asyncio.sleep(0.3)
    assert session.client.status == "live"  # an interrupt does not end the call
    await until(lambda: len(w.watch.audio) > audio_before + 20, 5, what="the avatar speaking again after the interrupt")
    await session.aclose()


async def test_actions_work_across_the_real_room(world):
    w = world
    from livekit.plugins.avarole import Action

    seen = []
    session = await start(w, actions=[Action("apply_coupon", "Apply", lambda a: seen.append(a) or {"discount": 10})])
    await avatar_visible(w)
    await until(lambda: any(p == "/host/actions" for p, _ in w.stub.posts), 5, what="the actions being offered")
    await w.stub.send(type="host_action", id="1", name="apply_coupon", args={"code": "X"})
    await until(lambda: any(p == "/host/result" for p, _ in w.stub.posts), 5, what="the action's result")
    assert seen == [{"code": "X"}]
    await session.aclose()


async def test_ten_seconds_steady_no_errors_no_leaks(world, caplog):
    w = world
    caplog.set_level(logging.WARNING)
    task, _ = await start_tone(w.customer, 1000)
    w.cleanup.append(task)
    base_tasks = len(asyncio.all_tasks())
    session = await start(w)
    await avatar_visible(w)
    a0, v0, f0 = len(w.watch.audio), len(w.watch.video_px), len(w.stub.frames)
    t0 = time.monotonic()
    await asyncio.sleep(10)
    secs = time.monotonic() - t0
    video_fps = (len(w.watch.video_px) - v0) / secs
    mic_fps = (len(w.stub.frames) - f0) / secs
    audio_s = sum(len(x) for x in w.watch.audio[a0:]) / 48000
    assert video_fps > 8, video_fps  # the avatar's picture keeps a steady rate
    assert 14 < mic_fps < 19, mic_fps  # the person's voice, in real time
    assert audio_s > 0.9 * secs, audio_s  # and the avatar's voice has no gaps
    await session.aclose()
    await asyncio.sleep(1.0)
    leftovers = [t for t in asyncio.all_tasks() if not t.done()]
    assert len(leftovers) <= base_tasks + 12, "tasks left behind: %d (before: %d)" % (len(leftovers), base_tasks)
    bad = [r for r in caplog.records if r.levelno >= logging.ERROR and "avarole" in r.name]
    assert not bad, [r.getMessage() for r in bad]


async def test_a_call_that_cannot_start_leaves_nothing_behind(world):
    from livekit.plugins.avarole import AvaroleError

    w = world
    w.stub.reject = True
    before = len(asyncio.all_tasks())
    session = AvatarSession("agent-1", client=w.make_client())
    try:
        await session.start(w.worker)
        raise AssertionError("a refused call must raise")
    except AvaroleError:
        pass
    await asyncio.sleep(0.5)
    assert len(asyncio.all_tasks()) <= before + 1, "tasks left behind after a refused call"
    assert not w.watch.sources  # and nothing was published into the room
    assert session.client.status in ("ended", "error")


async def test_when_avarole_s_avatar_leaves_its_copy_leaves_the_room_too(world):
    w = world
    session = await start(w)
    await avatar_visible(w)
    await w.stub._stop_avatar()  # Avarole's side stops publishing
    await until(
        lambda: {rtc.TrackKind.KIND_AUDIO, rtc.TrackKind.KIND_VIDEO} <= w.watch.gone,
        15,
        what="the frozen copy being removed from the room",
    )
    assert session.client.status == "live"  # the call itself is still up
    await session.aclose()


async def test_five_calls_in_a_row_in_the_same_room_leave_no_leaks(world):
    w = world
    task, _ = await start_tone(w.customer, 1000)
    w.cleanup.append(task)
    for round_ in range(5):
        w.watch.audio.clear()
        w.watch.video_px.clear()
        w.watch.gone.clear()
        w.stub.avatar_room = None
        session = await start(w)
        await avatar_visible(w)
        await until(lambda r=round_: len(w.stub.frames) > 10 * (r + 1), what="the person being heard")
        await session.aclose()
        await until(lambda: rtc.TrackKind.KIND_VIDEO in w.watch.gone, 10, what="the avatar leaving")
        await asyncio.sleep(0.3)
    mine = [
        t
        for t in asyncio.all_tasks()
        if "/livekit/plugins/avarole/" in str(getattr(t.get_coro(), "cr_code", "")) + str(t.get_coro())
    ]
    assert not mine, "the plugin left tasks behind: %s" % mine
    assert not [
        p for p in w.customer.remote_participants.values() if p.identity == "agent-worker" and p.track_publications
    ]


async def test_someone_who_joins_the_room_later_is_heard(world):
    from tests.e2e.conftest import token

    w = world
    session = await start(w)
    await avatar_visible(w)
    late = rtc.Room()
    await late.connect("ws://127.0.0.1:7880", token("late", w.room_name))
    w.rooms.append(late)
    t, _ = await start_tone(late, 1500)
    w.cleanup.append(t)
    n = len(w.stub.frames)
    await until(lambda: len(w.stub.frames) >= n + 40, 10, what="the late joiner being heard")
    assert abs(dominant_hz(w.stub.heard()[-16000:], 16000) - 1500) < 20
    await session.aclose()


async def test_thirty_seconds_of_everything_going_wrong(world, caplog):
    """People joining, talking over each other and leaving; the call connection dropping twice; barge-ins."""
    from tests.e2e.conftest import token

    w = world
    caplog.set_level(logging.WARNING)
    task, _ = await start_tone(w.customer, 1000)
    w.cleanup.append(task)
    session = await start(w)
    await avatar_visible(w)
    t0 = time.monotonic()

    async def visitor(i, hz):
        room = rtc.Room()
        await room.connect("ws://127.0.0.1:7880", token("visitor-%d" % i, w.room_name))
        tone, _ = await start_tone(room, hz)
        await asyncio.sleep(4)
        tone.cancel()
        await room.disconnect()

    visitors = []
    for i in range(6):
        visitors.append(asyncio.ensure_future(visitor(i, 600 + 150 * i)))
        await asyncio.sleep(1.5)
        if i in (1, 3):
            await w.stub.drop()  # the call connection drops
        if i in (0, 2, 4):
            await w.stub.send(type="interrupt")  # the person talks over the avatar
        await asyncio.sleep(2.5)
    await asyncio.gather(*visitors)
    assert time.monotonic() - t0 > 20

    assert session.client.status == "live"
    assert not w.watch.gone, "the avatar must stay in the room through all of it"
    gaps = np.diff(w.stub.frame_times)
    assert gaps.max() < 4.0, "Avarole stopped hearing the room for %.1fs" % gaps.max()
    assert len(w.stub.frame_times) > 20 * 14  # about real time throughout
    assert all(len(f) == 1920 for f in w.stub.frames)
    assert w.stub.sockets == 3  # two drops, two reconnects, no extras
    await session.aclose()
    bad = [r for r in caplog.records if r.levelno >= logging.ERROR and "avarole" in r.name]
    assert not bad, [r.getMessage() for r in bad]


async def test_a_720p_avatar_reaches_the_room_at_full_size(world):
    """Avarole's real avatar is 1280x720. Full size must reach the room, not a simulcast layer's small one.

    (WebRTC may scale the picture down on a machine that cannot keep up, so the test asks that full size arrives,
    not that nothing smaller ever does.)"""
    w = world
    w.stub.size = (1280, 720)
    session = await start(w)
    await avatar_visible(w, timeout=60)
    await until(lambda: (1280, 720) in w.watch.video_sizes, 30, what="a full-size 1280x720 frame")
    await session.aclose()
