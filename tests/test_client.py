import asyncio
import json

import pytest

from livekit.plugins.avarole import Action, AvaroleClient, AvaroleError, ScreenInfo
from livekit.plugins.avarole.client import FRAME_BYTES
from tests.conftest import until


def make(svc, **kw):
    return AvaroleClient("agent-1", "sk_good", app_url=svc.url, service_url=svc.url, **kw)


async def test_the_call_opens_with_the_agent_key_and_source(svc):
    async with make(svc, user_id="u1", contact={"email": "a@b.co"}) as call:
        assert call.status == "live" and len(call.session_id) == 24
        q = svc.queries[0]
        assert q["agent_id"] == "agent-1" and q["api_key"] == "sk_good" and q["source"] == "sdk"
        assert q["session_id"] == call.session_id and q["user_id"] == "u1" and q["contact_email"] == "a@b.co"
    assert call.status == "ended"


async def test_audio_is_sent_in_exact_60ms_frames_whatever_the_chunk_size(svc):
    async with make(svc) as call:
        await call.send_audio(b"\x01" * 3000)
        await call.send_audio(b"\x02" * 840)  # 3840 bytes in total: two frames, nothing left over
        await until(lambda: len(svc.frames) == 2)
        assert all(len(f) == FRAME_BYTES for f in svc.frames)
        await call.send_audio(b"\x03" * 100)  # a part frame waits
        await asyncio.sleep(0.05)
        assert len(svc.frames) == 2


async def test_a_muted_call_sends_no_audio(svc):
    async with make(svc) as call:
        call.mute(True)
        await call.send_audio(b"\0" * FRAME_BYTES * 3)
        call.mute(False)
        await call.send_audio(b"\0" * FRAME_BYTES)
        await until(lambda: len(svc.frames) == 1)


async def test_what_the_assistant_says_and_does_reaches_your_listeners(svc):
    seen = {}
    call = make(svc)
    for ev in ("transcript", "caption", "emotion", "agent_status", "interrupt", "limit", "ending"):
        call.on(ev, lambda v, ev=ev: seen.__setitem__(ev, v))
    async with call:
        for m in (
            {"type": "transcript", "text": "hi"},
            {"type": "response_chunk", "text": "Hello"},
            {"type": "emotion", "emotion": "happy"},
            {"type": "status", "msg": "looking that up"},
            {"type": "interrupt"},
            {"type": "session_limit", "seconds": 600},
            {"type": "session_ending", "seconds": 30},
        ):
            await svc.ws.send_str(json.dumps(m))
        await until(lambda: len(seen) == 7)
    assert seen == {
        "transcript": "hi",
        "caption": "Hello",
        "emotion": "happy",
        "agent_status": "looking that up",
        "interrupt": None,
        "limit": 600.0,
        "ending": 30.0,
    }


async def test_a_listener_that_raises_does_not_break_the_call(svc):
    call = make(svc)
    got = []
    call.on("caption", lambda v: 1 / 0)
    call.on("caption", got.append)
    async with call:
        await svc.ws.send_str(json.dumps({"type": "response_chunk", "text": "x"}))
        await until(lambda: got == ["x"])


async def test_an_unknown_event_name_is_refused():
    with pytest.raises(ValueError):
        AvaroleClient("a").on("captions", print)


async def test_actions_are_offered_run_and_answered(svc):
    calls = []

    async def apply(args):
        calls.append(args)
        return {"discount": 10}

    call = make(
        svc,
        actions=[
            Action(
                "apply_coupon",
                "Apply a coupon",
                apply,
                params={"type": "object", "properties": {"code": {"type": "string"}}},
            )
        ],
    )
    call.set_screen(ScreenInfo("Checkout", "2 items", [{"id": "email", "label": "Email", "filled": True}]))
    async with call:
        await until(lambda: any(p == "/host/actions" for p, _ in svc.posts))
        offered = next(b for p, b in svc.posts if p == "/host/actions")
        assert offered["session_id"] == call.session_id and offered["actions"][0]["name"] == "apply_coupon"
        assert next(b for p, b in svc.posts if p == "/host/context")["screen"]["name"] == "Checkout"
        await svc.ws.send_str(
            json.dumps({"type": "host_action", "id": "1", "name": "apply_coupon", "args": {"code": "X"}})
        )
        await until(lambda: any(p == "/host/result" for p, _ in svc.posts))
    res = next(b for p, b in svc.posts if p == "/host/result")
    assert (
        calls == [{"code": "X"}]
        and res["id"] == "1"
        and res["result"] == {"ok": True, "status": "ok", "data": {"discount": 10}}
    )


async def test_a_bad_action_is_not_accepted():
    call = AvaroleClient("a")
    out = call.register_actions([Action("Bad Name", "x", lambda a: 1), Action("good_one", "x", lambda a: 1), "nope"])
    assert out == {"added": ["good_one"], "rejected": ["Bad Name", "nope"]}


async def test_unknown_failing_and_unconfirmed_actions_report_back_instead_of_running(svc):
    ran = []

    def boom(args):
        raise RuntimeError("db down")

    call = make(
        svc,
        actions=[
            Action("cancel_order", "Cancel the order", lambda a: ran.append(1), confirm=True),
            Action("explode", "Fails", boom),
        ],
    )
    async with call:
        for i, name in enumerate(("nothing_here", "cancel_order", "explode")):
            await svc.ws.send_str(json.dumps({"type": "host_action", "id": str(i), "name": name}))
        await until(lambda: sum(p == "/host/result" for p, _ in svc.posts) == 3)
    out = {b["name"]: b["result"] for p, b in svc.posts if p == "/host/result"}
    assert out["nothing_here"]["status"] == "unknown_action"
    assert out["cancel_order"]["status"] == "declined" and ran == []  # no way to ask the user: never runs
    assert out["explode"] == {"ok": False, "status": "error", "reason": "db down"}


async def test_a_confirm_action_runs_only_when_the_user_allows_it(svc):
    ran = []
    asked = []

    async def confirm(title, body):
        asked.append(body)
        return len(asked) > 1  # refuse the first time, allow the second

    call = make(
        svc,
        confirm=confirm,
        actions=[Action("cancel_order", "Cancel the order", lambda a: ran.append(1) or "done", confirm=True)],
    )
    async with call:
        for i in range(2):
            await svc.ws.send_str(json.dumps({"type": "host_action", "id": str(i), "name": "cancel_order"}))
        await until(lambda: sum(p == "/host/result" for p, _ in svc.posts) == 2)
    results = [b["result"]["status"] for p, b in svc.posts if p == "/host/result"]
    assert results == ["declined", "ok"] and ran == [1] and "Cancel the order" in asked[0]


async def test_page_tools_are_answered_as_not_supported(svc):
    async with make(svc):
        await svc.ws.send_str(json.dumps({"type": "nav_action", "id": "n1", "tool": "click"}))
        await until(lambda: any(p == "/nav/result" for p, _ in svc.posts))
    body = next(b for p, b in svc.posts if p == "/nav/result")
    assert body["result"]["status"] == "not_supported" and body["id"] == "n1"


async def test_the_avatar_join_details_come_from_the_call_service(svc):
    async with make(svc) as call:
        join = await call.livekit_join()
        assert (
            join.url == "wss://lk.example" and join.token == "tok-" + call.session_id and join.room == call.session_id
        )
        await call.viewer_ready()
        assert any(p == "/livekit/viewer-ready" for p, _ in svc.posts)


async def test_an_expired_session_ends_the_call(svc):
    call = make(svc)
    expired = []
    call.on("expired", expired.append)
    async with call:
        await svc.ws.send_str(json.dumps({"type": "session_expired", "minutes": 15}))
        await until(lambda: call.status == "ended")
    assert expired == [15.0]


async def test_a_dropped_connection_comes_back_by_itself(svc):
    svc.drop_first = True
    call = make(svc)
    statuses = []
    call.on("status", statuses.append)
    async with call:
        await until(
            lambda: (
                svc.sockets == 2
                and statuses[-3:] == ["reconnecting", "live"][-3:]
                or statuses[-2:] == ["reconnecting", "live"]
            ),
            6,
        )
        await call.send_audio(b"\0" * FRAME_BYTES)
        await until(lambda: len(svc.frames) == 1)  # audio flows on the new socket
    assert statuses[:3] == ["connecting", "live", "reconnecting"]


async def test_a_refused_call_raises(svc):
    svc.reject_ws = True
    with pytest.raises(AvaroleError):
        await make(svc).start(timeout=2)


async def test_agent_settings_and_past_calls_can_be_read(svc):
    call = make(svc)
    assert (await call.agent_config())["avatar_display_name"] == "Nancy"
    assert (await call.sessions(limit=5))[0]["limit"] == "5"
    with pytest.raises(AvaroleError):
        await AvaroleClient("agent-1", "sk_bad", app_url=svc.url).sessions()
    with pytest.raises(AvaroleError):
        await AvaroleClient("agent-1", app_url=svc.url).sessions()
    await call.stop()
