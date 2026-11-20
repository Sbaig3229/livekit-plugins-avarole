# Changelog

## 0.1.1
Found by running the plugin against a real LiveKit server (WebRTC), which the first release had only been tested
against with stand-ins.
- Several people in the room are now mixed into one microphone signal. Before, their audio frames were interleaved,
  so Avarole heard everything at double speed.
- The avatar's voice and picture are removed from your room when the call ends, when your room closes (also when it is
  closed from your side), or when Avarole's avatar leaves. Before they stayed published, frozen.
- `AvatarSession(client=..., actions=[...])` no longer ignores `actions`.
- A barge-in now clears the avatar's queued speech at once.
- A call that fails to start leaves no tasks or tracks behind; background failures are logged instead of swallowed.
- `agent_config()` and `sessions()` no longer leave an HTTP connection open.

## 0.1.0
First release.
