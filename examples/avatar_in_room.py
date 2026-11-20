"""An Avarole avatar in a LiveKit room.

export LIVEKIT_URL=... LIVEKIT_API_KEY=... LIVEKIT_API_SECRET=...
export AVAROLE_AGENT_ID=... AVAROLE_API_KEY=...
python avatar_in_room.py dev
"""

import os

from livekit.agents import JobContext, WorkerOptions, cli

from livekit.plugins import avarole


async def entrypoint(ctx: JobContext):
    await ctx.connect()
    session = avarole.AvatarSession(agent_id=os.environ["AVAROLE_AGENT_ID"], api_key=os.environ["AVAROLE_API_KEY"])
    session.client.on("transcript", lambda text: print("user:", text))
    session.client.on("caption", lambda text: print("avatar:", text))
    await session.start(ctx.room)


if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint))
