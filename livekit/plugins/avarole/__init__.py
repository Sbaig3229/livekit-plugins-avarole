"""Avarole plugin for LiveKit Agents.

    from livekit.plugins import avarole

    session = avarole.AvatarSession(agent_id="...", api_key="...")
    await session.start(ctx.room)          # the avatar joins your room and talks to whoever is in it

``AvaroleClient`` is the same call without LiveKit, for any app that can supply 16 kHz microphone audio.
"""

from livekit.agents import Plugin

from .avatar import AvatarSession
from .client import Action, AvaroleClient, AvaroleError, LiveKitJoin, ScreenInfo
from .log import logger
from .version import __version__

__all__ = ["Action", "AvaroleClient", "AvaroleError", "AvatarSession", "LiveKitJoin", "ScreenInfo", "__version__"]


class AvarolePlugin(Plugin):
    def __init__(self) -> None:
        super().__init__(__name__, __version__, __package__, logger)


Plugin.register_plugin(AvarolePlugin())

# Hide the internals from `help()` and autocompletion.
__pdoc__ = {"AvarolePlugin": False}
for _n in [n for n in dir() if not n.startswith("_") and n not in __all__]:
    __pdoc__[_n] = False
