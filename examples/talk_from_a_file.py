"""Talk to an Avarole assistant from a 16 kHz mono 16-bit WAV file, and print what it says. No LiveKit needed.

export AVAROLE_AGENT_ID=... AVAROLE_API_KEY=...
python talk_from_a_file.py question.wav
"""

import asyncio
import os
import sys
import wave

from livekit.plugins.avarole import AvaroleClient


async def main(path: str):
    with wave.open(path, "rb") as w:
        assert (w.getframerate(), w.getnchannels(), w.getsampwidth()) == (16000, 1, 2), "need 16 kHz mono 16-bit"
        pcm = w.readframes(w.getnframes())
    async with AvaroleClient(os.environ["AVAROLE_AGENT_ID"], os.environ["AVAROLE_API_KEY"]) as call:
        call.on("transcript", lambda t: print("you:", t))
        call.on("caption", lambda t: print("avatar:", t, end="", flush=True))
        for i in range(0, len(pcm), 1920):  # in real time, 60 ms at a time
            await call.send_audio(pcm[i : i + 1920])
            await asyncio.sleep(0.06)
        await asyncio.sleep(15)  # let the answer arrive


asyncio.run(main(sys.argv[1]))
