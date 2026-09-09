"""Streaming speech — speak each sentence as the model produces it.

Waiting for a whole answer before saying a word is what makes a local
assistant feel slow: the model may take ten seconds to finish, and nothing
comes out of the speakers until it does. This speaks the first sentence as
soon as it lands, so the reply starts within a second of the model starting.

It also handles the two things that make an assistant feel interruptible:

  * `stop()` — barge-in. Cuts playback off mid-sentence and drops the queue.
  * `<ACTION>` and `<think>` suppression — commands NixOrb runs, and a
    reasoning model's private working, are held back rather than read
    aloud, without waiting for the whole response to arrive first.
"""
from __future__ import annotations

import asyncio
import contextlib
import logging
import re
from typing import Any

from nixorb.llm.reasoning import ACTION_TAGS, REASONING_TAGS, suppressors

log = logging.getLogger(__name__)

# End of a sentence: terminal punctuation followed by whitespace or the end.
_SENTENCE_END = re.compile(r"[.!?…](?=\s|$)|[\n\r]{2,}")

# Don't bother synthesising a fragment shorter than this; wait for more.
MIN_SENTENCE_CHARS = 12
# ...unless the buffer grows past this with no punctuation in sight.
MAX_BUFFER_CHARS = 240



def split_sentences(buffer: str) -> tuple[list[str], str]:
    """Split off complete sentences, returning (sentences, remainder)."""
    sentences: list[str] = []
    start = 0

    for match in _SENTENCE_END.finditer(buffer):
        end = match.end()
        candidate = buffer[start:end].strip()
        if len(candidate) >= MIN_SENTENCE_CHARS:
            sentences.append(candidate)
            start = end

    remainder = buffer[start:]
    if not sentences and len(remainder) > MAX_BUFFER_CHARS:
        # A model that never punctuates should still make noise. Break at the
        # last word boundary so we do not slice a word in half.
        cut = remainder.rfind(" ", 0, MAX_BUFFER_CHARS)
        if cut > 0:
            sentences.append(remainder[:cut].strip())
            remainder = remainder[cut:]

    return sentences, remainder


class Speaker:
    """Speaks an incoming token stream, sentence by sentence."""

    def __init__(self, engine: Any, streaming: bool = True) -> None:
        self._engine = engine
        self._streaming = streaming
        self._buffer = ""
        self._filter = suppressors(ACTION_TAGS, REASONING_TAGS)
        self._task: asyncio.Task | None = None
        self._queue: asyncio.Queue[str | None] | None = None
        self._stopped = False

    @property
    def speaking(self) -> bool:
        return self._task is not None and not self._task.done()

    # ── Lifecycle ────────────────────────────────────────────────── #

    async def start(self) -> None:
        """Begin a new utterance."""
        await self.stop()
        self._buffer = ""
        self._filter = suppressors(ACTION_TAGS, REASONING_TAGS)
        self._stopped = False
        if not self._streaming:
            return
        self._queue = asyncio.Queue()
        self._task = asyncio.create_task(self._drain(), name="tts-stream")

    async def feed(self, chunk: str) -> None:
        """Add model output; speak whatever completes a sentence."""
        if not self._streaming or self._stopped or not chunk:
            return

        text = self._suppress(chunk)
        if not text:
            return

        self._buffer += text
        sentences, self._buffer = split_sentences(self._buffer)
        for sentence in sentences:
            await self._enqueue(sentence)

    async def finish(self, timeout: float = 120.0) -> None:
        """Speak whatever is left and wait for playback to end."""
        if not self._streaming:
            return

        tail = self._buffer.strip()
        self._buffer = ""
        if tail and not self._stopped:
            await self._enqueue(tail)

        await self._enqueue(None)
        if self._task is not None:
            with contextlib.suppress(TimeoutError, asyncio.CancelledError):
                await asyncio.wait_for(asyncio.shield(self._task), timeout)
        self._task = None

    async def stop(self) -> None:
        """Barge-in: cut playback off now and drop anything queued.

        Only reaches for the engine when something is actually playing: an
        engine is entitled to treat stop() as terminal, and start() calls
        this to clear the decks before every utterance.
        """
        was_speaking = self.speaking
        self._stopped = True

        if was_speaking:
            engine_stop = getattr(self._engine, "stop", None)
            if callable(engine_stop):
                with contextlib.suppress(Exception):
                    engine_stop()

        if self._task is not None and not self._task.done():
            self._task.cancel()
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await self._task
        self._task = None
        self._queue = None

    # ── Fallback for non-streaming engines ───────────────────────── #

    async def say(self, text: str) -> None:
        """Speak a complete string in one go."""
        if text and text.strip() and not self._stopped:
            await self._engine.speak(text.strip())

    # ── Internals ────────────────────────────────────────────────── #

    def _suppress(self, chunk: str) -> str:
        """Drop <ACTION> and <think> blocks as they stream past.

        Both are the model talking to itself rather than to the listener:
        one is a command NixOrb runs, the other is a reasoning model
        working the problem out. Neither should be read aloud.
        """
        return self._filter.feed(chunk)

    async def _enqueue(self, item: str | None) -> None:
        if self._queue is not None and not self._stopped:
            await self._queue.put(item)

    async def _drain(self) -> None:
        """Speak queued sentences in order, one at a time."""
        queue = self._queue
        if queue is None:
            return
        try:
            while True:
                sentence = await queue.get()
                if sentence is None:
                    return
                if self._stopped:
                    continue
                try:
                    await self._engine.speak(sentence)
                except asyncio.CancelledError:
                    raise
                except Exception as exc:
                    log.warning("TTS: could not speak a sentence: %s", exc)
        except asyncio.CancelledError:
            log.debug("TTS: playback interrupted")
            raise
