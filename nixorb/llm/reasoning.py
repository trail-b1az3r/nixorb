"""Keeping a reasoning model's private thinking out of the reply.

Reasoning models answer in two parts: a `<think>` block where they work
the problem out, then the actual answer. NixOrb stripped `<ACTION>` blocks
before speaking but nothing else, so the orb read the whole thought aloud
and the answer arrived — if it arrived at all — long after the listener
had given up.

The suppression is streaming-safe: text is released as it arrives, and
only the characters that could still turn out to be part of an opening tag
are held back. Nothing waits for the end of the response, so speech still
starts a sentence or two in.
"""
from __future__ import annotations

import logging
import re
from collections.abc import Iterable, Sequence

log = logging.getLogger(__name__)

#: Tag pairs whose contents are the model talking to itself. `<think>` is
#: Qwen/DeepSeek-R1 style; the others cover the same idea under other names.
REASONING_TAGS: tuple[tuple[str, str], ...] = (
    ("<think>", "</think>"),
    ("<thought>", "</thought>"),
    ("<reasoning>", "</reasoning>"),
    ("<|begin_of_thought|>", "<|end_of_thought|>"),
)

#: Commands NixOrb runs. Held back for the same reason, and by the same code.
ACTION_TAGS: tuple[tuple[str, str], ...] = (("<ACTION>", "</ACTION>"),)


class TagSuppressor:
    """Drops `<tag>…</tag>` spans from a stream, chunk by chunk.

    Feed it whatever arrives and emit what comes back. `flush()` at the end
    releases anything held only because it might have begun a tag.
    """

    def __init__(self, tags: Sequence[tuple[str, str]] = REASONING_TAGS) -> None:
        self._tags = tuple(tags)
        self._openers = tuple(open_ for open_, _ in self._tags)
        self._held = ""
        self._closer: str | None = None
        #: Set once a closing tag turns up with no opener — some chat
        #: templates put the opener in the prompt, so the model's own
        #: output starts already inside the thought.
        self.saw_unopened_close = False

    @property
    def in_tag(self) -> bool:
        return self._closer is not None

    def feed(self, chunk: str) -> str:
        """Return the part of `chunk` that is not inside a suppressed tag."""
        if not chunk:
            return ""

        out: list[str] = []
        self._held += chunk

        while self._held:
            if self._closer is not None:
                index = self._held.find(self._closer)
                if index < 0:
                    # Keep only what could still complete the closing tag.
                    self._held = self._held[-(len(self._closer) - 1):] \
                        if len(self._closer) > 1 else ""
                    return "".join(out)
                self._held = self._held[index + len(self._closer):]
                self._closer = None
                continue

            opener, position = self._first_opener(self._held)
            if opener is not None:
                out.append(self._held[:position])
                self._held = self._held[position + len(opener):]
                self._closer = dict(self._tags)[opener]
                continue

            keep = self._partial_tag_length(self._held)
            if keep:
                out.append(self._held[:-keep])
                self._held = self._held[-keep:]
            else:
                out.append(self._held)
                self._held = ""
            return "".join(out)

        return "".join(out)

    def flush(self) -> str:
        """Release anything held back that never became a tag."""
        if self._closer is not None:
            # An unterminated thought: everything after it was thinking.
            self._held = ""
            return ""
        tail, self._held = self._held, ""
        return tail

    def _first_opener(self, text: str) -> tuple[str | None, int]:
        best: tuple[str | None, int] = (None, -1)
        for opener in self._openers:
            index = text.find(opener)
            if index >= 0 and (best[1] < 0 or index < best[1]):
                best = (opener, index)
        return best

    def _partial_tag_length(self, text: str) -> int:
        """How many trailing characters could still start an opening tag."""
        longest = max(len(opener) for opener in self._openers)
        for size in range(min(longest - 1, len(text)), 0, -1):
            suffix = text[-size:]
            if any(opener.startswith(suffix) for opener in self._openers):
                return size
        return 0


def strip_tags(
    text: str, tags: Sequence[tuple[str, str]] = REASONING_TAGS
) -> str:
    """Remove tag spans from a complete string.

    Also handles a closing tag with no opener: some chat templates put the
    opening `<think>` in the prompt, so the model's own output begins
    inside the thought and only the close is visible. Everything up to
    that close is thinking.
    """
    for opener, closer in tags:
        text = re.sub(
            re.escape(opener) + r".*?" + re.escape(closer),
            "",
            text,
            flags=re.DOTALL,
        )
        # An unmatched opener means it never finished: drop the remainder.
        cut = text.find(opener)
        if cut >= 0:
            text = text[:cut]
        # An unmatched closer means it started before the text did.
        end = text.rfind(closer)
        if end >= 0:
            text = text[end + len(closer):]
    return text.strip()


def strip_reasoning(text: str) -> str:
    """Remove thinking from a complete response."""
    return strip_tags(text, REASONING_TAGS)


def suppressors(*groups: Iterable[tuple[str, str]]) -> TagSuppressor:
    """One suppressor covering several tag groups."""
    tags: list[tuple[str, str]] = []
    for group in groups:
        tags.extend(group)
    return TagSuppressor(tags)
