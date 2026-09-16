"""Local when it works, T1 when it doesn't.

A local model is the point of NixOrb, and a local model that will not load
is the most common way the orb ends up mute: it starts, listens,
transcribes — and then has nothing to answer with. A GGUF whose
architecture the bundled llama.cpp predates does exactly this, and so does
a first run with nothing downloaded.

`llm_backend = "auto"` health-checks the local backend once, at startup,
and uses it if it answers. If it does not, and T1 is configured, the turn
goes to T1 instead and the log says why. The decision is made once per
process rather than per turn: a local model that failed to load will fail
identically every time, and re-deciding costs a model load each turn.
"""
from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from nixorb.settings import Settings

log = logging.getLogger(__name__)


class AutoBackend:
    """Prefers a local backend, falls back to T1."""

    name = "auto"

    def __init__(self, settings: Settings, local: Any, remote: Any) -> None:
        self._settings = settings
        self._local = local
        self._remote = remote
        self._chosen: Any = None
        self._reason = ""

    # ── which one ────────────────────────────────────────────────── #

    async def _pick(self) -> Any:
        """Choose once: a local model that will not load never will."""
        if self._chosen is not None:
            return self._chosen

        if self._local is not None:
            health = await self._local.health_check()
            if health.get("ok"):
                self._chosen = self._local
                self._reason = "local model is healthy"
                log.info("LLM: using the local backend (%s)", self._local.model)
                return self._chosen
            local_error = health.get("error", "unknown error")
        else:
            local_error = "no local backend is configured"

        if self._remote is None:
            # Nothing to fall back to: keep the local one so its own error
            # reaches the user rather than a vaguer one from here.
            self._chosen = self._local
            self._reason = local_error
            log.error(
                "LLM: local backend unavailable (%s) and T1 is not "
                "configured — set t1_base_url to fall back to it.",
                local_error,
            )
            return self._chosen

        remote_health = await self._remote.health_check()
        if not remote_health.get("ok"):
            self._chosen = self._local
            self._reason = local_error
            log.error(
                "LLM: neither backend is available. Local: %s. T1: %s.",
                local_error, remote_health.get("error", "unknown"),
            )
            return self._chosen

        self._chosen = self._remote
        self._reason = local_error
        log.warning(
            "LLM: the local model is unavailable (%s) — answering through "
            "T1 at %s instead.",
            local_error, getattr(self._settings, "t1_base_url", ""),
        )
        return self._chosen

    # ── the backend interface, forwarded ─────────────────────────── #

    @property
    def last_tool_calls(self) -> list[dict[str, Any]]:
        active = self._chosen or self._local or self._remote
        return getattr(active, "last_tool_calls", [])

    @last_tool_calls.setter
    def last_tool_calls(self, value: list[dict[str, Any]]) -> None:
        active = self._chosen or self._local or self._remote
        if active is not None:
            active.last_tool_calls = value

    @property
    def model(self) -> str:
        active = self._chosen or self._local or self._remote
        return getattr(active, "model", "unknown")

    @property
    def active(self) -> Any:
        """Whichever backend was chosen, or None before the first check."""
        return self._chosen

    async def health_check(self) -> dict[str, Any]:
        chosen = await self._pick()
        if chosen is None:
            return {
                "ok": False,
                "error": "no LLM backend is configured",
                "models": [],
            }
        health = await chosen.health_check()
        if health.get("ok") and chosen is self._remote:
            health = dict(health)
            health["error"] = f"using T1 because {self._reason}"
        return health

    async def stream(
        self, messages: list[dict], tools: list[dict] | None = None
    ) -> AsyncIterator[str]:
        chosen = await self._pick()
        if chosen is None:
            raise RuntimeError("No LLM backend is available")
        async for chunk in chosen.stream(messages, tools=tools):
            yield chunk

    async def generate(self, messages: list[dict]) -> str:
        return "".join([chunk async for chunk in self.stream(messages)])

    async def close(self) -> None:
        for backend in (self._local, self._remote):
            if backend is not None:
                await backend.close()
