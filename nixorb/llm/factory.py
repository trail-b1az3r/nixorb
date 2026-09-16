"""Pick an LLM backend from settings.

    llm_backend = "ollama"        # local Ollama server, no Python deps
    llm_backend = "huggingface"   # any local model: GGUF, or transformers
    llm_backend = "openai"        # any OpenAI-compatible HTTP endpoint

Before this existed, main.py always constructed OllamaBackend regardless of
``settings.llm_backend`` — the setting was logged in the startup banner but
never consulted, so choosing "huggingface" silently kept running Ollama and
ran Ollama's health check against a HuggingFace repo id.
"""
from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from typing import TYPE_CHECKING, Any, Protocol

from nixorb.core.event_bus import Event, bus

if TYPE_CHECKING:
    from nixorb.settings import Settings

log = logging.getLogger(__name__)

_ALIASES = {"hf": "huggingface", "transformers": "huggingface",
            "local": "huggingface", "openai-compatible": "openai",
            "vllm": "openai", "lmstudio": "openai",
            "hypernix": "t1", "t1api": "t1", "remote": "t1"}

BACKENDS = ("auto", "ollama", "huggingface", "openai", "t1")


class LLMBackend(Protocol):
    """What main.py needs from any language model backend."""

    last_tool_calls: list[dict[str, Any]]

    @property
    def model(self) -> str:
        """The model id actually in use, for logs and `nixorb status`."""
        ...

    def stream(
        self, messages: list[dict], tools: list[dict] | None = None
    ) -> AsyncIterator[str]: ...

    async def generate(self, messages: list[dict]) -> str: ...

    async def health_check(self) -> dict[str, Any]: ...

    async def close(self) -> None: ...


def normalise_backend(name: str | None) -> str:
    key = (name or "ollama").strip().lower()
    return _ALIASES.get(key, key)


def build_t1(settings: Settings) -> Any:
    """The T1 API backend, or None when it is not configured."""
    if not str(getattr(settings, "t1_base_url", "") or "").strip():
        return None
    from nixorb.llm.t1_backend import T1Backend

    return T1Backend(settings)


def build_llm(settings: Settings) -> Any:
    """Build the configured LLM backend."""
    backend = normalise_backend(getattr(settings, "llm_backend", None))

    if backend == "auto":
        # Local when it loads, T1 when it does not. The commonest way the
        # orb ends up mute is a local model that will not load at all.
        from nixorb.llm.auto_backend import AutoBackend

        local_name = normalise_backend(
            getattr(settings, "llm_local_backend", "huggingface")
        )
        if local_name in ("auto", "t1"):
            local_name = "huggingface"
        local = build_llm(_with_backend(settings, local_name))
        remote = build_t1(settings)
        log.info(
            "LLM: auto — local '%s', %s",
            local_name,
            f"T1 at {settings.t1_base_url}" if remote else "no T1 configured",
        )
        return AutoBackend(settings, local, remote)

    if backend == "t1":
        from nixorb.llm.t1_backend import T1Backend

        log.info(
            "LLM: using the T1 API at %s, model '%s'",
            getattr(settings, "t1_base_url", ""),
            getattr(settings, "t1_model", "") or "(routed by T1)",
        )
        return T1Backend(settings)

    if backend == "huggingface":
        from nixorb.llm.hf_llm_backend import HuggingFaceLLMBackend

        settings = _resolve_model_name(settings)
        log.info("LLM: using HuggingFace backend, model '%s'", settings.llm_model)
        return HuggingFaceLLMBackend(settings)

    if backend == "openai":
        from nixorb.llm.openai_compat_backend import OpenAICompatBackend

        log.info(
            "LLM: using OpenAI-compatible backend at %s, model '%s'",
            getattr(settings, "openai_base_url", ""), settings.llm_model,
        )
        return OpenAICompatBackend(settings)

    if backend != "ollama":
        log.warning(
            "LLM: unknown llm_backend '%s' (choose one of %s) — using ollama",
            getattr(settings, "llm_backend", None), ", ".join(BACKENDS),
        )

    from nixorb.llm.ollama_backend import OllamaBackend

    log.info("LLM: using Ollama backend, model '%s'", settings.llm_model)
    return OllamaBackend(settings)


def _resolve_model_name(settings: Settings) -> Settings:
    """Let `llm_model` be a short catalogue name.

    hypernix ships a catalogue of ~115 models, so "qwen3.5-4b" can mean
    "Qwen/Qwen3.5-4B" without the user having to know the owner. A name
    that is already a repo id, or that the catalogue does not know, is
    left exactly as written.
    """
    name = str(getattr(settings, "llm_model", "") or "")
    if not name or "/" in name:
        return settings

    from nixorb.utils.hypernix_client import HypernixClient

    resolved = HypernixClient(settings).resolve_repo_id(name)
    if resolved == name:
        return settings

    log.info("LLM: '%s' is '%s' in the hypernix catalogue", name, resolved)
    try:
        return settings.model_copy(update={"llm_model": resolved})
    except AttributeError:  # pragma: no cover
        return settings


def _with_backend(settings: Settings, backend: str) -> Settings:
    """A copy of `settings` naming a different llm_backend.

    Used so `auto` can build its local half through the same factory
    rather than a second, drifting copy of the dispatch.
    """
    try:
        return settings.model_copy(update={"llm_backend": backend})
    except AttributeError:  # pragma: no cover - a stand-in in tests
        import copy

        clone = copy.copy(settings)
        clone.llm_backend = backend
        return clone


create_llm = build_llm


class OfflineFallbackManager:
    """Switch to a second backend after the first fails repeatedly.

    Useful when the primary is a network-dependent daemon and the fallback
    is an in-process model (or the other way round).
    """

    FAIL_THRESHOLD = 3

    def __init__(self, primary: Any, fallback: Any) -> None:
        self._primary = primary
        self._fallback = fallback
        self._fail_count = 0
        self._using_fallback = False
        self.last_tool_calls: list[dict[str, Any]] = []

    @property
    def active(self) -> Any:
        return self._fallback if self._using_fallback else self._primary

    @property
    def model(self) -> str:
        return str(getattr(self.active, "model", "unknown"))

    async def health_check(self) -> dict[str, Any]:
        return await self.active.health_check()

    async def close(self) -> None:
        for backend in (self._primary, self._fallback):
            close = getattr(backend, "close", None)
            if close is not None:
                await close()

    async def stream(
        self, messages: list[dict], tools: list[dict] | None = None
    ) -> AsyncIterator[str]:
        failed = False
        try:
            async for chunk in self.active.stream(messages, tools):
                yield chunk
            self._fail_count = 0
        except Exception as exc:
            failed = True
            self._fail_count += 1
            log.error(
                "LLM backend error (%d/%d): %s",
                self._fail_count, self.FAIL_THRESHOLD, exc,
            )

        if not failed:
            self.last_tool_calls = list(
                getattr(self.active, "last_tool_calls", [])
            )
            return

        if self._fail_count >= self.FAIL_THRESHOLD and not self._using_fallback:
            self._using_fallback = True
            log.warning("Switching to the fallback LLM backend")
            await bus.emit(
                Event.LOG,
                data={"level": "warning",
                      "msg": "⚠️ Primary model unreachable — switched to fallback"},
                source="OfflineFallbackManager",
            )

        if self._using_fallback:
            async for chunk in self._fallback.stream(messages, tools):
                yield chunk
            self.last_tool_calls = list(
                getattr(self._fallback, "last_tool_calls", [])
            )

    async def generate(self, messages: list[dict]) -> str:
        return "".join([chunk async for chunk in self.stream(messages)])
