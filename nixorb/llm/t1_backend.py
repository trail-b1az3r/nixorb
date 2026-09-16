"""The T1 API as a language-model backend.

NixOrb runs its model locally, which is the point — until the local model
will not load. A GGUF whose architecture the bundled llama.cpp predates, a
machine with no GPU and no patience, a first run with nothing downloaded
yet: in every one of those the orb starts, listens, transcribes, and then
has nothing to answer with.

T1 is hypernix's hosted API. Point NixOrb at one and it answers over HTTP
instead. `llm_backend = "auto"` prefers whatever is local and only reaches
for T1 when the local model cannot serve, which is the arrangement most
people actually want: local when it works, an answer when it doesn't.

Chat runs over T1's *hyperlink* sessions, so the conversation is persisted
server-side and a reconnect keeps its thread. The SDK's chat call is
request/response rather than token-streaming, so `stream()` yields one
piece — honest about that rather than faking a trickle.
"""
from __future__ import annotations

import asyncio
import logging
from collections.abc import AsyncIterator
from typing import TYPE_CHECKING, Any

from nixorb.core.event_bus import Event, bus

if TYPE_CHECKING:
    from nixorb.settings import Settings

log = logging.getLogger(__name__)

#: Set on the module so tests can see which errors are treated as T1's.
SDK_IMPORT_HINT = (
    "The T1 backend needs hypernix. Install it with: pip install 'nixorb[t1]'"
)


class T1Error(RuntimeError):
    """The T1 backend could not serve a request."""


def _sdk() -> Any:
    """Import the T1 SDK, or explain how to get it."""
    try:
        from hypernix import t1sdk
    except ImportError as exc:
        from nixorb.hf import explain_import_error

        raise T1Error(
            explain_import_error(exc, "hypernix", "The T1 backend", extra="t1")
        ) from exc
    return t1sdk


class T1Backend:
    """Talks to a T1 API server."""

    name = "t1"

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._base_url = str(getattr(settings, "t1_base_url", "") or "").rstrip("/")
        self._key = str(getattr(settings, "t1_api_key", "") or "") or None
        self._model_id = str(getattr(settings, "t1_model", "") or "")
        self._timeout = float(getattr(settings, "t1_timeout", 30.0) or 30.0)
        self._client: Any = None
        self._session_id: str | None = None
        self.last_tool_calls: list[dict[str, Any]] = []

    @property
    def model(self) -> str:
        return self._model_id or "(routed by T1)"

    # ── connection ───────────────────────────────────────────────── #

    def _connect(self) -> Any:
        """Build the client. Blocking; the SDK is synchronous."""
        if self._client is not None:
            return self._client
        if not self._base_url:
            raise T1Error(
                "t1_base_url is not set. Point it at your T1 API server, "
                "e.g. t1_base_url = \"https://t1.example.org\"."
            )

        t1sdk = _sdk()
        self._client = t1sdk.T1Client(
            self._base_url, credential=self._key, timeout=self._timeout
        )
        return self._client

    async def _client_async(self) -> Any:
        return await asyncio.to_thread(self._connect)

    async def close(self) -> None:
        self._client = None
        self._session_id = None

    # ── health ───────────────────────────────────────────────────── #

    async def health_check(self) -> dict[str, Any]:
        """Is the server reachable, and does this key work on it?"""
        try:
            client = await self._client_async()
        except T1Error as exc:
            return {"ok": False, "error": str(exc), "models": []}

        def _probe() -> dict[str, Any]:
            t1sdk = _sdk()
            try:
                if not client.ping():
                    return {
                        "ok": False,
                        "error": f"T1 server at {self._base_url} is not answering",
                        "models": [],
                    }
                models = [m.model_id for m in client.list_models()]
            except t1sdk.T1AuthError as exc:
                return {
                    "ok": False,
                    "error": (
                        f"T1 refused the key ({exc}). Set t1_api_key to a valid "
                        "T1 key, or mint one from your T1 account page."
                    ),
                    "models": [],
                }
            except t1sdk.T1Error as exc:
                return {"ok": False, "error": f"T1: {exc}", "models": []}
            except Exception as exc:
                return {"ok": False, "error": f"T1: {exc}", "models": []}

            if self._model_id and self._model_id not in models:
                return {
                    "ok": False,
                    "error": (
                        f"T1 has no model '{self._model_id}'. Available: "
                        + (", ".join(sorted(models)[:8]) or "none")
                    ),
                    "models": models,
                }
            return {"ok": True, "error": "", "models": models}

        return await asyncio.to_thread(_probe)

    # ── chat ─────────────────────────────────────────────────────── #

    def _ensure_session(self, system_prompt: str) -> str:
        if self._session_id is not None:
            return self._session_id
        client = self._connect()
        created = client.hyperlink_create_session(
            title="NixOrb",
            model_id=self._model_id,
            system_prompt=system_prompt,
        )
        session_id = str(
            created.get("session_id") or created.get("id") or ""
        )
        if not session_id:
            raise T1Error(f"T1 did not return a session id: {created!r}")
        self._session_id = session_id
        log.info("LLM: T1 session %s open on %s", session_id, self._base_url)
        return session_id

    @staticmethod
    def _split(messages: list[dict]) -> tuple[str, str]:
        """The system prompt, and the latest thing the user said."""
        system = " ".join(
            str(m.get("content") or "")
            for m in messages
            if m.get("role") == "system"
        ).strip()
        latest = ""
        for message in reversed(messages):
            if message.get("role") == "user":
                latest = str(message.get("content") or "")
                break
        return system, latest

    def _turn(self, messages: list[dict]) -> str:
        """One blocking chat turn."""
        t1sdk = _sdk()
        system, content = self._split(messages)
        if not content:
            return ""

        try:
            session_id = self._ensure_session(system)
            reply = self._client.hyperlink_chat(
                session_id,
                content,
                model_id=self._model_id or None,
                max_tokens=int(getattr(self._settings, "llm_max_tokens", 4096)),
            )
        except t1sdk.T1QuotaError as exc:
            raise T1Error(
                f"T1 quota reached: {exc}. Check `nixorb status`, or wait for "
                "the window to roll over."
            ) from exc
        except t1sdk.T1AuthError as exc:
            raise T1Error(f"T1 refused the key: {exc}") from exc
        except t1sdk.T1Error as exc:
            raise T1Error(f"T1: {exc}") from exc

        return _reply_text(reply)

    async def stream(
        self, messages: list[dict], tools: list[dict] | None = None
    ) -> AsyncIterator[str]:
        """Yield the reply. T1's chat is request/response, so this is one piece."""
        self.last_tool_calls = []
        await bus.emit(
            Event.LLM_START, data={"model": self.model},
            source="T1Backend", priority=3,
        )

        text = await asyncio.to_thread(self._turn, messages)
        if text:
            await bus.emit(
                Event.LLM_CHUNK, data={"chunk": text},
                source="T1Backend", priority=3,
            )
            yield text

        await bus.emit(
            Event.LLM_DONE, data={"model": self.model},
            source="T1Backend", priority=3,
        )

    async def generate(self, messages: list[dict]) -> str:
        return "".join([chunk async for chunk in self.stream(messages)])


def _reply_text(payload: Any) -> str:
    """Pull the assistant's words out of whatever shape T1 returned.

    The endpoint is documented as lifting the reply out for you, but the
    key it lands under has moved between versions, so try the likely ones
    rather than hard-coding a guess that breaks on upgrade.
    """
    if isinstance(payload, str):
        return payload
    if not isinstance(payload, dict):
        return str(payload or "")

    for key in ("reply", "content", "text", "output", "answer"):
        value = payload.get(key)
        if isinstance(value, str) and value.strip():
            return value
        if isinstance(value, dict):
            nested = value.get("content") or value.get("text")
            if isinstance(nested, str) and nested.strip():
                return nested

    message = payload.get("message")
    if isinstance(message, dict):
        content = message.get("content")
        if isinstance(content, str):
            return content

    messages = payload.get("messages")
    if isinstance(messages, list):
        for entry in reversed(messages):
            if isinstance(entry, dict) and entry.get("role") == "assistant":
                content = entry.get("content")
                if isinstance(content, str):
                    return content

    log.warning("LLM: T1 reply had no recognisable text: %r", payload)
    return ""
