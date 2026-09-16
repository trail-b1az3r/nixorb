"""Build the TTS backend named by settings.

    tts_backend = "piper"        # offline, AUR piper-tts, espeak fallback
    tts_backend = "huggingface"  # any TTS model on the Hub
    tts_backend = "glados"       # the GLaDOS voice, via SpeechT5
    tts_backend = "openai"       # any OpenAI-compatible speech endpoint
    tts_backend = "espeak"       # force the always-available fallback
"""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from nixorb.settings import Settings

log = logging.getLogger(__name__)

_ALIASES = {"hf": "huggingface", "transformers": "huggingface",
            "espeak-ng": "espeak", "piper-tts": "piper"}

BACKENDS = ("piper", "huggingface", "glados", "openai", "espeak")

# Tried in this order when the configured backend cannot run. Piper is not
# first: it needs a separately-installed binary and a downloaded voice, so
# landing there by default meant landing on espeak in practice.
DEFAULT_FALLBACKS = ("huggingface", "piper", "espeak")


def normalise_backend(name: str | None) -> str:
    key = (name or "piper").strip().lower()
    return _ALIASES.get(key, key)


def _build_one(backend: str, settings: Settings) -> tuple[Any, str]:
    """Construct one backend, or say why it cannot run here.

    The reason matters: "cannot run" alone is the sort of message that
    sends people looking in the wrong place, when what they need is one
    pip command.
    """
    if backend == "huggingface":
        from nixorb.tts.hf_tts import HuggingFaceTTS

        hf_engine = HuggingFaceTTS(settings)
        if hf_engine.available:
            return hf_engine, ""
        return None, (
            "transformers is not installed — install it with: "
            "pip install 'nixorb[hf]'"
        )

    if backend == "glados":
        from nixorb.tts.glados_tts import GladosTTS

        return GladosTTS(settings), ""

    if backend == "openai":
        from nixorb.tts.openai_tts import OpenAITTS

        return OpenAITTS(settings), ""

    if backend in ("piper", "espeak"):
        from nixorb.tts.piper_tts import PiperTTS, find_piper_binary

        if backend == "piper" and find_piper_binary() is None:
            return None, (
                "the piper-tts binary is not installed — "
                "install it with: yay -S piper-tts"
            )
        piper = PiperTTS(settings)
        if backend == "espeak":
            # The user asked for espeak; do not quietly upgrade to Piper.
            piper._piper_available = False
        if piper.available:
            return piper, ""
        return None, "neither piper-tts nor espeak-ng is installed"

    raise ValueError(f"unknown TTS backend {backend!r}")


def resolve_chain(settings: Settings) -> list[str]:
    """The backends to try, in order, starting with the configured one.

    Piper used to be the hardcoded landing place for every failure, so a
    machine where the chosen backend could not run ended up on Piper — and
    then, with no voice model, on espeak — without ever having been asked.
    The order is a setting now, and the configured backend always leads.
    """
    chosen = normalise_backend(getattr(settings, "tts_backend", None))
    if chosen not in BACKENDS:
        log.warning(
            "TTS: unknown tts_backend %r (choose one of %s)",
            getattr(settings, "tts_backend", None), ", ".join(BACKENDS),
        )
        chosen = "huggingface"

    configured = getattr(settings, "tts_fallbacks", None)
    if configured is None:
        fallbacks = list(DEFAULT_FALLBACKS)
    else:
        fallbacks = [normalise_backend(name) for name in configured]

    chain = [chosen]
    for name in fallbacks:
        if name not in chain and name in BACKENDS:
            chain.append(name)
    return chain


def build_tts(settings: Settings) -> Any:
    """Build the configured TTS engine, falling back only as told to.

    Every step is logged: silently ending up on a different voice than the
    one configured is the kind of thing that reads as "the setting does
    nothing".
    """
    chain = resolve_chain(settings)

    for position, backend in enumerate(chain):
        try:
            engine, reason = _build_one(backend, settings)
        except Exception as exc:
            log.warning("TTS: %s is unavailable (%s)", backend, exc)
            continue

        if engine is None:
            log.warning("TTS: %s cannot run here — %s", backend, reason)
            continue

        if position:
            log.warning(
                "TTS: '%s' could not be used — speaking through '%s' instead. "
                "Set tts_fallbacks to choose a different order.",
                chain[0], backend,
            )
        else:
            log.info("TTS: using the %s backend", backend)
        return engine

    log.error(
        "TTS: none of %s could run — the orb will be silent. "
        "Install espeak-ng for a last resort.", ", ".join(chain),
    )
    from nixorb.tts.piper_tts import PiperTTS

    return PiperTTS(settings)


create_tts = build_tts
