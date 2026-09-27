"""Pick a speech-to-text engine from settings.

    asr_backend = "faster_whisper"   # CTranslate2 Whisper, self-contained
    asr_backend = "huggingface"      # any ASR model on the Hub
    asr_backend = "nemotron"         # NVIDIA Nemotron 3.5, native streaming
    asr_backend = "vosk"             # Kaldi, offline, tiny, no torch

All three expose the same surface (``name``, ``supports_streaming``,
``is_loaded``, ``preload``, ``unload``, ``stop_recording``,
``record_and_transcribe``, ``stream_transcribe``), so callers never need to
know which one they got.
"""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from nixorb.settings import Settings

log = logging.getLogger(__name__)

# Spellings people reasonably type, mapped to the canonical name.
_ALIASES = {
    "whisper": "faster_whisper",
    "faster-whisper": "faster_whisper",
    "fasterwhisper": "faster_whisper",
    "hf": "huggingface",
    "transformers": "huggingface",
    "nvidia": "nemotron",
    "nemotron-3.5": "nemotron",
    "kaldi": "vosk",
}

BACKENDS = ("faster_whisper", "huggingface", "nemotron", "vosk")


def normalise_backend(name: str | None) -> str:
    return _ALIASES.get((name or "faster_whisper").strip().lower(),
                        (name or "faster_whisper").strip().lower())


def is_nemotron_model(model_id: str) -> bool:
    """True for a Nemotron ASR checkpoint, whatever the backend says."""
    lowered = (model_id or "").lower()
    return "nemotron" in lowered and "asr" in lowered


def build_asr(settings: Settings) -> Any:
    """Build the configured ASR engine."""
    backend = normalise_backend(getattr(settings, "asr_backend", None))
    model_id = getattr(settings, "asr_model", "") or ""

    # Nemotron's whole point is cache-aware streaming, which the generic
    # pipeline cannot do. Route its checkpoints to the native backend even
    # when asr_backend was left at "huggingface" — otherwise the default
    # config quietly loses streaming.
    if backend == "huggingface" and is_nemotron_model(model_id):
        log.info("ASR: %s is a Nemotron checkpoint — using the native backend",
                 model_id)
        backend = "nemotron"

    # ...and the reverse: "nemotron" with a non-Nemotron model would fail to
    # load at all, so fall back to the generic engine rather than crashing.
    if backend == "nemotron" and model_id and not is_nemotron_model(model_id):
        log.warning(
            "ASR: asr_backend='nemotron' but '%s' is not a Nemotron "
            "checkpoint — using the huggingface backend instead", model_id,
        )
        backend = "huggingface"

    if backend == "vosk":
        from nixorb.asr.vosk_asr import VoskASREngine

        log.info("ASR: using Vosk backend, model '%s'", model_id or "auto")
        return VoskASREngine(settings)

    if backend == "nemotron":
        from nixorb.asr.nemotron_asr import NemotronASREngine

        log.info("ASR: using Nemotron backend, model '%s'", model_id)
        return NemotronASREngine(settings)

    if backend == "huggingface":
        from nixorb.asr.hf_asr_engine import HFASREngine

        log.info("ASR: using HuggingFace backend, model '%s'", model_id)
        return HFASREngine(settings)

    if backend != "faster_whisper":
        log.warning(
            "ASR: unknown asr_backend '%s' (choose one of %s) — "
            "falling back to faster_whisper",
            getattr(settings, "asr_backend", None), ", ".join(BACKENDS),
        )

    from nixorb.asr.whisper_engine import WhisperEngine

    log.info("ASR: using faster-whisper backend, model '%s'", model_id)
    return WhisperEngine(settings)


# Both names are in use across the codebase; keep them pointing at one thing.
create_asr = build_asr
