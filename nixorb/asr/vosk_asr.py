"""Vosk — offline speech recognition with no torch and no downloads to plan.

faster-whisper is the sensible default, but it is still a few hundred
megabytes of CTranslate2 and wants a couple of gigabytes of RAM for
anything above `base`. Vosk is Kaldi: a 40 MB model, a few hundred
megabytes of RAM, entirely offline, and it fetches its own model by
language the first time it runs.

That makes it the floor NixOrb can always reach — a Raspberry Pi, a
low-memory laptop, a machine where nothing ML-shaped is installed. It is
less accurate than Whisper and it does not punctuate. It also never fails
to load, which the alternatives cannot say.
"""
from __future__ import annotations

import contextlib
import json
import logging
from typing import TYPE_CHECKING, Any

import numpy as np

from nixorb.asr.base import SAMPLE_RATE, ASREngine

if TYPE_CHECKING:
    from nixorb.settings import Settings

log = logging.getLogger(__name__)

#: Vosk names its models `vosk-model-small-en-us-0.15` and so on. Given a
#: bare language it downloads the small model for it.
DEFAULT_LANGUAGE = "en-us"


class VoskASREngine(ASREngine):
    """Offline ASR through Vosk/Kaldi."""

    name = "vosk"
    supports_streaming = False
    #: It needs nothing that can break, so there is nothing to fall back to.
    whisper_fallback = False

    def __init__(self, settings: Settings) -> None:
        super().__init__(settings)
        self._model_name = str(getattr(settings, "asr_model", "") or "")
        self._language = _language_of(settings)

    # ── loading ──────────────────────────────────────────────────── #

    def _load(self) -> Any:
        try:
            import vosk
        except ImportError as exc:
            from nixorb.hf import explain_import_error

            raise RuntimeError(
                explain_import_error(exc, "vosk", "Vosk speech recognition",
                                     extra="vosk")
            ) from exc

        # Quiet: Vosk logs its Kaldi graph construction at full volume.
        with contextlib.suppress(Exception):
            vosk.SetLogLevel(-1)

        from pathlib import Path

        candidate = Path(self._model_name).expanduser() if self._model_name else None
        try:
            if candidate is not None and candidate.is_dir():
                log.info("ASR: loading Vosk model from %s", candidate)
                return vosk.Model(model_path=str(candidate))
            if self._model_name and self._model_name.startswith("vosk-model"):
                log.info("ASR: fetching Vosk model %s", self._model_name)
                return vosk.Model(model_name=self._model_name)
            log.info("ASR: fetching the small Vosk model for %s", self._language)
            return vosk.Model(lang=self._language)
        except Exception as exc:
            raise RuntimeError(
                f"Vosk could not load a model for '{self._model_name or self._language}': "
                f"{exc}. Set asr_model to a language like 'en-us', a model "
                "name like 'vosk-model-small-en-us-0.15', or the path of an "
                "unpacked model directory."
            ) from exc

    def _release(self, model: Any) -> None:
        del model

    # ── transcription ────────────────────────────────────────────── #

    def _transcribe(self, audio: np.ndarray) -> str:
        if self._model is None:
            raise RuntimeError("Vosk model not loaded")

        import vosk

        recogniser = vosk.KaldiRecognizer(self._model, float(SAMPLE_RATE))
        recogniser.SetWords(False)

        # Vosk wants 16-bit little-endian PCM, which is what the recorder
        # captures before it is normalised to float32.
        pcm = np.clip(np.asarray(audio, dtype=np.float32), -1.0, 1.0)
        pcm16 = (pcm * 32767.0).astype("<i2").tobytes()

        recogniser.AcceptWaveform(pcm16)
        return _text_of(recogniser.FinalResult())


def _language_of(settings: Any) -> str:
    """The Vosk language tag to fetch a model for."""
    raw = str(getattr(settings, "asr_language", "") or "").strip().lower()
    if not raw or raw == "auto":
        return DEFAULT_LANGUAGE
    # "en" -> "en-us"; "en-GB" -> "en-gb"; a full tag passes through.
    if raw == "en":
        return DEFAULT_LANGUAGE
    return raw.replace("_", "-")


def _text_of(result: str) -> str:
    """Pull the transcript out of Vosk's JSON result."""
    try:
        return str(json.loads(result).get("text") or "").strip()
    except (ValueError, AttributeError):
        log.debug("ASR: unparseable Vosk result %r", result[:200])
        return ""
