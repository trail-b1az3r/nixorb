"""Kokoro — a good neural voice that does not need torch.

The Hugging Face TTS backend is the flexible one, and it wants torch plus
a matching transformers. On a machine without that stack — which is most
machines that have not been set up for ML — it cannot run, and NixOrb
falls back to Piper's binary or to espeak's robot.

Kokoro is an 82M-parameter model that runs through onnxruntime. No torch,
no CUDA, no transformers version to match, a few hundred megabytes, and it
sounds like a person. That makes it the best voice available to a plain
install, which is why it is here.

It needs two files — the model and the voice pack — and they are not
bundled with the pip package. `tts_kokoro_model` and `tts_kokoro_voices`
take paths; left blank, NixOrb fetches them from the Hub once and caches
them under ~/.local/share/nixorb/kokoro.
"""
from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from nixorb.core.event_bus import Event, bus

if TYPE_CHECKING:
    from nixorb.settings import Settings

log = logging.getLogger(__name__)

#: Where the two files are cached when they are not given as paths.
CACHE_DIR = Path.home() / ".local" / "share" / "nixorb" / "kokoro"

#: The ONNX export and its voice pack. Both are needed; the pip package
#: ships neither.
DEFAULT_REPO = "onnx-community/Kokoro-82M-v1.0-ONNX"
DEFAULT_MODEL_FILE = "onnx/model_quantized.onnx"
DEFAULT_VOICES_FILE = "voices.bin"

DEFAULT_VOICE = "af_heart"


class KokoroUnavailable(RuntimeError):
    """Kokoro cannot run — the package or its model files are missing."""


class KokoroTTS:
    """Neural text-to-speech through onnxruntime."""

    name = "kokoro"

    def __init__(self, settings: Settings | None = None) -> None:
        self._settings = settings
        self._voice = _voice_name(getattr(settings, "tts_voice", "") if settings else "")
        self._speed = float(getattr(settings, "tts_speed", 1.0) or 1.0)
        self._language = str(getattr(settings, "tts_language", "") or "en-us")
        self._model_path = str(getattr(settings, "tts_kokoro_model", "") or "")
        self._voices_path = str(getattr(settings, "tts_kokoro_voices", "") or "")
        self._engine: Any = None
        self._stopped = False

    # ── availability ─────────────────────────────────────────────── #

    @property
    def available(self) -> bool:
        """True when the package is importable. Files are fetched lazily."""
        import importlib.util

        try:
            return importlib.util.find_spec("kokoro_onnx") is not None
        except (ImportError, ValueError):
            return False

    def stop(self) -> None:
        self._stopped = True
        try:
            import sounddevice as sd

            sd.stop()
        except Exception:
            pass

    # ── loading ──────────────────────────────────────────────────── #

    def _resolve_files(self) -> tuple[str, str]:
        """The model and voice-pack paths, downloading them if need be."""
        if self._model_path and self._voices_path:
            for path in (self._model_path, self._voices_path):
                if not Path(path).expanduser().is_file():
                    raise KokoroUnavailable(
                        f"{path} does not exist. tts_kokoro_model and "
                        "tts_kokoro_voices must both point at real files, or "
                        "be left blank to download them."
                    )
            return (
                str(Path(self._model_path).expanduser()),
                str(Path(self._voices_path).expanduser()),
            )

        try:
            from huggingface_hub import hf_hub_download
        except ImportError as exc:  # pragma: no cover - a base dependency
            raise KokoroUnavailable(
                "huggingface_hub is needed to fetch the Kokoro model"
            ) from exc

        from nixorb import hf

        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        token = hf.token(self._settings)
        got: list[str] = []
        for filename in (DEFAULT_MODEL_FILE, DEFAULT_VOICES_FILE):
            try:
                got.append(
                    hf_hub_download(
                        repo_id=DEFAULT_REPO,
                        filename=filename,
                        local_dir=str(CACHE_DIR),
                        token=token,
                    )
                )
            except Exception as exc:
                raise KokoroUnavailable(
                    f"could not fetch {filename} from {DEFAULT_REPO}: {exc}. "
                    "Download the Kokoro ONNX model and voice pack by hand and "
                    "set tts_kokoro_model and tts_kokoro_voices to them."
                ) from exc
        return got[0], got[1]

    def _load(self) -> Any:
        try:
            from kokoro_onnx import Kokoro
        except ImportError as exc:
            from nixorb.hf import explain_import_error

            raise KokoroUnavailable(
                explain_import_error(exc, "kokoro_onnx", "The Kokoro voice",
                                     extra="kokoro")
            ) from exc

        model_path, voices_path = self._resolve_files()
        log.info("TTS: loading Kokoro from %s", model_path)
        return Kokoro(model_path, voices_path)

    def _synthesise(self, text: str) -> tuple[np.ndarray, int]:
        if self._engine is None:
            self._engine = self._load()
        audio, rate = self._engine.create(
            text, voice=self._voice, speed=self._speed, lang=self._language
        )
        return np.asarray(audio, dtype=np.float32), int(rate)

    # ── speaking ─────────────────────────────────────────────────── #

    async def speak(self, text: str) -> None:
        if not text.strip():
            return
        self._stopped = False
        await bus.emit(Event.TTS_START, source="KokoroTTS")
        try:
            audio, rate = await asyncio.to_thread(self._synthesise, text)
            if not self._stopped:
                await asyncio.to_thread(self._play, audio, rate)
        except KokoroUnavailable as exc:
            log.error("TTS: %s", exc)
        except Exception as exc:
            log.error("TTS: Kokoro failed: %s", exc)
        finally:
            await bus.emit(Event.TTS_DONE, source="KokoroTTS")

    def _play(self, audio: np.ndarray, rate: int) -> None:
        import sounddevice as sd

        volume = float(getattr(self._settings, "tts_volume", 1.0) or 1.0)
        sd.play(audio * volume, samplerate=rate, blocking=True)
        sd.wait()

    async def synthesize_to_file(self, text: str, output_path: Path) -> bool:
        try:
            audio, rate = await asyncio.to_thread(self._synthesise, text)
        except Exception as exc:
            log.error("TTS: Kokoro could not synthesise: %s", exc)
            return False
        try:
            import soundfile as sf

            await asyncio.to_thread(sf.write, str(output_path), audio, rate)
        except Exception as exc:
            log.error("TTS: could not write %s: %s", output_path, exc)
            return False
        return True


def _voice_name(configured: str) -> str:
    """Kokoro voices are short ids like `af_heart`, not prose.

    `tts_voice` is shared with every other backend, and its default is a
    sentence describing a voice for the Hugging Face voice-design models.
    Handing that to Kokoro picks nothing.
    """
    text = (configured or "").strip()
    if not text:
        return DEFAULT_VOICE
    # A Kokoro voice id: two letters, an underscore, a name.
    if len(text) <= 24 and " " not in text and "_" in text:
        return text
    log.info(
        "TTS: tts_voice is not a Kokoro voice id — using '%s'. Kokoro voices "
        "look like 'af_heart', 'am_michael', 'bf_emma'.", DEFAULT_VOICE,
    )
    return DEFAULT_VOICE
