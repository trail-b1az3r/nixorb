"""NixOrb Piper TTS — offline text-to-speech.

Uses Piper (https://github.com/rhasspy/piper) for fast, local, neural
text-to-speech. Falls back to espeak-ng if Piper is not installed.
"""
from __future__ import annotations

import asyncio
import io
import logging
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from nixorb.core.event_bus import Event, bus

if TYPE_CHECKING:
    from nixorb.settings import Settings

log = logging.getLogger(__name__)

# Default Piper voice model
DEFAULT_VOICE = "en_US-lessac-medium"
PIPER_VOICES_DIR = Path.home() / ".local" / "share" / "piper" / "voices"

# The AUR's piper-tts package installs its binary as `piper-tts`, because
# Arch's `piper` package is the gaming-mouse configuration tool — an entirely
# unrelated program that would happily be found first. Prefer the unambiguous
# name and only fall back to `piper` for other distros and pip installs.
PIPER_BINARIES = ("piper-tts", "piper")

# Where voice models end up. The AUR piper-voices packages use the upstream
# nested layout (<lang>/<locale>/<name>/<quality>/), so those are searched
# recursively rather than by exact path.
VOICE_DIRS_FLAT = (
    PIPER_VOICES_DIR,
    Path.home() / ".piper" / "voices",
    Path.home() / ".local" / "share" / "piper-voices",
    Path("/usr/share/piper-voices"),
    Path("/usr/share/piper/voices"),
)
VOICE_DIRS_NESTED = (
    PIPER_VOICES_DIR,
    Path("/usr/share/piper-voices"),
    Path.home() / ".local" / "share" / "piper-voices",
)


def _hf_token(settings: Settings | None) -> str | None:
    """A Hugging Face token, if one is configured — voices are public, but
    a token still matters behind an authenticating proxy."""
    from nixorb import hf

    return hf.token(settings)


def find_piper_binary() -> str | None:
    """Return the Piper executable to use, or None if it is not installed."""
    for name in PIPER_BINARIES:
        path = shutil.which(name)
        if path:
            return path
    return None


class PiperTTS:
    """Offline TTS using Piper with fallback to espeak-ng."""

    def __init__(self, settings: Settings | None = None) -> None:
        if settings:
            self._voice = self._piper_voice_from(settings.tts_voice)
            self._speed = settings.tts_speed
            self._volume = settings.tts_volume
        else:
            self._voice = DEFAULT_VOICE
            self._speed = 1.0
            self._volume = 1.0

        self._hf_token = _hf_token(settings)
        self._download_voices = bool(
            getattr(settings, "tts_download_voices", True)
        ) if settings else True
        # Resolved once per process: a 60 MB download must not be attempted
        # per sentence, and a failure must not be re-logged per sentence.
        self._voice_model: Path | None = None
        self._voice_failed = False

        self._piper = find_piper_binary()
        self._piper_available = self._piper is not None
        self._espeak_available = shutil.which("espeak-ng") is not None
        self._aplay_available = shutil.which("aplay") is not None
        self._stopped = False

    name = "piper"

    @staticmethod
    def _piper_voice_from(configured: str) -> str:
        """The Piper voice to use, given whatever `tts_voice` holds.

        `tts_voice` is shared with the Hugging Face backend, whose default
        is a sentence describing a voice ("A calm, clear-voiced woman…")
        because voice-design models take one. Piper cannot use that, and
        handing it over produced nothing but a silent drop to espeak on a
        stock config. A value Piper cannot name is not an error — it just
        means the setting was written for another backend.
        """
        from nixorb.tts import piper_voices

        text = (configured or "").strip()
        if not text:
            return DEFAULT_VOICE
        if piper_voices.looks_like_path(text) or piper_voices.parse_voice_name(text):
            return text

        log.info(
            "TTS: tts_voice is a description, not a Piper voice name — "
            "using '%s'. Set tts_voice to a name like 'en_GB-alan-medium' "
            "to pick another; it is downloaded on first use.",
            DEFAULT_VOICE,
        )
        return DEFAULT_VOICE

    def stop(self) -> None:
        """Cut playback off mid-sentence (barge-in)."""
        self._stopped = True
        try:
            import sounddevice as sd

            sd.stop()
        except Exception:
            pass

    def _find_voice_model(self) -> Path | None:
        """Find the Piper voice model file."""
        filename = f"{self._voice}.onnx"

        for directory in VOICE_DIRS_FLAT:
            candidate = directory / filename
            if candidate.exists():
                return candidate

        # AUR piper-voices keeps upstream's nested layout, e.g.
        # /usr/share/piper-voices/en/en_US/lessac/medium/en_US-lessac-medium.onnx
        for directory in VOICE_DIRS_NESTED:
            if not directory.is_dir():
                continue
            try:
                match = next(directory.rglob(filename), None)
            except OSError:
                continue
            if match is not None:
                return match

        return None

    def _resolve_voice_model(self) -> Path | None:
        """The model for the configured voice, downloading it if needed.

        Piper ships no voices and the installer fetches exactly one, so
        without this every other value of `tts_voice` found nothing and
        dropped silently to espeak — which looked like Piper supporting a
        single voice.
        """
        if self._voice_model is not None:
            return self._voice_model
        if self._voice_failed:
            # Already tried and explained why; do not re-download per
            # sentence, and do not repeat the message on every one.
            return None

        from nixorb.tts import piper_voices

        try:
            model = piper_voices.ensure(
                self._voice,
                search=lambda _voice: self._find_voice_model(),
                dest=PIPER_VOICES_DIR,
                token=self._hf_token,
                allow_download=self._download_voices,
            )
        except piper_voices.VoiceUnavailable as exc:
            self._voice_failed = True
            log.warning("TTS: %s — using espeak-ng instead", exc)
            return None
        except Exception as exc:
            self._voice_failed = True
            log.warning(
                "TTS: could not prepare Piper voice '%s' (%s) — using "
                "espeak-ng instead", self._voice, exc,
            )
            return None

        self._voice_model = model
        return model

    @property
    def available(self) -> bool:
        """True if any TTS engine is installed."""
        return self._piper_available or self._espeak_available

    async def speak(self, text: str) -> None:
        """Speak the given text aloud."""
        if not text or not text.strip():
            return

        text = text.strip()
        self._stopped = False
        log.info("TTS: speaking '%s…'", text[:60])
        await bus.emit(Event.TTS_START, data={"text": text[:200]},
                       source="PiperTTS")

        if self._piper_available:
            await self._speak_piper(text)
        elif self._espeak_available:
            log.info("TTS: piper not installed — using espeak-ng")
            await self._speak_espeak(text)
        else:
            msg = (
                "No TTS engine installed — NixOrb will stay silent. "
                "Install the AUR's piper-tts, or espeak-ng."
            )
            log.error("TTS: %s", msg)
            await bus.emit(Event.TTS_ERROR, data={"error": msg},
                           source="PiperTTS")
            await bus.emit(
                Event.LOG,
                data={"level": "warning", "msg": f"🔇 {msg}"},
                source="PiperTTS",
            )
            return

        await bus.emit(Event.TTS_DONE, source="PiperTTS")

    def _speak_piper_sync(self, text: str) -> None:
        """Synchronous Piper TTS (runs in executor)."""
        model_path = self._resolve_voice_model()

        if model_path is None:
            self._speak_espeak_sync(text)
            return

        config_path = model_path.with_suffix(".onnx.json")

        try:
            # Run piper to generate WAV audio
            proc = subprocess.Popen(
                [
                    str(self._piper),
                    "--model", str(model_path),
                    "--config", str(config_path) if config_path.exists() else "",
                    "--output_file", "-",
                    "--length-scale", str(1.0 / self._speed),
                ],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            stdout, stderr = proc.communicate(text.encode("utf-8"), timeout=30)

            if proc.returncode != 0:
                log.error("TTS: piper failed: %s", stderr.decode())
                self._speak_espeak_sync(text)
                return

            # Play the audio
            self._play_wav_bytes(stdout)

        except subprocess.TimeoutExpired:
            log.error("TTS: piper timed out")
            proc.kill()
        except Exception as exc:
            log.error("TTS: piper error: %s", exc)
            self._speak_espeak_sync(text)

    def _speak_espeak_sync(self, text: str) -> None:
        """Synchronous espeak-ng TTS fallback."""
        try:
            # Generate WAV with espeak-ng
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
                wav_path = f.name

            subprocess.run(
                ["espeak-ng", "-w", wav_path, "-s", "150", text],
                check=True,
                capture_output=True,
                timeout=30,
            )

            # Read and play
            import wave

            with wave.open(wav_path, "rb") as wf:
                data = wf.readframes(wf.getnframes())
                self._play_pcm(data, wf.getframerate(), wf.getnchannels())

            Path(wav_path).unlink(missing_ok=True)

        except Exception as exc:
            log.error("TTS: espeak-ng error: %s", exc)

    def _play_wav_bytes(self, wav_data: bytes) -> None:
        """Play WAV audio data using sounddevice."""
        try:
            import wave

            with io.BytesIO(wav_data) as f:
                with wave.open(f, "rb") as wf:
                    data = wf.readframes(wf.getnframes())
                    sample_rate = wf.getframerate()
                    channels = wf.getnchannels()
                    self._play_pcm(data, sample_rate, channels)
        except Exception as exc:
            log.error("TTS: WAV playback error: %s", exc)

    def _play_pcm(self, data: bytes, sample_rate: int, channels: int) -> None:
        """Play raw PCM audio data."""
        if self._stopped:
            return
        try:
            import sounddevice as sd

            # Convert bytes to numpy array
            audio = np.frombuffer(data, dtype=np.int16).astype(np.float32) / 32768.0

            if channels > 1:
                audio = audio.reshape(-1, channels)

            # Apply volume
            audio = audio * self._volume

            # Play
            sd.play(audio, samplerate=sample_rate)
            sd.wait()

        except Exception as exc:
            log.error("TTS: PCM playback error: %s", exc)

    async def _speak_piper(self, text: str) -> None:
        """Async wrapper for Piper TTS."""
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, self._speak_piper_sync, text)

    async def _speak_espeak(self, text: str) -> None:
        """Async wrapper for espeak TTS."""
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, self._speak_espeak_sync, text)

    async def synthesize_to_file(self, text: str, output_path: Path) -> bool:
        """Synthesize speech to a WAV file."""
        if not self._piper_available:
            return False

        model_path = self._find_voice_model()
        if model_path is None:
            return False

        try:
            proc = await asyncio.create_subprocess_exec(
                str(self._piper),
                "--model", str(model_path),
                "--output_file", str(output_path),
                "--length-scale", str(1.0 / self._speed),
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            await proc.communicate(text.encode("utf-8"))
            return proc.returncode == 0
        except Exception as exc:
            log.error("TTS: synthesize error: %s", exc)
            return False
