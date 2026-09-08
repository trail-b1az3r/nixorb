"""NixOrb settings — Pydantic v2 settings with TOML persistence."""
from __future__ import annotations

import os
import tomllib
from pathlib import Path

import tomli_w
from pydantic import BaseModel

_CONFIG_ENV = "NIXORB_CONFIG"
_DEFAULT_CONFIG = Path.home() / ".config" / "nixorb" / "config.toml"


def _config_path() -> Path:
    return Path(os.environ[_CONFIG_ENV]) if _CONFIG_ENV in os.environ else _DEFAULT_CONFIG


class Settings(BaseModel):
    """NixOrb configuration — all user-tunable parameters."""

    # ── Orb UI ───────────────────────────────────────────────────── #
    orb_x: int | None = None
    orb_y: int | None = None
    orb_size: int = 88
    orb_opacity: float = 1.0
    hotkey: str = "Ctrl+Alt+Space"

    # ── ASR ──────────────────────────────────────────────────────── #
    # "faster_whisper" — CTranslate2 Whisper models (fast, GTX 1080-friendly).
    # "huggingface"    — any transformers ASR model/pipeline (Whisper variants,
    #                     distil-whisper, wav2vec2, Parakeet, Moonshine, ...).
    asr_backend: str = "huggingface"
    # Model identifier, interpreted per-backend: a faster-whisper size/repo
    # (e.g. "large-v3") when asr_backend="faster_whisper", or any HF repo id
    # (e.g. "openai/whisper-large-v3-turbo", "nvidia/parakeet-tdt-0.6b-v2")
    # when asr_backend="huggingface".
    asr_model: str = "nvidia/nemotron-3.5-asr-streaming-0.6b"
    asr_language: str = "en"
    hf_token: str = ""
    # Chunk-and-transcribe partial results while the user is still talking,
    # instead of waiting for silence. Emits Event.ASR_PARTIAL_TRANSCRIPT.
    asr_streaming: bool = False
    asr_streaming_chunk_seconds: float = 2.5
    # Nemotron only. Its cache-aware encoder streams natively rather than
    # re-transcribing a rolling buffer, and this is the latency dial:
    # right-context in 80ms frames, 0 | 3 | 6 | 13 → 80/320/560/1120 ms.
    asr_nemotron_lookahead: int = 3
    microphone_index: int | None = None
    # Preferred input device by name substring (case-insensitive), e.g. "USB".
    # Survives reboots/replugs better than an index, which PipeWire/ALSA can
    # renumber. Tried before microphone_index if both are set.
    microphone_name: str = ""
    # 0.0 (least sensitive, needs a loud/close voice) .. 1.0 (most sensitive,
    # picks up quiet/distant speech but also more room noise). 0.5 reproduces
    # the original fixed threshold.
    mic_sensitivity: float = 0.5

    # ── LLM ──────────────────────────────────────────────────────── #
    # "ollama"      — local Ollama server (default, no API keys)
    # "huggingface" — any local HF model: safetensors via transformers, or a
    #                 GGUF file/repo via llama-cpp-python (auto-detected)
    # "openai"      — any OpenAI-compatible HTTP API (OpenAI, vLLM, LM Studio,
    #                 llama.cpp server, TGI, ...)
    llm_backend: str = "huggingface"
    # Model identifier, interpreted per-backend: an Ollama tag when
    # llm_backend="ollama", or any HF repo id / local path when
    # llm_backend="huggingface", or a model name for the OpenAI-compatible
    # endpoint when llm_backend="openai".
    llm_model: str = "empero-ai/Qwen3.8-2B-Distill-GGUF"
    ollama_host: str = "http://localhost:11434"
    # Specific GGUF filename to pick out of a multi-file HF repo, e.g.
    # "model.Q4_K_M.gguf". Leave blank to auto-pick the first .gguf, or to
    # load full-precision/safetensors weights via transformers instead.
    # Q4_K_M is the repo's recommended quant — 1.31 GB, comfortable on 8 GB VRAM.
    llm_gguf_file: str = "Qwen3.8-2B-Q4_K_M.gguf"
    openai_api_key: str = ""
    openai_base_url: str = "https://api.openai.com/v1"
    # 4-bit quantisation for the non-GGUF (transformers) path, so a larger
    # model fits beside ASR on 8 GB. Needs bitsandbytes: nixorb[quant].
    llm_hf_load_in_4bit: bool = False
    # NOTE: this is a reasoning model — every response opens with a
    # <think>...</think> block per its own model card. Nothing currently
    # strips that before it's spoken/logged.
    llm_system_prompt: str = (
        "You are NixOrb, a voice assistant running on Arch Linux with KDE "
        "Plasma 6. You're direct, dryly funny, and don't pad your answers. "
        "Your reply is read aloud, so answer the way a person would out "
        "loud: one or two short sentences, the answer first, no preamble, "
        "no markdown, no bullet lists, no code blocks unless asked. Spell "
        "out numbers and units the way you would say them. If you need to "
        "run a shell command, put it in <ACTION>…</ACTION> and say briefly "
        "what you are doing. Only give a long answer when asked for detail. "
        "You can execute bash commands, search the web, capture the screen, "
        "and remember conversations."
    )
    llm_max_tokens: int = 512
    llm_temperature: float = 0.7
    # Seconds to wait for the user to answer a command-confirmation dialog.
    action_confirm_timeout: float = 60.0

    # ── TTS ──────────────────────────────────────────────────────── #
    # "piper" | "glados" | "huggingface" | "openai"
    tts_backend: str = "huggingface"
    # For tts_backend="huggingface": Breeze-TTS-2 is a voice-design model (no
    # named presets) — this is fed to it as a natural-language voice instruction.
    tts_voice: str = "A calm, clear-voiced woman with a dry, confident wit and unhurried delivery."
    tts_speed: float = 1.0
    tts_volume: float = 1.0
    # HF repo id for tts_backend="huggingface" (e.g. "microsoft/speecht5_tts",
    # "suno/bark-small", "hexgrad/Kokoro-82M", any text-to-speech pipeline).
    # NOTE: Breeze-TTS-2 needs ~7.7 GB VRAM minimum (12 GB recommended per its
    # own card) — confirm this actually fits your card before relying on it.
    tts_hf_repo: str = "BreezeBlue/Breeze-TTS-2"
    # Piper ships no voice models. Any name from
    # https://huggingface.co/rhasspy/piper-voices is fetched on first use
    # into ~/.local/share/piper/voices; turn this off to keep NixOrb
    # entirely offline and use only voices already on disk.
    tts_download_voices: bool = True
    # SpeechT5 needs a speaker x-vector: a .npy path, a cmu-arctic-xvectors
    # index, or blank for the default voice. Ignored by other models.
    tts_hf_speaker: str = ""
    # Speak each sentence as the model produces it, rather than waiting for
    # the whole answer. This is most of what makes the orb feel responsive.
    tts_streaming: bool = True

    # ── Wake Word ────────────────────────────────────────────────── #
    # openwakeword ships as the optional "wakeword" extra, so this stays
    # off until the user installs it and opts in.
    wake_word_enabled: bool = False
    # A bundled openwakeword pretrained name ("alexa", "hey_jarvis",
    # "hey_mycroft", "timer", "weather"), or an absolute path to a custom
    # .onnx/.tflite model you trained yourself. "hey_nixorb" is not a real
    # model — nobody has trained one — so this falls back to "hey_jarvis"
    # at runtime with a warning until you point it at a real one.
    # Comma-separate multiple names/paths to activate on any of them, e.g.
    # "~/.local/share/nixorb/wakeword/hey_nixorb.onnx,~/.local/share/nixorb/wakeword/nixorb.onnx".
    wake_word_model: str = "hey_nixorb"
    wake_word_sensitivity: float = 0.5

    # ── Conversation ─────────────────────────────────────────────── #
    # Triggering while the orb is talking stops it and starts listening,
    # rather than queueing behind the current answer.
    barge_in: bool = True
    # After answering, keep listening this long for a follow-up without
    # needing the hotkey again. 0 disables it.
    follow_up_seconds: float = 6.0
    # Cap spoken replies when tts_streaming is off.
    tts_max_sentences: int = 6

    # ── Hugging Face ─────────────────────────────────────────────── #
    # "auto" | "cuda" | "cpu"
    hf_device: str = "auto"
    # Override the model cache location (default: ~/.cache/huggingface).
    hf_cache_dir: str = ""
    # Off by default: this executes Python from the model repo. Some models
    # (custom architectures) do not load without it.
    hf_trust_remote_code: bool = False

    # ── Features ─────────────────────────────────────────────────── #
    screen_capture_enabled: bool = True
    web_search_enabled: bool = True
    web_search_max_results: int = 4
    clipboard_enabled: bool = True
    require_action_confirmation: bool = True
    # bubblewrap sandbox for <ACTION> commands. Off by default: the
    # sandbox is read-only with no network, so an approved command that
    # writes a file or installs a package fails for no visible reason.
    sandbox_actions: bool = False
    memory_enabled: bool = True
    plugins_enabled: bool = True

    # ── VRAM ─────────────────────────────────────────────────────── #
    vram_total_mb: int = 8192
    vram_system_reserve_mb: int = 512
    vram_safety_buffer_mb: int = 256

    # ── Paths ────────────────────────────────────────────────────── #
    plugin_dir: str = str(Path.home() / ".local" / "share" / "nixorb" / "plugins")
    memory_dir: str = str(Path.home() / ".local" / "share" / "nixorb" / "memory")

    @classmethod
    def load(cls) -> Settings:
        """Load settings from config file, creating defaults if missing."""
        p = _config_path()
        p.parent.mkdir(parents=True, exist_ok=True)
        if p.exists():
            try:
                with open(p, "rb") as f:
                    data = tomllib.load(f)
                return cls(**data)
            except Exception as exc:
                import logging
                logging.getLogger(__name__).error(
                    "Config load failed, using defaults: %s", exc
                )
        return cls()

    def save(self) -> None:
        """Persist current settings to config file."""
        p = _config_path()
        p.parent.mkdir(parents=True, exist_ok=True)
        data = {k: v for k, v in self.model_dump().items() if v is not None}
        with open(p, "wb") as f:
            tomli_w.dump(data, f)
