"""Pick settings this machine can actually run, and write them down.

The shipped defaults aim high — a Nemotron streaming ASR, a reasoning
GGUF, a Hugging Face voice model. On a machine with the right stack that
is a good assistant. On a machine without it, every stage fails in a
different place and the orb starts, listens, and says nothing.

`nixorb setup` looks at the machine first and chooses what will run
there, preferring the boring option every time: faster-whisper over
Nemotron, Ollama over an in-process GGUF, Piper over a neural voice. It
explains every choice, and `--dry-run` shows the config without writing
it.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

from nixorb.machine import Machine, probe

log = logging.getLogger(__name__)


def _installed(module: str) -> bool:
    """Is the module importable-in-principle, without importing it?"""
    import importlib.util

    try:
        return importlib.util.find_spec(module) is not None
    except (ImportError, ValueError):
        return False

#: faster-whisper sizes by how much memory the machine has. These run on
#: CPU, which is the point: no torch, no CUDA, no architecture surprises.
WHISPER_BY_RAM = (
    (16.0, "small"),
    (8.0, "base"),
    (0.0, "tiny"),
)

#: A small, widely-supported GGUF that current llama.cpp builds can load.
#: Not the sharpest model available — one that works without a rebuild.
SAFE_GGUF_REPO = "Qwen/Qwen2.5-1.5B-Instruct-GGUF"
SAFE_GGUF_FILE = "qwen2.5-1.5b-instruct-q4_k_m.gguf"

#: Ships with piper-voices and downloads in seconds.
SAFE_PIPER_VOICE = "en_US-lessac-medium"


@dataclass
class Choice:
    """One setting, the value chosen, and why."""

    key: str
    value: Any
    why: str


@dataclass
class Plan:
    """What `nixorb setup` intends to write."""

    choices: list[Choice] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    machine: Machine | None = None

    def add(self, key: str, value: Any, why: str) -> None:
        self.choices.append(Choice(key, value, why))

    def as_settings(self) -> dict[str, Any]:
        return {choice.key: choice.value for choice in self.choices}


def whisper_size_for(ram_gb: float) -> str:
    """The largest faster-whisper size this much memory can hold."""
    for floor, size in WHISPER_BY_RAM:
        if ram_gb >= floor:
            return size
    return "tiny"


def choose_asr(machine: Machine, plan: Plan) -> None:
    """Prefer the engine that needs the least to go right.

    faster-whisper is a self-contained CTranslate2 model: no torch, no
    CUDA, no transformers version to match. Nemotron is better when its
    stack lines up, and worse than silence when it does not.
    """
    ok, reason = machine.can_run_nemotron
    if ok:
        plan.add("asr_backend", "nemotron", "the Nemotron stack is complete here")
        plan.add(
            "asr_model", "nvidia/nemotron-3.5-asr-streaming-0.6b",
            "streaming ASR: partial transcripts while you are still talking",
        )
        return

    if machine.faster_whisper:
        size = whisper_size_for(machine.ram_gb)
        plan.add(
            "asr_backend", "faster_whisper",
            f"Nemotron is not usable here ({reason}); faster-whisper needs "
            "nothing but itself",
        )
        plan.add(
            "asr_model", size,
            f"sized for {machine.ram_gb or '?'} GB of RAM, runs on CPU",
        )
        return

    if _installed("vosk"):
        plan.add(
            "asr_backend", "vosk",
            "the only speech recognition installed — offline, and it fetches "
            "its own model",
        )
        plan.add("asr_model", "en-us", "the small English model, about 40 MB")
        return

    plan.warnings.append(
        "No speech recognition is installed. Either works: "
        "pip install faster-whisper (more accurate), or "
        "pip install 'nixorb[vosk]' (40 MB, no torch)."
    )
    plan.add("asr_backend", "faster_whisper", "the usual choice, once installed")
    plan.add("asr_model", "base", "a small default")


def choose_llm(machine: Machine, plan: Plan, settings: Any = None) -> None:
    """Prefer a model that is already running over one we must load."""
    if machine.ollama_reachable:
        plan.add("llm_backend", "ollama", "Ollama is already running here")
        plan.add("llm_model", "llama3.2", "a small Ollama tag; `ollama pull` it")
        return

    if str(getattr(settings, "t1_base_url", "") or "").strip():
        plan.add(
            "llm_backend", "auto",
            "T1 is configured, so a local failure still gets an answer",
        )
        return

    ok, reason = machine.can_run_gguf
    if ok:
        plan.add("llm_backend", "huggingface", "llama-cpp-python is installed")
        plan.add("llm_model", SAFE_GGUF_REPO, "small, and loadable by current llama.cpp")
        plan.add("llm_gguf_file", SAFE_GGUF_FILE, "the Q4_K_M quant, about 1 GB")
        if machine.ram_gb and machine.ram_gb < 8:
            plan.warnings.append(
                f"{machine.ram_gb} GB of RAM is tight for any local model — "
                "consider Ollama, or set t1_base_url for a hosted answer."
            )
        return

    if machine.ollama_binary:
        plan.add("llm_backend", "ollama", "Ollama is installed but not running")
        plan.add("llm_model", "llama3.2", "start it with: ollama serve")
        plan.warnings.append(
            "Ollama is installed but not answering. Run `ollama serve`, then "
            "`ollama pull llama3.2`."
        )
        return

    plan.add("llm_backend", "auto", "nothing local is available yet")
    plan.warnings.append(
        f"No language model is available ({reason}). The quickest fix is "
        "Ollama: install it, then `ollama pull llama3.2`. Or set "
        "t1_base_url and t1_api_key to answer through the T1 API."
    )


def choose_tts(machine: Machine, plan: Plan) -> None:
    """Prefer a voice that exists over one that must be fetched and built."""
    if _installed("kokoro_onnx"):
        # The best voice a plain machine can reach: onnxruntime, no torch.
        plan.add("tts_backend", "kokoro", "kokoro-onnx is installed — neural, no torch")
        plan.add("tts_voice", "af_heart", "a Kokoro voice id")
        plan.add(
            "tts_fallbacks", ["piper", "espeak"],
            "if the Kokoro model cannot be fetched",
        )
        return

    if machine.piper_binary:
        plan.add("tts_backend", "piper", f"piper-tts is installed ({machine.piper_binary})")
        plan.add("tts_voice", SAFE_PIPER_VOICE, "downloaded on first use")
        plan.add(
            "tts_fallbacks", ["espeak"],
            "espeak-ng if a voice cannot be fetched",
        )
        return

    ok, _ = machine.can_run_hf_models
    if ok:
        plan.add("tts_backend", "huggingface", "transformers and torch are installed")
        plan.add("tts_hf_repo", "microsoft/speecht5_tts", "small, and CPU-friendly")
        plan.add("tts_fallbacks", ["piper", "espeak"], "if the model will not load")
        return

    if machine.espeak:
        plan.add("tts_backend", "espeak", "the only voice installed here")
        plan.warnings.append(
            "espeak-ng sounds robotic. For a natural voice on this machine: "
            "pip install 'nixorb[kokoro]' (no torch needed), or yay -S piper-tts"
        )
        return

    plan.add("tts_backend", "piper", "nothing is installed yet")
    plan.warnings.append(
        "No speech synthesis is installed — the orb will be silent. "
        "Best on this machine: pip install 'nixorb[kokoro]' (neural, no "
        "torch). Or: yay -S piper-tts, or espeak-ng."
    )


def build_plan(settings: Any = None, machine: Machine | None = None) -> Plan:
    """Decide what this machine should be configured to use."""
    machine = machine or probe(settings=settings)
    plan = Plan(machine=machine)

    choose_asr(machine, plan)
    choose_llm(machine, plan, settings)
    choose_tts(machine, plan)

    plan.warnings.extend(machine.notes)
    return plan


def apply_plan(plan: Plan, settings: Any) -> Any:
    """Return `settings` with the plan's choices applied."""
    updates = plan.as_settings()
    try:
        return settings.model_copy(update=updates)
    except AttributeError:  # pragma: no cover - a stand-in in tests
        import copy

        clone = copy.copy(settings)
        for key, value in updates.items():
            setattr(clone, key, value)
        return clone


def configure_on_first_run(settings: Any) -> Any:
    """Choose settings on the very first start, save them, and use them.

    Nothing in the shipped defaults knows what is installed on this
    machine, so a first launch without the full stack fails at whichever
    piece is missing — the orb appears, listens, and never answers. That
    is the single commonest way NixOrb looks broken on a fresh install.

    This makes the same choices `nixorb setup` makes, writes them to the
    user's config, and returns the adjusted settings so this start already
    uses them. It happens exactly once: afterwards the config file exists,
    and a config the user has edited is never touched.
    """
    from nixorb.settings import config_path

    path = config_path()
    if path.exists():
        return settings

    log.info("First start — choosing settings this machine can actually run")
    # cuda=False: the torch CUDA probe costs seconds and only feeds a note.
    plan = build_plan(settings, machine=probe(settings=settings, cuda=False))
    for choice in plan.choices:
        log.info("  %s = %r  (%s)", choice.key, choice.value, choice.why)
    for warning in plan.warnings:
        log.warning("  %s", warning)

    configured = apply_plan(plan, settings)
    try:
        configured.save()
        log.info(
            "Saved to %s — edit it, or re-run `nixorb setup` after installing "
            "something", path,
        )
    except Exception as exc:
        # A read-only home or a bad path must not stop the orb starting;
        # the choices are still better than the defaults for this start.
        log.warning("Could not write %s (%s) — using the choices anyway", path, exc)
    return configured
