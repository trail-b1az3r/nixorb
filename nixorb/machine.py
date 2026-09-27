"""What this machine can actually run.

Every failure this project has produced in the field has been the same
shape: a default pointing at a model the machine could not load. A
Nemotron checkpoint needing transformers 5.13 and a numpy numba would
accept; a GGUF needing a llama.cpp newer than the wheel; a Hugging Face
TTS needing torch that was never installed. In each case NixOrb started
fine and then had nothing to say.

So: look first. Everything here is a read-only probe that answers in
under a second and never raises — a machine report you can base a
recommendation on, and that `nixorb check` can print.
"""
from __future__ import annotations

import logging
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path

log = logging.getLogger(__name__)

#: numba trails numpy by a release, and the Nemotron ASR stack pulls numba
#: in. This is the pairing that broke transcription in the field.
NUMBA_MAX_NUMPY = (2, 5)

#: Where Nemotron's architecture landed.
NEMOTRON_MIN_TRANSFORMERS = (5, 13)


def _version_of(module: str) -> str:
    try:
        from importlib.metadata import version

        return version(module)
    except Exception:
        return ""


def _tuple(raw: str) -> tuple[int, ...]:
    parts: list[int] = []
    for chunk in str(raw).split("."):
        digits = ""
        for ch in chunk:
            if not ch.isdigit():
                break
            digits += ch
        if not digits:
            break
        parts.append(int(digits))
    return tuple(parts)


def _installed(module: str) -> bool:
    """Is the module importable-in-principle, without importing it?"""
    import importlib.util

    try:
        return importlib.util.find_spec(module.split(".")[0]) is not None
    except (ImportError, ValueError):
        return False


@dataclass
class Machine:
    """A read-only picture of what is available here."""

    python: str = ""
    ram_gb: float = 0.0
    disk_free_gb: float = 0.0
    gpu_name: str = ""
    vram_gb: float = 0.0

    torch: str = ""
    torch_cuda: bool = False
    transformers: str = ""
    numpy: str = ""
    llama_cpp: str = ""
    faster_whisper: str = ""
    hypernix: str = ""

    piper_binary: str = ""
    espeak: bool = False
    ollama_binary: bool = False
    ollama_reachable: bool = False

    notes: list[str] = field(default_factory=list)

    # ── derived capability ───────────────────────────────────────── #

    @property
    def has_gpu(self) -> bool:
        return bool(self.gpu_name) and self.vram_gb > 0

    @property
    def can_run_nemotron(self) -> tuple[bool, str]:
        """Nemotron needs a specific, easily-broken stack."""
        if not self.transformers:
            return False, "transformers is not installed"
        if _tuple(self.transformers) < NEMOTRON_MIN_TRANSFORMERS:
            want = ".".join(str(n) for n in NEMOTRON_MIN_TRANSFORMERS)
            return False, f"transformers {self.transformers} is older than {want}"
        if not self.torch:
            return False, "torch is not installed"
        if self.numpy and _tuple(self.numpy)[:2] >= NUMBA_MAX_NUMPY:
            return False, (
                f"numpy {self.numpy} is newer than numba accepts "
                f"(pin numpy<{'.'.join(str(n) for n in NUMBA_MAX_NUMPY)})"
            )
        return True, ""

    @property
    def can_run_hf_models(self) -> tuple[bool, str]:
        if not self.transformers:
            return False, "transformers is not installed"
        if not self.torch:
            return False, "torch is not installed"
        return True, ""

    @property
    def can_run_gguf(self) -> tuple[bool, str]:
        if not self.llama_cpp:
            return False, "llama-cpp-python is not installed"
        return True, ""

    @property
    def can_speak(self) -> bool:
        return bool(self.piper_binary or self.espeak) or self.can_run_hf_models[0]


def _read_ram_gb() -> float:
    try:
        with open("/proc/meminfo") as handle:
            for line in handle:
                if line.startswith("MemTotal:"):
                    return round(int(line.split()[1]) / 1_048_576, 1)
    except OSError:
        pass
    return 0.0


def _read_disk_free_gb(where: Path | None = None) -> float:
    try:
        usage = shutil.disk_usage(str(where or Path.home()))
        return round(usage.free / 1_000_000_000, 1)
    except OSError:
        return 0.0


def _read_gpu() -> tuple[str, float]:
    if not shutil.which("nvidia-smi"):
        return "", 0.0
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=4, check=False,
        ).stdout.strip().splitlines()
    except (OSError, subprocess.SubprocessError):
        return "", 0.0
    if not out:
        return "", 0.0
    parts = [p.strip() for p in out[0].split(",")]
    name = parts[0] if parts else ""
    try:
        vram = round(float(parts[1]) / 1024, 1)
    except (IndexError, ValueError):
        vram = 0.0
    return name, vram


def _torch_cuda() -> bool:
    """Does torch see a usable CUDA device?

    Imported in a subprocess: importing torch takes seconds and, on a
    broken install, can abort the interpreter outright.
    """
    if not _installed("torch"):
        return False
    try:
        result = subprocess.run(
            [sys.executable, "-c",
             "import torch;print(int(torch.cuda.is_available()))"],
            capture_output=True, text=True, timeout=60, check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return False
    return result.stdout.strip().endswith("1")


def _ollama_reachable(host: str = "http://localhost:11434") -> bool:
    import urllib.error
    import urllib.request

    try:
        with urllib.request.urlopen(f"{host}/api/tags", timeout=2) as response:
            return 200 <= response.status < 300
    except Exception:
        return False


def probe(
    *,
    settings: object | None = None,
    deep: bool = True,
    cuda: bool = True,
) -> Machine:
    """Look at this machine. Never raises.

    `deep=False` skips both checks that cost real time: a subprocess
    `import torch`, and a connection to Ollama. `cuda=False` keeps the
    Ollama check — it answers in milliseconds, and it decides which
    language model can be used — but skips the torch import, which takes
    several seconds on a cold page cache and is only ever needed to
    report a CPU-only build.
    """
    from nixorb.tts.piper_tts import find_piper_binary

    machine = Machine(
        python=".".join(str(n) for n in sys.version_info[:3]),
        ram_gb=_read_ram_gb(),
        disk_free_gb=_read_disk_free_gb(),
        torch=_version_of("torch"),
        transformers=_version_of("transformers"),
        numpy=_version_of("numpy"),
        llama_cpp=_version_of("llama-cpp-python"),
        faster_whisper=_version_of("faster-whisper"),
        hypernix=_version_of("hypernix"),
        piper_binary=find_piper_binary() or "",
        espeak=shutil.which("espeak-ng") is not None,
        ollama_binary=shutil.which("ollama") is not None,
    )
    machine.gpu_name, machine.vram_gb = _read_gpu()

    if deep:
        if cuda:
            machine.torch_cuda = _torch_cuda()
        host = str(
            getattr(settings, "ollama_host", "") or "http://localhost:11434"
        )
        machine.ollama_reachable = _ollama_reachable(host)

    # Only worth saying when the CUDA check actually ran — torch_cuda is
    # False both for a CPU-only build and for a check that was skipped.
    if deep and cuda and machine.gpu_name and not machine.torch_cuda and machine.torch:
        machine.notes.append(
            f"{machine.gpu_name} is present but torch cannot use it — "
            "you have a CPU-only torch build"
        )
    if os.geteuid() == 0:
        machine.notes.append(
            "running as root: NixOrb refuses to execute commands as root"
        )
    return machine
