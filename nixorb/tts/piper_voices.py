"""Finding — and fetching — Piper voice models.

Piper ships no voices. The installer downloads exactly one
(`en_US-lessac-medium`), and `PiperTTS` only ever looked on disk, so setting
`tts_voice` to any other Piper voice found nothing and dropped silently to
espeak-ng. Every voice but one appeared to be unsupported.

Upstream's layout is fully deterministic, which is what makes fetching any
of them possible without an index:

    <lang>/<locale>/<speaker>/<quality>/<locale>-<speaker>-<quality>.onnx

`en_GB-alan-medium` lives at `en/en_GB/alan/medium/`, `ca_ES-upc_ona-x_low`
at `ca/ca_ES/upc_ona/x_low/`. Speaker names use underscores and never
hyphens, so a voice name splits on "-" into exactly three parts — the
locale, the speaker, and the quality — and the language directory is the
locale's prefix. Each directory holds the model and its `.onnx.json`
config, and Piper needs both.
"""
from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)

VOICES_REPO = "rhasspy/piper-voices"

# Real models are 20-115 MB. Anything far smaller is an error page, an LFS
# pointer, or a partial transfer — none of which Piper can load, and all of
# which would otherwise be cached as if they were a voice.
MIN_MODEL_BYTES = 1_000_000

# locale-speaker-quality, e.g. en_GB-southern_english_female-low.
#
# The speaker part is \w, not [A-Za-z0-9_]: upstream has pt_PT-tugão-medium,
# and an ASCII-only class silently refuses to fetch it. It also starts with a
# digit in vi_VN-25hours_single-low, so no leading-letter rule either. What
# it never contains is a hyphen — that is what makes three-way splitting
# safe for names like en_GB-southern_english_female-low.
_VOICE_NAME = re.compile(
    r"^(?P<locale>[a-z]{2,3}_[A-Za-z]{2,4})-(?P<speaker>[^\W-]+)-"
    r"(?P<quality>x_low|low|medium|high)$",
    re.UNICODE,
)


class VoiceUnavailable(RuntimeError):
    """A voice could not be found locally or fetched."""


def parse_voice_name(voice: str) -> tuple[str, str, str] | None:
    """Split a Piper voice name into (locale, speaker, quality).

    None for anything that is not one — a path, a URL, or a description
    meant for a different TTS backend.
    """
    match = _VOICE_NAME.match((voice or "").strip())
    if match is None:
        return None
    return match.group("locale"), match.group("speaker"), match.group("quality")


def voice_repo_path(voice: str) -> str | None:
    """Where `voice` lives inside the piper-voices repo, or None."""
    parsed = parse_voice_name(voice)
    if parsed is None:
        return None
    locale, speaker, quality = parsed
    language = locale.split("_")[0]
    return f"{language}/{locale}/{speaker}/{quality}/{voice}.onnx"


def looks_like_path(value: str) -> bool:
    """Is this a filesystem path to a model rather than a voice name?"""
    text = (value or "").strip()
    return text.endswith(".onnx") or text.startswith(("/", "~", "./", "../"))


def resolve_path(value: str) -> Path | None:
    """The model file `value` points at, if it points at one that exists."""
    if not looks_like_path(value):
        return None
    candidate = Path(value).expanduser()
    return candidate if candidate.is_file() else None


def validate(model: Path) -> None:
    """Raise unless `model` and its config look like a usable voice.

    A download that returned an HTML error page, or stopped halfway, is
    worse than no download at all: it is cached, found next time, and
    fails inside Piper with nothing pointing back here.
    """
    if not model.is_file():
        raise VoiceUnavailable(f"{model} was not written")

    size = model.stat().st_size
    if size < MIN_MODEL_BYTES:
        raise VoiceUnavailable(
            f"{model.name} is only {size} bytes — too small to be a voice "
            "model (an error page or a partial download)"
        )

    config = model.with_suffix(".onnx.json")
    if not config.is_file():
        raise VoiceUnavailable(f"{config.name} is missing — Piper needs both files")
    try:
        json.loads(config.read_text())
    except (OSError, ValueError) as exc:
        raise VoiceUnavailable(f"{config.name} is not readable JSON: {exc}") from exc


def download(voice: str, dest: Path, token: str | None = None) -> Path:
    """Fetch `voice` and its config into `dest`, returning the model path."""
    relative = voice_repo_path(voice)
    if relative is None:
        raise VoiceUnavailable(
            f"'{voice}' is not a Piper voice name. Piper voices look like "
            "'en_US-lessac-medium' — locale, speaker, quality. Browse them "
            f"at https://huggingface.co/{VOICES_REPO}, or set tts_voice to "
            "the path of a .onnx file you already have."
        )

    try:
        from huggingface_hub import hf_hub_download
    except ImportError as exc:  # pragma: no cover - it is a base dependency
        raise VoiceUnavailable(
            "huggingface_hub is needed to download a Piper voice: "
            "pip install huggingface_hub"
        ) from exc

    dest.mkdir(parents=True, exist_ok=True)
    log.info("TTS: fetching Piper voice '%s' from %s", voice, VOICES_REPO)

    model: Path | None = None
    for name in (relative, relative + ".json"):
        try:
            got = hf_hub_download(
                repo_id=VOICES_REPO,
                filename=name,
                local_dir=str(dest),
                token=token,
            )
        except Exception as exc:
            raise VoiceUnavailable(
                f"could not download '{voice}' ({name}): {exc}"
            ) from exc
        if model is None:
            model = Path(got)

    assert model is not None
    validate(model)
    log.info("TTS: Piper voice '%s' ready at %s", voice, model)
    return model


def ensure(
    voice: str,
    search: Any,
    dest: Path,
    token: str | None = None,
    allow_download: bool = True,
) -> Path:
    """Return a usable model for `voice`, fetching it only if needed.

    `search` is called first and should return a local path or None; disk
    always wins, so an offline machine with the voice already present never
    reaches for the network.
    """
    if looks_like_path(voice):
        direct = resolve_path(voice)
        if direct is None:
            # Saying "not a Piper voice name" about an obvious path sends
            # the reader looking for a typo in the wrong thing.
            raise VoiceUnavailable(
                f"tts_voice points at '{voice}', which is not a file that "
                "exists. Give the path of a .onnx voice model, or a voice "
                "name like 'en_US-lessac-medium' to have it downloaded."
            )
        validate(direct)
        return direct

    found = search(voice)
    if found is not None:
        return Path(found)

    if not allow_download:
        raise VoiceUnavailable(
            f"Piper voice '{voice}' is not installed and downloading is off "
            "(tts_download_voices = false)"
        )

    return download(voice, dest, token)
