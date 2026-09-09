"""What Python environment is NixOrb actually installed into?

Installing NixOrb into an interpreter that is shared with other AI projects
is the most common way it appears broken without ever raising an error of
its own. NixOrb needs ``transformers>=5.13`` — that is where the default
Nemotron ASR architecture landed — and a lot of tooling in this space still
pins ``transformers<=4.5x``. Whichever of the two is installed second wins,
pip prints one warning, and the loser fails much later from somewhere deep
inside ``from_pretrained``.

Nothing here can repair that; two packages with disjoint pins genuinely
cannot share one interpreter. What it can do is name the collision at the
moment the user asks, instead of letting them debug a model loader.
"""
from __future__ import annotations

import re
import sys
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

DISTRIBUTION = "nixorb"

_CANONICAL = re.compile(r"[-_.]+")

# Markers are evaluated with an empty `extra`, so requirements guarded by
# `extra == "..."` evaluate False and drop out: an optional dependency the
# user never asked for is not a conflict.
_NO_EXTRA = {"extra": ""}


def canonical(name: str) -> str:
    """Normalise a distribution name the way PEP 503 does."""
    return _CANONICAL.sub("-", name.strip()).lower()


@dataclass(frozen=True)
class Conflict:
    """An installed package whose pin the installed version violates."""

    package: str
    installed: str
    holder: str
    holder_version: str
    specifier: str

    def describe(self) -> str:
        pin = self.specifier or "(any)"
        return (
            f"{self.holder} {self.holder_version} needs {self.package}{pin}, "
            f"but {self.package} {self.installed} is installed"
        )


@dataclass(frozen=True)
class Unsatisfied:
    """One of NixOrb's own requirements the environment no longer meets."""

    package: str
    specifier: str
    installed: str | None

    def describe(self) -> str:
        have = f"{self.installed} is installed" if self.installed else "it is missing"
        return f"NixOrb needs {self.package}{self.specifier}, but {have}"


def parse_requirement(text: str) -> tuple[str, str] | None:
    """Return (canonical name, specifier) for a requirement that applies here.

    Returns None for anything unparseable, or whose environment markers
    exclude it — a `python_version < "3.11"` pin says nothing about 3.14.
    """
    try:
        from packaging.requirements import Requirement
    except ImportError:  # pragma: no cover - packaging ships with our deps
        return None

    try:
        req = Requirement(text)
    except Exception:
        return None

    if req.marker is not None:
        try:
            if not req.marker.evaluate(_NO_EXTRA):
                return None
        except Exception:
            return None

    return canonical(req.name), str(req.specifier)


def _satisfies(specifier: str, version: str) -> bool:
    """Does `version` fall inside `specifier`? Unparseable pins pass."""
    if not specifier:
        return True
    try:
        from packaging.specifiers import SpecifierSet
    except ImportError:  # pragma: no cover
        return True
    try:
        # prereleases=True so a legitimate 5.16.1rc1 is not reported as a
        # violation of >=5.13 that the user cannot act on.
        return SpecifierSet(specifier).contains(version, prereleases=True)
    except Exception:
        return True


def find_conflicts(
    ours: Iterable[str],
    installed: Mapping[str, str],
    requirements: Mapping[str, Sequence[str]],
) -> list[Conflict]:
    """Packages NixOrb shares with something whose pin they now violate.

    Only packages NixOrb itself depends on (`ours`) are considered — the
    rest of the environment's internal disagreements are not ours to
    report on.
    """
    ours = {canonical(name) for name in ours}
    found: list[Conflict] = []

    for holder, reqs in sorted(requirements.items()):
        if canonical(holder) == DISTRIBUTION:
            continue
        for text in reqs:
            parsed = parse_requirement(text)
            if parsed is None:
                continue
            name, specifier = parsed
            if name not in ours:
                continue
            have = installed.get(name)
            if have is None or _satisfies(specifier, have):
                continue
            found.append(
                Conflict(
                    package=name,
                    installed=have,
                    holder=holder,
                    holder_version=installed.get(canonical(holder), "?"),
                    specifier=specifier,
                )
            )
    return found


def find_unsatisfied(
    our_requirements: Iterable[str],
    installed: Mapping[str, str],
) -> list[Unsatisfied]:
    """NixOrb's own pins the environment stopped meeting.

    The mirror image of `find_conflicts`: installing a `transformers<=4.57`
    project *after* NixOrb silently downgrades it and breaks us instead.
    """
    missing: list[Unsatisfied] = []
    for text in our_requirements:
        parsed = parse_requirement(text)
        if parsed is None:
            continue
        name, specifier = parsed
        have = installed.get(name)
        if have is not None and _satisfies(specifier, have):
            continue
        missing.append(Unsatisfied(name, specifier, have))
    return missing


def read_environment() -> tuple[dict[str, str], dict[str, list[str]]]:
    """Installed versions and declared requirements, both keyed by name."""
    from importlib.metadata import distributions

    installed: dict[str, str] = {}
    requirements: dict[str, list[str]] = {}

    for dist in distributions():
        try:
            name = dist.metadata["Name"]
        except Exception:
            continue
        if not name:
            continue
        key = canonical(name)
        # Duplicate .dist-info dirs happen; first one wins, as on sys.path.
        if key in installed:
            continue
        installed[key] = dist.version or "?"
        requirements[name] = list(dist.requires or [])

    return installed, requirements


def our_requirements() -> list[str]:
    """NixOrb's own declared requirements, or [] if it is not installed."""
    from importlib.metadata import PackageNotFoundError, distribution

    try:
        return list(distribution(DISTRIBUTION).requires or [])
    except PackageNotFoundError:
        return []
    except Exception:
        return []


def our_dependency_names() -> list[str]:
    """Canonical names of the packages NixOrb depends on here."""
    names: list[str] = []
    for text in our_requirements():
        parsed = parse_requirement(text)
        if parsed is not None:
            names.append(parsed[0])
    return names


def isolation() -> tuple[bool, str]:
    """Is this interpreter NixOrb's alone, and if not, whose is it?"""
    if sys.prefix != sys.base_prefix:
        return True, f"virtualenv — {sys.prefix}"

    try:
        import site

        user_site = site.getusersitepackages()
    except Exception:
        user_site = ""

    here = Path(__file__).resolve().parent.parent
    if user_site:
        try:
            here.relative_to(Path(user_site).resolve())
        except ValueError:
            pass
        else:
            return False, f"shared per-user site-packages — {user_site}"

    return False, f"shared system interpreter — {sys.prefix}"


def scan() -> tuple[list[Unsatisfied], list[Conflict]]:
    """Everything wrong with this environment, from both directions."""
    installed, requirements = read_environment()
    return (
        find_unsatisfied(our_requirements(), installed),
        find_conflicts(our_dependency_names(), installed, requirements),
    )


ISOLATION_ADVICE = (
    "Install NixOrb into its own environment:",
    "  python -m venv ~/.local/share/nixorb/venv",
    "  ~/.local/share/nixorb/venv/bin/pip install nixorb",
)


def problems() -> list[str]:
    """Only what needs acting on — empty when the environment is healthy."""
    unsatisfied, conflicts = scan()
    lines = [f"❌ {item.describe()}" for item in unsatisfied]
    lines += [f"⚠️  {conflict.describe()}" for conflict in conflicts]
    if conflicts:
        lines.append(
            "Their pins do not overlap, so one interpreter cannot hold both."
        )
    if lines and not isolation()[0]:
        lines.extend(ISOLATION_ADVICE)
    return lines


def action_report() -> list[str]:
    """Whether NixOrb can run commands on this machine, and what gates them.

    "It can't run things on my PC" has several causes that look identical
    from outside: running as root disables execution outright, a missing
    confirmation dialog denies every command, and bubblewrap being absent
    silently drops the sandbox.
    """
    import os
    import shutil

    from nixorb.settings import Settings

    settings = Settings.load()
    lines: list[str] = []

    if os.geteuid() == 0:
        lines.append(
            "  ❌ running as root — command execution is refused entirely; "
            "start NixOrb as your normal user"
        )
    else:
        lines.append("  ✅ command execution available")

    if settings.require_action_confirmation:
        lines.append(
            "  ✅ every command asks first (require_action_confirmation)"
        )
    else:
        lines.append(
            "  ⚠️  commands run without asking "
            "(require_action_confirmation = false)"
        )

    if settings.sandbox_actions:
        if shutil.which("bwrap"):
            lines.append("  ✅ commands run inside a bubblewrap sandbox")
        else:
            lines.append(
                "  ⚠️  sandbox_actions is on but bwrap is not installed — "
                "commands run unsandboxed. Install bubblewrap."
            )
    else:
        lines.append("  ⚠️  commands run unsandboxed (sandbox_actions = false)")

    return lines


def torch_report() -> list[str]:
    """Whether torch and torchaudio import, and which CUDA build they are.

    Worth its own section because the failure is invisible until a turn
    runs: transformers>=5 imports torchaudio at module scope, so a torch
    stack whose halves came from different indexes takes down
    `AutoProcessor.from_pretrained` with a linker error naming a `.so` and
    nothing else. torch is deliberately not a declared dependency — no one
    build is right for every machine — so absence is a note, not a failure.
    """
    from nixorb.hf import describe_native_import_error, duplicate_extensions

    lines: list[str] = []
    for name in ("torch", "torchaudio"):
        try:
            module = __import__(name)
        except BaseException as exc:
            # A half-installed native package raises whatever it likes —
            # ImportError, OSError, RuntimeError — so classify on the
            # message, not the type.
            advice = describe_native_import_error(exc)
            leftovers = duplicate_extensions(name)

            if advice is None and not leftovers:
                if isinstance(exc, ImportError) and "No module named" in str(exc):
                    lines.append(f"  ⚠️  {name} is not installed")
                else:
                    lines.append(f"  ❌ {name} will not import: {exc}")
                continue

            lines.append(f"  ❌ {name} is installed but will not import: {exc}")
            if leftovers:
                lines.append(
                    f"       {len(leftovers)} extension files where there "
                    "should be one — the loader cannot choose:"
                )
                lines.extend(f"         {path}" for path in leftovers)
            if advice:
                lines.extend(f"       {line}" for line in advice.splitlines())
            continue

        version = getattr(module, "__version__", "?")
        cuda = getattr(getattr(module, "version", None), "cuda", None)
        flavour = f"CUDA {cuda}" if cuda else "CPU"
        lines.append(f"  ✅ {name} {version} ({flavour})")

    return lines


def report() -> list[str]:
    """Human-readable environment lines for `nixorb check`."""
    import platform

    lines = [f"  ✅ Python {platform.python_version()} ({sys.executable})"]

    isolated, where = isolation()
    lines.append(f"  {'✅' if isolated else '⚠️ '} {where}")
    if not isolated:
        lines.append(
            "       NixOrb needs transformers>=5.13; projects that pin an "
            "older one cannot share this interpreter."
        )
        for advice in ISOLATION_ADVICE:
            lines.append(f"       {advice}")

    unsatisfied, conflicts = scan()
    for item in unsatisfied:
        lines.append(f"  ❌ {item.describe()}")
    for conflict in conflicts:
        lines.append(f"  ⚠️  {conflict.describe()}")

    if conflicts:
        lines.append(
            "       Their pins do not overlap; give each its own virtualenv."
        )
    if not unsatisfied and not conflicts:
        lines.append("  ✅ No dependency conflicts with other installed packages")

    lines.extend(torch_report())

    return lines


def full_report() -> list[str]:
    """Everything `nixorb check` prints about this installation."""
    return report() + ["", "Running commands:"] + action_report()
