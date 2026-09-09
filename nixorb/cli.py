"""NixOrb CLI — command-line interface.

Usage:
    nixorb start              Launch the GUI orb
    nixorb start --headless   Daemon mode (no Qt)
    nixorb trigger            Activate the running orb (for KDE shortcuts)
    nixorb quit               Shut down the running orb
    nixorb ask "query"        One-shot text query
    nixorb tts "text"         Speak text
    nixorb status             Show system status
    nixorb config             Open config in editor
    nixorb config --gui       Open the graphical settings window
    nixorb check              Check dependencies
    nixorb version            Show version
"""
from __future__ import annotations

import asyncio
import logging
import shutil
import subprocess

import typer

import nixorb
from nixorb.settings import Settings

app = typer.Typer(
    name="nixorb",
    help="NixOrb — AI assistant for Arch Linux + KDE Plasma 6",
    no_args_is_help=True,
)

log = logging.getLogger(__name__)


@app.command()
def start(
    headless: bool = typer.Option(False, "--headless", help="Run without GUI"),
) -> None:
    """Start NixOrb."""
    if headless:
        typer.echo("Headless mode not yet implemented — use GUI mode")
        raise typer.Exit(1)

    from nixorb.main import main

    raise typer.Exit(main() or 0)


def _control(command: str, success: str) -> None:
    """Send one command to the running instance, or explain why we can't."""
    from nixorb.core.ipc import send

    try:
        reply = send(command)
    except ConnectionError:
        typer.echo(
            "NixOrb is not running — start it with:  nixorb start", err=True
        )
        raise typer.Exit(1) from None

    if reply.startswith("ok"):
        typer.echo(success)
    else:
        typer.echo(f"❌ {reply}", err=True)
        raise typer.Exit(1)


@app.command()
def trigger() -> None:
    """Activate the running orb (bind this to a KDE global shortcut)."""
    _control("trigger", "🔔 NixOrb activated")


@app.command()
def quit() -> None:  # noqa: A001 — this is the user-facing command name
    """Ask the running NixOrb to shut down."""
    _control("quit", "👋 NixOrb shutting down")


@app.command()
def ask(
    query: str = typer.Argument(..., help="Text query to send to the AI"),
) -> None:
    """Send a one-shot query to the AI."""

    async def _ask() -> None:
        settings = Settings.load()
        from nixorb.llm.factory import create_llm

        llm = create_llm(settings)
        messages = [
            {"role": "system", "content": settings.llm_system_prompt},
            {"role": "user", "content": query},
        ]

        try:
            health = await llm.health_check()
            if not health["ok"]:
                typer.echo(f"❌ {health['error']}", err=True)
                if settings.llm_backend == "ollama":
                    typer.echo(
                        f"   Try: ollama pull {settings.llm_model}", err=True
                    )
                raise typer.Exit(1)

            typer.echo(f"🤔 Querying {settings.llm_backend}/{llm.model}…")
            response = await llm.generate(messages)
            typer.echo(f"\n🤖 {response}")
        except typer.Exit:
            raise
        except Exception as exc:
            typer.echo(f"❌ Error: {exc}", err=True)
            raise typer.Exit(1) from exc
        finally:
            await llm.close()

    asyncio.run(_ask())


@app.command()
def tts(
    text: str = typer.Argument(..., help="Text to speak"),
) -> None:
    """Speak text using TTS."""

    async def _speak() -> None:
        settings = Settings.load()
        from nixorb.tts.tts_factory import create_tts

        tts_engine = create_tts(settings)
        typer.echo(f"🔊 Speaking: {text}")
        await tts_engine.speak(text)

    asyncio.run(_speak())


@app.command()
def status() -> None:
    """Show NixOrb system status."""
    settings = Settings.load()

    typer.echo("═" * 50)
    typer.echo(f"  NixOrb {nixorb.__version__}")
    typer.echo("═" * 50)

    from nixorb.core.ipc import is_running, socket_path

    typer.echo("\n🌐 Orb:")
    if is_running():
        typer.echo(f"  ✅ Running (control socket: {socket_path()})")
        typer.echo("     Activate with: nixorb trigger")
    else:
        typer.echo("  ⭘ Not running — start it with: nixorb start")

    # ── Models ──
    typer.echo("\n🤖 Language model:")
    typer.echo(f"  Backend: {settings.llm_backend}")

    async def _probe() -> tuple[dict, str]:
        from nixorb.llm.factory import create_llm

        llm = create_llm(settings)
        try:
            return await llm.health_check(), llm.model
        finally:
            await llm.close()

    health, llm_model = asyncio.run(_probe())
    typer.echo(f"  Model: {llm_model}")
    if settings.llm_backend == "ollama":
        typer.echo(f"  Host: {settings.ollama_host}")
    if health["ok"]:
        typer.echo(f"  ✅ Reachable, '{llm_model}' available")
    else:
        typer.echo(f"  ❌ {health['error']}")

    typer.echo("\n🎙 Speech recognition:")
    from nixorb.asr.factory import create_asr

    asr = create_asr(settings)
    typer.echo(f"  Backend: {asr.name}")
    typer.echo(f"  Model: {settings.asr_model}")
    typer.echo(f"  Language: {settings.asr_language or 'auto'}")
    if asr.supports_streaming:
        from nixorb.asr.nemotron_asr import LOOKAHEAD_LATENCY_MS

        lookahead = settings.asr_nemotron_lookahead
        latency = LOOKAHEAD_LATENCY_MS.get(lookahead, "?")
        typer.echo(
            f"  Streaming: {'on' if settings.asr_streaming else 'off'} "
            f"(lookahead {lookahead} ≈ {latency}ms)"
        )

    typer.echo("\n🔊 Speech synthesis:")
    typer.echo(f"  Backend: {settings.tts_backend}")
    if settings.tts_backend == "huggingface":
        typer.echo(f"  Model: {settings.tts_hf_repo}")
    else:
        typer.echo(f"  Voice: {settings.tts_voice}")

    # Check dependencies
    typer.echo("\n📦 Dependencies:")
    deps = {
        "piper-tts": "TTS engine (AUR: piper-tts)",
        "wl-paste": "Clipboard (Wayland)",
        "grim": "Screenshot",
        "espeak-ng": "TTS fallback",
        "bwrap": "Sandbox",
        "ollama": "AI backend",
    }
    for dep, desc in deps.items():
        found = "✅" if shutil.which(dep) else "❌"
        typer.echo(f"  {found} {dep} — {desc}")

    # Silent when the environment is healthy; `nixorb check` always reports.
    from nixorb import envcheck

    trouble = envcheck.problems()
    if trouble:
        typer.echo("\n🐍 Python environment:")
        for line in trouble:
            typer.echo(f"  {line}")

    # VRAM
    typer.echo("\n🎮 GPU:")
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name,memory.free,memory.total", "--format=csv,noheader"],
            timeout=2,
            stderr=subprocess.DEVNULL,
        ).decode().strip()
        name, free, total = [x.strip() for x in out.split(",")]
        typer.echo(f"  {name}")
        typer.echo(f"  VRAM: {free} / {total} MB free")
    except Exception:
        typer.echo("  (No NVIDIA GPU detected)")

    typer.echo("")


@app.command()
def config(
    gui: bool = typer.Option(
        False, "--gui", "-g", help="Open the graphical settings window instead of a text editor."
    ),
) -> None:
    """Open NixOrb configuration in default editor (or --gui for a graphical dialog)."""
    # settings.config_path(), not a second hardcoded copy: with
    # NIXORB_CONFIG set this used to open a file NixOrb never reads, so
    # every edit made here was discarded.
    from nixorb.settings import config_path as _resolve_config_path

    config_path = _resolve_config_path()
    config_path.parent.mkdir(parents=True, exist_ok=True)
    if not config_path.exists():
        Settings().save()

    if gui:
        from PySide6.QtWidgets import QApplication

        from nixorb.settings import Settings as _Settings
        from nixorb.ui.settings_window import SettingsWindow

        # Held for the duration of .exec() — required to keep the QApplication alive.
        _app = QApplication.instance() or QApplication([])
        window = SettingsWindow(_Settings.load())
        window.exec()
        return

    editor = shutil.which("nano") or shutil.which("vim") or shutil.which("vi")
    if editor:
        subprocess.call([editor, str(config_path)])
    else:
        typer.echo(f"Config file: {config_path}")


@app.command()
def check() -> None:
    """Check system dependencies and the Python environment."""
    from nixorb import envcheck

    typer.echo("Python environment:")
    for line in envcheck.full_report():
        typer.echo(line)

    typer.echo("")
    required = ["python", "pip"]
    recommended = [
        "wl-paste", "wl-copy", "grim", "espeak-ng",
        "bwrap", "ollama", "nvidia-smi",
    ]

    typer.echo("Required:")
    for dep in required:
        found = shutil.which(dep)
        typer.echo(f"  {'✅' if found else '❌'} {dep}")

    typer.echo("\nRecommended:")

    # Piper needs its own lookup: the AUR package installs `piper-tts`, and
    # a bare `piper` on Arch is the gaming-mouse tool, not a speech engine.
    from nixorb.tts.piper_tts import find_piper_binary

    piper = find_piper_binary()
    typer.echo(
        f"  {'✅' if piper else '❌'} piper-tts"
        + (f" ({piper})" if piper else "  — install with: yay -S piper-tts")
    )

    for dep in recommended:
        found = shutil.which(dep)
        typer.echo(f"  {'✅' if found else '❌'} {dep}")


@app.command()
def version() -> None:
    """Show NixOrb version."""
    typer.echo(f"NixOrb {nixorb.__version__}")


if __name__ == "__main__":
    app()
