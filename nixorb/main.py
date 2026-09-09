"""NixOrb main entry point — AI assistant daemon.

Architecture:
  Qt Main Thread     asyncio Event Loop        Thread Pool
  ───────────────    ──────────────────        ───────────
  OrbWindow (QML) ←  EventBus                 Whisper inference
  SettingsWindow  ←  LLM streaming             Piper TTS
  NixOrbTray      ←  VRAMManager               Command execution
  HotkeyManager      PluginLoader
                     VectorMemory (ChromaDB)

Pipeline:
  Hotkey/WakeWord → Record Audio → Whisper STT → Ollama LLM →
  Piper TTS → Speak + Execute Actions
"""
from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import os
import re
import signal
import sys
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)

# Keywords for feature detection
_SCREEN_KW = frozenset({
    "screen", "looking at", "what's on", "what is on",
    "see my screen", "my display", "show me my",
})
# How many times the model may call tools before we insist on an answer.
MAX_TOOL_ROUNDS = 4

# Upper bound on one capture, used to size the follow-up wait.
MAX_RECORDING_SECONDS = 30.0

# How many follow-ups one activation may chain before you press the hotkey
# again. Stops a room with a television in it from talking to itself forever.
MAX_FOLLOW_UPS = 8


class _FollowUp:
    """Marks a turn that already has its transcript, so it is not re-recorded."""

    def __init__(self, transcript: str) -> None:
        self.data = {"transcript": transcript, "follow_up": True}

_WEB_KW = frozenset({
    "search", "look up", "google", "find out", "what is",
    "who is", "when did", "latest", "news", "current",
    "today", "right now", "recently",
})


def _strip_actions(text: str) -> str:
    """Remove <ACTION> commands and <think> reasoning from text for TTS."""
    from nixorb.llm.reasoning import ACTION_TAGS, REASONING_TAGS, strip_tags

    return strip_tags(text, ACTION_TAGS + REASONING_TAGS)


def _wants_screen(text: str) -> bool:
    """Check if the user is asking about their screen."""
    return any(kw in text.lower() for kw in _SCREEN_KW)


def _wants_web(text: str) -> bool:
    """Check if the user wants a web search."""
    return any(kw in text.lower() for kw in _WEB_KW)


def _disable_crashing_accessibility_bridge() -> None:
    """Prevent KDE Plasma AT-SPI accessibility bridge crash.

    On KDE Plasma sessions, Qt auto-constructs a QSpiAccessibleBridge
    which crashes inside PySide6. Setting QT_ACCESSIBILITY=0 prevents this.
    """
    os.environ.setdefault("QT_ACCESSIBILITY", "0")


def _select_qt_platform() -> None:
    """Select a working Qt platform plugin.

    On Wayland sessions without a native Qt Wayland plugin, force xcb
    (XWayland) which is more reliable with pip-installed PySide6.
    """
    if os.environ.get("QT_QPA_PLATFORM"):
        return

    session_type = os.environ.get("XDG_SESSION_TYPE", "").lower()
    has_wayland = bool(os.environ.get("WAYLAND_DISPLAY"))
    has_x11 = bool(os.environ.get("DISPLAY"))

    if session_type == "wayland" or has_wayland:
        if has_x11:
            os.environ["QT_QPA_PLATFORM"] = "xcb"
            log.info("Qt: Wayland session → using xcb via XWayland")
        else:
            os.environ.setdefault("QT_QPA_PLATFORM", "wayland")
            log.warning(
                "Qt: No XWayland — attempting native wayland plugin. "
                "Install qt6-wayland if this fails."
            )
    elif not has_x11:
        log.error(
            "Qt: No display detected — NixOrb needs a graphical session."
        )


async def _async_main(settings, app) -> None:
    """Main async orchestrator — initializes all components."""
    from PySide6.QtWidgets import QSystemTrayIcon

    from nixorb.core.event_bus import Event, bus
    from nixorb.core.ipc import IPCServer
    from nixorb.core.vram_manager import vram
    from nixorb.memory.vector_store import VectorMemory
    from nixorb.ui.orb_window import OrbWindow
    from nixorb.ui.tray_icon import NixOrbTray
    from nixorb.utils.logger import setup_logging

    setup_logging(log_to_file=True)

    await bus.start()
    log.info("NixOrb %s starting", __import__("nixorb").__version__)

    # Bound up front so the finally-block below can clean up whatever
    # startup managed to create before it failed.
    asr = None
    llm = None
    wake_word = None
    ipc = None
    speaker = None
    background: set[asyncio.Task] = set()

    try:
        # ── Initialize UI ────────────────────────────────────────────── #
        if QSystemTrayIcon.isSystemTrayAvailable():
            tray = NixOrbTray(settings, app)
            tray.show()
            log.info("Tray: system tray icon active")
        else:
            log.warning("Tray: system tray not available")

        orb = OrbWindow(settings, app)
        orb.show()
        orb.log_visibility()

        # ── Initialize core services ─────────────────────────────────── #
        await vram.start_monitor(poll_interval=6.0)

        # Memory. Importing chromadb and opening its SQLite store takes seconds
        # and is fully synchronous — on the loop it freezes the UI before the orb
        # has even painted, which is indistinguishable from a hang.
        memory = await asyncio.to_thread(VectorMemory, settings.memory_dir)

        # ASR — faster-whisper, any HuggingFace ASR model, or Nemotron's
        # native cache-aware streaming, per settings.asr_backend.
        from nixorb.asr.factory import build_asr
        asr = build_asr(settings)
        log.info("ASR: backend '%s' (%s)", asr.name, settings.asr_model)

        # LLM — Ollama, a local HuggingFace model (GGUF or transformers), or
        # any OpenAI-compatible endpoint. This used to always be
        # OllamaBackend regardless of what was configured, which is why
        # changing llm_backend appeared to do nothing.
        from nixorb.llm.factory import build_llm
        llm = build_llm(settings)
        llm_model = settings.llm_model

        # Health check — backend-agnostic; each backend explains how to fix
        # itself (pull an Ollama tag, install a missing package, check a
        # HF repo id, reach an OpenAI-compatible server) rather than every
        # backend being told to "ollama pull" a HuggingFace repo id.
        health = await llm.health_check()
        if health["ok"]:
            log.info("LLM: %s ready with model '%s'", settings.llm_backend, llm_model)
        else:
            log.warning("LLM: %s", health.get("error", "Unknown error"))
            if settings.llm_backend == "ollama":
                log.info(
                    "LLM: Run 'ollama pull %s' to download the model",
                    settings.llm_model,
                )
            await bus.emit(
                Event.LOG,
                data={"level": "warning", "msg": f"⚠ LLM: {health.get('error', '')}"},
                source="startup",
            )

        # TTS — Piper, GLaDOS, any HuggingFace TTS model, or OpenAI TTS, per
        # settings.tts_backend. Previously always PiperTTS regardless of
        # what was configured.
        from nixorb.tts.tts_factory import build_tts
        tts = build_tts(settings)
        log.info("TTS: backend '%s'", getattr(tts, "name", settings.tts_backend))

        # Wraps whichever engine we got, so each sentence is spoken as the
        # model produces it instead of after the whole answer lands.
        from nixorb.tts.speaker import Speaker
        speaker = Speaker(tts, streaming=settings.tts_streaming)

        # Confirmation dialog handler — must be registered *before* the executor
        # can emit its first ACTION_REQUESTED, or the request goes unanswered.
        from nixorb.ui.confirm_dialog import register_confirmation_handler
        register_confirmation_handler(bus)

        # Action executor. It refuses to construct when running as root —
        # that should cost us <ACTION> support, not the entire assistant.
        from nixorb.action.executor import ActionExecutor
        try:
            executor = ActionExecutor(settings)
        except RuntimeError as exc:
            log.warning("Actions disabled: %s", exc)
            executor = None
            await bus.emit(
                Event.LOG,
                data={"level": "warning", "msg": f"⚠ Actions disabled: {exc}"},
                source="startup",
            )

        # Plugin loader
        from nixorb.plugins.loader import PluginLoader
        plugin_loader = PluginLoader(settings.plugin_dir)
        if settings.plugins_enabled:
            plugin_loader.load_all()

        # Let timer_plugin speak its reminders through the active TTS
        # backend, not just a desktop notification. set_timer() fires from
        # a plain threading.Timer (not the event loop), so bridge it back
        # with run_coroutine_threadsafe.
        try:
            from nixorb.plugins.builtin.timer_plugin import set_speak_callback

            _main_loop = asyncio.get_running_loop()

            def _speak_reminder(text: str) -> None:
                asyncio.run_coroutine_threadsafe(tts.speak(text), _main_loop)

            set_speak_callback(_speak_reminder)
        except Exception as exc:
            log.debug("Timer: could not wire TTS speak callback: %s", exc)

        # ── Control socket ───────────────────────────────────────── #
        # This is what makes `nixorb trigger` work, and with it any KDE
        # global shortcut bound to that command.
        async def _ipc_trigger(_cmd: str) -> str:
            await bus.emit(Event.HOTKEY_TRIGGERED, source="ipc")
            return "ok: triggered"

        async def _ipc_status(_cmd: str) -> str:
            health = await llm.health_check()
            return (
                f"ok: llm={settings.llm_backend}/{llm_model} "
                f"{'up' if health['ok'] else 'down'} "
                f"asr={asr.name} tts={getattr(tts, 'name', settings.tts_backend)} "
                f"actions={'on' if executor else 'off'}"
            )

        async def _ipc_quit(_cmd: str) -> str:
            await bus.emit(Event.SHUTDOWN, source="ipc")
            return "ok: shutting down"

        ipc = IPCServer({
            "trigger": _ipc_trigger,
            "status": _ipc_status,
            "quit": _ipc_quit,
        })
        if not await ipc.start():
            log.warning(
                "IPC: control socket unavailable — `nixorb trigger` will not "
                "reach this instance"
            )

        # ── Preload ASR model ────────────────────────────────────────── #
        async def _preload_asr() -> None:
            try:
                await asr.preload()
                await bus.emit(
                    Event.LOG,
                    data={"level": "info", "msg": "✅ ASR model ready"},
                    source="startup",
                )
            except Exception as exc:
                log.warning("ASR preload failed: %s", exc)
                await bus.emit(
                    Event.LOG,
                    data={"level": "warning", "msg": f"⚠ ASR preload failed: {exc}"},
                    source="startup",
                )

        # Tasks are kept in a set (declared above): a bare create_task()
        # reference can be garbage-collected while the coroutine is still
        # suspended, which silently drops the work.
        def _spawn(coro, name: str) -> asyncio.Task:
            task = asyncio.create_task(coro, name=name)
            background.add(task)
            task.add_done_callback(background.discard)
            return task

        _spawn(_preload_asr(), "asr-preload")

        # ── Hotkey manager ───────────────────────────────────────────── #
        from nixorb.ui.hotkey import HotkeyManager
        HotkeyManager(settings).start()

        # ── Wake word detector ───────────────────────────────────────── #
        wake_word = None
        if settings.wake_word_enabled:
            from nixorb.asr.wake_word import WakeWordDetector
            wake_word = WakeWordDetector(settings)
            _spawn(wake_word.run_forever(), "wake-word")

        # ── Mic mute state ───────────────────────────────────────────── #
        mic_muted = False

        async def _on_mic_muted(payload) -> None:
            nonlocal mic_muted
            mic_muted = bool((payload.data or {}).get("muted", False))
            log.info("Mic %s", "muted" if mic_muted else "unmuted")

        bus.subscribe(Event.MIC_MUTED, _on_mic_muted)

        # ── Main conversation handler ────────────────────────────────── #
        conversation: list[dict[str, str]] = [
            {"role": "system", "content": settings.llm_system_prompt}
        ]

        async def _stream_with_tools(messages, tools, depth: int = 0,
                                     on_text=None) -> str:
            """Stream one reply, dispatching plugin tool calls and retrying.

            Ollama returns tool calls instead of text; without this the model
            would call a plugin and the turn would end with an empty answer.
            """
            chunks: list[str] = []
            async for chunk in llm.stream(messages, tools=tools):
                chunks.append(chunk)
                if on_text is not None:
                    await on_text(chunk)
            text = "".join(chunks)

            calls = list(llm.last_tool_calls)
            if not calls or depth >= MAX_TOOL_ROUNDS:
                if calls:
                    log.warning(
                        "Tool loop hit the %d-round limit — answering as-is",
                        MAX_TOOL_ROUNDS,
                    )
                return text

            messages.append(
                {"role": "assistant", "content": text, "tool_calls": calls}
            )
            for call in calls:
                name = call.get("name", "")
                args = call.get("arguments") or {}
                if isinstance(args, str):
                    try:
                        args = json.loads(args)
                    except json.JSONDecodeError:
                        args = {}
                log.info("🔧 Tool call: %s(%s)", name, args)
                result = await plugin_loader.dispatch(name, args)
                log.info("🔧 Tool result: %s → %s", name, result[:120])
                await bus.emit(
                    Event.LOG,
                    data={"level": "info", "msg": f"🔧 {name} → {result[:160]}"},
                    source="main",
                )
                messages.append(
                    {"role": "tool", "name": name, "content": result}
                )

            return await _stream_with_tools(messages, tools, depth + 1, on_text)

        async def _handle_turn(payload) -> None:
            """Handle a single conversation turn."""
            nonlocal mic_muted

            if mic_muted:
                log.debug("Mic muted — ignoring trigger")
                return

            await bus.emit(Event.ORB_LISTENING, source="main")
            log.info("🎙 Listening…")

            # Record and transcribe. Streaming backends (Nemotron) emit
            # partials while you are still speaking; the rest return once.
            supplied = (getattr(payload, "data", None) or {}).get("transcript")
            if supplied:
                transcript = supplied
            elif settings.asr_streaming and asr.supports_streaming:
                pieces: list[str] = []
                async for piece in asr.stream_transcribe():
                    pieces.append(piece)
                    await bus.emit(
                        Event.TRANSCRIPT_READY,
                        data={"text": "".join(pieces), "partial": True},
                        source="main",
                    )
                transcript = "".join(pieces).strip()
            else:
                transcript = (await asr.record_and_transcribe()) or ""

            if not transcript:
                # A crash in the model is not silence. Saying "No speech
                # detected" for both hid a real fault behind a normal one.
                failure = getattr(asr, "last_error", "")
                if failure:
                    log.error("Transcription failed: %s", failure)
                    await bus.emit(
                        Event.LOG,
                        data={"level": "error",
                              "msg": f"❌ Could not transcribe: {failure}"},
                        source="main",
                    )
                else:
                    log.info("No speech detected")
                await bus.emit(Event.ORB_IDLE, source="main")
                return

            log.info(
                "📝 Transcript: %s%s",
                transcript[:200], "…" if len(transcript) > 200 else "",
            )
            await bus.emit(
                Event.LOG,
                data={"level": "info", "msg": f"🎙 You: {transcript}"},
                source="main",
            )

            # Build user message with context
            user_msg = transcript

            # Add memory context. Embedding runs ONNX inference, so it goes
            # off-thread: on the loop, the first query froze the orb (and the
            # control socket) for over ten seconds.
            if settings.memory_enabled:
                mem_ctx = await asyncio.to_thread(
                    memory.build_context_block, transcript
                )
                if mem_ctx:
                    user_msg = mem_ctx + transcript

            # Check clipboard
            if settings.clipboard_enabled and "clipboard" in transcript.lower():
                from nixorb.action.clipboard import read_clipboard
                clip = await read_clipboard()
                if clip:
                    user_msg += f"\n\n[Clipboard]:\n{clip}"
                    log.debug("Clipboard injected (%d chars)", len(clip))

            # Check web search
            if settings.web_search_enabled and _wants_web(transcript):
                try:
                    from nixorb.utils.web_search import search_formatted
                    log.info("🔍 Web search: %s", transcript[:60])
                    web_ctx = await search_formatted(transcript, settings.web_search_max_results)
                    user_msg += f"\n\n{web_ctx}"
                except Exception as exc:
                    log.warning("Web search failed: %s", exc)

            # Check screen capture
            if settings.screen_capture_enabled and _wants_screen(transcript):
                await bus.emit(Event.SCREEN_CAPTURE_REQ, source="main")
                try:
                    from nixorb.vision.screen_capture import ScreenCapture
                    screen = ScreenCapture()
                    desc = await screen.describe()
                    user_msg += f"\n\n[Screen]: {desc}"
                    await bus.emit(Event.SCREEN_CAPTURE_DONE, source="main")
                except Exception as exc:
                    log.warning("Screen capture failed: %s", exc)
                    await bus.emit(Event.SCREEN_CAPTURE_DONE, source="main")

            # Add to conversation
            conversation.append({"role": "user", "content": user_msg})

            # Unload Whisper to free VRAM for LLM
            await asr.unload()
            await bus.emit(Event.ORB_THINKING, source="main")
            log.info("🤔 Querying LLM: %s/%s", settings.llm_backend, llm_model)

            # Stream the answer, speaking each sentence as it lands.
            await bus.emit(Event.ORB_SPEAKING, source="main")
            await speaker.start()
            try:
                tools = plugin_loader.get_tool_definitions() or None
                response = await _stream_with_tools(
                    conversation, tools, on_text=speaker.feed
                )

            except Exception as exc:
                log.error("LLM error: %s", exc)
                await bus.emit(Event.LLM_ERROR, data={"error": str(exc)}, source="main")
                await bus.emit(Event.ORB_ERROR, source="main")
                await asyncio.sleep(2)
                await bus.emit(Event.ORB_IDLE, source="main")
                return

            # Add response to conversation
            conversation.append({"role": "assistant", "content": response})
            log.info("🤖 Response (%d chars): %s", len(response), response[:100])
            await bus.emit(
                Event.LOG,
                data={"level": "info", "msg": f"🤖 NixOrb: {response[:200]}"},
                source="main",
            )

            # Store in memory (also an embedding call — same treatment).
            if settings.memory_enabled:
                await asyncio.to_thread(
                    memory.store,
                    f"User: {transcript}\nAssistant: {response[:600]}",
                    {"type": "conversation"},
                )

            # Execute any actions
            action_results = (
                await executor.handle_llm_output(response) if executor else []
            )
            if action_results:
                # Report stderr too — the model needs to see failures, not
                # just silence, or it will happily claim the command worked.
                result_texts = [
                    str(r) for r in action_results if r.stdout or r.stderr
                ]
                if result_texts:
                    result_msg = "\n\n".join(result_texts)
                    conversation.append(
                        {"role": "user", "content": f"<RESULT>\n{result_msg}\n</RESULT>"}
                    )
                    try:
                        followup = await _stream_with_tools(
                            conversation, tools, on_text=speaker.feed
                        )
                        if followup:
                            conversation.append(
                                {"role": "assistant", "content": followup}
                            )
                            response = followup
                    except Exception as exc:
                        log.warning("Follow-up LLM call failed: %s", exc)

            # Copy code blocks to clipboard
            if settings.clipboard_enabled:
                from nixorb.action.clipboard import write_clipboard
                code_blocks = re.findall(r"```(?:\w+)?\n(.*?)```", response, re.DOTALL)
                if code_blocks:
                    await write_clipboard(code_blocks[-1].strip())
                    log.debug("Copied code block to clipboard")

            # Finish speaking. With streaming on, most of this already came
            # out of the speakers while the model was still generating.
            if settings.tts_streaming:
                await speaker.finish()
            else:
                speech_text = _strip_actions(response)
                sentences = re.split(r"(?<=[.!?])\s+", speech_text)
                limit = max(1, int(settings.tts_max_sentences))
                tts_text = (
                    " ".join(sentences[:limit])
                    if len(sentences) > limit
                    else speech_text
                )
                if tts_text:
                    log.info("🔊 Speaking: %s", tts_text[:80])
                    await speaker.say(tts_text)

            await bus.emit(Event.ORB_IDLE, source="main")

            # Trim conversation history
            if len(conversation) > 22:
                conversation[1:] = conversation[-20:]

        async def _listen_for_follow_up() -> str | None:
            """Briefly listen again so you can just keep talking."""
            if settings.follow_up_seconds <= 0 or mic_muted:
                return None

            log.info("👂 Listening for a follow-up (%.0fs)",
                     settings.follow_up_seconds)
            await bus.emit(Event.ORB_LISTENING, source="follow-up")
            try:
                heard = await asyncio.wait_for(
                    asr.record_and_transcribe(),
                    timeout=settings.follow_up_seconds + MAX_RECORDING_SECONDS,
                )
            except TimeoutError:
                heard = None
            except Exception as exc:
                log.debug("Follow-up listen failed: %s", exc)
                heard = None

            if not heard:
                await bus.emit(Event.ORB_IDLE, source="follow-up")
                return None
            return heard

        # ── Trigger wiring ───────────────────────────────────────────── #
        # A turn takes seconds to minutes. Bus handlers are awaited inline by the
        # dispatch loop, so running the turn *as* a handler stops every other
        # event dead: the orb never changes colour, log lines never arrive, and
        # the executor's ACTION_REQUESTED can never be delivered — a real
        # deadlock, since the turn is itself waiting on the answer. Spawn it.
        turn_lock = asyncio.Lock()

        async def _run_turn(payload) -> None:
            if turn_lock.locked():
                log.info("Trigger ignored — a turn is already in progress")
                return
            async with turn_lock:
                try:
                    current = payload
                    for _ in range(MAX_FOLLOW_UPS + 1):
                        await _handle_turn(current)
                        heard = await _listen_for_follow_up()
                        if not heard:
                            break
                        log.info("↩ Follow-up: %s", heard[:80])
                        current = _FollowUp(heard)
                    else:
                        log.info(
                            "Follow-up limit (%d) reached — press the hotkey "
                            "to continue", MAX_FOLLOW_UPS,
                        )
                except asyncio.CancelledError:
                    raise
                except Exception as exc:
                    log.exception("Turn failed: %s", exc)
                    await bus.emit(
                        Event.LOG,
                        data={"level": "error", "msg": f"❌ Turn failed: {exc}"},
                        source="main",
                    )
                    await bus.emit(Event.ORB_ERROR, source="main")
                    await asyncio.sleep(2)
                    await bus.emit(Event.ORB_IDLE, source="main")

        async def _on_trigger(payload) -> None:
            # Barge-in: talking over the orb should stop it, the way you can
            # cut off any assistant worth using — not queue behind it.
            if settings.barge_in and speaker.speaking:
                log.info("⏹ Barge-in — stopping playback")
                await speaker.stop()
            _spawn(_run_turn(payload), "turn")

        bus.subscribe(Event.HOTKEY_TRIGGERED, _on_trigger, priority=2)
        bus.subscribe(Event.WAKE_WORD_DETECTED, _on_trigger, priority=2)
        bus.subscribe(Event.ORB_CLICKED, _on_trigger, priority=2)

        # Log handler
        async def _log_to_python(payload) -> None:
            data = payload.data or {}
            level = data.get("level", "info")
            msg = data.get("msg", "")
            getattr(
                log,
                level if level in ("debug", "info", "warning", "error") else "info",
            )("[bus] %s", msg)

        bus.subscribe(Event.LOG, _log_to_python)

        # ── Shutdown handling ────────────────────────────────────────── #
        stop_event = asyncio.Event()

        def _on_qt_quit() -> None:
            log.info("Qt quit signal received")
            stop_event.set()

        app.aboutToQuit.connect(_on_qt_quit)

        async def _on_shutdown(_payload) -> None:
            stop_event.set()

        bus.subscribe(Event.SHUTDOWN, _on_shutdown)

        # ── Ready ────────────────────────────────────────────────────── #
        log.info(
            "✅ NixOrb %s ready — hotkey: %s | LLM: %s/%s | ASR: %s/%s | TTS: %s",
            __import__("nixorb").__version__,
            settings.hotkey,
            settings.llm_backend, llm_model,
            asr.name, settings.asr_model,
            getattr(tts, "name", settings.tts_backend),
        )
        await bus.emit(
            Event.LOG,
            data={
                "level": "success",
                "msg": (
                    f"✅ NixOrb ready | hotkey: {settings.hotkey} "
                    f"| LLM: {llm_model} | ASR: {asr.name}"
                ),
            },
            source="startup",
        )

        # Wait for shutdown
        await stop_event.wait()

    finally:
        log.info("Shutting down…")

        if wake_word is not None:
            wake_word.stop()

        if speaker is not None:
            # Its drain task outlives the turn otherwise, and asyncio
            # complains that a pending task was destroyed.
            with contextlib.suppress(Exception):
                await speaker.stop()

        if ipc is not None:
            with contextlib.suppress(Exception):
                await ipc.stop()

        for task in list(background):
            task.cancel()
        if background:
            await asyncio.gather(*background, return_exceptions=True)

        steps: list[tuple[str, Any]] = []
        if llm is not None:
            steps.append(("LLM close", llm.close))
        if asr is not None:
            steps.append(("ASR unload", asr.unload))
        steps.append(("VRAM stop", vram.stop))
        steps.append(("EventBus stop", bus.stop))

        for step, make_coro in steps:
            try:
                await asyncio.wait_for(make_coro(), timeout=10.0)
            except Exception as exc:
                log.warning("Shutdown: %s failed: %s", step, exc)

        log.info("NixOrb stopped")


def main() -> int:
    """Entry point — initializes Qt and starts the async loop."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
        datefmt="%H:%M:%S",
    )

    # Prevent Qt crashes on KDE. Both must happen before QApplication exists.
    _disable_crashing_accessibility_bridge()
    _select_qt_platform()

    try:
        import qasync
        from PySide6.QtWidgets import QApplication
    except ImportError as exc:
        log.error(
            "NixOrb cannot start: %s\n"
            "  Install the GUI dependencies:  pip install PySide6 qasync",
            exc,
        )
        return 1

    from nixorb.core.ipc import is_running, socket_path
    from nixorb.settings import Settings

    if is_running():
        log.error(
            "NixOrb is already running (control socket: %s).\n"
            "  Activate it with:  nixorb trigger\n"
            "  Stop it with:      nixorb quit",
            socket_path(),
        )
        return 1

    settings = Settings.load()

    # Create Qt application
    existing = QApplication.instance()
    app = existing if isinstance(existing, QApplication) else QApplication(sys.argv)
    app.setApplicationName("NixOrb")
    app.setApplicationVersion(__import__("nixorb").__version__)
    app.setOrganizationName("NixOrb")
    app.setQuitOnLastWindowClosed(False)

    # qasync: integrate asyncio with the Qt event loop
    loop = qasync.QEventLoop(app)
    asyncio.set_event_loop(loop)

    # qasync schedules every asyncio callback through one QObject
    # (its internal `_SimpleTimer`) via QObject.startTimer(). That object is
    # only usable from the thread that first calls startTimer() on it. If
    # anything reaches it from another thread before this one does — a
    # platform-theme/DBus integration thread some desktops (KDE Plasma
    # included) spin up during QApplication startup, for instance — Qt
    # refuses with "QObject::startTimer: Timers can only be used with
    # threads started with QThread" (and the equivalent QSocketNotifier
    # warning for any fd registered the same way), and every asyncio
    # callback in the process silently stops firing from then on —
    # including the ones needed to even start NixOrb's own startup
    # coroutine, which is why nothing after this point would otherwise log
    # anything at all. Running one trivial coroutine to completion here,
    # before anything else touches Qt or asyncio, claims that first call
    # for this thread while the loop is still idle.
    loop.run_until_complete(asyncio.sleep(0))

    # Qt swallows SIGINT, so Ctrl-C would otherwise do nothing at all.
    def _handle_signal(signum, _frame) -> None:
        log.info("Received %s — shutting down", signal.Signals(signum).name)
        from nixorb.core.event_bus import Event, bus

        # Not app.quit(): that stops the qasync loop underneath _async_main,
        # so its cleanup runs with no running loop and silently does nothing.
        bus.emit_sync(Event.SHUTDOWN, source="signal")

    for sig in (signal.SIGINT, signal.SIGTERM):
        with contextlib.suppress(ValueError, OSError):
            signal.signal(sig, _handle_signal)

    # Give Python a chance to run the signal handler between Qt events.
    from PySide6.QtCore import QTimer

    sigtimer = QTimer()
    sigtimer.start(200)
    sigtimer.timeout.connect(lambda: None)

    status = 0
    try:
        with loop:
            loop.run_until_complete(_async_main(settings, app))
    except KeyboardInterrupt:
        log.info("KeyboardInterrupt — shutting down")
    except Exception as exc:
        # Without this, a startup failure is a bare traceback on a terminal
        # the user probably is not looking at.
        log.exception("NixOrb failed to start: %s", exc)
        log.error(
            "Full log: %s",
            Path.home() / ".local" / "share" / "nixorb" / "logs" / "nixorb.log",
        )
        status = 1
    return status


if __name__ == "__main__":
    sys.exit(main())
