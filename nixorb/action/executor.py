"""NixOrb action executor — bash command execution from LLM output.

Parses <ACTION>…</ACTION> blocks out of an LLM response and runs them,
asking the user first for anything dangerous.

The confirmation handshake:

    ActionExecutor            EventBus              confirm_dialog
    ──────────────            ────────              ──────────────
    emit ACTION_REQUESTED  ──────────────────────►  show dialog
    await pending[req_id]                           │
    resolve pending[req_id] ◄── ACTION_CONFIRMED ───┘
                                /ACTION_DENIED

Both halves key off ``request_id``. Earlier versions created the future
locally and never handed it to anybody, so *every* command sat for the full
timeout and was then reported as denied, with no dialog ever shown.
"""
from __future__ import annotations

import asyncio
import contextlib
import logging
import os
import re
import shutil
import uuid
from dataclasses import dataclass
from typing import TYPE_CHECKING

from nixorb.core.event_bus import Event, EventPayload, bus

if TYPE_CHECKING:
    from nixorb.settings import Settings

log = logging.getLogger(__name__)

# Maximum output capture size
MAX_OUTPUT_BYTES = 50_000
# Command timeout (patched in tests)
TIMEOUT_SECONDS = 30.0
# Fallback if settings carries no action_confirm_timeout
DEFAULT_CONFIRM_TIMEOUT = 60.0

# Commands that are never run, confirmed or not.
HARD_DENYLIST = (
    "rm -rf /",
    "rm -rf /*",
    "mkfs.",
    "dd if=/dev/zero of=/dev/sd",
    "dd if=/dev/random of=/dev/sd",
    "> /dev/sd",
    ":(){ :|:& };:",
)

# Substrings that make a command "dangerous" enough to need a click.
REQUIRE_CONFIRM = (
    "rm -rf", "rm -r", "dd ", "mkfs", "fdisk", "parted",
    "chmod -R", "chown -R", "pacman -R", "pacman -S",
    "systemctl stop", "systemctl disable", "kill ", "pkill",
    "curl", "wget", "pip install", "pip uninstall", "sudo",
)


def is_hard_denied(command: str) -> bool:
    """True for commands that must never run."""
    lowered = " ".join(command.lower().split())
    return any(pattern in lowered for pattern in HARD_DENYLIST)


def is_high_risk(command: str) -> bool:
    """True for commands worth flagging extra-loudly in the dialog.

    This is a hint for the UI, not a security boundary: `echo $(rm -rf ~)`,
    `bash -c …` and `python -c …` all sail past any substring list. The
    boundary is require_action_confirmation, which gates *every* command.
    """
    lowered = command.lower()
    return any(pattern in lowered for pattern in REQUIRE_CONFIRM)


@dataclass
class ActionResult:
    """Result of executing an action."""

    command: str
    stdout: str = ""
    stderr: str = ""
    returncode: int = 0
    approved: bool = True
    timed_out: bool = False

    @property
    def success(self) -> bool:
        return self.approved and not self.timed_out and self.returncode == 0

    def __str__(self) -> str:
        parts = [f"$ {self.command}"]
        if self.stdout:
            parts.append(self.stdout.rstrip())
        if self.stderr:
            parts.append(f"[stderr] {self.stderr.rstrip()}")
        if self.returncode:
            parts.append(f"[exit {self.returncode}]")
        return "\n".join(parts)


class ActionExecutor:
    """Executes bash commands from LLM responses."""

    def __init__(self, settings: Settings) -> None:
        # Running the assistant as root would let a hallucinated command do
        # unbounded damage; refuse rather than sandbox our way around it.
        if os.geteuid() == 0:
            raise RuntimeError(
                "NixOrb refuses to run as root — command execution would be "
                "unrestricted. Start it as your normal user."
            )

        self._settings = settings
        self._sandbox_available = shutil.which("bwrap") is not None
        self._pending: dict[str, asyncio.Future[bool]] = {}

        bus.subscribe(Event.ACTION_CONFIRMED, self._on_confirmed)
        bus.subscribe(Event.ACTION_DENIED, self._on_denied)

    # ── confirmation handshake ───────────────────────────────────── #

    def _resolve(self, payload: EventPayload, approved: bool) -> None:
        request_id = (payload.data or {}).get("request_id", "")
        future = self._pending.get(request_id)
        if future is not None and not future.done():
            future.set_result(approved)

    async def _on_confirmed(self, payload: EventPayload) -> None:
        self._resolve(payload, True)

    async def _on_denied(self, payload: EventPayload) -> None:
        self._resolve(payload, False)

    @property
    def _confirm_timeout(self) -> float:
        return float(
            getattr(self._settings, "action_confirm_timeout",
                    DEFAULT_CONFIRM_TIMEOUT)
        )

    async def _request_approval(self, command: str) -> bool:
        """Ask the UI for approval, and wait for the answer."""
        request_id = uuid.uuid4().hex[:8]
        loop = asyncio.get_running_loop()
        future: asyncio.Future[bool] = loop.create_future()
        self._pending[request_id] = future

        try:
            # emit (not emit_sync) so the request is queued before we wait —
            # emit_sync would fire-and-forget and could race the await.
            await bus.emit(
                Event.ACTION_REQUESTED,
                data={
                    "command": command,
                    "request_id": request_id,
                    "high_risk": is_high_risk(command),
                },
                source="ActionExecutor",
                priority=1,
            )
            return await asyncio.wait_for(future, timeout=self._confirm_timeout)
        except TimeoutError:
            log.warning(
                "Action: no answer within %.0fs for '%s' — treating as denied. "
                "Is anything subscribed to ACTION_REQUESTED?",
                self._confirm_timeout, command,
            )
            return False
        finally:
            self._pending.pop(request_id, None)

    # ── parsing ──────────────────────────────────────────────────── #

    def _extract_actions(self, text: str) -> list[str]:
        """Extract <ACTION> commands from the answer — not from the thinking.

        A reasoning model rehearses inside `<think>`: it will write a
        command out, consider it, and often decide against it. Running one
        from there executes something the model explicitly rejected, so the
        reasoning is removed before anything is extracted. Only commands
        the model put in its actual answer are eligible.
        """
        from nixorb.llm.reasoning import REASONING_TAGS, strip_tags

        answer = strip_tags(text, REASONING_TAGS)
        if answer != text.strip():
            considered = re.findall(
                r"<ACTION>(.*?)</ACTION>", text, re.DOTALL | re.IGNORECASE
            )
            kept = re.findall(
                r"<ACTION>(.*?)</ACTION>", answer, re.DOTALL | re.IGNORECASE
            )
            if len(considered) > len(kept):
                log.info(
                    "Action: ignored %d command(s) the model only weighed up "
                    "inside its reasoning",
                    len(considered) - len(kept),
                )

        pattern = r"<ACTION>(.*?)</ACTION>"
        commands = re.findall(pattern, answer, re.DOTALL | re.IGNORECASE)
        return [cmd.strip() for cmd in commands if cmd.strip()]

    def _build_command(self, cmd: str) -> list[str]:
        """Build the argv, optionally wrapped in a bubblewrap sandbox."""
        want_sandbox = bool(getattr(self._settings, "sandbox_actions", False))
        if want_sandbox and self._sandbox_available:
            return [
                "bwrap",
                "--unshare-net",
                "--ro-bind", "/", "/",
                "--tmpfs", "/tmp",
                "--proc", "/proc",
                "--dev", "/dev",
                "--die-with-parent",
                "bash", "-c", cmd,
            ]
        if want_sandbox and not self._sandbox_available:
            log.warning(
                "Action: sandbox_actions is on but bwrap is not installed — "
                "running unsandboxed. Install bubblewrap."
            )
        return ["bash", "-c", cmd]

    # ── execution ────────────────────────────────────────────────── #

    async def _execute_command(self, command: str) -> ActionResult:
        """Execute a single command, confirming first when needed."""
        if is_hard_denied(command):
            log.warning("Action: hard-denied '%s'", command)
            return ActionResult(
                command=command,
                stderr="Command denied: matches the hard denylist.",
                returncode=-1,
                approved=False,
            )

        # Confirmation gates every command, not just ones matching a pattern
        # list — anything the model emits can shell out to anything else.
        if self._settings.require_action_confirmation:
            if not await self._request_approval(command):
                log.info("Action: denied '%s'", command)
                return ActionResult(
                    command=command,
                    stderr="Command denied by the user.",
                    returncode=-1,
                    approved=False,
                )

        proc = None
        try:
            log.info("Action: executing '%s'", command)
            proc = await asyncio.create_subprocess_exec(
                *self._build_command(command),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                limit=MAX_OUTPUT_BYTES,
            )
            stdout, stderr = await asyncio.wait_for(
                proc.communicate(), timeout=TIMEOUT_SECONDS
            )

            result = ActionResult(
                command=command,
                stdout=stdout.decode("utf-8", errors="replace")[:5000],
                stderr=stderr.decode("utf-8", errors="replace")[:2000],
                returncode=proc.returncode or 0,
            )
            log.info(
                "Action: '%s' → rc=%d, stdout=%d chars",
                command, result.returncode, len(result.stdout),
            )
            return result

        except TimeoutError:
            log.error("Action: command timed out '%s'", command)
            if proc is not None and proc.returncode is None:
                proc.kill()
                # Reap it so we do not leak a zombie or an "unawaited" warning.
                with contextlib.suppress(TimeoutError, ProcessLookupError):
                    await asyncio.wait_for(proc.wait(), timeout=5.0)
            return ActionResult(
                command=command,
                stderr=f"Command timed out after {TIMEOUT_SECONDS:g}s",
                returncode=-1,
                timed_out=True,
            )
        except FileNotFoundError as exc:
            log.error("Action: cannot run '%s': %s", command, exc)
            return ActionResult(
                command=command, stderr=str(exc), returncode=-1
            )
        except Exception as exc:
            log.error("Action: error executing '%s': %s", command, exc)
            return ActionResult(
                command=command, stderr=str(exc), returncode=-1
            )

    async def handle_llm_output(self, text: str) -> list[ActionResult]:
        """Extract and execute every action in an LLM response."""
        commands = self._extract_actions(text)
        if not commands:
            return []

        log.info("Action: found %d command(s) to execute", len(commands))
        results: list[ActionResult] = []
        for cmd in commands:
            result = await self._execute_command(cmd)
            results.append(result)

            await bus.emit(
                Event.ACTION_RESULT,
                data={
                    "command": result.command,
                    "stdout": result.stdout,
                    "stderr": result.stderr,
                    "returncode": result.returncode,
                    "success": result.success,
                },
                source="ActionExecutor",
            )

        return results
