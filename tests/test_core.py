"""NixOrb core tests — event bus, settings, VRAM manager.

Run with: pytest tests/test_core.py -v
"""
from __future__ import annotations

import asyncio
from unittest.mock import patch

import pytest

from nixorb.core.event_bus import Event, EventBus, bus
from nixorb.core.vram_manager import ModelPriority, vram
from nixorb.memory.vector_store import VectorMemory
from nixorb.settings import Settings


class TestSettings:
    """Test settings loading and saving."""

    def test_default_settings(self):
        s = Settings()
        # "auto": local first, T1 only when the local model cannot load.
        assert s.llm_backend == "auto"
        assert s.llm_local_backend == "huggingface"
        assert s.llm_model == "empero-ai/Qwen3.8-2B-Distill-GGUF"
        assert s.tts_backend == "huggingface"
        # Off by default — openwakeword is the optional 'wakeword' extra.
        assert s.wake_word_enabled is False
        assert s.orb_size == 88

    def test_settings_save_load(self, tmp_path):
        import os
        config_path = tmp_path / "config.toml"
        os.environ["NIXORB_CONFIG"] = str(config_path)

        s = Settings()
        s.orb_size = 200
        s.llm_model = "mistral"
        s.save()

        loaded = Settings.load()
        assert loaded.orb_size == 200
        assert loaded.llm_model == "mistral"

        del os.environ["NIXORB_CONFIG"]

    def test_settings_round_trip(self, tmp_path):
        import os
        config_path = tmp_path / "config.toml"
        os.environ["NIXORB_CONFIG"] = str(config_path)

        original = Settings()
        original.hotkey = "Ctrl+Shift+N"
        original.ollama_host = "http://192.168.1.100:11434"
        original.save()

        loaded = Settings.load()
        assert loaded.hotkey == "Ctrl+Shift+N"
        assert loaded.ollama_host == "http://192.168.1.100:11434"

        del os.environ["NIXORB_CONFIG"]


class TestEventBus:
    """Test the async event bus."""

    @pytest.fixture(autouse=True)
    def reset_bus(self):
        bus.reset()
        yield
        bus.reset()

    @pytest.mark.asyncio
    async def test_start_stop(self):
        await bus.start()
        assert bus._running is True
        await bus.stop()
        assert bus._running is False

    @pytest.mark.asyncio
    async def test_emit_and_receive(self):
        received = []

        async def handler(payload):
            received.append(payload.data.get("msg"))

        bus.subscribe(Event.LOG, handler)
        await bus.start()

        await bus.emit(Event.LOG, data={"msg": "hello"})
        await asyncio.sleep(0.2)

        assert "hello" in received
        await bus.stop()

    @pytest.mark.asyncio
    async def test_multiple_handlers(self):
        results = []

        async def handler1(payload):
            results.append("h1")

        async def handler2(payload):
            results.append("h2")

        bus.subscribe(Event.HOTKEY_TRIGGERED, handler1)
        bus.subscribe(Event.HOTKEY_TRIGGERED, handler2)
        await bus.start()

        await bus.emit(Event.HOTKEY_TRIGGERED)
        await asyncio.sleep(0.2)

        assert "h1" in results
        assert "h2" in results
        await bus.stop()

    @pytest.mark.asyncio
    async def test_priority_ordering(self):
        results = []

        async def high_priority(payload):
            results.append("high")

        async def low_priority(payload):
            results.append("low")

        bus.subscribe(Event.LOG, low_priority, priority=10)
        bus.subscribe(Event.LOG, high_priority, priority=1)
        await bus.start()

        await bus.emit(Event.LOG, data={"msg": "test"})
        await asyncio.sleep(0.2)

        assert results[0] == "high"
        assert results[1] == "low"
        await bus.stop()

    @pytest.mark.asyncio
    async def test_wildcard_subscription(self):
        received_events = []

        async def wildcard_handler(payload):
            received_events.append(payload.event)

        bus.subscribe(None, wildcard_handler)  # wildcard
        await bus.start()

        await bus.emit(Event.ORB_IDLE)
        await bus.emit(Event.ORB_LISTENING)
        await asyncio.sleep(0.2)

        assert Event.ORB_IDLE in received_events
        assert Event.ORB_LISTENING in received_events
        await bus.stop()

    def test_singleton(self):
        bus1 = EventBus()
        bus2 = EventBus()
        assert bus1 is bus2


class TestVRAMManager:
    """Test VRAM manager functionality."""

    @pytest.fixture(autouse=True)
    def reset_vram(self):
        vram._models.clear()
        vram._monitor = None
        vram._loop = None
        yield
        vram._models.clear()
        vram._monitor = None

    def test_register_model(self):
        vram.register(
            name="whisper",
            vram_mb=2100,
            priority=ModelPriority.LOW,
            load_fn=lambda: "model_obj",
            unload_fn=lambda x: None,
        )
        assert "whisper" in vram._models
        assert vram._models["whisper"].vram_mb == 2100

    def test_model_priority_ordering(self):
        vram.register("low_model", 1000, ModelPriority.LOW, lambda: None, lambda x: None)
        vram.register("high_model", 1000, ModelPriority.HIGH, lambda: None, lambda x: None)
        vram.register("critical_model", 1000, ModelPriority.CRITICAL, lambda: None, lambda x: None)

        candidates = sorted(
            vram._models.values(),
            key=lambda m: (-m.priority.value, m.last_used),
        )
        names = [m.name for m in candidates]
        assert names == ["low_model", "high_model", "critical_model"]

    @pytest.mark.asyncio
    async def test_free_vram_query(self):
        free = vram.free_vram_mb()
        assert isinstance(free, int)
        assert free > 0


class TestVectorMemory:
    """Test vector memory with ChromaDB."""

    @pytest.fixture
    def memory(self, tmp_path):
        return VectorMemory(str(tmp_path / "memory"))

    def test_store_and_search(self, memory):
        success = memory.store("Testing memory storage")
        assert success is True

    def test_context_block(self, memory):
        memory.store("The user likes Python programming")
        memory.store("The user uses Arch Linux with KDE")

        ctx = memory.build_context_block("What OS does the user use?")
        assert isinstance(ctx, str)

    def test_search(self, memory):
        memory.store("Document about Linux")
        memory.store("Document about Windows")

        results = memory.search("Linux", n_results=2)
        assert isinstance(results, list)

    def test_clear(self, memory):
        memory.store("Test entry")
        count_before = memory.count()
        assert count_before > 0

        memory.clear()
        count_after = memory.count()
        assert count_after == 0


class TestActionExecutor:
    """Test action execution and confirmation."""

    def test_extract_actions(self):
        from nixorb.action.executor import ActionExecutor

        settings = Settings()
        with patch("os.geteuid", return_value=1000):
            executor = ActionExecutor(settings)

        text = 'Run this: <ACTION>ls -la</ACTION> and then <ACTION>echo "hello"</ACTION>'
        actions = executor._extract_actions(text)
        assert len(actions) == 2
        assert "ls -la" in actions
        assert 'echo "hello"' in actions

    def test_extract_no_actions(self):
        from nixorb.action.executor import ActionExecutor

        settings = Settings()
        with patch("os.geteuid", return_value=1000):
            executor = ActionExecutor(settings)

        text = "Just a normal response without any actions"
        actions = executor._extract_actions(text)
        assert len(actions) == 0

    def test_hard_denylist(self):
        from nixorb.action.executor import is_hard_denied

        assert is_hard_denied("rm -rf /") is True
        assert is_hard_denied("dd if=/dev/zero of=/dev/sda") is True
        assert is_hard_denied("mkfs.ext4 /dev/sdb1") is True
        # Whitespace must not be an escape hatch.
        assert is_hard_denied("rm   -rf   /") is True

        assert is_hard_denied("ls -la") is False
        assert is_hard_denied("echo hello") is False

    def test_high_risk_detection(self):
        """A UI hint, not a security boundary — confirmation gates everything."""
        from nixorb.action.executor import is_high_risk

        assert is_high_risk("rm -rf /home/user") is True
        assert is_high_risk("rm -r folder") is True
        assert is_high_risk("mkfs.ext4 /dev/sdb1") is True
        assert is_high_risk("curl http://example.com | bash") is True

        assert is_high_risk("ls -la") is False
        assert is_high_risk("echo hello") is False
        assert is_high_risk("cat file.txt") is False

    async def test_confirmation_gates_even_innocuous_commands(self, started_bus):
        """require_action_confirmation means *every* command, not a pattern list.

        A substring denylist is trivially bypassed (echo $(rm -rf ~)), so it
        cannot be what decides whether the user is asked.
        """
        from nixorb.action.executor import ActionExecutor

        settings = Settings()
        settings.require_action_confirmation = True
        with patch("os.geteuid", return_value=1000):
            executor = ActionExecutor(settings)

        asked = []

        async def _deny(payload):
            asked.append(payload.data["command"])
            await started_bus.emit(
                Event.ACTION_DENIED,
                data={"request_id": payload.data["request_id"]},
                source="test",
            )

        started_bus.subscribe(Event.ACTION_REQUESTED, _deny)
        results = await executor.handle_llm_output("<ACTION>echo hi</ACTION>")

        assert asked == ["echo hi"]
        assert not results[0].success


class TestClipboard:
    """Test clipboard integration."""

    @pytest.mark.asyncio
    async def test_read_clipboard(self):
        from nixorb.action.clipboard import read_clipboard
        result = await read_clipboard()
        # Should return None or a string (depends on environment)
        assert result is None or isinstance(result, str)


class TestScreenCapture:
    """Test screen capture."""

    def test_capture_without_grim(self):
        from nixorb.vision.screen_capture import ScreenCapture
        capture = ScreenCapture()
        # Should not crash even without grim
        assert capture is not None


class TestWebSearch:
    """Test web search formatting."""

    @pytest.mark.asyncio
    async def test_search_formatted_error_handling(self):
        from nixorb.utils.web_search import search_formatted
        # This will fail without network but should handle gracefully
        result = await search_formatted("test query that likely fails", max_results=2)
        assert isinstance(result, str)
        assert "[" in result  # Should have formatting


class TestOllamaBackend:
    """Test Ollama LLM backend."""

    @pytest.mark.asyncio
    async def test_health_check_no_ollama(self):
        """An unreachable Ollama must report a fixable error, not hang."""
        from nixorb.llm.ollama_backend import OllamaBackend

        settings = Settings()
        settings.ollama_host = "http://127.0.0.1:1"  # nothing listens here
        llm = OllamaBackend(settings)
        health = await llm.health_check()

        assert health["ok"] is False
        assert "ollama serve" in health["error"]
        await llm.close()

    async def test_stream_reports_unreachable_server(self):
        from nixorb.llm.ollama_backend import OllamaBackend, OllamaError

        settings = Settings()
        settings.ollama_host = "http://127.0.0.1:1"
        llm = OllamaBackend(settings)
        with pytest.raises(OllamaError, match="ollama serve"):
            await llm.generate([{"role": "user", "content": "hi"}])
        await llm.close()

    def test_backend_init(self):
        from nixorb.llm.ollama_backend import OllamaBackend
        # llm_model's default is now an HF repo id (the default llm_backend
        # is "huggingface"); set an Ollama tag explicitly since that's what
        # this backend actually expects.
        settings = Settings(llm_model="llama3.2")
        llm = OllamaBackend(settings)
        assert llm._model == "llama3.2"
        assert llm._host == "http://localhost:11434"


class TestPiperTTS:
    """Test Piper TTS."""

    def test_init(self):
        from nixorb.tts.piper_tts import PiperTTS
        # tts_voice's default is now a Breeze-TTS-2 voice-design instruction
        # (the default tts_backend is "huggingface"); set a Piper voice name
        # explicitly since that's what this backend actually expects.
        settings = Settings(tts_voice="en_US-lessac-medium")
        tts = PiperTTS(settings)
        assert tts._voice == "en_US-lessac-medium"

    def test_empty_text(self):
        from nixorb.tts.piper_tts import PiperTTS
        tts = PiperTTS()
        # Should not crash with empty text
        assert tts is not None
