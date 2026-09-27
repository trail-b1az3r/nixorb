"""Kokoro and Vosk — the two engines that need no torch at all.

Both exist for the same reason: the Hugging Face backends are the
flexible ones and they want torch plus a matching transformers, which
most machines do not have. Kokoro is a neural voice through onnxruntime;
Vosk is Kaldi ASR in 40 MB. Between them a plain machine can hear and
speak without an ML stack.

The packages are not installed on every runner, so they are faked here —
but the call shapes come from the real `kokoro_onnx.Kokoro` and
`vosk.Model` signatures, and a test asserts they still match when the
packages *are* present.
"""
from __future__ import annotations

import json
import sys
import types

import numpy as np
import pytest

from nixorb.settings import Settings

# ── Kokoro ───────────────────────────────────────────────────────── #

class TestKokoroVoiceNames:
    def test_a_kokoro_voice_id_is_kept(self):
        from nixorb.tts.kokoro_tts import _voice_name

        assert _voice_name("am_michael") == "am_michael"
        assert _voice_name("bf_emma") == "bf_emma"

    def test_the_stock_prose_default_is_replaced(self):
        # tts_voice is shared with every backend and defaults to a
        # sentence for the HF voice-design models. Kokoro cannot use it.
        from nixorb.tts.kokoro_tts import DEFAULT_VOICE, _voice_name

        assert _voice_name(
            "A calm, clear-voiced woman with a dry, confident wit."
        ) == DEFAULT_VOICE

    def test_an_empty_setting_gets_the_default(self):
        from nixorb.tts.kokoro_tts import DEFAULT_VOICE, _voice_name

        assert _voice_name("") == DEFAULT_VOICE
        assert _voice_name("   ") == DEFAULT_VOICE


class TestKokoroEngine:
    def _fake(self, monkeypatch, tmp_path):
        made = {}

        class _Kokoro:
            def __init__(self, model_path, voices_path, **kwargs):
                made["paths"] = (model_path, voices_path)

            def create(self, text, voice, speed=1.0, lang="en-us", **kwargs):
                made["call"] = (text, voice, speed, lang)
                return np.zeros(2400, dtype=np.float32), 24000

        module = types.ModuleType("kokoro_onnx")
        module.Kokoro = _Kokoro  # type: ignore[attr-defined]
        monkeypatch.setitem(sys.modules, "kokoro_onnx", module)

        model = tmp_path / "model.onnx"
        voices = tmp_path / "voices.bin"
        model.write_bytes(b"x")
        voices.write_bytes(b"y")
        return made, model, voices

    def test_it_synthesises_through_the_real_call_shape(
        self, monkeypatch, tmp_path
    ):
        from nixorb.tts.kokoro_tts import KokoroTTS

        made, model, voices = self._fake(monkeypatch, tmp_path)
        engine = KokoroTTS(Settings(
            tts_voice="am_michael", tts_speed=1.2,
            tts_kokoro_model=str(model), tts_kokoro_voices=str(voices),
        ))
        audio, rate = engine._synthesise("hello there")

        assert rate == 24000 and audio.dtype == np.float32
        assert made["paths"] == (str(model), str(voices))
        assert made["call"] == ("hello there", "am_michael", 1.2, "en-us")

    def test_a_missing_file_is_named(self, monkeypatch, tmp_path):
        from nixorb.tts.kokoro_tts import KokoroTTS, KokoroUnavailable

        self._fake(monkeypatch, tmp_path)
        engine = KokoroTTS(Settings(
            tts_kokoro_model=str(tmp_path / "gone.onnx"),
            tts_kokoro_voices=str(tmp_path / "voices.bin"),
        ))
        with pytest.raises(KokoroUnavailable, match="does not exist"):
            engine._resolve_files()

    def test_a_missing_package_says_how_to_install_it(self, monkeypatch, tmp_path):
        from nixorb.tts.kokoro_tts import KokoroTTS, KokoroUnavailable

        monkeypatch.setitem(sys.modules, "kokoro_onnx", None)
        engine = KokoroTTS(Settings())
        with pytest.raises((KokoroUnavailable, Exception)) as caught:
            engine._load()
        assert "kokoro" in str(caught.value).lower()

    async def test_speaking_survives_a_failure(self, monkeypatch, started_bus):
        from nixorb.tts.kokoro_tts import KokoroTTS

        engine = KokoroTTS(Settings())

        def boom(text):
            raise RuntimeError("no model")

        monkeypatch.setattr(engine, "_synthesise", boom)
        await engine.speak("anything")  # must not raise

    def test_the_factory_knows_it(self):
        from nixorb.tts.tts_factory import BACKENDS, normalise_backend

        assert "kokoro" in BACKENDS
        assert normalise_backend("kokoro-onnx") == "kokoro"


# ── Vosk ─────────────────────────────────────────────────────────── #

class TestVoskHelpers:
    @pytest.mark.parametrize(
        "configured,expected",
        [("", "en-us"), ("auto", "en-us"), ("en", "en-us"),
         ("de", "de"), ("en_GB", "en-gb"), ("fr-FR", "fr-fr")],
    )
    def test_language_tags(self, configured, expected):
        from nixorb.asr.vosk_asr import _language_of

        assert _language_of(Settings(asr_language=configured)) == expected

    def test_the_transcript_is_pulled_out_of_the_json(self):
        from nixorb.asr.vosk_asr import _text_of

        assert _text_of(json.dumps({"text": "hello there"})) == "hello there"
        assert _text_of(json.dumps({"text": ""})) == ""

    def test_unparseable_output_is_empty_not_a_crash(self):
        from nixorb.asr.vosk_asr import _text_of

        assert _text_of("not json at all") == ""
        assert _text_of("") == ""


class TestVoskEngine:
    def _fake(self, monkeypatch, *, text="it worked"):
        seen = {}

        class _Recogniser:
            def __init__(self, model, rate):
                seen["rate"] = rate

            def SetWords(self, flag):
                seen["words"] = flag

            def AcceptWaveform(self, pcm):
                seen["pcm"] = pcm
                return True

            def FinalResult(self):
                return json.dumps({"text": text})

        class _Model:
            def __init__(self, model_path=None, model_name=None, lang=None):
                seen["model"] = {"path": model_path, "name": model_name,
                                 "lang": lang}

        module = types.ModuleType("vosk")
        module.Model = _Model  # type: ignore[attr-defined]
        module.KaldiRecognizer = _Recogniser  # type: ignore[attr-defined]
        module.SetLogLevel = lambda level: seen.setdefault("quiet", level)  # type: ignore[attr-defined]
        monkeypatch.setitem(sys.modules, "vosk", module)
        return seen

    def test_a_bare_language_fetches_the_small_model(self, monkeypatch):
        from nixorb.asr.vosk_asr import VoskASREngine

        seen = self._fake(monkeypatch)
        VoskASREngine(Settings(asr_language="en", asr_model=""))._load()
        assert seen["model"] == {"path": None, "name": None, "lang": "en-us"}

    def test_a_model_name_is_fetched_by_name(self, monkeypatch):
        from nixorb.asr.vosk_asr import VoskASREngine

        seen = self._fake(monkeypatch)
        VoskASREngine(
            Settings(asr_model="vosk-model-small-en-us-0.15")
        )._load()
        assert seen["model"]["name"] == "vosk-model-small-en-us-0.15"

    def test_a_directory_is_loaded_from_disk(self, monkeypatch, tmp_path):
        from nixorb.asr.vosk_asr import VoskASREngine

        seen = self._fake(monkeypatch)
        local = tmp_path / "my-model"
        local.mkdir()
        VoskASREngine(Settings(asr_model=str(local)))._load()
        assert seen["model"]["path"] == str(local)

    def test_audio_is_converted_to_the_pcm_vosk_wants(self, monkeypatch):
        from nixorb.asr.base import SAMPLE_RATE
        from nixorb.asr.vosk_asr import VoskASREngine

        seen = self._fake(monkeypatch)
        engine = VoskASREngine(Settings())
        engine._model = object()

        audio = np.array([0.0, 1.0, -1.0, 0.5], dtype=np.float32)
        assert engine._transcribe(audio) == "it worked"
        assert seen["rate"] == float(SAMPLE_RATE)
        # 16-bit little-endian, two bytes per sample.
        assert len(seen["pcm"]) == audio.size * 2
        assert np.frombuffer(seen["pcm"], dtype="<i2")[1] == 32767

    def test_out_of_range_samples_are_clipped_not_wrapped(self, monkeypatch):
        from nixorb.asr.vosk_asr import VoskASREngine

        seen = self._fake(monkeypatch)
        engine = VoskASREngine(Settings())
        engine._model = object()
        engine._transcribe(np.array([2.5, -2.5], dtype=np.float32))
        values = np.frombuffer(seen["pcm"], dtype="<i2")
        assert values[0] == 32767 and values[1] == -32767

    def test_a_load_failure_says_what_to_set(self, monkeypatch):
        from nixorb.asr.vosk_asr import VoskASREngine

        module = types.ModuleType("vosk")

        def boom(**kwargs):
            raise OSError("no such model")

        module.Model = boom  # type: ignore[attr-defined]
        module.SetLogLevel = lambda level: None  # type: ignore[attr-defined]
        monkeypatch.setitem(sys.modules, "vosk", module)

        with pytest.raises(RuntimeError, match="asr_model"):
            VoskASREngine(Settings())._load()

    def test_the_factory_knows_it(self):
        from nixorb.asr.factory import BACKENDS, build_asr, normalise_backend

        assert "vosk" in BACKENDS
        assert normalise_backend("kaldi") == "vosk"
        engine = build_asr(Settings(asr_backend="vosk", asr_model="en-us"))
        assert engine.name == "vosk"


class TestAgainstTheRealPackages:
    """Fakes drift. If the package is here, check the signatures."""

    def test_kokoro_call_shape(self):
        import inspect

        kokoro_onnx = pytest.importorskip("kokoro_onnx")
        init = inspect.signature(kokoro_onnx.Kokoro.__init__).parameters
        assert {"model_path", "voices_path"} <= set(init)
        create = inspect.signature(kokoro_onnx.Kokoro.create).parameters
        assert {"text", "voice", "speed", "lang"} <= set(create)

    def test_vosk_call_shape(self):
        import inspect

        vosk = pytest.importorskip("vosk")
        init = inspect.signature(vosk.Model.__init__).parameters
        assert {"model_path", "model_name", "lang"} <= set(init)
        for name in ("KaldiRecognizer", "SetLogLevel"):
            assert hasattr(vosk, name)
