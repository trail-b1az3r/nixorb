"""Making Piper work with any voice, not just the one the installer fetched.

Piper ships no voice models. `install.sh` downloads exactly one —
`en_US-lessac-medium` — and `PiperTTS` only ever looked on disk, so setting
`tts_voice` to any other Piper voice found nothing and dropped silently to
espeak-ng. From outside it looked like Piper supported a single voice.

The paths here are checked against the real contents of
huggingface.co/rhasspy/piper-voices, listed from the Hub. The awkward ones
are in `REAL_VOICES` on purpose: a non-ASCII speaker (`tugão`), a speaker
starting with a digit (`25hours_single`), multi-word speakers, and the
`x_low` quality whose own underscore breaks naive splitting.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from nixorb.tts import piper_voices
from nixorb.tts.piper_voices import (
    MIN_MODEL_BYTES,
    VoiceUnavailable,
    ensure,
    looks_like_path,
    parse_voice_name,
    validate,
    voice_repo_path,
)

# voice name -> its real path in rhasspy/piper-voices.
REAL_VOICES = {
    "en_US-lessac-medium": "en/en_US/lessac/medium/en_US-lessac-medium.onnx",
    "en_GB-alan-medium": "en/en_GB/alan/medium/en_GB-alan-medium.onnx",
    "en_GB-southern_english_female-low":
        "en/en_GB/southern_english_female/low/en_GB-southern_english_female-low.onnx",
    "en_GB-northern_english_male-medium":
        "en/en_GB/northern_english_male/medium/en_GB-northern_english_male-medium.onnx",
    "ca_ES-upc_ona-x_low": "ca/ca_ES/upc_ona/x_low/ca_ES-upc_ona-x_low.onnx",
    "de_DE-thorsten_emotional-medium":
        "de/de_DE/thorsten_emotional/medium/de_DE-thorsten_emotional-medium.onnx",
    "pt_PT-tugão-medium": "pt/pt_PT/tugão/medium/pt_PT-tugão-medium.onnx",
    "vi_VN-25hours_single-low":
        "vi/vi_VN/25hours_single/low/vi_VN-25hours_single-low.onnx",
    "ja_JA-hi_fi_captain-medium":
        "ja/ja_JA/hi_fi_captain/medium/ja_JA-hi_fi_captain-medium.onnx",
    "kk_KZ-issai-high": "kk/kk_KZ/issai/high/kk_KZ-issai-high.onnx",
    "es_ES-mls_10246-low": "es/es_ES/mls_10246/low/es_ES-mls_10246-low.onnx",
    "zh_CN-huayan-x_low": "zh/zh_CN/huayan/x_low/zh_CN-huayan-x_low.onnx",
}


def write_voice(directory: Path, voice: str, size: int = MIN_MODEL_BYTES + 1):
    """A pair of files shaped like a real voice."""
    directory.mkdir(parents=True, exist_ok=True)
    model = directory / f"{voice}.onnx"
    model.write_bytes(b"\0" * size)
    model.with_suffix(".onnx.json").write_text(
        json.dumps({"audio": {"sample_rate": 22050}, "espeak": {"voice": "en-us"}})
    )
    return model


class TestVoiceNames:
    @pytest.mark.parametrize("voice,expected", sorted(REAL_VOICES.items()))
    def test_every_real_shape_maps_to_its_real_path(self, voice, expected):
        assert voice_repo_path(voice) == expected

    def test_the_language_directory_comes_from_the_locale(self):
        locale, speaker, quality = parse_voice_name("pt_BR-faber-medium")
        assert (locale, speaker, quality) == ("pt_BR", "faber", "medium")
        assert voice_repo_path("pt_BR-faber-medium").startswith("pt/pt_BR/")

    @pytest.mark.parametrize(
        "value",
        [
            "A calm, clear-voiced woman with a dry, confident wit.",
            "/home/u/voices/custom.onnx",
            "en_US-lessac",
            "lessac",
            "en_US-lessac-ultra",
            "",
            "   ",
        ],
    )
    def test_things_that_are_not_piper_voice_names(self, value):
        assert parse_voice_name(value) is None
        assert voice_repo_path(value) is None

    def test_a_hyphenated_description_is_not_mistaken_for_a_voice(self):
        # The stock tts_voice is prose with hyphens in it; splitting on "-"
        # without anchoring would have produced a nonsense download.
        assert parse_voice_name("A dry, clear-voiced woman - unhurried") is None


class TestLooksLikePath:
    @pytest.mark.parametrize(
        "value", ["/opt/voices/x.onnx", "~/v.onnx", "./v.onnx", "../v.onnx", "v.onnx"]
    )
    def test_paths(self, value):
        assert looks_like_path(value)

    @pytest.mark.parametrize("value", ["en_US-lessac-medium", "a description", ""])
    def test_not_paths(self, value):
        assert not looks_like_path(value)


class TestValidate:
    def test_a_good_pair_passes(self, tmp_path):
        validate(write_voice(tmp_path, "en_US-lessac-medium"))

    def test_an_html_error_page_is_rejected(self, tmp_path):
        # What a failed fetch actually leaves behind. Caching it would make
        # every later run fail inside Piper with nothing pointing here.
        model = tmp_path / "v.onnx"
        model.write_bytes(b"<!DOCTYPE html><html>404</html>")
        model.with_suffix(".onnx.json").write_text("{}")
        with pytest.raises(VoiceUnavailable, match="too small"):
            validate(model)

    def test_an_lfs_pointer_is_rejected(self, tmp_path):
        model = tmp_path / "v.onnx"
        model.write_text("version https://git-lfs.github.com/spec/v1\noid sha256:a\n")
        model.with_suffix(".onnx.json").write_text("{}")
        with pytest.raises(VoiceUnavailable, match="too small"):
            validate(model)

    def test_a_missing_config_is_rejected(self, tmp_path):
        model = tmp_path / "v.onnx"
        model.write_bytes(b"\0" * (MIN_MODEL_BYTES + 1))
        with pytest.raises(VoiceUnavailable, match="Piper needs both"):
            validate(model)

    def test_a_corrupt_config_is_rejected(self, tmp_path):
        model = write_voice(tmp_path, "v")
        model.with_suffix(".onnx.json").write_text("not json {{{")
        with pytest.raises(VoiceUnavailable, match="not readable JSON"):
            validate(model)

    def test_a_missing_model_is_rejected(self, tmp_path):
        with pytest.raises(VoiceUnavailable, match="was not written"):
            validate(tmp_path / "nope.onnx")


class TestEnsure:
    def test_disk_wins_and_nothing_is_downloaded(self, tmp_path, monkeypatch):
        existing = write_voice(tmp_path / "have", "en_GB-alan-medium")

        def explode(*a, **k):
            raise AssertionError("downloaded despite having it locally")

        monkeypatch.setattr(piper_voices, "download", explode)
        assert ensure(
            "en_GB-alan-medium", search=lambda v: existing, dest=tmp_path
        ) == existing

    def test_an_explicit_path_is_used_as_given(self, tmp_path):
        model = write_voice(tmp_path, "custom")
        assert ensure(str(model), search=lambda v: None, dest=tmp_path) == model

    def test_an_explicit_path_is_still_validated(self, tmp_path):
        bad = tmp_path / "bad.onnx"
        bad.write_bytes(b"tiny")
        bad.with_suffix(".onnx.json").write_text("{}")
        with pytest.raises(VoiceUnavailable):
            ensure(str(bad), search=lambda v: None, dest=tmp_path)

    def test_a_path_that_does_not_exist_says_so(self, tmp_path):
        with pytest.raises(VoiceUnavailable, match="not a file that exists"):
            ensure(str(tmp_path / "gone.onnx"), search=lambda v: None, dest=tmp_path)

    def test_downloading_can_be_turned_off(self, tmp_path):
        with pytest.raises(VoiceUnavailable, match="downloading is off"):
            ensure(
                "en_GB-alan-medium",
                search=lambda v: None,
                dest=tmp_path,
                allow_download=False,
            )

    def test_a_name_that_is_not_a_voice_explains_itself(self, tmp_path):
        with pytest.raises(VoiceUnavailable, match="not a Piper voice name") as caught:
            ensure("mumble", search=lambda v: None, dest=tmp_path)
        assert "en_US-lessac-medium" in str(caught.value)
        assert "piper-voices" in str(caught.value)


class TestDownload:
    """The hub call is mocked; the sandbox running these has no Hub access.

    What is checked here is everything around it: that the right repo file
    is asked for, that both halves are fetched, and that what lands is
    validated before being treated as a voice.
    """

    def _hub(self, monkeypatch, tmp_path, size=MIN_MODEL_BYTES + 1):
        asked = []

        def fake(repo_id, filename, local_dir, token=None, **kwargs):
            asked.append((repo_id, filename))
            # hf_hub_download replicates the repo layout under local_dir.
            out = Path(local_dir) / filename
            out.parent.mkdir(parents=True, exist_ok=True)
            if filename.endswith(".json"):
                out.write_text(json.dumps({"audio": {"sample_rate": 22050}}))
            else:
                out.write_bytes(b"\0" * size)
            return str(out)

        monkeypatch.setattr("huggingface_hub.hf_hub_download", fake)
        return asked

    def test_it_fetches_the_model_and_its_config(self, tmp_path, monkeypatch):
        asked = self._hub(monkeypatch, tmp_path)
        model = piper_voices.download("en_GB-alan-medium", tmp_path)

        assert asked == [
            ("rhasspy/piper-voices", "en/en_GB/alan/medium/en_GB-alan-medium.onnx"),
            ("rhasspy/piper-voices", "en/en_GB/alan/medium/en_GB-alan-medium.onnx.json"),
        ]
        assert model.name == "en_GB-alan-medium.onnx"
        assert model.with_suffix(".onnx.json").is_file()

    def test_the_result_lands_where_the_nested_search_looks(
        self, tmp_path, monkeypatch
    ):
        # hf_hub_download keeps the repo's directory layout, so PiperTTS
        # has to search PIPER_VOICES_DIR recursively, not just flat.
        self._hub(monkeypatch, tmp_path)
        model = piper_voices.download("en_GB-alan-medium", tmp_path)
        assert model.relative_to(tmp_path).as_posix() == (
            "en/en_GB/alan/medium/en_GB-alan-medium.onnx"
        )

    def test_a_truncated_download_is_not_accepted(self, tmp_path, monkeypatch):
        self._hub(monkeypatch, tmp_path, size=64)
        with pytest.raises(VoiceUnavailable, match="too small"):
            piper_voices.download("en_GB-alan-medium", tmp_path)

    def test_a_hub_failure_names_the_voice_and_the_file(self, tmp_path, monkeypatch):
        def boom(repo_id, filename, **kwargs):
            raise OSError("403 Forbidden")

        monkeypatch.setattr("huggingface_hub.hf_hub_download", boom)
        with pytest.raises(VoiceUnavailable) as caught:
            piper_voices.download("en_GB-alan-medium", tmp_path)
        message = str(caught.value)
        assert "en_GB-alan-medium" in message
        assert "403 Forbidden" in message


class TestPiperTTSWiring:
    """What the engine does with whatever `tts_voice` happens to hold."""

    def _engine(self, monkeypatch, tmp_path, voice, **extra):
        from nixorb.settings import Settings
        from nixorb.tts import piper_tts

        monkeypatch.setattr(piper_tts, "PIPER_VOICES_DIR", tmp_path)
        monkeypatch.setattr(
            piper_tts, "VOICE_DIRS_FLAT", (tmp_path,), raising=False
        )
        monkeypatch.setattr(
            piper_tts, "VOICE_DIRS_NESTED", (tmp_path,), raising=False
        )
        return piper_tts.PiperTTS(Settings(tts_voice=voice, **extra))

    def test_a_named_voice_is_kept(self, monkeypatch, tmp_path):
        engine = self._engine(monkeypatch, tmp_path, "en_GB-alan-medium")
        assert engine._voice == "en_GB-alan-medium"

    def test_a_path_is_kept(self, monkeypatch, tmp_path):
        engine = self._engine(monkeypatch, tmp_path, "/opt/voices/mine.onnx")
        assert engine._voice == "/opt/voices/mine.onnx"

    def test_the_stock_description_falls_back_to_a_real_voice(
        self, monkeypatch, tmp_path
    ):
        # The shipped default of tts_voice is prose for the HF voice-design
        # backend. Handing it to Piper produced silence-by-espeak on a
        # stock config, which is most of "it doesn't download the model".
        from nixorb.tts.piper_tts import DEFAULT_VOICE

        engine = self._engine(
            monkeypatch,
            tmp_path,
            "A calm, clear-voiced woman with a dry, confident wit.",
        )
        assert engine._voice == DEFAULT_VOICE

    def test_an_empty_setting_falls_back_too(self, monkeypatch, tmp_path):
        from nixorb.tts.piper_tts import DEFAULT_VOICE

        assert self._engine(monkeypatch, tmp_path, "")._voice == DEFAULT_VOICE

    def test_a_voice_on_disk_is_found_without_downloading(
        self, monkeypatch, tmp_path
    ):
        write_voice(tmp_path, "en_GB-alan-medium")
        engine = self._engine(monkeypatch, tmp_path, "en_GB-alan-medium")
        monkeypatch.setattr(
            piper_voices, "download",
            lambda *a, **k: pytest.fail("downloaded a voice already on disk"),
        )
        assert engine._resolve_voice_model().name == "en_GB-alan-medium.onnx"

    def test_a_download_happens_once_not_per_sentence(self, monkeypatch, tmp_path):
        calls = []

        def fake_download(voice, dest, token=None):
            calls.append(voice)
            return write_voice(dest, voice)

        engine = self._engine(monkeypatch, tmp_path, "en_GB-alan-medium")
        monkeypatch.setattr(piper_voices, "download", fake_download)

        for _ in range(5):
            assert engine._resolve_voice_model() is not None
        assert calls == ["en_GB-alan-medium"], "a 60 MB fetch per sentence"

    def test_a_failure_is_reported_once_and_then_left_alone(
        self, monkeypatch, tmp_path, caplog
    ):
        attempts = []

        def always_fails(voice, dest, token=None):
            attempts.append(voice)
            raise VoiceUnavailable("no network")

        engine = self._engine(monkeypatch, tmp_path, "en_GB-alan-medium")
        monkeypatch.setattr(piper_voices, "download", always_fails)

        with caplog.at_level("WARNING"):
            for _ in range(4):
                assert engine._resolve_voice_model() is None

        assert attempts == ["en_GB-alan-medium"]
        warnings = [r for r in caplog.records if "no network" in r.getMessage()]
        assert len(warnings) == 1, "the same failure was logged per sentence"

    def test_downloading_can_be_disabled_in_settings(self, monkeypatch, tmp_path):
        engine = self._engine(
            monkeypatch, tmp_path, "en_GB-alan-medium", tts_download_voices=False
        )
        monkeypatch.setattr(
            piper_voices, "download",
            lambda *a, **k: pytest.fail("downloaded with downloads turned off"),
        )
        assert engine._resolve_voice_model() is None
