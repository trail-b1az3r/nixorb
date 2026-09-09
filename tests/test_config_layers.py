"""Why changing the config appeared to do nothing.

Four separate reasons, all reported as one symptom:

  * `config/default.toml` is shipped as package data and documented as the
    defaults, and no code ever read it. Editing it did nothing, ever.
  * A mistyped key name was dropped by pydantic without a word, so
    `tts_backendd` looked identical to a setting that has no effect.
  * One badly-typed value failed validation for the whole file, so every
    other setting in it silently reverted to its default.
  * `nixorb config` opened a hardcoded path rather than the one that gets
    loaded, so with NIXORB_CONFIG set it edited a file nothing reads.
"""
from __future__ import annotations

import importlib

import pytest

import nixorb.settings as settings_module
from nixorb.settings import Settings


@pytest.fixture
def config(tmp_path, monkeypatch):
    """Point NixOrb at a throwaway user config and packaged default."""
    user = tmp_path / "config.toml"
    packaged = tmp_path / "default.toml"
    monkeypatch.setenv("NIXORB_CONFIG", str(user))
    monkeypatch.setenv("NIXORB_DEFAULT_CONFIG", str(packaged))

    def load(user_text: str | None = None, packaged_text: str | None = None):
        if packaged_text is None:
            packaged.unlink(missing_ok=True)
        else:
            packaged.write_text(packaged_text)
        if user_text is None:
            user.unlink(missing_ok=True)
        else:
            user.write_text(user_text)
        return Settings.load()

    load.user = user
    load.packaged = packaged
    return load


class TestUserConfigApplies:
    def test_a_setting_in_the_user_config_takes_effect(self, config):
        assert config('tts_speed = 1.7\n').tts_speed == 1.7

    def test_several_settings_all_take_effect(self, config):
        loaded = config(
            'tts_speed = 1.7\nllm_model = "my-model"\norb_size = 120\n'
        )
        assert (loaded.tts_speed, loaded.llm_model, loaded.orb_size) == (
            1.7, "my-model", 120,
        )

    def test_no_config_at_all_is_just_defaults(self, config):
        assert config().orb_size == Settings().orb_size


class TestPackagedDefaults:
    def test_the_shipped_default_toml_is_actually_read(self, config):
        # It is installed as shared data and documented as the defaults.
        # Nothing read it, so editing it did nothing at all.
        assert config(packaged_text='llm_model = "from-defaults"\n').llm_model == (
            "from-defaults"
        )

    def test_the_user_config_wins_over_it(self, config):
        loaded = config(
            user_text='tts_speed = 2.0\n',
            packaged_text='tts_speed = 1.0\nllm_model = "from-defaults"\n',
        )
        assert loaded.tts_speed == 2.0
        assert loaded.llm_model == "from-defaults"

    def test_a_missing_packaged_file_is_not_an_error(self, config):
        assert config('tts_speed = 1.5\n').tts_speed == 1.5

    def test_the_real_shipped_file_is_found_in_a_checkout(self, monkeypatch):
        # A source checkout must resolve config/default.toml, or developers
        # edit it and see nothing happen — the original complaint.
        monkeypatch.delenv("NIXORB_DEFAULT_CONFIG", raising=False)
        importlib.reload(settings_module)
        found = settings_module.packaged_defaults()
        assert found is not None and found.name == "default.toml"

    def test_the_shipped_file_is_entirely_valid(self, monkeypatch):
        # Every key in it must be a real setting; an unknown one there
        # would warn on every single start.
        monkeypatch.delenv("NIXORB_DEFAULT_CONFIG", raising=False)
        importlib.reload(settings_module)
        path = settings_module.packaged_defaults()
        assert path is not None
        data = settings_module._read_toml(path)
        assert data, "default.toml is empty or unreadable"
        assert not (set(data) - set(Settings.model_fields))
        assert settings_module._drop_invalid(Settings, data, str(path)) == data


class TestOneBadKeyCostsOneKey:
    def test_the_rest_of_the_file_still_applies(self, config, caplog):
        with caplog.at_level("ERROR"):
            loaded = config('orb_size = "big"\ntts_speed = 1.7\nllm_model = "mine"\n')
        assert loaded.tts_speed == 1.7
        assert loaded.llm_model == "mine"
        assert loaded.orb_size == Settings().orb_size
        assert any("orb_size" in r.getMessage() for r in caplog.records)

    def test_several_bad_keys_are_each_named(self, config, caplog):
        with caplog.at_level("ERROR"):
            loaded = config(
                'orb_size = "big"\ntts_speed = "quick"\nllm_model = "mine"\n'
            )
        assert loaded.llm_model == "mine"
        logged = " ".join(r.getMessage() for r in caplog.records)
        assert "orb_size" in logged and "tts_speed" in logged

    def test_a_bad_value_in_the_packaged_file_does_not_kill_the_user_config(
        self, config
    ):
        loaded = config(
            user_text='llm_model = "mine"\n', packaged_text='orb_size = "big"\n'
        )
        assert loaded.llm_model == "mine"

    def test_unparseable_toml_is_reported_not_swallowed(self, config, caplog):
        with caplog.at_level("ERROR"):
            loaded = config("this is not = valid toml [[[\n")
        assert loaded.orb_size == Settings().orb_size
        assert any("could not be read" in r.getMessage() for r in caplog.records)


class TestUnknownKeys:
    def test_a_typo_is_named_rather_than_ignored(self, config, caplog):
        with caplog.at_level("WARNING"):
            config('tts_backendd = "espeak"\n')
        messages = " ".join(r.getMessage() for r in caplog.records)
        assert "tts_backendd" in messages
        assert "has no effect" in messages

    def test_a_near_miss_gets_a_suggestion(self, config, caplog):
        with caplog.at_level("WARNING"):
            config('tts_backendd = "espeak"\n')
        assert "did you mean 'tts_backend'" in " ".join(
            r.getMessage() for r in caplog.records
        )

    def test_valid_keys_produce_no_warning(self, config, caplog):
        with caplog.at_level("WARNING"):
            config('tts_backend = "espeak"\ntts_speed = 1.2\n')
        assert not [
            r for r in caplog.records if "not a NixOrb setting" in r.getMessage()
        ]

    def test_an_unknown_key_does_not_lose_the_valid_ones(self, config):
        loaded = config('tts_backendd = "espeak"\ntts_speed = 1.9\n')
        assert loaded.tts_speed == 1.9


class TestConfigPathIsSingleSourced:
    def test_the_env_var_is_honoured(self, tmp_path, monkeypatch):
        target = tmp_path / "elsewhere.toml"
        monkeypatch.setenv("NIXORB_CONFIG", str(target))
        importlib.reload(settings_module)
        assert settings_module.config_path() == target

    def test_the_cli_edits_the_file_that_is_loaded(self, tmp_path, monkeypatch):
        # `nixorb config` used to open a hardcoded ~/.config path, so with
        # NIXORB_CONFIG set every edit went to a file nothing reads.
        import inspect

        from nixorb import cli

        source = inspect.getsource(cli.config)
        assert "config_path()" in source
        assert '".config" / "nixorb"' not in source

    def test_save_and_load_round_trip(self, tmp_path, monkeypatch):
        monkeypatch.setenv("NIXORB_CONFIG", str(tmp_path / "c.toml"))
        monkeypatch.delenv("NIXORB_DEFAULT_CONFIG", raising=False)
        importlib.reload(settings_module)
        original = settings_module.Settings(tts_speed=1.33, llm_model="round-trip")
        original.save()
        assert settings_module.Settings.load().tts_speed == 1.33
        assert settings_module.Settings.load().llm_model == "round-trip"
