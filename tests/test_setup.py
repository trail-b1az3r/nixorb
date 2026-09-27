"""Choosing settings this machine can actually run.

Every failure this project produced in the field was the same shape: a
default pointing at a model the machine could not load. A Nemotron
checkpoint needing transformers 5.13 and a numpy numba would accept; a
GGUF needing a newer llama.cpp than the wheel; a Hugging Face voice
needing torch that was never installed. NixOrb started fine every time
and then had nothing to say.

`nixorb setup` looks first and prefers the boring option.
"""
from __future__ import annotations

import pytest

from nixorb.machine import Machine
from nixorb.settings import Settings
from nixorb.setup_wizard import (
    SAFE_GGUF_REPO,
    Plan,
    apply_plan,
    build_plan,
    choose_asr,
    choose_llm,
    choose_tts,
    whisper_size_for,
)


def machine(**kwargs) -> Machine:
    """A machine with nothing installed, then whatever you name."""
    return Machine(python="3.12.3", ram_gb=16.0, **kwargs)


def plan_for(chooser, m: Machine, settings=None) -> Plan:
    plan = Plan(machine=m)
    if chooser is choose_llm:
        chooser(m, plan, settings)
    else:
        chooser(m, plan)
    return plan


def value(plan: Plan, key: str):
    for choice in plan.choices:
        if choice.key == key:
            return choice.value
    return None


class TestWhisperSizing:
    @pytest.mark.parametrize(
        "ram,expected",
        [(32.0, "small"), (16.0, "small"), (12.0, "base"), (8.0, "base"),
         (4.0, "tiny"), (0.0, "tiny")],
    )
    def test_size_matches_memory(self, ram, expected):
        assert whisper_size_for(ram) == expected


class TestNemotronGating:
    """The exact stack that failed in the field, checked before choosing."""

    def test_a_complete_stack_gets_nemotron(self):
        m = machine(transformers="5.16.1", torch="2.7.1", numpy="2.4.6")
        assert m.can_run_nemotron == (True, "")
        assert value(plan_for(choose_asr, m), "asr_backend") == "nemotron"

    def test_numpy_too_new_for_numba_rules_it_out(self):
        # This is what actually happened: numba trails numpy by a release
        # and the Nemotron stack pulls numba in.
        m = machine(transformers="5.16.1", torch="2.7.1", numpy="2.5.0",
                    faster_whisper="1.2.1")
        ok, why = m.can_run_nemotron
        assert not ok
        assert "numpy 2.5.0" in why and "numba" in why
        assert value(plan_for(choose_asr, m), "asr_backend") == "faster_whisper"

    def test_transformers_too_old_rules_it_out(self):
        m = machine(transformers="4.44.0", torch="2.7.1", faster_whisper="1.2.1")
        ok, why = m.can_run_nemotron
        assert not ok and "4.44.0" in why
        assert value(plan_for(choose_asr, m), "asr_backend") == "faster_whisper"

    def test_no_torch_rules_it_out(self):
        m = machine(transformers="5.16.1", faster_whisper="1.2.1")
        assert m.can_run_nemotron == (False, "torch is not installed")

    def test_the_reason_reaches_the_user(self):
        m = machine(transformers="4.0.0", torch="2.7.1", faster_whisper="1.2.1")
        why = plan_for(choose_asr, m).choices[0].why
        assert "Nemotron is not usable here" in why


class TestASRChoice:
    def test_vosk_when_it_is_the_only_one(self, monkeypatch):
        monkeypatch.setattr(
            "nixorb.setup_wizard._installed", lambda name: name == "vosk"
        )
        plan = plan_for(choose_asr, machine())
        assert value(plan, "asr_backend") == "vosk"
        assert value(plan, "asr_model") == "en-us"

    def test_nothing_installed_says_what_to_install(self, monkeypatch):
        monkeypatch.setattr("nixorb.setup_wizard._installed", lambda name: False)
        plan = plan_for(choose_asr, machine())
        assert plan.warnings
        assert "faster-whisper" in " ".join(plan.warnings)
        # Still writes a usable value rather than leaving it unset.
        assert value(plan, "asr_backend")


class TestLLMChoice:
    def test_a_running_ollama_wins(self):
        m = machine(ollama_reachable=True, llama_cpp="0.3.0")
        assert value(plan_for(choose_llm, m), "llm_backend") == "ollama"

    def test_gguf_when_llama_cpp_is_there(self):
        m = machine(llama_cpp="0.3.0")
        plan = plan_for(choose_llm, m)
        assert value(plan, "llm_backend") == "huggingface"
        assert value(plan, "llm_model") == SAFE_GGUF_REPO

    def test_t1_configured_means_auto(self):
        plan = plan_for(
            choose_llm, machine(), Settings(t1_base_url="https://t1.example")
        )
        assert value(plan, "llm_backend") == "auto"

    def test_ollama_installed_but_stopped_says_so(self):
        m = machine(ollama_binary=True)
        plan = plan_for(choose_llm, m)
        assert value(plan, "llm_backend") == "ollama"
        assert "ollama serve" in " ".join(plan.warnings)

    def test_nothing_available_explains_the_quickest_fix(self):
        plan = plan_for(choose_llm, machine())
        assert "ollama" in " ".join(plan.warnings).lower()
        assert "t1_base_url" in " ".join(plan.warnings)

    def test_tight_memory_is_flagged(self):
        m = Machine(ram_gb=4.0, llama_cpp="0.3.0")
        assert "tight" in " ".join(plan_for(choose_llm, m).warnings)


class TestTTSChoice:
    def test_kokoro_wins_when_installed(self, monkeypatch):
        # The best voice a plain machine can reach: no torch at all.
        monkeypatch.setattr(
            "nixorb.setup_wizard._installed", lambda name: name == "kokoro_onnx"
        )
        plan = plan_for(choose_tts, machine(espeak=True))
        assert value(plan, "tts_backend") == "kokoro"
        assert value(plan, "tts_voice") == "af_heart"

    def test_piper_when_its_binary_is_there(self, monkeypatch):
        monkeypatch.setattr("nixorb.setup_wizard._installed", lambda name: False)
        plan = plan_for(choose_tts, machine(piper_binary="/usr/bin/piper-tts"))
        assert value(plan, "tts_backend") == "piper"
        assert value(plan, "tts_voice") == "en_US-lessac-medium"

    def test_huggingface_when_the_stack_is_complete(self, monkeypatch):
        monkeypatch.setattr("nixorb.setup_wizard._installed", lambda name: False)
        m = machine(transformers="5.16.1", torch="2.7.1")
        assert value(plan_for(choose_tts, m), "tts_backend") == "huggingface"

    def test_espeak_last_and_says_how_to_do_better(self, monkeypatch):
        monkeypatch.setattr("nixorb.setup_wizard._installed", lambda name: False)
        plan = plan_for(choose_tts, machine(espeak=True))
        assert value(plan, "tts_backend") == "espeak"
        assert "kokoro" in " ".join(plan.warnings).lower()

    def test_nothing_at_all_warns_about_silence(self, monkeypatch):
        monkeypatch.setattr("nixorb.setup_wizard._installed", lambda name: False)
        plan = plan_for(choose_tts, machine())
        assert "silent" in " ".join(plan.warnings)


class TestPlan:
    def test_every_choice_carries_a_reason(self, monkeypatch):
        monkeypatch.setattr("nixorb.setup_wizard._installed", lambda name: False)
        plan = build_plan(Settings(), machine(faster_whisper="1.2.1"))
        assert plan.choices
        for choice in plan.choices:
            assert choice.why, f"{choice.key} was chosen without a reason"

    def test_the_plan_applies_onto_settings(self, monkeypatch):
        monkeypatch.setattr("nixorb.setup_wizard._installed", lambda name: False)
        plan = build_plan(Settings(), machine(faster_whisper="1.2.1"))
        updated = apply_plan(plan, Settings())
        assert updated.asr_backend == value(plan, "asr_backend")

    def test_applying_does_not_mutate_the_original(self, monkeypatch):
        monkeypatch.setattr("nixorb.setup_wizard._installed", lambda name: False)
        original = Settings()
        before = original.asr_backend
        apply_plan(build_plan(original, machine(faster_whisper="1.2.1")), original)
        assert original.asr_backend == before

    def test_machine_notes_reach_the_warnings(self):
        m = machine(faster_whisper="1.2.1")
        m.notes.append("something worth knowing")
        assert "something worth knowing" in build_plan(Settings(), m).warnings

    def test_every_key_is_a_real_setting(self, monkeypatch):
        # A plan naming a key Settings does not have would be silently
        # dropped on load — the exact bug 2.0.15 fixed.
        monkeypatch.setattr("nixorb.setup_wizard._installed", lambda name: False)
        for m in (
            machine(faster_whisper="1.2.1"),
            machine(piper_binary="/usr/bin/piper-tts", ollama_reachable=True),
            machine(transformers="5.16.1", torch="2.7.1", numpy="2.4.0",
                    llama_cpp="0.3.0"),
        ):
            for choice in build_plan(Settings(), m).choices:
                assert choice.key in Settings.model_fields, choice.key


class TestProbe:
    def test_it_never_raises_on_this_machine(self):
        from nixorb.machine import probe

        report = probe(deep=False)
        assert report.python
        assert isinstance(report.notes, list)

    def test_a_gpu_torch_cannot_use_is_called_out(self):
        from nixorb.machine import probe

        report = probe(deep=False)
        # Only meaningful when there is a GPU; the check is that probing
        # with deep=False does not invent one.
        assert report.torch_cuda is False

    def test_cuda_false_skips_the_torch_subprocess(self, monkeypatch):
        # Importing torch to ask about CUDA costs seconds on a cold cache,
        # and the answer only ever feeds a note. The first start skips it.
        import nixorb.machine as machine_module

        def explode():
            raise AssertionError("torch should not have been imported")

        monkeypatch.setattr(machine_module, "_torch_cuda", explode)
        monkeypatch.setattr(machine_module, "_ollama_reachable", lambda host: True)
        report = machine_module.probe(deep=True, cuda=False)
        assert report.ollama_reachable is True

    def test_skipping_the_cuda_check_does_not_claim_a_cpu_only_build(
        self, monkeypatch
    ):
        # torch_cuda is False both for a CPU-only torch and for a check
        # that never ran; only the first deserves the note.
        import nixorb.machine as machine_module

        monkeypatch.setattr(
            machine_module, "_read_gpu", lambda: ("GeForce RTX 4090", 24.0)
        )
        monkeypatch.setattr(
            machine_module, "_version_of",
            lambda name: "2.6.0" if name == "torch" else "",
        )
        monkeypatch.setattr(machine_module, "_ollama_reachable", lambda host: False)

        quiet = machine_module.probe(deep=True, cuda=False)
        assert not any("cannot use it" in note for note in quiet.notes)

        monkeypatch.setattr(machine_module, "_torch_cuda", lambda: False)
        loud = machine_module.probe(deep=True, cuda=True)
        assert any("cannot use it" in note for note in loud.notes)


class TestFirstRun:
    """The first start must not be the one that fails.

    Someone who pip-installs NixOrb and runs it never sees `nixorb setup`.
    Left alone they get the shipped defaults, which aim at a stack they may
    not have, and the orb starts, listens, and says nothing.
    """

    @pytest.fixture
    def home(self, tmp_path, monkeypatch):
        config = tmp_path / "config.toml"
        monkeypatch.setenv("NIXORB_CONFIG", str(config))
        monkeypatch.delenv("NIXORB_DEFAULT_CONFIG", raising=False)
        # Nothing installed, so the choices are predictable.
        monkeypatch.setattr(
            "nixorb.setup_wizard.probe",
            lambda **kwargs: machine(faster_whisper="1.1.0", espeak=True),
        )
        monkeypatch.setattr("nixorb.setup_wizard._installed", lambda name: False)
        return config

    def test_it_writes_a_config_when_there_is_none(self, home):
        from nixorb.setup_wizard import configure_on_first_run

        assert not home.exists()
        configured = configure_on_first_run(Settings())
        assert home.exists(), "the first start should leave a config behind"
        assert configured.asr_backend == "faster_whisper"

    def test_the_returned_settings_are_the_chosen_ones(self, home):
        from nixorb.setup_wizard import configure_on_first_run

        # Not just written for next time — this start has to use them.
        configured = configure_on_first_run(Settings(asr_backend="nemotron"))
        assert configured.asr_backend == "faster_whisper"
        assert configured.tts_backend == "espeak"

    def test_what_it_wrote_is_what_loads_next_time(self, home):
        from nixorb.setup_wizard import configure_on_first_run

        configure_on_first_run(Settings())
        assert Settings.load().asr_backend == "faster_whisper"

    def test_an_existing_config_is_never_touched(self, home):
        from nixorb.setup_wizard import configure_on_first_run

        home.write_text('asr_backend = "nemotron"\n')
        before = home.read_text()
        configured = configure_on_first_run(Settings.load())
        assert home.read_text() == before
        assert configured.asr_backend == "nemotron"

    def test_a_config_that_cannot_be_written_still_starts(
        self, home, monkeypatch, caplog
    ):
        from nixorb.setup_wizard import configure_on_first_run

        def refuse(self):
            raise OSError("read-only file system")

        monkeypatch.setattr(Settings, "save", refuse)
        with caplog.at_level("WARNING"):
            configured = configure_on_first_run(Settings())
        # The choices still apply to this start, which is the point.
        assert configured.asr_backend == "faster_whisper"
        assert any("read-only" in r.getMessage() for r in caplog.records)
