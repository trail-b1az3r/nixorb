"""Which commands are eligible to run, and which are only being weighed up.

A reasoning model rehearses inside `<think>`: it writes a command out,
considers it, and often decides against it. Extraction ran over the raw
response, so a command the model had explicitly rejected was still found
and executed. Commands now come from the answer only.
"""
from __future__ import annotations

import pytest

from nixorb.action.executor import ActionExecutor

# Unbound: ActionExecutor refuses to construct as root, and these are pure.
extract = ActionExecutor._extract_actions


class TestExtraction:
    def test_a_command_in_the_answer_runs(self):
        assert extract(None, "Doing it.<ACTION>uptime</ACTION>") == ["uptime"]

    def test_several_commands_all_run(self):
        text = "<ACTION>ls</ACTION> then <ACTION>pwd</ACTION>"
        assert extract(None, text) == ["ls", "pwd"]

    def test_no_commands_is_empty(self):
        assert extract(None, "Just chatting.") == []

    def test_whitespace_is_trimmed(self):
        assert extract(None, "<ACTION>\n  ls -l \n</ACTION>") == ["ls -l"]

    def test_an_empty_block_is_ignored(self):
        assert extract(None, "<ACTION>   </ACTION>") == []

    def test_the_tag_is_case_insensitive(self):
        assert extract(None, "<action>ls</action>") == ["ls"]


class TestReasoningIsNotExecuted:
    def test_a_command_the_model_rejected_does_not_run(self):
        reply = (
            "<think>I could run <ACTION>rm -rf /</ACTION> but that would wipe "
            "the disk, so no.</think>"
            "Listing your files.<ACTION>ls -l</ACTION>"
        )
        assert extract(None, reply) == ["ls -l"]

    def test_thinking_alone_yields_nothing_to_run(self):
        reply = "<think>Maybe <ACTION>shutdown now</ACTION>? Better not.</think>No."
        assert extract(None, reply) == []

    def test_it_is_logged_when_one_is_ignored(self, caplog):
        reply = "<think><ACTION>dangerous</ACTION></think>Fine.<ACTION>ls</ACTION>"
        with caplog.at_level("INFO"):
            assert extract(None, reply) == ["ls"]
        assert any(
            "only weighed up" in r.getMessage() for r in caplog.records
        )

    def test_an_unterminated_thought_runs_nothing(self):
        # Generation cut off mid-thought must not leave a live command.
        reply = "Sure.<think>first I would <ACTION>rm -rf ~</ACTION>"
        assert extract(None, reply) == []

    @pytest.mark.parametrize("opener,closer", [
        ("<think>", "</think>"),
        ("<reasoning>", "</reasoning>"),
        ("<thought>", "</thought>"),
    ])
    def test_every_reasoning_tag_shields_its_contents(self, opener, closer):
        reply = f"{opener}<ACTION>bad</ACTION>{closer}<ACTION>good</ACTION>"
        assert extract(None, reply) == ["good"]

    def test_a_normal_reply_is_unaffected(self, caplog):
        with caplog.at_level("INFO"):
            assert extract(None, "Sure.<ACTION>date</ACTION>") == ["date"]
        assert not [
            r for r in caplog.records if "only weighed up" in r.getMessage()
        ]
