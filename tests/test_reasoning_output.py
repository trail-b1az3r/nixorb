"""Why the orb read its own thinking aloud, run together, with no answer.

Three faults produced one experience:

  * `_extract_tool_calls` ended in `.strip()` and ran on every streamed
    token. llama.cpp emits tokens with the leading space attached, so
    stripping each one gave "Iamthinkingaboutthis", and a token that was
    only a space became "" and was dropped at the `if cleaned:` guard.
  * Nothing stripped `<think>`. NixOrb held back `<ACTION>` blocks and
    nothing else, so a reasoning model's private working was spoken.
  * `llm_max_tokens` was 512 — less than a `<think>` block, so generation
    stopped before the answer began.
"""
from __future__ import annotations

import pytest

from nixorb.llm.reasoning import (
    ACTION_TAGS,
    REASONING_TAGS,
    TagSuppressor,
    strip_reasoning,
    strip_tags,
    suppressors,
)
from nixorb.settings import Settings


class TestStreamedSpacing:
    def test_tokens_keep_their_spaces(self):
        from nixorb.llm.hf_llm_backend import HuggingFaceLLMBackend

        backend = HuggingFaceLLMBackend(Settings())
        tokens = ["I", " am", " thinking", ".", " ", "Answer", ":", " 42"]
        out = "".join(
            cleaned
            for cleaned, _ in (backend._extract_tool_calls(t) for t in tokens)
            if cleaned
        )
        assert out == "I am thinking. Answer: 42"

    def test_a_whitespace_only_token_is_not_dropped(self):
        from nixorb.llm.hf_llm_backend import HuggingFaceLLMBackend

        cleaned, _ = HuggingFaceLLMBackend(Settings())._extract_tool_calls(" ")
        assert cleaned == " "

    def test_tool_calls_are_still_removed(self):
        from nixorb.llm.hf_llm_backend import HuggingFaceLLMBackend

        backend = HuggingFaceLLMBackend(Settings())
        text = 'Sure. <tool_call>{"name": "x", "arguments": {}}</tool_call> Done.'
        cleaned, calls = backend._extract_tool_calls(text)
        assert "tool_call" not in cleaned
        assert [c["name"] for c in calls] == ["x"]
        assert cleaned == "Sure.  Done."


class TestStreamingSuppression:
    def test_a_thought_is_never_emitted(self):
        stream = ["<think>", "The user wants", " the time.", "</think>",
                  "It is", " half past two."]
        s = TagSuppressor()
        assert "".join(s.feed(c) for c in stream) + s.flush() == (
            "It is half past two."
        )

    def test_a_tag_split_across_chunks_is_still_caught(self):
        # The case a naive replace() on each chunk misses entirely.
        stream = ["<th", "ink>", "musing", ".</thi", "nk>", "Answer."]
        s = TagSuppressor()
        assert "".join(s.feed(c) for c in stream) + s.flush() == "Answer."

    def test_one_character_at_a_time(self):
        text = "<think>hidden</think>said"
        s = TagSuppressor()
        assert "".join(s.feed(ch) for ch in text) + s.flush() == "said"

    def test_text_without_tags_streams_through_unchanged(self):
        s = TagSuppressor()
        chunks = ["Hello", " there", ", how", " are you?"]
        assert "".join(s.feed(c) for c in chunks) + s.flush() == (
            "Hello there, how are you?"
        )

    def test_speech_is_not_held_back_waiting_for_the_end(self):
        # Buffering the whole response would delay the first spoken word
        # until generation finished, which defeats streaming TTS.
        s = TagSuppressor()
        assert s.feed("The answer is ") == "The answer is "

    def test_an_unterminated_thought_swallows_the_rest(self):
        s = TagSuppressor()
        out = s.feed("Fine.<think>still going and never closes")
        assert out + s.flush() == "Fine."

    def test_actions_and_thoughts_together(self):
        s = suppressors(ACTION_TAGS, REASONING_TAGS)
        stream = ["<think>", "plan", "</think>", "Running it. ",
                  "<ACTION>", "rm -rf /", "</ACTION>", "Done."]
        assert "".join(s.feed(c) for c in stream) + s.flush() == (
            "Running it. Done."
        )

    @pytest.mark.parametrize(
        "opener,closer",
        [t for t in REASONING_TAGS],
    )
    def test_every_known_reasoning_tag(self, opener, closer):
        s = TagSuppressor()
        assert s.feed(f"{opener}private{closer}public") + s.flush() == "public"


class TestWholeStringStripping:
    def test_a_complete_thought_is_removed(self):
        assert strip_reasoning("<think>musing</think>The answer.") == "The answer."

    def test_a_closing_tag_with_no_opener(self):
        # Some chat templates put the opening <think> in the prompt, so the
        # model's own output starts already inside the thought.
        assert strip_reasoning("musing about it</think>The answer.") == "The answer."

    def test_an_opener_that_never_closes(self):
        assert strip_reasoning("Fine.<think>still going") == "Fine."

    def test_untagged_text_is_untouched(self):
        assert strip_reasoning("Just an answer.") == "Just an answer."

    def test_main_strips_both_kinds(self):
        from nixorb.main import _strip_actions

        text = "<think>plan</think>Running it.<ACTION>ls</ACTION>"
        assert _strip_actions(text) == "Running it."

    def test_action_stripping_still_works_alone(self):
        assert strip_tags("Doing it.<ACTION>ls -l</ACTION>", ACTION_TAGS) == "Doing it."


class TestTokenBudget:
    def test_there_is_room_for_a_thought_and_an_answer(self):
        # 512 stopped inside the <think> block, so the answer never came.
        assert Settings().llm_max_tokens >= 2048


class TestSpeakerNeverSpeaksThinking:
    async def test_a_reasoning_reply_speaks_only_the_answer(self, started_bus):
        from nixorb.tts.speaker import Speaker

        spoken: list[str] = []

        class _Engine:
            name = "fake"

            async def speak(self, text):
                spoken.append(text)

            def stop(self):
                pass

        speaker = Speaker(_Engine(), streaming=True)
        await speaker.start()
        for chunk in ["<think>", "The user asked the time. ",
                      "I will check it now.", "</think>",
                      "It is half past two.", " Anything else?"]:
            await speaker.feed(chunk)
        await speaker.finish(timeout=5.0)

        said = " ".join(spoken)
        assert "half past two" in said
        assert "think" not in said
        assert "The user asked" not in said

    async def test_the_state_resets_between_turns(self, started_bus):
        # An unterminated thought in one turn must not silence the next.
        from nixorb.tts.speaker import Speaker

        spoken: list[str] = []

        class _Engine:
            name = "fake"

            async def speak(self, text):
                spoken.append(text)

            def stop(self):
                pass

        speaker = Speaker(_Engine(), streaming=True)
        await speaker.start()
        await speaker.feed("<think>never closed")
        await speaker.finish(timeout=5.0)

        spoken.clear()
        await speaker.start()
        await speaker.feed("A clean answer this time.")
        await speaker.finish(timeout=5.0)
        assert "clean answer" in " ".join(spoken)
