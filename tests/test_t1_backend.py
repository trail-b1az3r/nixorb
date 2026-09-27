"""The T1 API as a backend, and choosing between it and the local model.

A local model that will not load is the commonest way the orb ends up
mute: it starts, listens, transcribes, and then has nothing to answer
with. `llm_backend = "auto"` health-checks the local model once and
answers through T1 when it cannot serve.

The SDK is not installed on every machine that runs these tests, so the
client is faked — but the call shapes are taken from the real
`hypernix.t1sdk.T1Client` signatures, and one test asserts they still
match when hypernix *is* installed.
"""
from __future__ import annotations

import sys
import types

import pytest

from nixorb.llm.auto_backend import AutoBackend
from nixorb.llm.factory import BACKENDS, build_llm, build_t1, normalise_backend
from nixorb.llm.t1_backend import T1Backend, T1Error, _reply_text
from nixorb.settings import Settings

T1_SETTINGS = dict(base := {"t1_base_url": "https://t1.example", "t1_api_key": "k"})


class _FakeErrors(Exception):
    pass


def install_fake_sdk(monkeypatch, client):
    """Put a stand-in `hypernix.t1sdk` on sys.modules."""

    class T1Error_(Exception):
        pass

    class T1AuthError_(T1Error_):
        pass

    class T1QuotaError_(T1Error_):
        pass

    sdk = types.SimpleNamespace(
        T1Client=lambda *a, **k: client,
        T1Error=T1Error_,
        T1AuthError=T1AuthError_,
        T1QuotaError=T1QuotaError_,
    )
    hypernix = types.ModuleType("hypernix")
    hypernix.t1sdk = sdk  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "hypernix", hypernix)
    monkeypatch.setitem(sys.modules, "hypernix.t1sdk", sdk)
    return sdk


class _Client:
    def __init__(self, *, models=("qwen3-8b",), reachable=True):
        self._models = list(models)
        self._reachable = reachable
        self.sessions: list[dict] = []
        self.turns: list[tuple] = []

    def ping(self):
        return self._reachable

    def list_models(self):
        return [types.SimpleNamespace(model_id=m) for m in self._models]

    def hyperlink_create_session(self, *, title="", model_id="", system_prompt=""):
        self.sessions.append(
            {"title": title, "model_id": model_id, "system_prompt": system_prompt}
        )
        return {"session_id": "sess-1"}

    def hyperlink_chat(self, session_id, content, *, model_id=None, max_tokens=None,
                       attachment_ids=None):
        self.turns.append((session_id, content, model_id, max_tokens))
        return {"reply": f"answered: {content}"}


class TestFactoryRouting:
    def test_t1_is_a_known_backend(self):
        assert "t1" in BACKENDS and "auto" in BACKENDS

    @pytest.mark.parametrize("alias", ["t1", "hypernix", "t1api", "remote"])
    def test_aliases_reach_t1(self, alias):
        assert normalise_backend(alias) == "t1"

    def test_build_t1_needs_a_base_url(self):
        assert build_t1(Settings()) is None
        assert build_t1(Settings(t1_base_url="https://t1.example")) is not None

    def test_auto_builds_both_halves(self):
        built = build_llm(Settings(llm_backend="auto", **T1_SETTINGS))
        assert isinstance(built, AutoBackend)
        assert built._remote is not None

    def test_auto_without_t1_still_builds(self):
        built = build_llm(Settings(llm_backend="auto"))
        assert isinstance(built, AutoBackend)
        assert built._remote is None

    def test_auto_does_not_recurse_into_itself(self):
        # llm_local_backend = "auto" would otherwise build AutoBackends forever.
        built = build_llm(Settings(llm_backend="auto", llm_local_backend="auto"))
        assert not isinstance(built._local, AutoBackend)


class TestHealth:
    async def test_a_reachable_server_is_healthy(self, monkeypatch):
        install_fake_sdk(monkeypatch, _Client())
        health = await T1Backend(Settings(**T1_SETTINGS)).health_check()
        assert health["ok"]
        assert health["models"] == ["qwen3-8b"]

    async def test_an_unreachable_server_says_so(self, monkeypatch):
        install_fake_sdk(monkeypatch, _Client(reachable=False))
        health = await T1Backend(Settings(**T1_SETTINGS)).health_check()
        assert not health["ok"]
        assert "not answering" in health["error"]

    async def test_a_missing_base_url_is_explained(self):
        health = await T1Backend(Settings()).health_check()
        assert not health["ok"]
        assert "t1_base_url" in health["error"]

    async def test_a_model_the_server_does_not_have(self, monkeypatch):
        install_fake_sdk(monkeypatch, _Client(models=("a", "b")))
        health = await T1Backend(
            Settings(t1_model="nope", **T1_SETTINGS)
        ).health_check()
        assert not health["ok"]
        assert "no model 'nope'" in health["error"]
        assert "a, b" in health["error"]

    async def test_a_refused_key_says_which_setting(self, monkeypatch):
        client = _Client()
        sdk = install_fake_sdk(monkeypatch, client)

        def refuse():
            raise sdk.T1AuthError("bad key")

        client.ping = refuse
        health = await T1Backend(Settings(**T1_SETTINGS)).health_check()
        assert not health["ok"]
        assert "t1_api_key" in health["error"]


class TestChat:
    async def test_a_turn_reaches_the_server(self, monkeypatch, started_bus):
        client = _Client()
        install_fake_sdk(monkeypatch, client)
        backend = T1Backend(Settings(**T1_SETTINGS))

        messages = [
            {"role": "system", "content": "Be brief."},
            {"role": "user", "content": "What time is it?"},
        ]
        assert await backend.generate(messages) == "answered: What time is it?"
        assert client.sessions[0]["system_prompt"] == "Be brief."
        assert client.turns[0][1] == "What time is it?"

    async def test_the_session_is_reused_across_turns(self, monkeypatch, started_bus):
        client = _Client()
        install_fake_sdk(monkeypatch, client)
        backend = T1Backend(Settings(**T1_SETTINGS))
        for text in ("one", "two", "three"):
            await backend.generate([{"role": "user", "content": text}])
        assert len(client.sessions) == 1, "a new session per turn loses the thread"
        assert len(client.turns) == 3

    async def test_only_the_latest_user_message_is_sent(
        self, monkeypatch, started_bus
    ):
        # The server persists the thread itself, so resending it would
        # duplicate every previous turn.
        client = _Client()
        install_fake_sdk(monkeypatch, client)
        await T1Backend(Settings(**T1_SETTINGS)).generate([
            {"role": "user", "content": "old"},
            {"role": "assistant", "content": "reply"},
            {"role": "user", "content": "new"},
        ])
        assert client.turns[0][1] == "new"

    async def test_a_quota_error_names_the_limit(self, monkeypatch, started_bus):
        client = _Client()
        sdk = install_fake_sdk(monkeypatch, client)

        def over(*a, **k):
            raise sdk.T1QuotaError("out of tokens")

        client.hyperlink_chat = over
        with pytest.raises(T1Error, match="quota"):
            await T1Backend(Settings(**T1_SETTINGS)).generate(
                [{"role": "user", "content": "hi"}]
            )

    async def test_an_empty_message_asks_nothing(self, monkeypatch, started_bus):
        client = _Client()
        install_fake_sdk(monkeypatch, client)
        assert await T1Backend(Settings(**T1_SETTINGS)).generate([]) == ""
        assert client.turns == []


class TestReplyShapes:
    """The key the reply lands under has moved between T1 versions."""

    @pytest.mark.parametrize(
        "payload,expected",
        [
            ({"reply": "hello"}, "hello"),
            ({"content": "hello"}, "hello"),
            ({"text": "hello"}, "hello"),
            ({"message": {"content": "hello"}}, "hello"),
            ({"messages": [{"role": "assistant", "content": "hello"}]}, "hello"),
            ({"reply": {"content": "hello"}}, "hello"),
            ("hello", "hello"),
        ],
    )
    def test_the_text_is_found(self, payload, expected):
        assert _reply_text(payload) == expected

    def test_an_unrecognisable_reply_is_empty_and_logged(self, caplog):
        with caplog.at_level("WARNING"):
            assert _reply_text({"surprise": 1}) == ""
        assert any("no recognisable text" in r.getMessage() for r in caplog.records)


class TestAutoSelection:
    class _Backend:
        def __init__(self, ok, name):
            self._ok, self.model, self.last_tool_calls = ok, name, []
            self.closed = False
            self.checks = 0

        async def health_check(self):
            self.checks += 1
            return {"ok": self._ok, "error": "" if self._ok else "will not load"}

        async def stream(self, messages, tools=None):
            yield f"[{self.model}]"

        async def close(self):
            self.closed = True

    async def test_a_healthy_local_model_is_used(self):
        local = self._Backend(True, "local")
        auto = AutoBackend(Settings(), local, self._Backend(True, "t1"))
        assert await auto.generate([]) == "[local]"
        assert auto.model == "local"

    async def test_a_broken_local_model_falls_through_to_t1(self):
        auto = AutoBackend(
            Settings(**T1_SETTINGS), self._Backend(False, "local"),
            self._Backend(True, "t1"),
        )
        assert await auto.generate([]) == "[t1]"

    async def test_the_choice_is_made_once(self):
        # Re-deciding every turn costs a model load each time.
        local = self._Backend(True, "local")
        auto = AutoBackend(Settings(), local, None)
        for _ in range(4):
            await auto.generate([])
        assert local.checks == 1

    async def test_with_no_t1_the_local_error_survives(self):
        auto = AutoBackend(Settings(), self._Backend(False, "local"), None)
        health = await auto.health_check()
        assert not health["ok"]
        assert "will not load" in health["error"]

    async def test_both_down_reports_both(self, caplog):
        auto = AutoBackend(
            Settings(**T1_SETTINGS), self._Backend(False, "local"),
            self._Backend(False, "t1"),
        )
        with caplog.at_level("ERROR"):
            await auto.health_check()
        assert any("neither backend" in r.getMessage() for r in caplog.records)

    async def test_closing_closes_both(self):
        local, remote = self._Backend(True, "l"), self._Backend(True, "r")
        await AutoBackend(Settings(), local, remote).close()
        assert local.closed and remote.closed

    async def test_tool_calls_pass_through_to_the_active_backend(self):
        local = self._Backend(True, "local")
        auto = AutoBackend(Settings(), local, None)
        await auto.health_check()
        auto.last_tool_calls = [{"name": "x"}]
        assert local.last_tool_calls == [{"name": "x"}]
        assert auto.last_tool_calls == [{"name": "x"}]


class TestAgainstTheRealSDK:
    def test_the_calls_match_the_installed_sdk(self):
        """Fakes drift. If hypernix is here, check the signatures."""
        import inspect

        t1sdk = pytest.importorskip("hypernix.t1sdk")
        client = t1sdk.T1Client

        create = inspect.signature(client.hyperlink_create_session).parameters
        assert {"title", "model_id", "system_prompt"} <= set(create)

        chat = inspect.signature(client.hyperlink_chat).parameters
        assert {"session_id", "content", "model_id", "max_tokens"} <= set(chat)

        init = inspect.signature(client.__init__).parameters
        assert {"base_url", "credential", "timeout"} <= set(init)

        for name in ("ping", "list_models"):
            assert callable(getattr(client, name))

        for error in ("T1Error", "T1AuthError", "T1QuotaError"):
            assert hasattr(t1sdk, error)


class TestDefaultIsResilient:
    """The default must not leave the orb mute when the model won't load."""

    def test_auto_is_the_default_backend(self):
        assert Settings().llm_backend == "auto"

    def test_auto_prefers_the_local_model(self):
        assert Settings().llm_local_backend == "huggingface"

    def test_with_no_t1_configured_nothing_is_contacted(self):
        settings = Settings()
        assert settings.t1_base_url == ""
        assert build_t1(settings) is None
