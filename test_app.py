"""Regression guards for the OpenAI client usage in app.py.

The repository only uses the Chat Completions endpoint. Upstream spec updates
have removed the Assistants, Conversations, Evals and audio-translation paths
(and assistant-only request properties), so the request sent for a chat
completion must stay limited to the supported Chat Completions fields and the
client must not reach for any of the removed namespaces.
"""

from types import SimpleNamespace

import app


class FakeCompletions:
    def __init__(self):
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content="  ok  "))]
        )


class FakeChat:
    def __init__(self, completions):
        self.completions = completions


class FakeClient:
    def __init__(self):
        self._completions = FakeCompletions()
        self.chat = FakeChat(self._completions)


def test_call_model_uses_supported_chat_completion_request(monkeypatch):
    fake = FakeClient()
    monkeypatch.setattr(app, "client", fake)

    result = app.call_model("hello", max_tokens=42)

    assert result == "ok"
    assert len(fake._completions.calls) == 1

    payload = fake._completions.calls[0]
    assert payload["model"] == app.MODEL
    assert payload["messages"] == [{"role": "user", "content": "hello"}]
    assert payload["max_tokens"] == 42
    # ``spend_control`` was removed upstream and must never be sent.
    assert "spend_control" not in payload


class TrackingClient:
    """Records which top-level client namespaces app.py touches."""

    def __init__(self):
        completions = FakeCompletions()
        object.__setattr__(self, "_accessed", [])
        object.__setattr__(self, "_completions", completions)
        object.__setattr__(self, "chat", FakeChat(completions))

    def __getattribute__(self, name):
        if not name.startswith("_"):
            object.__getattribute__(self, "_accessed").append(name)
        return object.__getattribute__(self, name)


def test_call_model_stays_on_chat_completions_only(monkeypatch):
    """The upstream spec removed the Assistants, Conversations, Evals and
    audio-translation paths. app.py must keep using only Chat Completions and
    must not send assistant-only request properties."""
    fake = TrackingClient()
    monkeypatch.setattr(app, "client", fake)

    assert app.call_model("hello") == "ok"

    # No removed namespace (assistants, conversations, evals, audio) is used.
    assert fake._accessed == ["chat"]

    payload = fake._completions.calls[0]
    assert set(payload) == {"model", "messages", "max_tokens"}
    for removed_field in ("response_format", "tool_resources", "chunking_strategy"):
        assert removed_field not in payload
