"""Regression guards for the OpenAI client usage in app.py.

The repository only uses the Chat Completions endpoint. The upstream spec
update removed the ``spend_control`` request property and unrelated agents /
moderations operations, so the request sent for a chat completion must stay
limited to the supported Chat Completions fields.
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
