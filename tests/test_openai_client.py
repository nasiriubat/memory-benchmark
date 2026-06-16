"""Direct OpenAI client."""

from __future__ import annotations

from benchmark.llms.openai_client import OpenAIClient


def test_openai_client_uses_max_completion_tokens(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    captured: dict = {}

    def fake_retry(params):
        captured.update(params)
        class _Msg:
            content = "ok"

        class _Choice:
            message = _Msg()

        class _Resp:
            model = "gpt-5.5"
            choices = [_Choice()]
            usage = {"prompt_tokens": 1, "completion_tokens": 1}

            def model_dump(self):
                return {"model": self.model, "choices": [], "usage": self.usage}

        return _Resp()

    client = OpenAIClient(requested_model="gpt-5.5", resolved_model="gpt-5.5")
    monkeypatch.setattr(client, "_call_with_retry", fake_retry)
    client.generate([{"role": "user", "content": "hi"}])

    assert "max_completion_tokens" in captured
    assert captured["max_completion_tokens"] == 4096
    assert "max_tokens" not in captured
    assert "temperature" not in captured


def test_openai_client_sends_temperature_for_gpt4o(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    captured: dict = {}

    def fake_retry(params):
        captured.update(params)

        class _Msg:
            content = "ok"

        class _Choice:
            message = _Msg()

        class _Resp:
            model = "gpt-4o"
            choices = [_Choice()]
            usage = {"prompt_tokens": 1, "completion_tokens": 1}

            def model_dump(self):
                return {"model": self.model, "choices": [], "usage": self.usage}

        return _Resp()

    client = OpenAIClient(
        requested_model="gpt-4o", resolved_model="gpt-4o", temperature=0.0
    )
    monkeypatch.setattr(client, "_call_with_retry", fake_retry)
    client.generate([{"role": "user", "content": "hi"}])

    assert captured.get("temperature") == 0.0
