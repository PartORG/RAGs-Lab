from types import SimpleNamespace

import ollama
import pytest

from rag_lab import cli
from storage import add_user


def installed(*names):
    return lambda: SimpleNamespace(models=[SimpleNamespace(model=n) for n in names])


@pytest.fixture(autouse=True)
def plenty_of_ram(monkeypatch):
    monkeypatch.setattr(cli, "ram_gb", lambda: 32.0)


def test_has_model_matches_the_latest_tag_but_not_other_sizes():
    assert cli.has_model(["nomic-embed-text:latest"], "nomic-embed-text")
    assert cli.has_model(["qwen3:8b"], "qwen3:8b")
    assert not cli.has_model(["qwen3:4b"], "qwen3:8b")


def test_doctor_passes_when_ollama_models_and_an_account_are_there(monkeypatch, capsys):
    monkeypatch.setattr(ollama, "list", installed(*cli.REQUIRED_MODELS))
    add_user("ann", "long enough")
    assert cli.doctor(fix=False) == 0
    out = capsys.readouterr().out
    assert "[ok  ] accounts: ann" in out
    assert "[warn] model" in out  # the optional models are missing, which is not a failure


def test_doctor_fails_with_a_fix_when_ollama_is_not_installed(monkeypatch, capsys):
    def unreachable():
        raise ConnectionError("refused")

    monkeypatch.setattr(ollama, "list", unreachable)
    monkeypatch.setattr(cli.shutil, "which", lambda _: None)
    monkeypatch.delenv("OLLAMA_HOST", raising=False)
    assert cli.doctor(fix=False) == 1
    out = capsys.readouterr().out
    assert "[FAIL] Ollama: not installed" in out
    assert f"fix: {cli.ollama_fixes()[0]}" in out
    assert "[FAIL] accounts: none yet" in out


def test_doctor_fix_pulls_only_the_missing_required_models(monkeypatch, capsys):
    have = [cli.REQUIRED_MODELS[1]]
    monkeypatch.setattr(ollama, "list", lambda: installed(*have)())
    pulled = []
    monkeypatch.setattr(cli, "pull", lambda name: (pulled.append(name), have.append(name)))
    add_user("ann", "long enough")
    assert cli.doctor(fix=True) == 0
    assert pulled == [cli.CHAT_MODEL]


def test_commands_are_routed(monkeypatch):
    calls = []
    monkeypatch.setattr(cli, "adduser", lambda name: calls.append(("adduser", name)))
    monkeypatch.setattr(cli, "run", lambda: calls.append(("run",)))
    cli.main(["adduser", "ann"])
    cli.main([])
    assert calls == [("adduser", "ann"), ("run",)]
