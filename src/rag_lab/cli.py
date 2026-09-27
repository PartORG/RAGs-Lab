"""The `rag-lab` command.

rag-lab                   start the page (same as `rag-lab run`)
rag-lab adduser NAME      create an account, or reset its password
rag-lab deluser NAME      delete an account with all its documents
rag-lab doctor [--fix]    check Ollama, models, memory and disk; --fix pulls missing models
"""

import argparse
import ctypes
import os
import shutil
import sys
import threading
import webbrowser
from dataclasses import dataclass
from getpass import getpass
from pathlib import Path

import storage

CHAT_MODEL = os.environ.get("RAG_CHAT_MODEL", "qwen3:8b")
# Required: the page cannot answer or index without these. Optional: one strategy each needs it.
REQUIRED_MODELS = [CHAT_MODEL, "nomic-embed-text"]
OPTIONAL_MODELS = {
    os.environ.get("RAG_VISION_MODEL", "qwen2.5vl:3b"): "Multimodal RAG",
    os.environ.get(
        "RAG_DRAFT_MODEL", "llama3.2:3b"
    ): "Speculative RAG (falls back to the chat model)",
}
MIN_RAM_GB = 16  # a 5-6 GB chat model plus the reranker ran a 14 GB laptop out of memory
MIN_DISK_GB = 10  # models (~6 GB for the required ones) plus indexes


def run() -> None:
    from streamlit.web import cli

    # Uploaded documents stay on this machine: listen on localhost and send no usage statistics.
    # Defaults only, so the environment can override them (Docker sets the address to 0.0.0.0).
    os.environ.setdefault("STREAMLIT_SERVER_ADDRESS", "localhost")
    os.environ.setdefault("STREAMLIT_BROWSER_SERVER_ADDRESS", "localhost")
    os.environ.setdefault("STREAMLIT_BROWSER_GATHER_USAGE_STATS", "false")
    os.environ.setdefault("STREAMLIT_SERVER_HEADLESS", "true")
    os.environ.setdefault("STREAMLIT_SERVER_MAX_UPLOAD_SIZE", "50")
    if os.environ["STREAMLIT_SERVER_ADDRESS"] == "localhost":
        # Headless skips Streamlit's own browser opening (and its e-mail prompt); open it here.
        port = os.environ.get("STREAMLIT_SERVER_PORT", "8501")
        threading.Timer(2, webbrowser.open, [f"http://localhost:{port}"]).start()
    sys.argv = ["streamlit", "run", str(Path(__file__).with_name("app.py"))]
    sys.exit(cli.main())


def adduser(name: str) -> None:
    password = getpass(f"Password for {name}: ")
    if password != getpass("Again: "):
        sys.exit("The passwords differ.")
    try:
        storage.add_user(name, password)
    except ValueError as e:
        sys.exit(f"Not saved: {e}.")
    print(f"Saved {name}.")


def deluser(name: str) -> None:
    # No undo: the account's documents and every index built from them go too.
    if input(f"Delete {name} and all their documents? Type the name to confirm: ") != name:
        sys.exit("Not deleted.")
    try:
        found = storage.delete_user(name)
    except ValueError as e:
        sys.exit(f"Not deleted: {e}.")
    print(f"Deleted {name}." if found else f"No account {name}; removed any files left for it.")


@dataclass
class Check:
    name: str
    ok: bool
    detail: str
    fix: str = ""
    required: bool = True


def ollama_fixes() -> tuple[str, str]:
    """How to install Ollama, and how to start it, on this OS."""
    if sys.platform == "win32":
        return "winget install Ollama.Ollama", "start Ollama from the Start menu"
    if sys.platform == "darwin":
        return "brew install ollama (or https://ollama.com/download)", "open the Ollama app"
    return (
        "curl -fsSL https://ollama.com/install.sh | sh",
        "sudo systemctl start ollama (or run: ollama serve)",
    )


def has_model(installed: list[str], name: str) -> bool:
    """`nomic-embed-text` matches `nomic-embed-text:latest`; `qwen3:8b` only itself."""
    return name in installed or (
        ":" not in name and any(m.startswith(name + ":") for m in installed)
    )


def ram_gb() -> float | None:
    try:
        if sys.platform == "win32":

            class MemoryStatus(ctypes.Structure):
                _fields_ = [("length", ctypes.c_ulong), ("load", ctypes.c_ulong)] + [
                    (field, ctypes.c_ulonglong)
                    for field in ("total", "free", "page", "page_free", "virt", "virt_free", "ext")
                ]

            status = MemoryStatus(length=ctypes.sizeof(MemoryStatus))
            ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status))
            return status.total / 2**30
        return os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 2**30
    except (AttributeError, OSError, ValueError):
        return None


def checks() -> tuple[list[Check], list[str]]:
    """Every check, and the required models that are missing (for --fix)."""
    import ollama

    install, start = ollama_fixes()
    found: list[Check] = []
    missing: list[str] = []
    try:
        installed = [m.model for m in ollama.list().models]
        found.append(Check("Ollama", True, "running"))
    except (ConnectionError, OSError) as e:
        installed = None
        if shutil.which("ollama") or os.environ.get("OLLAMA_HOST"):
            found.append(Check("Ollama", False, f"not reachable ({e})", start))
        else:
            found.append(Check("Ollama", False, "not installed", install))
    if installed is not None:
        for name in REQUIRED_MODELS:
            ok = has_model(installed, name)
            found.append(
                Check(
                    f"model {name}",
                    ok,
                    "installed" if ok else "missing",
                    f"ollama pull {name}  (or: rag-lab doctor --fix)",
                )
            )
            if not ok:
                missing.append(name)
        for name, used_by in OPTIONAL_MODELS.items():
            ok = has_model(installed, name)
            found.append(
                Check(
                    f"model {name}",
                    ok,
                    "installed" if ok else f"missing, for {used_by}",
                    f"ollama pull {name}",
                    required=False,
                )
            )
    ram = ram_gb()
    if ram is not None:
        found.append(
            Check(
                "memory",
                ram >= MIN_RAM_GB,
                f"{ram:.0f} GB",
                f"under {MIN_RAM_GB} GB: close other programs, avoid bge-m3 and the "
                "reranker next to an 8B chat model",
                required=False,
            )
        )
    try:
        storage.DATA.mkdir(parents=True, exist_ok=True)
        free = shutil.disk_usage(storage.DATA).free / 2**30
        found.append(Check("data folder", True, str(storage.DATA)))
        found.append(
            Check(
                "free disk",
                free >= MIN_DISK_GB,
                f"{free:.0f} GB",
                f"models and indexes need about {MIN_DISK_GB} GB",
                required=False,
            )
        )
        users = storage.user_names()
        found.append(
            Check("accounts", bool(users), ", ".join(users) or "none yet", "rag-lab adduser NAME")
        )
    except OSError as e:
        found.append(
            Check(
                "data folder",
                False,
                f"{storage.DATA}: {e}",
                "set RAG_DATA to a folder you can write to",
            )
        )
    return found, missing


def pull(name: str) -> None:
    import ollama

    print(f"Pulling {name}…")
    for part in ollama.pull(name, stream=True):
        if part.total:
            print(f"\r  {part.status} {100 * (part.completed or 0) // part.total}%   ", end="")
    print()


def doctor(fix: bool) -> int:
    found, missing = checks()
    if fix and missing:
        for name in missing:
            pull(name)
        found, missing = checks()
    # ASCII markers: a Windows console with a legacy code page cannot print ✓.
    for check in found:
        mark = "ok  " if check.ok else ("FAIL" if check.required else "warn")
        print(f"[{mark}] {check.name}: {check.detail}")
        if not check.ok and check.fix:
            print(f"       fix: {check.fix}")
    failed = [c for c in found if c.required and not c.ok]
    print("\nReady: start with `rag-lab`." if not failed else f"\n{len(failed)} problem(s) to fix.")
    return 1 if failed else 0


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="rag-lab", description="Compare RAG strategies locally.")
    commands = parser.add_subparsers(dest="command")
    commands.add_parser("run", help="start the page (the default)")
    for command, text in (
        ("adduser", "create an account or reset its password"),
        ("deluser", "delete an account with all its documents"),
    ):
        commands.add_parser(command, help=text).add_argument("name")
    doctor_parser = commands.add_parser("doctor", help="check that everything needed is installed")
    doctor_parser.add_argument("--fix", action="store_true", help="pull missing required models")
    args = parser.parse_args(argv)
    if args.command == "adduser":
        adduser(args.name)
    elif args.command == "deluser":
        deluser(args.name)
    elif args.command == "doctor":
        sys.exit(doctor(args.fix))
    else:
        run()
