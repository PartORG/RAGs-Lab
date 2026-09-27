"""Start the RAG Lab web page: `uv run main.py`.

Create an account, or reset its password: `uv run main.py adduser NAME`.
Delete an account with all its documents: `uv run main.py deluser NAME`.
"""

import os
import sys
from getpass import getpass
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def adduser(name: str) -> None:
    from storage import add_user

    password = getpass(f"Password for {name}: ")
    if password != getpass("Again: "):
        sys.exit("The passwords differ.")
    try:
        add_user(name, password)
    except ValueError as e:
        sys.exit(f"Not saved: {e}.")
    print(f"Saved {name}.")


def deluser(name: str) -> None:
    from storage import delete_user

    # No undo: the account's documents and every index built from them go too.
    if input(f"Delete {name} and all their documents? Type the name to confirm: ") != name:
        sys.exit("Not deleted.")
    try:
        found = delete_user(name)
    except ValueError as e:
        sys.exit(f"Not deleted: {e}.")
    print(f"Deleted {name}." if found else f"No account {name}; removed any files left for it.")


COMMANDS = {"adduser": adduser, "deluser": deluser}

if __name__ == "__main__":
    if sys.argv[1:2] and sys.argv[1] in COMMANDS:
        if len(sys.argv) != 3:
            sys.exit(f"Usage: uv run main.py {sys.argv[1]} NAME")
        COMMANDS[sys.argv[1]](sys.argv[2])
        sys.exit()
    from streamlit.web import cli

    os.chdir(ROOT)  # Streamlit reads .streamlit/config.toml (localhost only) from the working dir
    sys.argv = ["streamlit", "run", str(ROOT / "src" / "frontend" / "app.py")]
    sys.exit(cli.main())
