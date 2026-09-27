"""Everything the lab keeps, in one SQLite file (`data/lab.db`): the accounts, and every index of
every user. Uploaded files and extracted figures stay files, under `data/users/<name>/`.

An index is one JSON document per (owner, name) row, the same JSON the index files used to hold.
SQLite makes each save atomic and serialises concurrent writers, so two users uploading at once
cannot corrupt each other, and a crash mid-save leaves the previous version.
"""

import hashlib
import hmac
import json
import os
import re
import secrets
import shutil
import sqlite3
from contextlib import closing
from dataclasses import dataclass
from pathlib import Path

DATA = Path(os.environ.get("RAG_DATA") or Path(__file__).resolve().parents[2] / "data")
DB = DATA / "lab.db"
# Names become folder names, so nothing that could climb out of data/users/.
USERNAME = re.compile(r"[a-z0-9_-]{1,32}")
MIN_PASSWORD = 8

SCHEMA = """
CREATE TABLE IF NOT EXISTS users (name TEXT PRIMARY KEY, salt BLOB NOT NULL, hash BLOB NOT NULL);
CREATE TABLE IF NOT EXISTS saved (
    owner TEXT NOT NULL, name TEXT NOT NULL, data TEXT NOT NULL, PRIMARY KEY (owner, name)
);
"""


def _run(sql: str, args: tuple = ()) -> list[tuple]:
    """One statement in its own connection and transaction: Streamlit serves every session from
    its own thread, and a sqlite3 connection must not cross threads."""
    DB.parent.mkdir(parents=True, exist_ok=True)
    with closing(sqlite3.connect(DB, timeout=30)) as db, db:
        db.execute("PRAGMA journal_mode=WAL")  # readers never wait for a writer
        db.executescript(SCHEMA)
        return db.execute(sql, args).fetchall()


@dataclass(frozen=True)
class Saved:
    """One saved index: a JSON document belonging to one user."""

    owner: str
    name: str

    def load(self) -> object | None:
        rows = _run("SELECT data FROM saved WHERE owner = ? AND name = ?", (self.owner, self.name))
        return json.loads(rows[0][0]) if rows else None

    def save(self, data: object) -> None:
        _run(
            "INSERT INTO saved VALUES (?, ?, ?) "
            "ON CONFLICT (owner, name) DO UPDATE SET data = excluded.data",
            (self.owner, self.name, json.dumps(data)),
        )


def saved_of(owner: str, prefix: str) -> list[Saved]:
    """Every saved index of this user whose name starts with the prefix."""
    rows = _run(
        "SELECT name FROM saved WHERE owner = ? AND substr(name, 1, ?) = ?",
        (owner, len(prefix), prefix),
    )
    return [Saved(owner, name) for (name,) in rows]


def user_dir(name: str) -> Path:
    if not USERNAME.fullmatch(name):
        raise ValueError(f"not a valid user name: {name!r}")
    return DATA / "users" / name


def _hash(password: str, salt: bytes) -> bytes:
    return hashlib.scrypt(password.encode(), salt=salt, n=2**14, r=8, p=1)


def add_user(name: str, password: str) -> None:
    """Create an account, or reset the password of an existing one."""
    user_dir(name)  # validates the name
    if len(password) < MIN_PASSWORD:
        raise ValueError(f"the password needs at least {MIN_PASSWORD} characters")
    salt = secrets.token_bytes(16)
    _run(
        "INSERT INTO users VALUES (?, ?, ?) "
        "ON CONFLICT (name) DO UPDATE SET salt = excluded.salt, hash = excluded.hash",
        (name, salt, _hash(password, salt)),
    )


def delete_user(name: str) -> bool:
    """Remove an account with all its indexes and files. False if there was no such account."""
    folder = user_dir(name)  # validates the name before anything is removed
    existed = bool(_run("SELECT 1 FROM users WHERE name = ?", (name,)))
    _run("DELETE FROM saved WHERE owner = ?", (name,))
    _run("DELETE FROM users WHERE name = ?", (name,))
    shutil.rmtree(folder, ignore_errors=True)
    return existed


def account_stamp(name: str) -> str | None:
    """Changes whenever the account is created, re-created or given a new password (the salt is
    new each time); None once it is deleted. A session and its cached indexes belong to one stamp,
    so a deleted account's open tab ends, and a new account under the same name never inherits
    the old one's indexes from memory."""
    rows = _run("SELECT salt FROM users WHERE name = ?", (name,))
    return rows[0][0].hex() if rows else None


def check_password(name: str, password: str) -> bool:
    rows = _run("SELECT salt, hash FROM users WHERE name = ?", (name,))
    # An unknown name costs the same scrypt as a known one, so timing does not tell them apart.
    salt, expected = rows[0] if rows else (b"\0" * 16, b"")
    return hmac.compare_digest(_hash(password, salt), expected) and bool(rows)
