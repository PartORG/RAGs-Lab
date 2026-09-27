#!/usr/bin/env bash
# Install RAG Lab on Linux (or macOS):
#   curl -fsSL https://raw.githubusercontent.com/PartORG/RAGs-Lab/main/install.sh | bash
#
# RAG_LAB_SOURCE   what to install: a release wheel, a folder, or a source archive URL
#                  (default: the main branch). e.g. RAG_LAB_SOURCE=. from a checkout.
# RAG_LAB_NONINTERACTIVE=1   answer every question "no" (CI): installs uv and rag-lab only.
set -euo pipefail

SOURCE="${RAG_LAB_SOURCE:-rag-lab @ https://github.com/PartORG/RAGs-Lab/archive/refs/heads/main.zip}"

ask() {  # stdin is this script when piped from curl, so read the answer from the terminal
    [ -n "${RAG_LAB_NONINTERACTIVE:-}" ] && return 1
    local answer
    read -r -p "$1 [y/N] " answer </dev/tty
    [[ "$answer" =~ ^[Yy] ]]
}

if ! command -v uv >/dev/null; then
    echo "Installing uv (the Python package manager this installs with)…"
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
fi

echo "Installing rag-lab (Python 3.13 and the CPU build of torch; about 1.5 GB)…"
# CPU torch: the in-process models run on the CPU, Ollama has the GPU; CUDA torch adds ~4 GB.
uv tool install --force --python 3.13 --torch-backend cpu "$SOURCE"
uv tool update-shell >/dev/null 2>&1 || true  # puts uv's tool folder on PATH for new shells
RAG_LAB="$(uv tool dir --bin)/rag-lab"

if ! command -v ollama >/dev/null && [ -z "${OLLAMA_HOST:-}" ]; then
    if ask "Ollama (runs the models) is not installed. Install it with its official script (needs sudo)?"; then
        curl -fsSL https://ollama.com/install.sh | sh
    else
        echo "Skipped. Install it later: https://ollama.com/download"
    fi
fi

echo
if ! "$RAG_LAB" doctor && ask "Pull the missing required models now (qwen3:8b is ~5 GB)?"; then
    "$RAG_LAB" doctor --fix || true
fi

if ask "Create an account now?"; then
    read -r -p "User name (a-z, 0-9, _ or -): " name </dev/tty
    "$RAG_LAB" adduser "$name" </dev/tty
fi

echo
echo "Done. Start RAG Lab with: rag-lab   (open a new terminal first if the command is not found)"
echo "Check the setup any time: rag-lab doctor"
