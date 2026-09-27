# Install RAG Lab on Windows (PowerShell):
#   powershell -ExecutionPolicy ByPass -c "irm https://raw.githubusercontent.com/PartORG/RAGs-Lab/main/install.ps1 | iex"
#
# $env:RAG_LAB_SOURCE   what to install: a release wheel, a folder, or a source archive URL
#                       (default: the main branch). e.g. $env:RAG_LAB_SOURCE="." from a checkout.
# $env:RAG_LAB_NONINTERACTIVE=1   answer every question "no" (CI): installs uv and rag-lab only.
$ErrorActionPreference = "Stop"

$Source = if ($env:RAG_LAB_SOURCE) { $env:RAG_LAB_SOURCE } else {
    "rag-lab @ https://github.com/PartORG/RAGs-Lab/archive/refs/heads/main.zip"
}

function Ask($question) {
    if ($env:RAG_LAB_NONINTERACTIVE) { return $false }
    return (Read-Host "$question [y/N]") -match '^[Yy]'
}

function Invoke-Checked {  # a failing native command does not stop a script by itself
    & $args[0] $args[1..($args.Count - 1)]
    if ($LASTEXITCODE) { throw "$($args[0]) failed with exit code $LASTEXITCODE" }
}

if (-not (Get-Command uv -ErrorAction SilentlyContinue)) {
    Write-Host "Installing uv (the Python package manager this installs with)..."
    powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
    $env:Path = "$env:USERPROFILE\.local\bin;$env:Path"
}

Write-Host "Installing rag-lab (Python 3.13 and the CPU build of torch; about 1.5 GB)..."
# CPU torch: the in-process models run on the CPU, Ollama has the GPU; CUDA torch adds ~4 GB.
Invoke-Checked uv tool install --force --python 3.13 --torch-backend cpu $Source
uv tool update-shell *> $null  # puts uv's tool folder on PATH for new terminals
$RagLab = Join-Path (uv tool dir --bin) "rag-lab.exe"

if (-not (Get-Command ollama -ErrorAction SilentlyContinue) -and -not $env:OLLAMA_HOST) {
    if (Ask "Ollama (runs the models) is not installed. Install it with winget?") {
        Invoke-Checked winget install --id Ollama.Ollama -e --accept-source-agreements --accept-package-agreements
        Write-Host "Ollama starts by itself after installing; if not, start it from the Start menu."
    } else {
        Write-Host "Skipped. Install it later: https://ollama.com/download"
    }
}

Write-Host ""
& $RagLab doctor
if ($LASTEXITCODE -and (Ask "Pull the missing required models now (qwen3:8b is ~5 GB)?")) {
    & $RagLab doctor --fix
}

if (Ask "Create an account now?") {
    $name = Read-Host "User name (a-z, 0-9, _ or -)"
    & $RagLab adduser $name
}

Write-Host ""
Write-Host "Done. Start RAG Lab with: rag-lab   (open a new terminal first if the command is not found)"
Write-Host "Check the setup any time: rag-lab doctor"
