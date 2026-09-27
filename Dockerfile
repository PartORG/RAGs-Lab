FROM python:3.14-slim

COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv
WORKDIR /app
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy PATH=/app/.venv/bin:$PATH

# Dependencies first, so a code change does not re-download torch; the uv cache stays out of
# the image.
COPY pyproject.toml uv.lock ./
RUN --mount=type=cache,target=/root/.cache/uv uv sync --locked --no-dev --no-install-project
COPY main.py ./
COPY .streamlit .streamlit
COPY src src
RUN --mount=type=cache,target=/root/.cache/uv uv sync --locked --no-dev

RUN useradd --create-home app && mkdir /data /opt/hf && chown app /data /opt/hf
USER app
# The reranker (~1.1 GB) is baked in, so the first Retrieve-and-Rerank question does not wait
# for Hugging Face. The optional local embedding models still download on first pick, here.
ENV HF_HOME=/opt/hf
RUN python -c "from sentence_transformers import CrossEncoder; CrossEncoder('BAAI/bge-reranker-base', device='cpu')"

# Accounts and indexes (lab.db) and uploads live on the /data volume.
ENV RAG_DATA=/data STREAMLIT_SERVER_ADDRESS=0.0.0.0
VOLUME /data
EXPOSE 8501
HEALTHCHECK --interval=30s --timeout=5s --start-period=30s \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8501/_stcore/health')"
CMD ["python", "main.py"]
