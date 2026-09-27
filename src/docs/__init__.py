"""The guides in this folder, cut into the one section the app's ℹ️ button shows.

Each dropdown entry points at its numbered `## N.` section. The numbers are fixed here rather than
matched on titles, because the dropdown labels and the headings were written for different readers.
"""

import re
from pathlib import Path

HERE = Path(__file__).resolve().parent

# Dropdown key -> (guide, section number). RAG strategies are keyed by their display name.
SECTIONS: dict[str, dict[str, int]] = {
    "RAGS.md": {
        "Naive RAG": 1,
        "Retrieve-and-Rerank": 2,
        "Hybrid RAG": 3,
        "Multi-Query / RAG-Fusion": 4,
        "HyDE": 5,
        "Conversational RAG": 6,
        "Hierarchical RAG": 7,
        "RAPTOR": 8,
        "Contextual RAG": 9,
        "GraphRAG": 10,
        "Multimodal RAG": 11,
        "Corrective RAG (CRAG)": 12,
        "Self-RAG": 13,
        "Adaptive RAG": 14,
        "Iterative / Multi-hop RAG": 15,
        "Speculative RAG": 16,
        "Agentic RAG": 17,
        "Branched / Multi-Source RAG": 18,
        "Cache-Augmented Generation (CAG)": 20,
        "Self-Query RAG": 21,
    },
    "CHUNKING_STRATEGIES.md": {
        "fixed": 1,
        "recursive": 2,
        "sentences": 3,
        "window": 4,
        "paragraphs": 5,
        "structure": 6,
        "tokens": 7,
        "semantic": 8,
        "llm": 9,
        "contextual": 10,
        "propositions": 11,
    },
    "EMBEDDINGS.md": {
        "nomic": 1,
        "mxbai": 2,
        "bge_small": 3,
        "bge_base": 4,
        "bge_m3": 5,
        "minilm_multilingual": 6,
    },
    "RETRIEVERS.md": {
        "similarity": 1,
        "mmr": 2,
        "threshold": 3,
        "bm25": 4,
        "hybrid": 5,
        "reranked": 6,
    },
}


def section(guide: str, key: str | None) -> str:
    """The `## N.` section for this dropdown key, heading included, up to the next `## `; the
    whole guide when there is no dropdown to follow (key None)."""
    text = (HERE / guide).read_text(encoding="utf-8")  # not the OS default: cp1252 on Windows
    if key is None:
        return text
    number = SECTIONS[guide][key]
    match = re.search(rf"^## {number}\. .*?(?=^## |\Z)", text, re.M | re.S)
    return match.group(0).rstrip().removesuffix("---").rstrip()


EDGE = re.compile(r'\s*(?:--\s*"([^"]*)"\s*-->|(-\.->|-->))(?:\|"([^"]*)"\|)?\s*')
NODE = re.compile(r'(\w+)\s*(\[\("|\["|\{"|\("|)(.*?)(?:"\)\]|"\]|"\}|"\))?$')
SHAPES = {'[("': "cylinder", '{"': "diamond", '("': "ellipse"}


def mermaid_to_dot(mermaid: str) -> str | None:
    """The guides' Mermaid flowcharts as Graphviz DOT, which Streamlit draws without a network
    call (Mermaid would need its script from a CDN). Only the subset the guides use: boxes,
    cylinders, diamonds, solid/dotted edges and edge labels. None for anything else."""
    lines = mermaid.strip().splitlines()
    direction = "TB" if lines[0].split()[-1] in ("TD", "TB") else "LR"
    out = [f"digraph {{ rankdir={direction}; node [shape=box, style=rounded];"]
    for line in lines[1:]:
        line = line.strip()
        if not line or line == "end" or line.startswith("subgraph"):
            continue  # ponytail: subgraph boxes are dropped, only their nodes and edges drawn
        parts = EDGE.split(line)
        ids = []
        for raw in parts[::4]:
            node = NODE.match(raw.strip())
            if not node:
                return None
            name, opening, label = node.groups()
            ids.append(name)
            if opening:
                shape = SHAPES.get(opening, "box")
                out.append(f'{name} [label="{label}", shape={shape}];')
        for i in range(len(ids) - 1):
            label, arrow, pipe_label = parts[4 * i + 1 : 4 * i + 4]
            attrs = [f'label="{label or pipe_label}"'] if label or pipe_label else []
            if arrow == "-.->":
                attrs.append("style=dashed")
            out.append(f"{ids[i]} -> {ids[i + 1]} [{', '.join(attrs)}];")
    return "\n".join(out) + "\n}"
