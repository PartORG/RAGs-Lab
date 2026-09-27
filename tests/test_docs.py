import re

import pytest

from chunking_strategies import CHUNKERS
from docs import SECTIONS, mermaid_to_dot, section
from embeddings import EMBEDDINGS
from retrievers import RETRIEVERS


@pytest.mark.parametrize(
    ("guide", "registry"),
    [
        ("CHUNKING_STRATEGIES.md", CHUNKERS),
        ("EMBEDDINGS.md", EMBEDDINGS),
        ("RETRIEVERS.md", RETRIEVERS),
    ],
)
def test_every_dropdown_entry_has_a_section(guide, registry):
    assert set(SECTIONS[guide]) == set(registry)


@pytest.mark.parametrize(
    ("guide", "key"), [(guide, key) for guide, keys in SECTIONS.items() for key in keys]
)
def test_section_is_one_numbered_section_and_its_diagrams_convert(guide, key):
    text = section(guide, key)
    assert text.startswith(f"## {SECTIONS[guide][key]}. ")
    assert len(re.findall(r"^## ", text, re.M)) == 1
    for diagram in re.findall(r"```mermaid\n(.*?)```", text, re.S):
        assert mermaid_to_dot(diagram) is not None


def test_mermaid_to_dot_shapes_labels_and_chains():
    dot = mermaid_to_dot(
        'flowchart TD\n  Q["Query"] --> VS[("DB")]\n  G{"ok?"} -- "no" --> R["Retry"] --> Q\n'
        '  A -.->|"swap"| A'
    )
    assert "rankdir=TB" in dot
    assert 'VS [label="DB", shape=cylinder]' in dot
    assert 'G [label="ok?", shape=diamond]' in dot
    assert 'G -> R [label="no"]' in dot
    assert "R -> Q []" in dot
    assert 'A -> A [label="swap", style=dashed]' in dot


def test_a_guide_without_a_dropdown_is_shown_whole_and_its_diagrams_convert():
    text = section("EVALUATION_STRATEGIES.md", None)
    assert text.startswith("# Evaluation Strategies")
    diagrams = re.findall(r"```mermaid\n(.*?)```", text, re.S)
    assert len(diagrams) == 15
    assert all(mermaid_to_dot(d) for d in diagrams)
