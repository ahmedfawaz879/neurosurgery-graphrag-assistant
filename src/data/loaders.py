"""Loaders for the literature corpus, gap taxonomy, and QA evaluation set.

`load_corpus()` implements both corpus paths behind one function, exactly as in
the notebook (`notebooks/neurosurgery_graphrag_assistant.ipynb`, cells 9-10):

- "shipped": the 14 real, cited papers in data/corpus/literature_corpus.jsonl.
- "real_pdf_dir": an arbitrary local PDF collection loaded via LlamaIndex's
  SimpleDirectoryReader. Automatic gap-tagging and citation lookup for
  arbitrary PDFs is deliberately NOT implemented -- see data/README.md for
  the required manual-tagging workflow before using this path for evaluation.
"""

from __future__ import annotations

import json
from pathlib import Path

from src.data.schemas import GapTaxonomy, Paper, QAItem

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CORPUS_PATH = REPO_ROOT / "data" / "corpus" / "literature_corpus.jsonl"
DEFAULT_QA_PATH = REPO_ROOT / "data" / "eval" / "qa_set.jsonl"
DEFAULT_TAXONOMY_PATH = REPO_ROOT / "data" / "gap_taxonomy.json"


def _load_jsonl(path: str | Path) -> list[dict]:
    records = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def _load_shipped_corpus(path: str | Path) -> list[Paper]:
    return [Paper.model_validate(r) for r in _load_jsonl(path)]


def load_real_pdf_corpus(pdf_dir: str) -> list[Paper]:
    """Loads a real PDF collection via LlamaIndex's SimpleDirectoryReader.

    Each returned Paper has `citation` and `url` set to "(fill in manually)" and
    `gap_tags` set to `[]` -- these must be assigned by hand (see data/README.md)
    before the output is usable for evaluation. Automatic gap-tagging or
    citation lookup for arbitrary PDFs is intentionally not implemented here,
    to avoid silently mis-tagging or mis-citing real literature.
    """
    from llama_index.core import SimpleDirectoryReader

    docs = SimpleDirectoryReader(pdf_dir).load_data()
    out = []
    for i, d in enumerate(docs):
        out.append(
            Paper(
                id=f"real_pdf_{i:03d}",
                title=d.metadata.get("file_name", f"real_pdf_{i:03d}"),
                citation="(fill in manually)",
                url="(fill in manually)",
                abstract=d.text[:2000],
                gap_tags=[],
            )
        )
    return out


def load_corpus(mode: str = "shipped", path: str | None = None) -> list[Paper]:
    """Loads the active paper corpus.

    Args:
        mode: "shipped" (default) or "real_pdf_dir".
        path: for "shipped", overrides the default literature_corpus.jsonl path;
              for "real_pdf_dir", the directory of PDFs to load.
    """
    if mode == "shipped":
        return _load_shipped_corpus(path or DEFAULT_CORPUS_PATH)
    if mode == "real_pdf_dir":
        if not path:
            raise ValueError("`path` (the PDF directory) is required when mode='real_pdf_dir'")
        return load_real_pdf_corpus(path)
    raise ValueError(f"Unknown corpus mode: {mode!r} (expected 'shipped' or 'real_pdf_dir')")


def load_qa_set(path: str | Path = DEFAULT_QA_PATH) -> list[QAItem]:
    return [QAItem.model_validate(r) for r in _load_jsonl(path)]


def load_gap_taxonomy(path: str | Path = DEFAULT_TAXONOMY_PATH) -> GapTaxonomy:
    with open(path, encoding="utf-8") as f:
        return GapTaxonomy.model_validate(json.load(f))
