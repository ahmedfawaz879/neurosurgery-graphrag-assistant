# Data

## The shipped corpus (`data/corpus/literature_corpus.jsonl`)

This is **real, published literature — 14 real papers, one selected per gap category** in
`data/gap_taxonomy.json`. It is **not a synthetic dataset**, and it is **not a systematic sample of the
field**: each paper was chosen as one clear, verifiable illustration of its gap category, not as the
definitive or most-cited work on that topic. A 14-paper convenience sample is a pipeline demonstration, not
a literature survey.

Every entry carries:

- `citation` and `url` — the real source, exactly as identified during authoring. These are the
  traceability link back to a verifiable paper and must never be dropped, invented, or altered.
- `abstract` — an **original paraphrase written for this project**, not copied or lightly reworded from
  the source's own abstract. Quoting is deliberately avoided entirely (not just kept short) so nothing here
  substitutes for reading the actual paper.
- `gap_tags` — one or more codes from `data/gap_taxonomy.json` that the paper's finding illustrates.

**Always verify a finding against the cited `url` before relying on it.** The paraphrase is a lossy
compression of the source, written by a human (and, downstream, re-summarized by an LLM in this
assistant's answers) — treat any specific claim this repo's assistant produces as a lead to check against
the primary source, not as a citation-grade statement on its own.

## The QA evaluation set (`data/eval/qa_set.jsonl`)

Eight questions of three types, gold-labeled against the real papers' actual reported findings:

- `local` — answerable from a single paper's content.
- `global` — answerable only by synthesizing across multiple papers (GraphRAG's community-summarization
  path exists specifically because no single paper answers these).
- `trap` — embeds a false premise about what a real paper found, to test whether the assistant fabricates
  confidently or corrects the record.

Each item's `paper_ids` reference `id`s in `literature_corpus.jsonl`, and `gold_gaps` reference codes in
`gap_taxonomy.json` — both relationships are asserted by `tests/test_data_loaders.py`.

## Loading a different/additional real PDF collection

`load_corpus(mode="real_pdf_dir", path=...)` loads an arbitrary local PDF collection you hold rights to,
via LlamaIndex's `SimpleDirectoryReader`, instead of (or in addition to) the shipped 14-paper corpus.

**This path deliberately does not auto-tag gaps or auto-fill citations.** Every `Paper` it returns has
`citation="(fill in manually)"`, `url="(fill in manually)"`, and `gap_tags=[]`. Automatically guessing a
gap category or generating a citation for an arbitrary PDF risks silently mis-tagging or mis-citing real
literature — a mistake that compounds every time the assistant answers a question sourced from that paper.

Before using `load_real_pdf_corpus()` output for evaluation:

1. For each returned entry, manually verify the paper's real citation and source URL, and fill in the
   `citation` / `url` fields.
2. Manually assign `gap_tags` by reading the paper (or, at minimum, its actual abstract) against
   `data/gap_taxonomy.json` — do not accept an LLM's guess as ground truth without a human check.
3. Only after both steps should entries from this path be mixed into an evaluation run — an untagged,
   uncited entry will fail `tests/test_data_loaders.py`'s provenance assertions by design.
