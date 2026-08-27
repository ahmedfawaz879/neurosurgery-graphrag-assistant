"""Triple + gap-code extraction from paper abstracts via constrained-JSON LLM extraction.

Ported from the notebook's Section 5 (`notebooks/neurosurgery_graphrag_assistant.ipynb`,
cell 19): `parse_llm_json()` and `extract_triples_and_gap()`.
"""

from __future__ import annotations

import json
import re
from typing import Any

from pydantic import BaseModel, ValidationInfo, field_validator

from src.data.schemas import Paper
from src.llm.backend import OpenAIBackend

TRIPLE_EXTRACTION_PROMPT = """Extract (subject, predicate, object) triples from
this paper abstract, and classify which gap category from this taxonomy it
primarily relates to (or null if none clearly apply).

Gap taxonomy:
{taxonomy}

Return ONLY JSON in this exact structure:

{{
  "triples": [
    {{
      "subject": "...",
      "predicate": "...",
      "object": "..."
    }}
  ],
  "gap_code": "G1"
}}

Do not include explanations.
Do not include Markdown code fences.

Title:
{title}

Abstract:
{abstract}
"""


class Triple(BaseModel):
    """A (subject, predicate, object) fact, optionally tagged with the gap
    category it was extracted under. `gap_code` is validated against the
    caller-supplied taxonomy at construction time (via `context=`) and
    coerced to `None` -- never silently accepted -- if it isn't a real code."""

    subject: str
    predicate: str
    object: str
    gap_code: str | None = None

    @field_validator("gap_code")
    @classmethod
    def _gap_code_must_be_in_taxonomy(cls, v: str | None, info: ValidationInfo) -> str | None:
        context = info.context or {}
        valid_codes = context.get("valid_gap_codes")
        if v is None or valid_codes is None:
            return v
        return v if v in valid_codes else None


def parse_llm_json(raw: str) -> dict[str, Any]:
    """Robustly parses JSON returned by an LLM. Handles: (1) plain JSON,
    (2) ```json ... ``` Markdown fences, (3) extra text surrounding a JSON object."""
    if not isinstance(raw, str):
        raise TypeError(f"Expected string from llm_backend.generate(), got {type(raw).__name__}")

    raw = raw.strip()

    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        pass

    cleaned = re.sub(r"^\s*```(?:json)?\s*", "", raw, flags=re.IGNORECASE)
    cleaned = re.sub(r"\s*```\s*$", "", cleaned).strip()

    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass

    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start != -1 and end != -1 and end > start:
        candidate = cleaned[start : end + 1]
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            pass

    raise json.JSONDecodeError("Could not parse JSON from LLM response", raw, 0)


def extract_triples_and_gap(
    paper: Paper,
    gap_taxonomy: dict[str, str],
    llm_backend: OpenAIBackend,
) -> tuple[list[Triple], str | None]:
    """Extracts triples and a paper-level gap classification from `paper`'s abstract.

    Fails closed: a malformed LLM response (unparseable JSON, wrong shapes) yields
    `([], None)` rather than raising or fabricating a plausible-looking result. An
    invalid `gap_code` (not a real taxonomy key) is coerced to `None`.
    """
    prompt = TRIPLE_EXTRACTION_PROMPT.format(
        taxonomy=json.dumps(gap_taxonomy, ensure_ascii=False, indent=2),
        title=paper.title,
        abstract=paper.abstract,
    )
    raw = llm_backend.generate(prompt, max_tokens=500, json_mode=True)

    try:
        data = parse_llm_json(raw)
    except Exception:
        return [], None

    valid_codes = set(gap_taxonomy.keys())
    gap_code = data.get("gap_code")
    if gap_code not in valid_codes:
        gap_code = None

    raw_triples = data.get("triples", [])
    if not isinstance(raw_triples, list):
        raw_triples = []

    triples: list[Triple] = []
    for t in raw_triples:
        if not isinstance(t, dict):
            continue
        subject = str(t.get("subject", "")).strip()
        predicate = str(t.get("predicate", "")).strip()
        obj = str(t.get("object", "")).strip()
        if not subject or not predicate or not obj:
            continue
        triples.append(
            Triple.model_validate(
                {"subject": subject, "predicate": predicate, "object": obj, "gap_code": gap_code},
                context={"valid_gap_codes": valid_codes},
            )
        )

    return triples, gap_code
