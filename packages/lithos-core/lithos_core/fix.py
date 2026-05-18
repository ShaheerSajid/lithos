"""lithos_core.fix — schema for the LLM-extracted fix metadata column.

Each rule row in the DB carries a ``fix_metadata_json`` column populated by
the LLM ingestion pass. It describes the *intent* of the rule and which
classes of fix-actions are appropriate — the structured complement to the
deck-derived ``constraint_json``.

The action-class strings are opaque tags at this layer: they're whatever the
LLM emits under constrained decoding, validated against a vocabulary the
fix engine maintains separately. Binding to concrete geometric operations
happens at runtime in the repair package; this schema only commits to the
*shape* of the metadata, not to any particular vocabulary.
"""
from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, ConfigDict


class FixBranch(BaseModel):
    """Conditional fix arm — for rules whose remedies depend on the violation context.

    Example: "if the violating shape is a power rail (width ≥ N), do not
    widen — shift the neighbour instead". One ``FixBranch`` captures one
    such if-then.
    """
    model_config = ConfigDict(frozen=True)

    condition: str                          # human-readable predicate from the PDF
    allowed_action_classes: list[str] = []
    forbidden_action_classes: list[str] = []
    notes: str = ""


class FixMetadata(BaseModel):
    """LLM-extracted fix guidance for one rule.

    ``allowed_action_classes`` / ``forbidden_action_classes`` are opaque tag
    sets — the binding to the fix engine's action vocabulary happens at
    runtime via lookup. Keeping them as strings lets us iterate on the
    vocabulary without touching the schema or re-running ingestion.
    """
    model_config = ConfigDict(frozen=True)

    intent: str = ""                        # one-line "why this rule exists"
    allowed_action_classes: list[str] = []
    forbidden_action_classes: list[str] = []
    affected_layers: list[str] = []         # layers the fix may touch
    branches: list[FixBranch] = []
    notes: str = ""                         # free-text PDF guidance
    pdf_page: Optional[int] = None          # best-effort page reference for human review
