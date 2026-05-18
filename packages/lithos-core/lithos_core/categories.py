"""lithos_core.categories — user-configurable rule category definitions.

A *category* is a foundry-topical grouping of DRC rules (e.g. ``"poly"``,
``"metal_low"``, ``"via"``, ``"antenna"``, ``"density"``). Categories drive
**targeted ingestion**: instead of asking the LLM to read a 2000-page rule
manual end-to-end, the user enables the categories they care about and the
ingestion pipeline only processes matching rules. New categories can be
added by editing the per-PDK ``categories.yaml`` — no code change required.

Mapping a rule to a category is by code prefix(es) and/or a regex pattern,
plus an optional PDF section heading pattern so the chunker can scope its
search. The first **enabled** category to claim a rule code (in ascending
``priority`` order) wins. A rule code with no matching enabled category
gets ``"unknown"`` in the DB.

Schema (sky130A example)::

    default_category: unknown
    categories:
      - name: poly
        code_prefixes: ["poly.", "p.", "P."]
        pdf_section_pattern: '^Polysilicon Layer.*'
        enabled: true
        priority: 10
        description: Gate poly width, spacing, endcap.

      - name: metal_low
        code_prefixes: ["li.", "li1.", "met1.", "M1.", "met2.", "M2."]
        enabled: true
        priority: 20

      - name: antenna
        code_prefixes: ["antenna.", "ant.", "A."]
        enabled: false      # defer until generator hits antenna failures
        priority: 90
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Optional

import yaml
from pydantic import BaseModel, ConfigDict, Field


class CategoryDef(BaseModel):
    """One category definition. Either ``code_prefixes`` or ``code_pattern``
    (or both) must be set; matching unions the two.
    """
    model_config = ConfigDict(frozen=True)

    name: str
    code_prefixes:       list[str] = Field(default_factory=list)
    code_pattern:        Optional[str] = None        # regex
    pdf_section_pattern: Optional[str] = None        # regex on heading text
    enabled:             bool = True
    priority:            int  = 100                   # lower = checked first
    description:         str  = ""

    def matches(self, rule_code: str) -> bool:
        """Return True if this category claims ``rule_code``."""
        for pfx in self.code_prefixes:
            if rule_code.startswith(pfx):
                return True
        if self.code_pattern is not None:
            if re.match(self.code_pattern, rule_code):
                return True
        return False


class CategoryConfig(BaseModel):
    """A PDK's category config. Use :meth:`match` to look up a rule code."""
    model_config = ConfigDict(frozen=True)

    categories:       list[CategoryDef]
    default_category: str = "unknown"
    config_path:      Optional[Path] = None

    def enabled(self) -> list[CategoryDef]:
        """Enabled categories, sorted by priority (lowest first)."""
        return sorted(
            (c for c in self.categories if c.enabled),
            key=lambda c: c.priority,
        )

    def match(self, rule_code: str) -> Optional[CategoryDef]:
        """Return the first enabled category to claim ``rule_code``, or None."""
        for cat in self.enabled():
            if cat.matches(rule_code):
                return cat
        return None

    def category_for(self, rule_code: str) -> str:
        """Return the category name for ``rule_code``, or :attr:`default_category`."""
        hit = self.match(rule_code)
        return hit.name if hit is not None else self.default_category

    def by_name(self, name: str) -> Optional[CategoryDef]:
        for cat in self.categories:
            if cat.name == name:
                return cat
        return None


def load_categories(path: Path | str) -> CategoryConfig:
    """Load a category config from a YAML file."""
    path = Path(path)
    with open(path) as f:
        data = yaml.safe_load(f) or {}
    return CategoryConfig(
        categories       = [CategoryDef(**c) for c in (data.get("categories") or [])],
        default_category = str(data.get("default_category", "unknown")),
        config_path      = path,
    )
