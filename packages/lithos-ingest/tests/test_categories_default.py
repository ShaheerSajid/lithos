"""The shipped default categories config loads and covers the major rule sets."""
from __future__ import annotations

from pathlib import Path

import pytest

from lithos_core.categories import CategoryConfig, load_categories


DEFAULT = Path(__file__).resolve().parents[1] / "lithos_ingest" / "data" / "categories_default.yaml"


def test_default_categories_file_loads():
    cfg = load_categories(DEFAULT)
    assert isinstance(cfg, CategoryConfig)
    assert cfg.default_category == "unknown"


def test_default_categories_cover_major_rule_sets():
    """The shipped default must claim every category cell generation needs."""
    cfg = load_categories(DEFAULT)
    names = {c.name for c in cfg.categories}
    assert names == {"poly", "diffusion", "well", "metal", "via"}, names


@pytest.mark.parametrize("code,expected", [
    # poly
    ("PO.W.1",         "poly"),
    ("POLY.S.2",       "poly"),
    ("GA.W.1",         "poly"),
    # diffusion
    ("OD.W.1",         "diffusion"),
    ("DIFF.S.2",       "diffusion"),
    ("RX.E.1",         "diffusion"),
    # well
    ("NW.W.1",         "well"),
    ("PW.S.1",         "well"),
    ("DNW.W.1",        "well"),
    # metal
    ("M1.W.1",         "metal"),
    ("M9.S.3",         "metal"),
    ("AP.W.1",         "metal"),
    # via
    ("CO.W.1",         "via"),
    ("VIA1.S.1",       "via"),
    ("V3.W.2",         "via"),
])
def test_classification(code: str, expected: str):
    cfg = load_categories(DEFAULT)
    assert cfg.category_for(code) == expected
