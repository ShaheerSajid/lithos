"""Validate the lithos cell-synthesis pipeline against the sky130A PDK.

Loads the metadata + bootstrap YAMLs in `pdks/sky130A/`, seeds an
in-memory :class:`lithos_core.RuleDB` with the public sky130 design-rule
thresholds, then synthesises every shipped cell template via
``lithos_layout.synthesize_cell``. Optionally runs the KLayout backend
against the real ``sky130A_mr.drc`` deck.

Usage::

    PYTHONPATH=<lithos workspace paths> \\
        python scripts/validate_sky130.py [--drc] [--out DIR]

Default ``--out`` is ``/tmp/lithos_sky130``. With ``--drc``, the script
sets ``PDK_ROOT=/usr/local/share/pdk`` if it isn't already exported and
runs the KLayout DRC backend on each emitted GDS.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from lithos_core import (
    Constraint,
    ConstraintBranch,
    EnclosureCheck,
    LayerRef,
    Rule,
    RuleDB,
    SpacingCheck,
    WidthCheck,
)
from lithos_core.metadata import load_metadata
from lithos_layout         import BootstrapRules, load_bootstrap_mapping, synthesize_cell


# ── Templates that ship with lithos (all 12) ────────────────────────────────

TEMPLATES = (
    "inverter", "nand2", "nand3", "nor2", "nor3",
    "aoi21", "oai21", "buffer", "row_driver",
    "bit_cell_6t", "dido", "tap_cell",
)


# ── sky130A rule thresholds (public DRM values) ─────────────────────────────
# (rule_code, kind, threshold_um) tuples; kind ∈ {"W","S","E"} (Width /
# Spacing / Enclosure). Width applies to layer-self; Spacing applies to
# layer self-spacing; Enclosure is generic — we just need ANY value to
# satisfy the RuleDB's threshold extraction.

_RULES: tuple[tuple[str, str, str, float], ...] = (
    # code           kind  layer      threshold (µm)
    # ── poly ──
    ("poly.1a",      "W",  "poly",    0.150),
    ("poly.2",       "S",  "poly",    0.210),
    ("poly.7",       "E",  "poly",    0.250),
    ("poly.8",       "E",  "poly",    0.130),

    # ── diff ──
    ("difftap.1",    "W",  "diff",    0.150),
    ("difftap.2",    "W",  "diff",    0.420),
    ("difftap.2b",   "E",  "diff",    0.250),
    ("difftap.3",    "S",  "diff",    0.270),
    ("difftap.9",    "S",  "diff",    0.340),
    ("difftap.11",   "S",  "diff",    0.130),

    # ── contact (licon1) ──
    ("licon.1",      "W",  "contact", 0.170),
    ("licon.2",      "S",  "contact", 0.170),
    ("licon.5a",     "E",  "contact", 0.040),
    ("licon.5c",     "E",  "contact", 0.060),
    ("licon.7",      "E",  "contact", 0.120),
    ("licon.8",      "E",  "contact", 0.050),
    ("licon.8a",     "E",  "contact", 0.080),

    # ── m0 (li1) ──
    ("li.1",         "W",  "m0",      0.170),
    ("li.2",         "S",  "m0",      0.170),
    ("li.5",         "E",  "m0",      0.000),
    ("li.5a",        "E",  "m0",      0.080),
    ("li.6",         "W",  "m0",      0.0561),    # area (µm²) packed as threshold

    # ── via_m0_m1 (mcon) ──
    ("mcon.1",       "W",  "via_m0_m1", 0.170),
    ("mcon.2",       "S",  "via_m0_m1", 0.190),

    # ── m1 (met1) ──
    ("m1.1",         "W",  "m1",      0.140),
    ("m1.2",         "S",  "m1",      0.140),
    ("m1.4",         "E",  "m1",      0.030),
    ("m1.5",         "E",  "m1",      0.060),
    ("m1.6",         "W",  "m1",      0.083),

    # ── via_m1_m2 (via1) ──
    ("via.1a",       "W",  "via_m1_m2", 0.150),
    ("via.2",        "S",  "via_m1_m2", 0.170),
    ("via.4a",       "E",  "via_m1_m2", 0.055),
    ("via.5a",       "E",  "via_m1_m2", 0.085),

    # ── m2 (met2) ──
    ("m2.1",         "W",  "m2",      0.140),
    ("m2.2",         "S",  "m2",      0.140),
    ("m2.4",         "E",  "m2",      0.055),
    ("m2.5",         "E",  "m2",      0.085),
    ("m2.6",         "W",  "m2",      0.0676),

    # ── nwell ──
    ("nwell.1",      "W",  "nwell",   0.840),
    ("nwell.2",      "S",  "nwell",   1.270),
    ("nwell.5",      "E",  "nwell",   0.180),

    # ── implant (nsdm/psdm) ──
    ("nsd.1",        "W",  "nimplant", 0.380),
    ("nsd.2",        "S",  "nimplant", 0.380),
    ("nsd.5a",       "E",  "nimplant", 0.125),
)


def _seed_rule_db(db_path: Path) -> RuleDB:
    db = RuleDB(db_path)
    db.open()
    db.set_pdk(name="sky130A", version="1.0.5",
               ingested_at="2026-05-20T00:00:00Z")
    for code, kind, layer, thr in _RULES:
        if kind == "W":
            check = WidthCheck(target=LayerRef(name=layer),
                               op=">=", threshold_um=thr)
        elif kind == "S":
            check = SpacingCheck(layer_a=LayerRef(name=layer),
                                 op=">=", threshold_um=thr)
        else:                                          # "E"
            check = EnclosureCheck(inner=LayerRef(name=layer),
                                   outer=LayerRef(name=layer),
                                   op=">=", threshold_um=thr)
        db.upsert_rule(Rule(
            code=code, category="geom", usage_class="geometry_primitive",
            constraint=Constraint(branches=[ConstraintBranch(check=check)]),
        ))
    return db


def _build_rules(out: Path) -> BootstrapRules:
    pdk_dir = Path(__file__).resolve().parent.parent / "pdks" / "sky130A"
    metadata = load_metadata(pdk_dir / "metadata.yaml")
    mapping  = load_bootstrap_mapping(pdk_dir / "bootstrap.yaml")
    db = _seed_rule_db(out / "rules.sky130A.db")
    return BootstrapRules(metadata, db, mapping)


def _maybe_drc_runner(rules: BootstrapRules, enable_drc: bool):
    if not enable_drc:
        return None
    os.environ.setdefault("PDK_ROOT", "/usr/local/share/pdk")
    deck = rules.metadata.drc_decks.get("klayout")
    if deck is None or not Path(os.path.expandvars(str(deck))).exists():
        print(f"  [drc] klayout deck not found at {deck}; skipping DRC.")
        return None
    from lithos_drc import KLayoutDRCRunner
    return KLayoutDRCRunner(rules.metadata)


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--drc", action="store_true",
                    help="Run KLayout DRC against sky130A_mr.drc after synthesis.")
    ap.add_argument("--out", type=Path, default=Path("/tmp/lithos_sky130"),
                    help="Output directory for GDS files (default: /tmp/lithos_sky130).")
    args = ap.parse_args(argv)

    args.out.mkdir(parents=True, exist_ok=True)
    rules  = _build_rules(args.out)
    runner = _maybe_drc_runner(rules, args.drc)

    print(f"lithos validation against sky130A — output dir: {args.out}")
    print(f"  DRC: {'KLayout backend on sky130A_mr.drc' if runner else 'disabled'}")
    print()

    n_ok      = 0
    n_failed  = 0
    n_dirty   = 0
    total_v   = 0
    for name in TEMPLATES:
        try:
            result = synthesize_cell(name, rules, {"w": 0.52, "l": 0.15},
                                     drc_runner=runner)
        except Exception as exc:                       # pragma: no cover
            print(f"  {name:14s} FAIL  {type(exc).__name__}: {exc}")
            n_failed += 1
            continue

        gds = args.out / f"{name}.gds"
        result.component.write_gds(str(gds))
        polys = result.component.get_polygons(by="tuple")
        n_polys = sum(len(v) for v in polys.values())
        n_v = len(result.violations)
        status = "OK  " if n_v == 0 else f"DRTY"
        if n_v: n_dirty += 1
        else:   n_ok    += 1
        total_v += n_v
        print(f"  {name:14s} {status}  {gds.stat().st_size:>6d}B  "
              f"polys={n_polys:3d}  violations={n_v}")

    print()
    print(f"summary: {n_ok}/{len(TEMPLATES)} clean, "
          f"{n_dirty} with violations, {n_failed} failed; "
          f"{total_v} total violations.")
    return 0 if n_failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
