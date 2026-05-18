"""lithos_drc.klayout_runner — KLayout batch-mode DRC backend.

Wraps the ``klayout -b -r <deck>`` invocation, reads the resulting
``.lyrdb`` XML report, and emits :class:`DRCViolation` instances whose
``rule`` field is the raw tool-emitted check name. The resolver in
:mod:`lithos_drc.resolver` maps that to the canonical foundry code via
the rule DB's alias table.

Configuration
-------------
The deck path comes from :class:`PDKMetadata`'s ``drc_decks["klayout"]``.
Override at runtime with the ``LITHOS_KLAYOUT_DECK`` environment variable.

KLayout knobs (used by foundry decks like sky130A's ``sky130A_mr.drc``)
are exposed as ``LITHOS_DRC_<KNOB>=<value>`` env vars::

    LITHOS_DRC_FEOL         (default true)
    LITHOS_DRC_BEOL         (default true)
    LITHOS_DRC_OFFGRID      (default true)
    LITHOS_DRC_SEAL         (default false)  # top-level only
    LITHOS_DRC_FLOATING_MET (default false)  # top-level only
    LITHOS_DRC_SRAM_EXCLUDE (default false)
    LITHOS_DRC_THR          (default = os.cpu_count())

These match sky130A's deck-exposed knobs. Other PDK decks ignore the
unknown ``-rd`` arguments harmlessly.
"""
from __future__ import annotations

import os
import subprocess
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Optional

from lithos_core.metadata import PDKMetadata

from lithos_drc.base import DRCRunner, DRCViolation


# Knobs the sky130A foundry deck reads via ``-rd``. Other decks ignore them.
_DEFAULT_KNOBS: dict[str, str] = {
    "feol":         "true",
    "beol":         "true",
    "offgrid":      "true",
    "seal":         "false",
    "floating_met": "false",
    "sram_exclude": "false",
}


class KLayoutDRCRunner(DRCRunner):
    """DRC backend driving KLayout's batch mode.

    Parameters
    ----------
    metadata
        PDK metadata; ``metadata.drc_decks["klayout"]`` is the default deck.
    klayout_exe
        Path or name of the ``klayout`` executable (default ``"klayout"``).
    """

    def __init__(self, metadata: PDKMetadata, *, klayout_exe: str = "klayout"):
        super().__init__(metadata)
        self.klayout_exe = klayout_exe

    @property
    def tool_name(self) -> str:
        return "klayout"

    def is_available(self) -> bool:
        try:
            subprocess.run(
                [self.klayout_exe, "-v"],
                capture_output=True, timeout=10,
            )
            return True
        except (FileNotFoundError, subprocess.TimeoutExpired):
            return False

    def _resolve_deck(self) -> Path:
        """Return the deck path, honouring ``LITHOS_KLAYOUT_DECK`` overrides."""
        override = os.environ.get("LITHOS_KLAYOUT_DECK")
        if override:
            p = Path(override)
            if p.is_file():
                return p
        deck = self.deck_path()
        if not deck.is_file():
            raise FileNotFoundError(
                f"KLayout DRC deck not found at {deck}. "
                f"Set the path in PDKMetadata.drc_decks['klayout'] or "
                f"export LITHOS_KLAYOUT_DECK=<path>."
            )
        return deck

    def _build_command(
        self,
        gds_path: Path,
        deck:     Path,
        report:   Path,
        cell_name: Optional[str],
    ) -> list[str]:
        """Compose the ``klayout -b`` invocation. Public for testability."""
        knobs = dict(_DEFAULT_KNOBS)
        for k in list(knobs):
            env_v = os.environ.get(f"LITHOS_DRC_{k.upper()}")
            if env_v is not None:
                knobs[k] = env_v
        thr = os.environ.get("LITHOS_DRC_THR") or str(os.cpu_count() or 4)

        cmd: list[str] = [
            self.klayout_exe, "-b",
            "-r",  str(deck),
            "-rd", f"input={gds_path}",
            "-rd", f"report={report}",
        ]
        for k, v in knobs.items():
            cmd += ["-rd", f"{k}={v}"]
        cmd += ["-rd", f"thr={thr}"]
        if cell_name:
            # Different decks use different names for this knob; pass both
            # so the deck picks the one it understands.
            cmd += ["-rd", f"top_cell={cell_name}", "-rd", f"topcell={cell_name}"]
        return cmd

    def run(
        self,
        gds_path: Path,
        cell_name: Optional[str] = None,
    ) -> list[DRCViolation]:
        gds_path = Path(gds_path).resolve()
        deck     = self._resolve_deck()

        with tempfile.TemporaryDirectory(prefix="lithos_klayout_") as tmpdir:
            report = Path(tmpdir) / "violations.lyrdb"
            cmd    = self._build_command(gds_path, deck, report, cell_name)
            result = subprocess.run(
                cmd, capture_output=True, text=True, timeout=300,
            )
            if result.returncode != 0:
                raise RuntimeError(
                    f"KLayout DRC failed (exit {result.returncode}):\n{result.stderr}"
                )
            if not report.exists():
                return []
            return parse_lyrdb(report)


# ── .lyrdb XML parser (pure, public for testing) ─────────────────────────────

_DBU = 0.001
"""KLayout default: 1 dbu = 1 nm = 0.001 µm. Used when point text parses
as an integer (dbu); when it parses as float it's already in µm."""


def parse_lyrdb(path: Path) -> list[DRCViolation]:
    """Parse a KLayout ``.lyrdb`` XML report into :class:`DRCViolation` list."""
    tree = ET.parse(path)
    root = tree.getroot()

    # Build category → description map (deck-author description text).
    cat_desc: dict[str, str] = {}
    for cat in root.findall(".//category"):
        name = cat.findtext("name", "").strip().strip("'\"")
        desc = cat.findtext("description", "") or ""
        cat_desc[name] = desc

    violations: list[DRCViolation] = []
    for item in root.findall(".//item"):
        rule = (item.findtext("category", "") or "").strip().strip("'\"")
        desc = cat_desc.get(rule, "")
        x, y = _centroid_from_item(item)
        value = _measured_value(item)
        violations.append(DRCViolation(
            rule=rule, description=desc, x=x, y=y, value=value,
        ))
    return violations


def _measured_value(item: ET.Element) -> Optional[float]:
    """Pull a single measured-value µm out of an RDB item if present.

    KLayout encodes the measurement in one of two ways: a plain numeric
    text node inside ``<values>/<value>``, or an ``edge-pair:`` geometry
    string from which we recover the minimum distance.
    """
    for val_el in item.findall("values/value"):
        text = (val_el.text or "").strip()
        try:
            return float(text)
        except ValueError:
            pass
        if text.lower().startswith("edge-pair:"):
            d = _edge_pair_distance(text)
            if d is not None:
                return d
    return None


def _centroid_from_item(item: ET.Element) -> tuple[float, float]:
    """Return (x_um, y_um) centroid for an RDB item's geometry.

    Two storage formats appear depending on KLayout version / deck style:

    1. Dedicated child element: ``<polygon>`` or ``<edge-pair>``.
    2. Inline in ``<values>/<value>`` as ``edge-pair: (...)|(...)`` or
       ``polygon: (...)``.
    """
    poly_el = item.find("polygon")
    if poly_el is not None and poly_el.text:
        pts = _parse_pts(poly_el.text)
        if pts:
            return _centroid(pts)

    ep_el = item.find("edge-pair")
    if ep_el is not None and ep_el.text:
        pts: list[tuple[float, float]] = []
        for seg in ep_el.text.split("/"):
            pts.extend(_parse_pts(seg))
        if pts:
            return _centroid(pts)

    for val_el in item.findall("values/value"):
        text = val_el.text or ""
        pts = _parse_geometry_text(text)
        if pts:
            return _centroid(pts)

    return 0.0, 0.0


def _parse_geometry_text(text: str) -> list[tuple[float, float]]:
    """Parse geometry from value text like ``edge-pair: (...)|(...)``."""
    for prefix in ("edge-pair:", "edge_pair:", "polygon:", "box:", "edge:"):
        if text.lower().startswith(prefix):
            text = text[len(prefix):]
            break
    else:
        return []
    pts: list[tuple[float, float]] = []
    for seg in text.replace("/", "|").split("|"):
        pts.extend(_parse_pts(seg))
    return pts


def _parse_pts(s: str) -> list[tuple[float, float]]:
    """Parse ``(x1,y1;x2,y2;...)`` → ``[(x_um, y_um), ...]``.

    Integer tokens are treated as dbu (multiply by ``_DBU``); float tokens
    are µm verbatim.
    """
    s = s.strip().strip("()")
    pts: list[tuple[float, float]] = []
    for token in s.split(";"):
        token = token.strip()
        if "," not in token:
            continue
        xs, ys = token.split(",", 1)
        try:
            pts.append((int(xs) * _DBU, int(ys) * _DBU))
        except ValueError:
            try:
                pts.append((float(xs), float(ys)))
            except ValueError:
                pass
    return pts


def _centroid(pts: list[tuple[float, float]]) -> tuple[float, float]:
    n = len(pts)
    return sum(p[0] for p in pts) / n, sum(p[1] for p in pts) / n


def _edge_pair_distance(text: str) -> Optional[float]:
    """Minimum distance between the two edges in an ``edge-pair: ...`` string.

    For spacing/width violations this equals the measured value reported
    by KLayout.
    """
    body = text.split(":", 1)[-1].strip()
    parts = body.replace("/", "|").split("|")
    if len(parts) < 2:
        return None
    edge_a = _parse_pts(parts[0])
    edge_b = _parse_pts(parts[1])
    if not edge_a or not edge_b:
        return None
    min_d = float("inf")
    for ax, ay in edge_a:
        for bx, by in edge_b:
            d = ((ax - bx) ** 2 + (ay - by) ** 2) ** 0.5
            if d < min_d:
                min_d = d
    return round(min_d, 6) if min_d < float("inf") else None
