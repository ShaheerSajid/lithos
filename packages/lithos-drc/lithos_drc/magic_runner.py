"""lithos_drc.magic_runner — Magic batch-mode DRC backend.

Drives Magic in batch mode against the PDK's ``.magicrc`` and ``.tech``
files; flattens the input GDS first (gdsfactory wraps cells in
``$$$CONTEXT_INFO$$$`` which Magic rejects); parses Magic's ``drc listall
why`` textual output into :class:`DRCViolation` instances.

Configuration
-------------
The Magic tech file path comes from :class:`PDKMetadata`'s
``drc_decks["magic"]`` (typically points at ``<pdk>.tech``). The
``.magicrc`` is located next to it or under ``$PDK_ROOT``.
"""
from __future__ import annotations

import os
import re
import subprocess
import tempfile
import warnings
from pathlib import Path
from typing import Optional

from lithos_core.metadata import PDKMetadata

from lithos_drc.base import DRCRunner, DRCViolation


# ── Layer / rule heuristics ──────────────────────────────────────────────────

_LAYER_FROM_RULE = re.compile(
    r"(nwell|diff|tap|poly|licon|li1|mcon|met[1-9]|via[1-9]|npc|nsdm|psdm)",
    re.IGNORECASE,
)


def _guess_layer(rule_text: str) -> str:
    """Best-effort logical layer name from a Magic rule description."""
    m = _LAYER_FROM_RULE.search(rule_text)
    return m.group(1).lower() if m else ""


def _guess_rule_id(rule_text: str) -> str:
    """Extract a compact rule ID like ``poly.2`` from Magic's description.

    Magic descriptions look like::

        P-diff distance to N-tap must be >= 0.130um (difftap.2)

    The parenthesised suffix is the rule code. Fallback: first
    letter-then-number-ish token.
    """
    m = re.search(r"\(([a-zA-Z0-9_.]+)\)\s*$", rule_text)
    if m:
        return m.group(1)
    m2 = re.search(r"([a-zA-Z]+[\d.]+)", rule_text)
    return m2.group(1) if m2 else "unknown"


# ── Runner ───────────────────────────────────────────────────────────────────

class MagicDRCRunner(DRCRunner):
    """DRC backend driving Magic in batch mode.

    Parameters
    ----------
    metadata
        PDK metadata; ``metadata.drc_decks["magic"]`` is the tech file path.
    magic_exe
        Path or name of the ``magic`` executable.
    """

    def __init__(self, metadata: PDKMetadata, *, magic_exe: str = "magic"):
        super().__init__(metadata)
        self.magic_exe = magic_exe

    @property
    def tool_name(self) -> str:
        return "magic"

    def is_available(self) -> bool:
        try:
            subprocess.run(
                [self.magic_exe, "--version"],
                capture_output=True, timeout=10,
            )
            return True
        except (FileNotFoundError, subprocess.TimeoutExpired):
            return False

    # ── rcfile discovery ────────────────────────────────────────────────

    def _find_rcfile(self) -> Path:
        """Locate the Magic ``.magicrc`` for the active PDK.

        Resolution order:
          1. ``LITHOS_MAGIC_RCFILE`` env var, if it points at a file.
          2. ``<PDK_ROOT>/<pdk_name>/libs.tech/magic/<pdk_name>.magicrc``.
          3. Next to the ``.tech`` file from ``metadata.drc_decks["magic"]``.
        """
        override = os.environ.get("LITHOS_MAGIC_RCFILE")
        if override:
            p = Path(override)
            if p.is_file():
                return p

        pdk_root = os.environ.get("PDK_ROOT", "/usr/local/share/pdk")
        pdk_name = self.metadata.name
        rc = Path(pdk_root) / pdk_name / "libs.tech" / "magic" / f"{pdk_name}.magicrc"
        if rc.is_file():
            return rc

        tech = self.metadata.drc_decks.get("magic")
        if tech is not None:
            rc2 = Path(tech).parent / f"{pdk_name}.magicrc"
            if rc2.is_file():
                return rc2

        raise FileNotFoundError(
            f"Cannot find Magic .magicrc for PDK {pdk_name!r}. "
            f"Export PDK_ROOT or LITHOS_MAGIC_RCFILE, or set "
            f"PDKMetadata.drc_decks['magic'] to the tech-file path "
            f"(magicrc lives alongside)."
        )

    # ── Execution ────────────────────────────────────────────────────────

    def run(
        self,
        gds_path: Path,
        cell_name: Optional[str] = None,
    ) -> list[DRCViolation]:
        gds_path = Path(gds_path).resolve()
        if not gds_path.exists():
            raise FileNotFoundError(f"GDS file not found: {gds_path}")

        rcfile = self._find_rcfile()
        if cell_name is None:
            cell_name = gds_path.stem

        with tempfile.TemporaryDirectory(prefix="lithos_magic_") as tmpdir:
            flat_gds = Path(tmpdir) / "flat.gds"
            cell_name = _flatten_gds(gds_path, flat_gds, cell_name)
            output_file = Path(tmpdir) / "drc_results.txt"
            tcl_script  = Path(tmpdir) / "run_drc.tcl"
            tcl_script.write_text(_generate_tcl(
                gds_path    = str(flat_gds),
                cell_name   = cell_name,
                output_file = str(output_file),
            ))

            env = os.environ.copy()
            env["MAGTYPE"] = "mag"

            proc = subprocess.run(
                [
                    self.magic_exe, "-dnull", "-noconsole",
                    "-rcfile", str(rcfile), str(tcl_script),
                ],
                env=env, cwd=tmpdir,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                universal_newlines=True,
                timeout=300,
            )
            if proc.returncode != 0:
                warnings.warn(
                    f"Magic exited with code {proc.returncode}. "
                    f"stderr: {proc.stderr[:500]}",
                    stacklevel=2,
                )
            if not output_file.exists():
                warnings.warn(
                    f"Magic DRC produced no output file. "
                    f"stdout: {proc.stdout[:500]}",
                    stacklevel=2,
                )
                return []

            return parse_magic_output(output_file.read_text())


# ── GDS flattening (pure helper, requires gdstk) ─────────────────────────────

def _flatten_gds(src: Path, dst: Path, cell_name: str) -> str:
    """Flatten ``src`` into a single named cell at ``dst``. Returns the
    sanitised cell name actually used.

    gdsfactory wraps cells in ``$$$CONTEXT_INFO$$$`` which Magic can't
    load; we re-emit a clean single-cell GDS to dodge that.
    """
    try:
        import gdstk                            # type: ignore[import-not-found]
    except ImportError as exc:                  # pragma: no cover - install hint
        raise ImportError(
            "lithos_drc.magic_runner requires gdstk for GDS flattening. "
            "Install with: pip install gdstk"
        ) from exc

    lib = gdstk.read_gds(str(src))
    target = None
    for c in lib.cells:
        if c.name == cell_name:
            target = c
            break
    if target is None:
        for c in lib.cells:
            if "$$$" not in c.name and c.polygons:
                target = c
                break
    if target is None:
        tops = lib.top_level()
        target = tops[0] if tops else lib.cells[0]
    target.flatten()

    safe = re.sub(r"[^a-zA-Z0-9_]", "_", target.name)
    target.name = safe

    out = gdstk.Library()
    out.add(target)
    out.write_gds(str(dst))
    return safe


# ── Tcl script (pure helper) ─────────────────────────────────────────────────

def _generate_tcl(gds_path: str, cell_name: str, output_file: str) -> str:
    """Generate the batch Tcl script driving Magic's DRC."""
    return f"""\
# Auto-generated Magic DRC script
crashbackups stop
drc euclidean on
drc style drc(full)
drc on
snap internal
gds flatglob *__example_*
gds flatten true
gds read {gds_path}
load {cell_name}
select top cell
expand
drc catchup
set allerrors [drc listall why]
set oscale [cif scale out]
set ofile [open {output_file} w]
puts $ofile "DRC errors for cell {cell_name}"
puts $ofile "--------------------------------------------"
foreach {{whytext rectlist}} $allerrors {{
   puts $ofile ""
   puts $ofile $whytext
   foreach rect $rectlist {{
       set llx [format "%.3f" [expr $oscale * [lindex $rect 0]]]
       set lly [format "%.3f" [expr $oscale * [lindex $rect 1]]]
       set urx [format "%.3f" [expr $oscale * [lindex $rect 2]]]
       set ury [format "%.3f" [expr $oscale * [lindex $rect 3]]]
       puts $ofile "$llx $lly $urx $ury"
   }}
}}
close $ofile
puts "DRC complete: [llength $allerrors] rule(s) with violations"
quit
"""


# ── Output parser (pure, public for testing) ─────────────────────────────────

_COORD_RE = re.compile(
    r"^\s*(-?[\d.]+)\s+(-?[\d.]+)\s+(-?[\d.]+)\s+(-?[\d.]+)\s*$"
)


def parse_magic_output(text: str) -> list[DRCViolation]:
    """Parse Magic's DRC output text into :class:`DRCViolation` list.

    Format::

        DRC errors for cell <name>
        --------------------------------------------

        <rule description text>
        x0 y0 x1 y1
        x0 y0 x1 y1
        ...

        <next rule description>
        ...
    """
    violations: list[DRCViolation] = []
    current_rule = ""
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("DRC errors") or line.startswith("---"):
            continue
        m = _COORD_RE.match(line)
        if m and current_rule:
            x0, y0, x1, y1 = (float(m.group(i)) for i in range(1, 5))
            cx = (x0 + x1) / 2
            cy = (y0 + y1) / 2
            violations.append(DRCViolation(
                rule        = _guess_rule_id(current_rule),
                description = current_rule,
                layer       = _guess_layer(current_rule),
                severity    = "error",
                x           = cx,
                y           = cy,
            ))
        else:
            current_rule = line
    return violations
