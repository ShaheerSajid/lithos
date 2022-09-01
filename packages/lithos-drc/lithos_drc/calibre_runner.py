"""lithos_drc.calibre_runner — Calibre DRC backend invoked via Docker.

Calibre isn't installed on most hosts; lithos calls it inside the
user's docker container (typically built with ``CALIBREHOME`` under
``/tools/mentor/...``). The runner:

1. Stages the GDS into a workdir that the container can see (under
   ``~/.cache/lithos_drc/calibre/``; ``$HOME`` is mounted into the
   container in the standard setup).
2. Writes a wrapper SVRF that points at the staged GDS, requests an
   ASCII RDB report, and ``INCLUDE``\\ s the foundry deck.
3. Runs ``docker run --rm`` (or ``docker exec`` against an existing
   container) with the appropriate license-env + network setup.
4. Parses the ASCII RDB into :class:`DRCViolation` objects.

Configuration env vars (all optional)::

    LITHOS_CALIBRE_DECK              overrides metadata.drc_decks["calibre"]
    LITHOS_CALIBRE_IMAGE             docker image (default: localhost/tools:latest)
    LITHOS_CALIBRE_NETWORK           docker network (default: tools-isolated)
    LITHOS_CALIBRE_CSHRC             rc file to source inside the container
                                     to put ``calibre`` on PATH
                                     (default: /root/.cshrc)
    LITHOS_CALIBRE_EXEC_CONTAINER    docker container name to ``docker exec``
                                     against instead of running a fresh one
    LITHOS_CALIBRE_WORKDIR           override staging dir (default ~/.cache/...)
"""
from __future__ import annotations

import os
import re
import shutil
import subprocess
import tempfile
import uuid
from pathlib import Path
from typing import Optional

from lithos_core.metadata import PDKMetadata

from lithos_drc.base import DRCRunner, DRCViolation


# ── Defaults ────────────────────────────────────────────────────────────────

_DEFAULT_IMAGE   = "localhost/tools:latest"
_DEFAULT_NETWORK = "tools-isolated"
_DEFAULT_CSHRC   = "/root/.cshrc"

_RVDB_ASCII = "output.results"
_SUMMARY    = "output.summary"
_WRAPPER    = "runset.svrf"


# ── Runner ──────────────────────────────────────────────────────────────────

class CalibreDRCRunner(DRCRunner):
    """DRC backend driving Calibre through Docker.

    The runner assumes the user has a tools container available that
    has ``/tools`` mounted at the same path as the host and the
    license server reachable on its network. The defaults match the
    setup produced by the user's ``runtools-secure.sh``.
    """

    def __init__(
        self,
        metadata: PDKMetadata,
        *,
        image:          str | None = None,
        network:        str | None = None,
        cshrc:          str | None = None,
        exec_container: str | None = None,
    ):
        super().__init__(metadata)
        self.image          = image          or os.environ.get("LITHOS_CALIBRE_IMAGE",           _DEFAULT_IMAGE)
        self.network        = network        or os.environ.get("LITHOS_CALIBRE_NETWORK",         _DEFAULT_NETWORK)
        self.cshrc          = cshrc          or os.environ.get("LITHOS_CALIBRE_CSHRC",           _DEFAULT_CSHRC)
        self.exec_container = exec_container or os.environ.get("LITHOS_CALIBRE_EXEC_CONTAINER", "")

    @property
    def tool_name(self) -> str:
        return "calibre"

    def is_available(self) -> bool:
        """True if the docker CLI is reachable (the actual calibre binary
        lives inside a container we don't probe here)."""
        try:
            subprocess.run(["docker", "version"],
                           capture_output=True, timeout=10)
            return True
        except (FileNotFoundError, subprocess.TimeoutExpired):
            return False

    def _resolve_deck(self) -> Path:
        """Return the deck path inside the container.

        We need a path the container can see. ``$LITHOS_CALIBRE_DECK``
        or ``metadata.drc_decks["calibre"]`` should be an absolute host
        path under a mounted directory (typically ``/home/...`` or
        ``/tools/...``); we don't translate paths.
        """
        override = os.environ.get("LITHOS_CALIBRE_DECK")
        if override:
            p = Path(override)
            if p.is_file():
                return p
        deck = self.deck_path()
        if not deck.is_file():
            raise FileNotFoundError(
                f"Calibre DRC deck not found at {deck}. "
                f"Set PDKMetadata.drc_decks['calibre'] or "
                f"export LITHOS_CALIBRE_DECK=<absolute deck path>."
            )
        return deck

    def _workdir_root(self) -> Path:
        """Return the staging-dir parent (mounted into the container)."""
        override = os.environ.get("LITHOS_CALIBRE_WORKDIR")
        if override:
            p = Path(override)
            p.mkdir(parents=True, exist_ok=True)
            return p
        p = Path.home() / ".cache" / "lithos_drc" / "calibre"
        p.mkdir(parents=True, exist_ok=True)
        return p

    def _stage(self, gds_path: Path, deck: Path, cell_name: Optional[str]) -> Path:
        """Copy the GDS into a fresh workdir and write the wrapper SVRF.

        Returns the workdir path. Caller is responsible for removing it.
        """
        workdir = self._workdir_root() / uuid.uuid4().hex[:12]
        workdir.mkdir(parents=True)

        # 1. Stage the GDS so the docker container can read it.
        staged_gds = workdir / "input.gds"
        shutil.copyfile(gds_path, staged_gds)

        # 2. Build the runset by COPYING the deck and substituting its
        #    placeholders. Foundry decks ship with sentinel paths
        #    (``"GDSFILENAME"`` / ``"TOPCELLNAME"`` / ``"DRC_RES.db"``
        #    in TSMC's case) that the user is supposed to edit per run.
        #    INCLUDE-wrapper doesn't work for decks that declare their
        #    own LAYOUT SYSTEM — Calibre errors on duplicates.
        primary = cell_name or _detect_top_cell(staged_gds) or "TOP"
        deck_text = deck.read_text()
        substitutions = {
            'LAYOUT PATH "GDSFILENAME"':       f'LAYOUT PATH "{staged_gds}"',
            'LAYOUT PRIMARY "TOPCELLNAME"':    f'LAYOUT PRIMARY "{primary}"',
            'DRC RESULTS DATABASE "DRC_RES.db"':
                f'DRC RESULTS DATABASE "{workdir / _RVDB_ASCII}" ASCII',
        }
        for needle, replacement in substitutions.items():
            deck_text = deck_text.replace(needle, replacement)
        wrapper = workdir / _WRAPPER
        wrapper.write_text(deck_text)
        return workdir

    def _build_command(self, workdir: Path) -> list[str]:
        """Compose the docker invocation. Public for testability.

        Uses ``docker exec`` when ``LITHOS_CALIBRE_EXEC_CONTAINER`` is
        set (faster — no container startup), otherwise ``docker run``.

        Calibre and the license env (``MGLS_LICENSE_FILE``) come from
        ``self.cshrc`` (typically ``/root/.cshrc`` inside the tools
        image). We source it explicitly because non-interactive csh
        doesn't read it on its own.
        """
        wrapper = workdir / _WRAPPER
        # csh: source the env, then run calibre. The shell's exit code
        # equals the last command's exit code (calibre's), which is
        # what subprocess.run cares about.
        csh_cmd = (
            f"source {self.cshrc}; "
            f"calibre -drc -hier -turbo {wrapper}"
        )

        if self.exec_container:
            return ["docker", "exec",
                    self.exec_container,
                    "csh", "-c", csh_cmd]

        # Fresh container per run. Mount $HOME and /tools so the
        # workdir + deck + binary are all reachable. ``--rm`` keeps the
        # host clean.
        home = str(Path.home())
        return [
            "docker", "run", "--rm",
            "--network", self.network,
            "-v", f"{home}:{home}",
            "-v", "/tools:/tools",
            self.image,
            "csh", "-c", csh_cmd,
        ]

    def run(
        self,
        gds_path: Path,
        cell_name: Optional[str] = None,
    ) -> list[DRCViolation]:
        gds_path = Path(gds_path).resolve()
        deck     = self._resolve_deck()
        workdir  = self._stage(gds_path, deck, cell_name)
        try:
            cmd = self._build_command(workdir)
            result = subprocess.run(
                cmd, capture_output=True, text=True, timeout=600,
            )
            # Calibre returns 0 even when violations exist; non-zero means
            # the tool itself failed (license, syntax, missing deck...).
            if result.returncode != 0:
                raise RuntimeError(
                    f"Calibre DRC failed (exit {result.returncode}):\n"
                    f"stdout:\n{result.stdout}\n"
                    f"stderr:\n{result.stderr}"
                )
            rvdb = workdir / _RVDB_ASCII
            if not rvdb.exists():
                return []
            return parse_rvdb_ascii(rvdb)
        finally:
            if not os.environ.get("LITHOS_CALIBRE_KEEP_WORKDIR"):
                shutil.rmtree(workdir, ignore_errors=True)


# ── ASCII RVDB parser (pure, public for testing) ────────────────────────────

# Calibre ASCII RVDB format. One rule block per checked rule.
#
#   $$$CONTEXT_INFO$$$ <version>             ← file header
#   RULE.NAME                                ← rule name (line by itself)
#   <count> <orig_count> <def_lines> <date>  ← stats
#   RULE.NAME { @ description @              ← rule definition opens
#     ...body...                             ← def_lines total counting the `{` line
#   }                                        ← rule definition closes
#   p <id> <vertex_count>                    ← polygon record (one violation)
#   <x1_dbu> <y1_dbu>                        ← vertex_count lines, integer dbu (nm)
#   <x2_dbu> <y2_dbu>
#   ...
#   e <id> 2 <x1_dbu> <y1_dbu> <x2_dbu> <y2_dbu>   ← edge record (rare)
#   ...
#   <next RULE.NAME>
#
# Rules with ``count == 0`` have no p/e records — only the definition.
# Coordinates are integer database units (1 dbu = 1 nm = 0.001 µm).

_RULE_NAME_RE = re.compile(r"^[A-Za-z][A-Za-z0-9_\.\-]*$")
_DBU_TO_UM    = 0.001


def parse_rvdb_ascii(path: Path) -> list[DRCViolation]:
    """Parse a Calibre ASCII RVDB into :class:`DRCViolation` objects.

    Each ``p`` (polygon) or ``e`` (edge) record under a rule becomes one
    violation. Coordinates in the RVDB are integer database units (nm);
    the returned ``DRCViolation`` uses microns.
    """
    lines = path.read_text(errors="replace").splitlines()
    i, n = 0, len(lines)
    violations: list[DRCViolation] = []

    while i < n:
        line = lines[i].strip()
        # Skip blank lines, the $$$CONTEXT_INFO$$$ header, and anything that
        # isn't the bare rule name we expect at the start of a block.
        if not _looks_like_rule_name(line):
            i += 1
            continue

        rule = line
        i += 1
        if i >= n:
            break

        # Stats line: "<count> <orig> <def_lines> <date string ...>".
        count, def_lines, ok = _parse_stats_line(lines[i])
        if not ok:
            continue
        i += 1

        # Skip the rule-definition body: ``def_lines`` lines starting
        # with ``RULE.NAME { @ ... @``.
        i += def_lines
        description = _extract_description(lines, i - def_lines)

        # Consume ``count`` violation records.
        for _ in range(count):
            if i >= n:
                break
            head = _parse_record_header(lines[i])
            if head is None:
                i += 1
                continue
            kind, vertex_count = head
            i += 1
            cx, cy = _consume_record_body(lines, i, kind, vertex_count)
            if kind == "p":
                i += vertex_count
            violations.append(DRCViolation(
                rule        = rule,
                description = description,
                x           = round(cx, 4),
                y           = round(cy, 4),
            ))

    return violations


# ── helpers ────────────────────────────────────────────────────────────────

def _looks_like_rule_name(line: str) -> bool:
    s = line.strip()
    return bool(s) and _RULE_NAME_RE.match(s) is not None and not _is_record_header(s)


def _is_record_header(line: str) -> bool:
    parts = line.strip().split()
    return len(parts) >= 2 and parts[0] in ("p", "e")


def _parse_stats_line(line: str) -> tuple[int, int, bool]:
    """Parse ``<count> <orig> <def_lines> <date...>`` → (count, def_lines, ok).

    Returns ``ok=False`` when the line doesn't look like a stats header
    (caller bails the current rule).
    """
    parts = line.strip().split()
    if len(parts) < 4:
        return 0, 0, False
    try:
        count     = int(parts[0])
        def_lines = int(parts[2])
        return count, def_lines, True
    except ValueError:
        return 0, 0, False


def _extract_description(lines: list[str], start: int) -> str:
    """Pull the ``@ description @`` comment from the rule's definition opener."""
    if 0 <= start < len(lines):
        m = re.search(r"@\s*(.+?)\s*@", lines[start])
        if m:
            return m.group(1).strip()
    return ""


def _parse_record_header(line: str) -> Optional[tuple[str, int]]:
    """Return ``(kind, vertex_count)`` for a polygon/edge header.

    * Polygon: ``p <id> <vertex_count>`` (vertices on subsequent lines).
    * Edge:    ``e <id> 2 <x1> <y1> <x2> <y2>`` (single line).
    """
    parts = line.strip().split()
    if len(parts) < 3 or parts[0] not in ("p", "e"):
        return None
    try:
        return parts[0], int(parts[2])
    except ValueError:
        return None


def _consume_record_body(
    lines: list[str], start: int, kind: str, vertex_count: int,
) -> tuple[float, float]:
    """Return polygon / edge centroid in µm."""
    if kind == "e":
        head_parts = lines[start - 1].strip().split()
        try:
            x1, y1 = int(head_parts[3]), int(head_parts[4])
            x2, y2 = int(head_parts[5]), int(head_parts[6])
        except (IndexError, ValueError):
            return 0.0, 0.0
        return ((x1 + x2) / 2) * _DBU_TO_UM, ((y1 + y2) / 2) * _DBU_TO_UM

    # Polygon: vertex_count coordinate lines after the header.
    pts: list[tuple[float, float]] = []
    for k in range(vertex_count):
        idx = start + k
        if idx >= len(lines):
            break
        parts = lines[idx].strip().split()
        if len(parts) >= 2:
            try:
                pts.append((int(parts[0]) * _DBU_TO_UM,
                            int(parts[1]) * _DBU_TO_UM))
            except ValueError:
                pass
    if not pts:
        return 0.0, 0.0
    cx = sum(p[0] for p in pts) / len(pts)
    cy = sum(p[1] for p in pts) / len(pts)
    return cx, cy


def _detect_top_cell(gds_path: Path) -> Optional[str]:
    """Best-effort: read the GDS header to find a candidate top cell name.

    We don't link against gdspy/klayout from this module — this is a
    minimal binary-GDS scan that grabs the FIRST ``STRNAME`` record (06
    record type). Callers can always pass ``cell_name`` explicitly.
    """
    try:
        data = gds_path.read_bytes()
    except OSError:
        return None
    # GDS record header: 2-byte length, 1-byte type, 1-byte data type.
    # STRNAME is type 0x06; ASCII data type 0x06.
    i = 0
    n = len(data)
    while i + 4 <= n:
        rec_len  = int.from_bytes(data[i:i+2], "big")
        rec_type = data[i+2]
        if rec_len < 4 or rec_len > n - i:
            break
        if rec_type == 0x06:  # STRNAME
            name = data[i+4:i+rec_len].rstrip(b"\x00").decode("ascii", "replace")
            return name or None
        i += rec_len
    return None
