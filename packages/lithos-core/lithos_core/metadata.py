"""lithos_core.metadata — PDK metadata YAML loader.

The metadata YAML carries the small hand-authored facts about a PDK that
*aren't* DRC rules: GDS layer numbers, manufacturing/routing grid, deck
file paths, port-label layer assignments, preferred routing directions.

The DRC rule content lives in a sibling SQLite DB produced by ``lithos-ingest``
(see :mod:`lithos_core.db`). Together, the YAML + the ``rules.<version>.db``
fully describe a PDK release.

Schema (sky130A example)::

    name: sky130A
    version: "1.0.5"
    layers:
      met2:  {layer: 69, datatype: 20}
      ...
    grid:
      manufacturing_um: 0.005
      routing_um:       0.005
    drc_decks:
      klayout: $PDK_ROOT/sky130A/libs.tech/klayout/...
      magic:   $PDK_ROOT/sky130A/libs.tech/magic/...
    preferred_direction:
      met1: horizontal
      met2: vertical
    label_layers:
      met1: [68, 16]
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import yaml


@dataclass(frozen=True)
class PDKMetadata:
    """Resolved PDK metadata. All paths are absolute or workspace-relative.

    ``devices`` carries the per-device-type structural facts that the
    layout generator needs but that don't belong in the rule DB: which
    physical layer plays the role of diffusion / gate / implant / bulk
    for an ``nmos`` or ``pmos``, whether the device sits inside an
    N-well, finger-count limits, and so on. Hand-authored in the same
    YAML as the layer map; loaded by :func:`load_metadata`.
    """

    name:                 str
    version:              str
    layers:               dict[str, tuple[int, int]]
    grid:                 dict[str, float]
    drc_decks:            dict[str, Path]
    preferred_direction:  dict[str, str] = field(default_factory=dict)
    label_layers:         dict[str, tuple[int, int]] = field(default_factory=dict)
    devices:              dict[str, dict] = field(default_factory=dict)
    metadata_path:        Optional[Path] = None

    def layer(self, name: str) -> tuple[int, int]:
        """Return the ``(gds_layer, gds_datatype)`` pair for a logical layer name."""
        try:
            return self.layers[name]
        except KeyError as exc:
            raise KeyError(
                f"Layer {name!r} not defined in PDK {self.name!r}. "
                f"Available: {sorted(self.layers)}"
            ) from exc

    def device(self, name: str) -> dict:
        """Return the device-type definition for ``name`` (e.g. ``"nmos"``)."""
        try:
            return self.devices[name]
        except KeyError as exc:
            raise KeyError(
                f"Device {name!r} not defined in PDK {self.name!r}. "
                f"Available: {sorted(self.devices)}"
            ) from exc

    @property
    def mfg_grid(self) -> float:
        return float(self.grid.get("manufacturing_um", 0.005))

    @property
    def routing_grid(self) -> float:
        return float(self.grid.get("routing_um", self.mfg_grid))


def load_metadata(path: Path | str) -> PDKMetadata:
    """Load a PDK metadata YAML, resolving deck paths and env vars.

    Path resolution for ``drc_decks`` entries (in priority order):

    1. ``DRC_DECK_<TOOL>`` environment variable (e.g. ``DRC_DECK_KLAYOUT``).
    2. ``$VAR`` expansion of the value in the YAML.
    3. If the result is relative, resolve it against the YAML's parent directory.
    """
    path = Path(path)
    with open(path) as f:
        data = yaml.safe_load(f) or {}

    if "name" not in data or "version" not in data:
        raise ValueError(
            f"PDK metadata at {path} must define both 'name' and 'version'."
        )

    layers = {}
    for layer_name, entry in (data.get("layers") or {}).items():
        if not isinstance(entry, dict) or "layer" not in entry or "datatype" not in entry:
            raise ValueError(
                f"Layer {layer_name!r} in {path} must be a mapping with "
                f"'layer' and 'datatype' keys."
            )
        layers[layer_name] = (int(entry["layer"]), int(entry["datatype"]))

    label_layers: dict[str, tuple[int, int]] = {}
    for layer_name, spec in (data.get("label_layers") or {}).items():
        if isinstance(spec, (list, tuple)) and len(spec) == 2:
            label_layers[layer_name] = (int(spec[0]), int(spec[1]))

    decks: dict[str, Path] = {}
    for tool, raw in (data.get("drc_decks") or {}).items():
        env_key = f"DRC_DECK_{tool.upper()}"
        resolved = os.environ.get(env_key) or os.path.expandvars(str(raw))
        p = Path(resolved)
        if not p.is_absolute():
            p = path.parent / p
        decks[tool] = p

    pdir = {
        str(layer): str(direction) if direction else ""
        for layer, direction in (data.get("preferred_direction") or {}).items()
    }

    devices = data.get("devices") or {}
    if not isinstance(devices, dict):
        raise ValueError(
            f"PDK metadata at {path}: 'devices' must be a mapping, got "
            f"{type(devices).__name__}"
        )

    return PDKMetadata(
        name                = str(data["name"]),
        version             = str(data["version"]),
        layers              = layers,
        grid                = {k: float(v) for k, v in (data.get("grid") or {}).items()},
        drc_decks           = decks,
        preferred_direction = pdir,
        label_layers        = label_layers,
        devices             = dict(devices),
        metadata_path       = path,
    )
