"""lithos_layout.synth.netlist — connectivity graph from device terminals.

Builds a :class:`NetGraph` that captures which device terminals share
each net. The auto-router uses this graph to decide what needs to be
routed and how.

The graph is constructed entirely from the ``devices`` and ``nets``
sections of a :class:`~lithos_layout.synth.loader.CellTemplate`.
No geometric information is needed — placement happens separately.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from lithos_layout.synth.loader import CellTemplate


# ── Data types ──────────────────────────────────────────────────────────────

@dataclass
class TerminalRef:
    """A reference to one device terminal (e.g. ``N_PD.D``)."""
    device:   str        # device instance name
    terminal: str        # "G", "D", or "S"

    @property
    def ref(self) -> str:
        return f"{self.device}.{self.terminal}"

    def __repr__(self) -> str:
        return self.ref


@dataclass
class NetInfo:
    """All information about a single net."""
    name:      str
    net_type:  str                          # "power" | "signal" | "internal"
    rail:      str = ""                     # "top" | "bottom" (power nets only)
    layer:     str = ""                     # preferred routing layer ("m0", "m1", ...)
    terminals: list[TerminalRef] = field(default_factory=list)

    @property
    def gate_terminals(self) -> list[TerminalRef]:
        return [t for t in self.terminals if t.terminal == "G"]

    @property
    def sd_terminals(self) -> list[TerminalRef]:
        return [t for t in self.terminals if t.terminal in ("S", "D")]

    @property
    def is_power(self) -> bool:
        return self.net_type == "power"

    @property
    def is_internal(self) -> bool:
        return self.net_type == "internal"


@dataclass
class NetGraph:
    """Connectivity graph: nets → terminal references.

    Attributes
    ----------
    nets :
        Net name → :class:`NetInfo` (type, rail, terminals).
    devices :
        Device name → ``{terminal: net_name}`` map.
    device_types :
        Device name → ``"nmos"`` | ``"pmos"``.
    """
    nets:         dict[str, NetInfo]
    devices:      dict[str, dict[str, str]]
    device_types: dict[str, str]

    def terminals_on_net(self, net_name: str) -> list[TerminalRef]:
        info = self.nets.get(net_name)
        return info.terminals if info else []

    def nets_for_device(self, dev_name: str) -> dict[str, str]:
        return self.devices.get(dev_name, {})


# ── Builder ─────────────────────────────────────────────────────────────────

def build_net_graph(template: CellTemplate) -> NetGraph:
    """Build a :class:`NetGraph` from a cell template."""
    nets: dict[str, NetInfo] = {}
    for name, nspec in template.nets.items():
        nets[name] = NetInfo(
            name     = name,
            net_type = nspec.net_type,
            rail     = nspec.rail,
            layer    = getattr(nspec, "layer", ""),
        )

    devices:      dict[str, dict[str, str]] = {}
    device_types: dict[str, str]            = {}

    for dev_name, dev_spec in template.devices.items():
        devices[dev_name]      = dict(dev_spec.terminals)
        device_types[dev_name] = dev_spec.device_type

        for term, net_name in dev_spec.terminals.items():
            if term == "B":
                continue                          # body terminal — not routed
            # Auto-create net entry if not declared (covers internal nets
            # the template author didn't bother listing explicitly).
            if net_name not in nets:
                nets[net_name] = NetInfo(name=net_name, net_type="internal")
            nets[net_name].terminals.append(
                TerminalRef(device=dev_name, terminal=term)
            )

    return NetGraph(nets=nets, devices=devices, device_types=device_types)
