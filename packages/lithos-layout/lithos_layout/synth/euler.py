"""lithos_layout.synth.euler — Euler-path transistor ordering.

Finds a common Eulerian path through the NMOS pull-down and PMOS
pull-up diffusion networks of a cell so that adjacent transistors in
the layout share a source/drain terminal. This minimises diffusion
cuts, reducing cell area and improving current-drive capability.

Theory
------
Represent the pull-up or pull-down network as an undirected graph:

  - **nodes** = diffusion nets (the nets connected to S / D terminals).
  - **edges** = transistors (each labelled with the device name).

An Eulerian path visits every *edge* exactly once, which translates to
placing every transistor exactly once with no wasted diffusion break
between adjacent transistors.

An Eulerian path exists iff the graph is connected and has exactly 0
or 2 nodes with odd degree. If a single ordering satisfies both the
NMOS and PMOS graphs simultaneously, diffusion is shared in both
rows.

Public surface
--------------
* :func:`euler_order` — recommended placement order for a template
  (Euler path when one exists, original device order otherwise).
* :func:`common_euler_order` — Euler-respecting ordering across NMOS
  and PMOS rows, or ``None`` if no path exists.
* :func:`build_diffusion_graph`, :func:`has_euler_path`,
  :func:`euler_path` — building blocks for callers that need finer
  control.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from lithos_layout.synth.loader import CellTemplate


# ── Graph types ─────────────────────────────────────────────────────────────

@dataclass
class _Edge:
    """One transistor represented as a graph edge."""
    dev_name: str
    u:        str   # net at one S/D terminal
    v:        str   # net at the other S/D terminal


@dataclass
class _Graph:
    """Undirected multigraph for diffusion-network analysis."""
    edges: list[_Edge]             = field(default_factory=list)
    adj:   dict[str, list[_Edge]]  = field(default_factory=dict)

    def add_edge(self, e: _Edge) -> None:
        self.edges.append(e)
        self.adj.setdefault(e.u, []).append(e)
        self.adj.setdefault(e.v, []).append(e)

    @property
    def nodes(self) -> list[str]:
        return list(self.adj)

    def degree(self, node: str) -> int:
        return len(self.adj.get(node, []))

    def is_connected(self) -> bool:
        if not self.adj:
            return True
        start = next(iter(self.adj))
        visited: set[str] = set()
        stack = [start]
        while stack:
            n = stack.pop()
            if n in visited:
                continue
            visited.add(n)
            for e in self.adj.get(n, []):
                nb = e.v if e.u == n else e.u
                if nb not in visited:
                    stack.append(nb)
        return len(visited) == len(self.adj)

    def odd_degree_nodes(self) -> list[str]:
        return [n for n in self.adj if self.degree(n) % 2 == 1]


# ── Graph construction ──────────────────────────────────────────────────────

def build_diffusion_graph(template: "CellTemplate", device_type: str) -> _Graph:
    """Build an undirected diffusion graph for all devices of ``device_type``.

    Parameters
    ----------
    template :
        Cell topology template.
    device_type :
        ``"nmos"`` or ``"pmos"``.

    Returns
    -------
    _Graph
        Nodes = diffusion nets, edges = transistors.
    """
    g = _Graph()
    for name, spec in template.devices.items():
        if spec.device_type != device_type:
            continue
        s_net = spec.terminals.get("S", f"{name}_S")
        d_net = spec.terminals.get("D", f"{name}_D")
        g.add_edge(_Edge(dev_name=name, u=s_net, v=d_net))
    return g


# ── Euler path ──────────────────────────────────────────────────────────────

def has_euler_path(g: _Graph) -> bool:
    """Return ``True`` if ``g`` has an Eulerian path (0 or 2 odd-degree nodes)."""
    if not g.edges:
        return True
    if not g.is_connected():
        return False
    odd = len(g.odd_degree_nodes())
    return odd in (0, 2)


def euler_path(g: _Graph) -> list[str] | None:
    """Return a list of device names forming an Eulerian path through ``g``.

    Uses Hierholzer's algorithm. Returns ``None`` if no Eulerian path
    exists. Adjacent entries in the returned list share a diffusion
    terminal (source / drain).
    """
    if not g.edges:
        return []
    if not has_euler_path(g):
        return None

    # Work on a mutable copy of adjacency lists.
    adj: dict[str, list[_Edge]] = {n: list(edges) for n, edges in g.adj.items()}
    used: set[int] = set()    # edge indices (id of _Edge objects)

    # Start from an odd-degree node when one exists (gives a path, not a circuit).
    odd = g.odd_degree_nodes()
    start = odd[0] if odd else next(iter(adj))

    # Hierholzer's algorithm — sequence of diffusion nodes visited.
    stack       = [start]
    path_nodes: list[str] = []

    while stack:
        v = stack[-1]
        moved = False
        while adj.get(v):
            e = adj[v].pop()
            eid = id(e)
            if eid in used:
                continue
            used.add(eid)
            nb = e.v if e.u == v else e.u
            adj.setdefault(nb, [])
            stack.append(nb)
            moved = True
            break
        if not moved:
            path_nodes.append(stack.pop())

    # Convert node sequence to edge (device) sequence.
    node_seq = list(reversed(path_nodes))
    if len(node_seq) < 2:
        return [e.dev_name for e in g.edges]

    edge_pool: dict[tuple[str, str], list[_Edge]] = {}
    for e in g.edges:
        edge_pool.setdefault((e.u, e.v), []).append(e)
        edge_pool.setdefault((e.v, e.u), []).append(e)

    dev_order: list[str] = []
    used_devs: set[str]  = set()
    for i in range(len(node_seq) - 1):
        u, v = node_seq[i], node_seq[i + 1]
        for e in edge_pool.get((u, v), []):
            if e.dev_name not in used_devs:
                dev_order.append(e.dev_name)
                used_devs.add(e.dev_name)
                break

    # Append any missed devices (defensive — shouldn't happen on a valid path).
    for e in g.edges:
        if e.dev_name not in used_devs:
            dev_order.append(e.dev_name)

    return dev_order


# ── Common Euler order ──────────────────────────────────────────────────────

def common_euler_order(template: "CellTemplate") -> list[str] | None:
    """Return a device ordering that is a valid Euler path for both the
    NMOS and PMOS diffusion networks simultaneously, when one exists.

    The returned list contains *all* devices, NMOS row first followed
    by the PMOS row. Within each row the Euler order applies, so
    adjacent same-row devices share a diffusion terminal.

    Returns ``None`` if no valid common ordering can be found.
    Callers should fall back to the template's original device order.
    """
    nmos_graph = build_diffusion_graph(template, "nmos")
    pmos_graph = build_diffusion_graph(template, "pmos")

    nmos_order = euler_path(nmos_graph)
    pmos_order = euler_path(pmos_graph)

    if nmos_order is None and pmos_order is None:
        return None

    result: list[str] = []
    result.extend(nmos_order or [
        d for d, s in template.devices.items() if s.device_type == "nmos"
    ])
    result.extend(pmos_order or [
        d for d, s in template.devices.items() if s.device_type == "pmos"
    ])
    return result


# ── Convenience ─────────────────────────────────────────────────────────────

def euler_order(template: "CellTemplate") -> list[str]:
    """Return the recommended device placement order for ``template``.

    Tries to find a common Euler path; falls back to the template's
    original device order if no Eulerian path exists for either
    network.
    """
    order = common_euler_order(template)
    if order is not None:
        return order
    return list(template.devices)
