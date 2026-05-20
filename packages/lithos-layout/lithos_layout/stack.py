"""lithos_layout.stack — canonical metal stack + via-transition lookup.

The lithos canonical stack runs bottom-to-top through::

    poly / diff  ──contact──▶  m0  ──via_m0_m1──▶  m1  ──via_m1_m2──▶  m2  …

For any two layers in this stack, :func:`via_stack_between` returns the
ordered list of :class:`ViaTransition` records the router needs to
draw a via stack between them. Each transition carries the via cut
size and the enclosure values (2-adjacent-edge and all-sides /
opposite) for the lower and upper metals.

The data is sourced from the active :class:`BootstrapRules`
bootstrap mapping. The semantic-key conventions are:

* ``contact.size_um`` — contact cut size.
* ``contact.enclosure_in_<lower>_um`` — contact enclosed by ``poly``
  or ``diff``.
* ``m0.enclosure_of_contact_2adj_um`` /
  ``m0.enclosure_of_contact_um`` — m0 around the contact (2 adjacent
  edges vs all sides).
* ``via_mX_mY.size_um`` — inter-metal via cut size.
* ``mX.enclosure_of_via_mX_mY_2adj_um`` /
  ``mX.enclosure_of_via_mX_mY_um`` — lower-metal enclosure (2 adjacent
  edges vs all sides).
* ``mY.enclosure_of_via_mX_mY_2adj_um`` /
  ``mY.enclosure_of_via_mX_mY_um`` — upper-metal enclosure (2 adjacent
  edges vs all sides).

Missing all-sides ("opposite") values fall back to the 2adj values,
which yields a symmetric (square) pad — DRC-safe but slightly larger
than minimum.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from lithos_layout.rules import BootstrapRules


# Canonical metal stack ordering. Indices below ``0`` (``poly`` / ``diff``)
# are "below" the first true metal and connect via ``contact``; indices
# from 0 upward are ``m0``, ``m1``, …. Routers that need more than 10
# metal layers can extend ``_METALS`` here without touching the
# transition logic.
_METALS: tuple[str, ...] = tuple(f"m{i}" for i in range(10))
"""Canonical bottom-to-top metal-layer names."""


@dataclass(frozen=True)
class ViaTransition:
    """One via cut between two adjacent layers in the stack.

    Attributes
    ----------
    via_layer :
        Logical via layer name (e.g. ``"contact"``, ``"via_m0_m1"``).
    via_size :
        Via cut size in µm (square).
    lower_metal :
        Logical layer name of the metal below the via.
    upper_metal :
        Logical layer name of the metal above the via.
    enc_lower :
        2-adjacent-edge enclosure of the via in ``lower_metal`` (µm).
    enc_upper :
        2-adjacent-edge enclosure of the via in ``upper_metal`` (µm).
    enc_lower_opp :
        Opposite-edge (all-sides) enclosure of the via in
        ``lower_metal`` (µm). Falls back to ``enc_lower`` when the
        bootstrap mapping doesn't carry a separate "all sides" value.
    enc_upper_opp :
        Opposite-edge enclosure in ``upper_metal``. Falls back to
        ``enc_upper`` when not separately mapped.
    """
    via_layer:      str
    via_size:       float
    lower_metal:    str
    upper_metal:    str
    enc_lower:      float
    enc_upper:      float
    enc_lower_opp:  float = 0.0
    enc_upper_opp:  float = 0.0


# ── Index helpers ───────────────────────────────────────────────────────────

def _stack_index(name: str) -> int:
    """Return the stack index for a layer.

    ``poly`` and ``diff`` are below ``m0`` and share an index of ``-1``
    (they both connect to ``m0`` via the same ``contact``). Any
    ``mN`` (for ``N`` in 0..9) returns ``N``.

    Raises
    ------
    KeyError
        If ``name`` is not part of the canonical stack.
    """
    if name in ("poly", "diff"):
        return -1
    try:
        return _METALS.index(name)
    except ValueError as exc:
        raise KeyError(
            f"Layer {name!r} is not part of the lithos canonical metal "
            f"stack {('poly', 'diff') + _METALS}"
        ) from exc


# ── Transition construction ─────────────────────────────────────────────────

def _enc_pair(
    rules: "BootstrapRules",
    section: str,
    key_prefix: str,
) -> tuple[float, float]:
    """Return ``(enc_2adj, enc_opp)`` for a section / via combination.

    The 2adj value comes from ``<section>.<key_prefix>_2adj_um`` if
    mapped, otherwise from ``<section>.<key_prefix>_um``. The opposite
    value comes from ``<section>.<key_prefix>_um`` and falls back to
    the 2adj value when not separately defined.
    """
    adj_key = f"{section}.{key_prefix}_2adj_um"
    all_key = f"{section}.{key_prefix}_um"
    enc_2adj = rules.get(adj_key) if rules.has(adj_key) else None
    enc_opp  = rules.get(all_key) if rules.has(all_key) else None
    if enc_2adj is None and enc_opp is None:
        return 0.0, 0.0
    if enc_2adj is None:
        return float(enc_opp), float(enc_opp)
    if enc_opp is None:
        return float(enc_2adj), float(enc_2adj)
    return float(enc_2adj), float(enc_opp)


def _contact_transition(
    rules: "BootstrapRules",
    lower: str,                # "poly" or "diff"
) -> ViaTransition:
    size = rules.get("contact.size_um")
    enc_lower_key = f"contact.enclosure_in_{lower}_um"
    enc_lower = rules.get(enc_lower_key) if rules.has(enc_lower_key) else 0.0
    enc_upper_2adj, enc_upper_opp = _enc_pair(rules, "m0", "enclosure_of_contact")
    return ViaTransition(
        via_layer     = "contact",
        via_size      = float(size),
        lower_metal   = lower,
        upper_metal   = "m0",
        enc_lower     = float(enc_lower),
        enc_upper     = enc_upper_2adj,
        enc_lower_opp = float(enc_lower),
        enc_upper_opp = enc_upper_opp,
    )


def _inter_metal_transition(
    rules: "BootstrapRules",
    lower_idx: int,                # m{lower_idx} → m{lower_idx + 1}
) -> ViaTransition:
    lower = f"m{lower_idx}"
    upper = f"m{lower_idx + 1}"
    via   = f"via_{lower}_{upper}"
    size  = rules.get(f"{via}.size_um")
    enc_lower_2adj, enc_lower_opp = _enc_pair(rules, lower, f"enclosure_of_{via}")
    enc_upper_2adj, enc_upper_opp = _enc_pair(rules, upper, f"enclosure_of_{via}")
    return ViaTransition(
        via_layer     = via,
        via_size      = float(size),
        lower_metal   = lower,
        upper_metal   = upper,
        enc_lower     = enc_lower_2adj,
        enc_upper     = enc_upper_2adj,
        enc_lower_opp = enc_lower_opp,
        enc_upper_opp = enc_upper_opp,
    )


# ── Public API ─────────────────────────────────────────────────────────────

def via_stack_between(
    rules:      "BootstrapRules",
    from_layer: str,
    to_layer:   str,
) -> list[ViaTransition]:
    """Return the ordered transitions to connect ``from_layer`` to ``to_layer``.

    Parameters
    ----------
    rules :
        The active :class:`BootstrapRules`.
    from_layer, to_layer :
        Logical layer names. Either may be a metal (``m0``, ``m1``, …)
        or one of the contact-fed layers (``poly``, ``diff``). The
        order is normalised internally — passing them swapped yields
        the same transition list.

    Returns
    -------
    list[ViaTransition]
        Empty when ``from_layer`` and ``to_layer`` resolve to the same
        stack position (e.g. both ``m0`` on a PDK where
        :pyattr:`BootstrapRules.m0_is_m1` is true). Otherwise one
        :class:`ViaTransition` per cut, ordered bottom-to-top.
    """
    i_from = _stack_index(from_layer)
    i_to   = _stack_index(to_layer)
    if i_from > i_to:
        i_from, i_to = i_to, i_from
        from_layer, to_layer = to_layer, from_layer

    # Same stack position → no via needed.
    if i_from == i_to:
        return []

    transitions: list[ViaTransition] = []

    # Optional contact → m0 hop when the lower end is poly or diff.
    if i_from == -1:
        # ``from_layer`` is either poly or diff (whichever has index -1).
        contact_lower = from_layer if from_layer in ("poly", "diff") else "diff"
        transitions.append(_contact_transition(rules, contact_lower))
        # After the contact hop, we're at m0.
        i_from = 0

    # Inter-metal hops m{idx} → m{idx+1}.
    for lower_idx in range(i_from, i_to):
        # When the PDK collapses m0 onto m1, the m0→m1 hop is a no-op.
        if lower_idx == 0 and getattr(rules, "m0_is_m1", False):
            continue
        transitions.append(_inter_metal_transition(rules, lower_idx))

    return transitions
