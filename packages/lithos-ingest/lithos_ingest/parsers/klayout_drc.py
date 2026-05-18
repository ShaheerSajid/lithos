"""lithos_ingest.parsers.klayout_drc — KLayout DRC (Ruby) deck parser.

Projects the **declarative subset** of KLayout DRC scripts into the canonical
:class:`lithos_core.ir.Constraint`. The subset is the one foundry decks
universally use: a sequence of variable assignments and chained method
calls terminating in ``.output(code, description)``. Conditionals, blocks,
metaprogramming, and arbitrary Ruby semantics are intentionally out of
scope — those constructs in real decks usually wrap *knob-driven* logic
(``feol``, ``beol``, …) which is invocation-time configuration, not rule
content. Run the deck through KLayout itself for those checks; use this
parser to ingest the rule catalogue.

Grammar (informally)::

    deck       := (assignment | rule_emit | comment)*
    assignment := IDENT '=' expr
    rule_emit  := expr '.' 'output' '(' STRING (',' STRING)? ')'
    comment    := '#' .*

    expr       := atom ('.' method_call)*
    method_call:= IDENT '(' args? ')'
    atom       := IDENT
                | NUMBER ('.' 'um' | '.' 'um2')?       # trailing `.um` strips
                | STRING
                | '(' expr ')'
                | 'input'    '(' args ')'              # input(layer, datatype)
                | 'polygons' '(' args ')'

    args       := value (',' value)*
    value      := expr | STRING | NUMBER | IDENT

Method semantics
----------------
Geometric checks (left operand is a layer; produce a check):

    .width(t [, mod...])           → WidthCheck
    .space(t [, mod...])           → SpacingCheck (same-layer)
    .separation(b, t [, mod...])   → SpacingCheck (cross-layer)
    .enclosing(b, t [, mod...])    → EnclosureCheck (left encloses b)
    .enclosed_by(b, t [, mod...])  → EnclosureCheck (left is enclosed by b)
    .with_area(lo, hi)             → AreaCheck (lo or hi None for open-ended)

Layer algebra (left operand is a layer; produce a derived layer):

    .and(b) .or(b) .not(b) .xor(b)               → LayerBool
    .inside(b) .outside(b) .interact(b)
    .covers(b) .touches(b) .overlapping(b)       → LayerSelect
    .sized(n) .size_by(n)                        → LayerSize

Trailing modifier identifiers (``projection``, ``square``, ``euclidian``,
``opposite``, ``transparent``, ``intra_polygon``, ``shielded``) are
recorded under :attr:`SpacingCheck.modifiers` for the matching checks
and otherwise ignored.

The rule emit method ``.output("code", "description")`` is the sink that
produces a :class:`ParsedRule`. The first arg becomes the rule code; the
second (optional) becomes the title. ``input(layer, datatype)`` and
``polygons(layer, datatype)`` are recognised as layer-creation primitives
and bound to the assignment LHS — the resulting layer name reported in
the IR is the variable name (e.g. ``"met2"``), which matches how the rule
manual refers to it.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional

from lithos_core.ir import (
    AreaCheck,
    Constraint,
    ConstraintBranch,
    EnclosureCheck,
    LayerBool,
    LayerExpr,
    LayerRef,
    LayerSelect,
    LayerSize,
    SpacingCheck,
    WidthCheck,
)

from lithos_ingest.parsers.types import ParsedRule


class KLayoutDRCParseError(SyntaxError):
    """Raised when the parser hits KLayout DRC it doesn't understand.

    Treat as "extend the parser" rather than "invalid script" — KLayout
    decks contain patterns we may not have taught yet.
    """


# ── Lexer ────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class _Tok:
    kind: str
    text: str
    line: int
    col:  int


_TOKEN_RE = re.compile(
    r"""
    (?P<COMMENT>  \#[^\n]* )
  | (?P<WS>       [\ \t]+ )
  | (?P<NL>       \n )
  | (?P<STRING>   "(?:[^"\\]|\\.)*" | '(?:[^'\\]|\\.)*' )
  | (?P<NUMBER>   -?\d+(?:\.\d+)?(?:[eE][-+]?\d+)? )
  | (?P<DOT>      \. )
  | (?P<COMMA>    , )
  | (?P<LPAREN>   \( )
  | (?P<RPAREN>   \) )
  | (?P<EQ>       = )
  | (?P<IDENT>    [A-Za-z_][A-Za-z0-9_]* )
    """,
    re.VERBOSE,
)


def _tokenize(src: str) -> list[_Tok]:
    toks: list[_Tok] = []
    line, line_start = 1, 0
    i = 0
    while i < len(src):
        m = _TOKEN_RE.match(src, i)
        if m is None:
            raise KLayoutDRCParseError(
                f"unexpected character {src[i]!r} at line {line}, "
                f"col {i - line_start + 1}"
            )
        kind = m.lastgroup
        text = m.group()
        col  = i - line_start + 1
        if kind == "WS" or kind == "COMMENT":
            pass
        elif kind == "NL":
            line += 1
            line_start = m.end()
        else:
            toks.append(_Tok(kind, text, line, col))
        i = m.end()
    toks.append(_Tok("EOF", "", line, 0))
    return toks


# ── Helpers / vocabulary ────────────────────────────────────────────────────

_MODIFIERS = {
    "projection", "square", "euclidian", "transparent", "opposite",
    "intra_polygon", "shielded", "whole_edges",
}
"""Bare identifier "keywords" used as modifier arguments to KLayout
geometric checks. Captured into ``SpacingCheck.modifiers`` and otherwise
informational."""

_LAYER_BOOL_BINARY = {"and", "or", "xor"}
"""Layer-algebra methods that produce a binary boolean — recorded as
:class:`LayerBool` with two operands."""

_LAYER_SUBTRACT = "not"
"""Special-case: KLayout's ``.not(b)`` is set subtraction → ``a AND (NOT b)``."""

_LAYER_SELECT_OPS = {
    "inside": "inside", "outside": "outside",
    "interact": "interact", "interacting": "interact",
    "covers": "covers",
    "touches": "touch", "touching": "touch",
    "overlapping": "interact",
}
"""KLayout selection methods → IR ``LayerSelect.op``."""

_LAYER_PRIMITIVES = {"input", "polygons", "labels", "edges"}
"""Built-in layer-creation functions. Their (layer, datatype) args are
informational for the parser — we identify layers by their bound name."""


# ── Parser ──────────────────────────────────────────────────────────────────

@dataclass
class _LayerBinding:
    """Map a Ruby variable name to the LayerExpr it was assigned."""
    name:        str
    layer_expr:  LayerExpr


# Value-bearing argument used inside a method call.
@dataclass(frozen=True)
class _Arg:
    """One method-call argument after parsing.

    Exactly one of ``layer`` / ``number`` / ``string`` is non-None for
    structured args; ``modifier`` carries trailing bare-identifier modifier
    tokens (``projection``, ``square``, …).
    """
    layer:    Optional[LayerExpr]   = None
    number:   Optional[float]       = None
    string:   Optional[str]         = None
    modifier: Optional[str]         = None


class _Parser:
    def __init__(self, toks: list[_Tok], source: str):
        self.toks = toks
        self.source = source
        self.i = 0
        self.bindings: dict[str, LayerExpr] = {}     # var → derived layer
        # `derived_layers` collects every named layer expression by source
        # order, so each parsed rule's Constraint carries the chain of
        # intermediates needed for the joiner to see what the deck built.
        self.derived: dict[str, LayerExpr] = {}

    # ── Token plumbing ───────────────────────────────────────────────────

    def _peek(self, offset: int = 0) -> _Tok:
        return self.toks[self.i + offset]

    def _bump(self) -> _Tok:
        t = self.toks[self.i]
        self.i += 1
        return t

    def _accept(self, kind: str, text: Optional[str] = None) -> Optional[_Tok]:
        t = self._peek()
        if t.kind == kind and (text is None or t.text == text):
            return self._bump()
        return None

    def _expect(self, kind: str, text: Optional[str] = None) -> _Tok:
        t = self._accept(kind, text)
        if t is None:
            cur = self._peek()
            want = kind if text is None else f"{kind}({text!r})"
            raise KLayoutDRCParseError(
                f"expected {want} but got {cur.kind} {cur.text!r} "
                f"at line {cur.line}, col {cur.col}"
            )
        return t

    # ── Top level ────────────────────────────────────────────────────────

    def parse_deck(self) -> list[ParsedRule]:
        rules: list[ParsedRule] = []
        while self._peek().kind != "EOF":
            if self._peek().kind == "IDENT" and self._peek(1).kind == "EQ":
                self._parse_assignment()
                continue
            rule = self._parse_statement_expr()
            if rule is not None:
                rules.append(rule)
        return rules

    # ── Statements ───────────────────────────────────────────────────────

    def _parse_assignment(self) -> None:
        name = self._expect("IDENT").text
        self._expect("EQ")
        expr = self._parse_expr()
        self.bindings[name] = expr
        self.derived[name]  = expr

    def _parse_statement_expr(self) -> Optional[ParsedRule]:
        """Parse a statement-level expression. If it terminates in
        ``.output(...)`` returns the resulting ParsedRule; otherwise None
        (a no-op expression in the deck)."""
        expr, last_method_chain = self._parse_expr_capturing_chain()
        # Look for a terminal `.output("code", "desc")` call in the chain.
        for entry in reversed(last_method_chain):
            if entry["method"] == "output":
                return self._build_rule_from_chain(
                    expr_before_output = entry["receiver"],
                    output_args        = entry["args"],
                )
        return None

    # ── Expressions ──────────────────────────────────────────────────────

    def _parse_expr(self) -> LayerExpr:
        expr, _ = self._parse_expr_capturing_chain()
        return expr

    def _parse_expr_capturing_chain(
        self,
    ) -> tuple[LayerExpr, list[dict]]:
        """Parse a chained expression, returning both the final LayerExpr
        and a flat record of every ``.method(args)`` invocation along the
        chain.

        The chain records let us spot a terminal ``.output(...)`` after
        the fact and build a :class:`ParsedRule` from whatever was on its
        receiver.
        """
        receiver, current_kind = self._parse_atom_with_kind()
        chain: list[dict] = []
        last_check: Optional[object] = None    # WidthCheck / SpacingCheck / ...
        while self._accept("DOT"):
            method_tok = self._expect("IDENT")
            method = method_tok.text

            # `.um` / `.um2` are no-op µm/µm² casts on bare numbers in
            # KLayout DRC — we already store everything in µm. Skip them.
            if method in ("um", "um2"):
                continue

            args: list[_Arg] = []
            if self._accept("LPAREN"):
                args = self._parse_args()
                self._expect("RPAREN")

            chain.append({
                "method":   method,
                "args":     args,
                "receiver": receiver,
            })

            new_receiver, new_kind, produced_check = self._apply_method(
                receiver, current_kind, method, args, method_tok,
            )
            receiver     = new_receiver
            current_kind = new_kind
            if produced_check is not None:
                last_check = produced_check

        # Stash the last-built check on the receiver tag chain so the
        # rule-emitter (.output) can find it.
        if last_check is not None:
            chain[-1].setdefault("_check", last_check)
        return receiver, chain

    def _parse_atom_with_kind(self) -> tuple[LayerExpr, str]:
        """Return (expr, kind) where kind ∈ {"layer", "number", "string", "primitive"}.

        ``kind`` lets the chain interpret subsequent method calls correctly:
        ``"layer"`` may have geometric methods called on it, etc. For the
        IR we only care about LayerExpr — strings and numbers can't carry
        through, but they can appear as standalone statements.
        """
        t = self._peek()
        if self._accept("LPAREN"):
            inner, kind = self._parse_atom_chain_inside_parens()
            self._expect("RPAREN")
            return inner, kind

        if t.kind == "IDENT":
            self._bump()
            name = t.text
            if name in _LAYER_PRIMITIVES:
                # input(layer, dt) / polygons(layer, dt) / labels(...) / edges(...)
                if self._accept("LPAREN"):
                    args = self._parse_args()
                    self._expect("RPAREN")
                    # args[0], args[1] are layer/datatype — informational.
                    # The "name" of this layer is unbound until the
                    # caller does `foo = input(...)`. We use a synthetic
                    # placeholder LayerRef tied to (layer, datatype).
                    L = args[0].number if args and args[0].number is not None else 0
                    D = args[1].number if len(args) > 1 and args[1].number is not None else 0
                    return LayerRef(name=f"input({int(L)},{int(D)})"), "layer"
                return LayerRef(name=name), "layer"
            if name in self.bindings:
                return LayerRef(name=name), "layer"
            # Bare IDENT not bound — treat as a forward reference, IR will
            # carry the name and joiner / cross-validation can flag it.
            return LayerRef(name=name), "layer"

        if t.kind == "NUMBER":
            self._bump()
            # Numbers can't head an interesting chain on their own, but
            # they may carry `.um` next — handled in caller.
            return LayerRef(name=f"<number:{t.text}>"), "number"

        if t.kind == "STRING":
            self._bump()
            return LayerRef(name=f"<string>"), "string"

        raise KLayoutDRCParseError(
            f"unexpected token {t.kind} {t.text!r} at line {t.line}, col {t.col} "
            f"(expected an expression atom)"
        )

    def _parse_atom_chain_inside_parens(self) -> tuple[LayerExpr, str]:
        """A parenthesised expr; reuse the chain parser inside."""
        expr, _chain = self._parse_expr_capturing_chain()
        return expr, "layer"

    def _parse_args(self) -> list[_Arg]:
        args: list[_Arg] = []
        if self._peek().kind == "RPAREN":
            return args
        while True:
            args.append(self._parse_one_arg())
            if not self._accept("COMMA"):
                break
        return args

    def _parse_one_arg(self) -> _Arg:
        t = self._peek()
        # A bare IDENT modifier (`projection`, ...) is special — it's a
        # standalone identifier with no method chain. Distinguish from a
        # variable reference by membership in _MODIFIERS and absence of
        # a DOT/LPAREN immediately after.
        if (
            t.kind == "IDENT"
            and t.text in _MODIFIERS
            and self._peek(1).kind not in ("DOT", "LPAREN")
        ):
            self._bump()
            return _Arg(modifier=t.text)
        if t.kind == "STRING":
            self._bump()
            return _Arg(string=_unquote(t.text))
        if t.kind == "NUMBER":
            self._bump()
            # Allow `0.14.um` etc. as the no-op µm cast.
            if self._peek().kind == "DOT" and self._peek(1).kind == "IDENT" \
                    and self._peek(1).text in ("um", "um2"):
                self._bump()  # DOT
                self._bump()  # um / um2
            return _Arg(number=float(t.text))
        # Otherwise: expression (likely a layer reference, possibly chained).
        expr, _chain = self._parse_expr_capturing_chain()
        return _Arg(layer=expr)

    # ── Method dispatch ──────────────────────────────────────────────────

    def _apply_method(
        self,
        receiver:     LayerExpr,
        receiver_kind: str,
        method:       str,
        args:         list[_Arg],
        method_tok:   _Tok,
    ) -> tuple[LayerExpr, str, Optional[object]]:
        """Return ``(new_receiver, new_kind, check_or_none)`` after applying ``method``."""

        # ── Rule sink: .output(code, [desc]) ─────────────────────────────
        if method == "output":
            # Doesn't change the receiver; the chain-walker picks this up.
            return receiver, receiver_kind, None

        # ── Geometric checks ─────────────────────────────────────────────
        if method == "width":
            threshold = _first_number(args)
            check = WidthCheck(
                target=receiver, op=">=", threshold_um=threshold,
            )
            return receiver, receiver_kind, check

        if method == "space":
            threshold = _first_number(args)
            modifiers = [a.modifier for a in args if a.modifier]
            check = SpacingCheck(
                layer_a=receiver, layer_b=None,
                op=">=", threshold_um=threshold,
                modifiers=modifiers,
            )
            return receiver, receiver_kind, check

        if method == "separation":
            other     = _first_layer(args)
            threshold = _first_number(args)
            modifiers = [a.modifier for a in args if a.modifier]
            check = SpacingCheck(
                layer_a=receiver, layer_b=other,
                op=">=", threshold_um=threshold,
                modifiers=modifiers,
            )
            return receiver, receiver_kind, check

        if method == "enclosing":
            other     = _first_layer(args)
            threshold = _first_number(args)
            check = EnclosureCheck(
                inner=other, outer=receiver,
                op=">=", threshold_um=threshold,
            )
            return receiver, receiver_kind, check

        if method == "enclosed_by":
            other     = _first_layer(args)
            threshold = _first_number(args)
            check = EnclosureCheck(
                inner=receiver, outer=other,
                op=">=", threshold_um=threshold,
            )
            return receiver, receiver_kind, check

        if method == "with_area":
            lo = _nth_number(args, 0)
            hi = _nth_number(args, 1)
            # We emit a single ">= lo" check when only lo is set; when both
            # are present we use the lo bound (typical "min area" pattern).
            threshold = lo if lo not in (None, 0.0) else (hi or 0.0)
            op = ">=" if lo is not None else "<="
            check = AreaCheck(
                target=receiver, op=op, threshold_um2=float(threshold or 0.0),
            )
            return receiver, receiver_kind, check

        # ── Layer algebra (binary) ───────────────────────────────────────
        if method in _LAYER_BOOL_BINARY:
            other = _first_layer(args)
            if other is None:
                raise KLayoutDRCParseError(
                    f".{method} requires a layer argument at line "
                    f"{method_tok.line}, col {method_tok.col}"
                )
            return (
                LayerBool(op=method, operands=[receiver, other]),
                "layer", None,
            )

        if method == _LAYER_SUBTRACT:
            other = _first_layer(args)
            if other is None:
                raise KLayoutDRCParseError(
                    f".not requires a layer argument at line "
                    f"{method_tok.line}, col {method_tok.col}"
                )
            return (
                LayerBool(op="and", operands=[
                    receiver, LayerBool(op="not", operands=[other]),
                ]),
                "layer", None,
            )

        # ── Layer selection ──────────────────────────────────────────────
        if method in _LAYER_SELECT_OPS:
            other = _first_layer(args)
            if other is None:
                raise KLayoutDRCParseError(
                    f".{method} requires a layer argument at line "
                    f"{method_tok.line}, col {method_tok.col}"
                )
            return (
                LayerSelect(
                    op=_LAYER_SELECT_OPS[method],
                    subject=receiver, reference=other,
                ),
                "layer", None,
            )

        # ── Sizing ───────────────────────────────────────────────────────
        if method in ("sized", "size_by", "size"):
            n = _first_number(args)
            return (
                LayerSize(operand=receiver, by_um=float(n)),
                "layer", None,
            )

        # ── Unrecognised methods are tolerated (pass-through) ────────────
        # Many KLayout decks call `.output_polygon`, `.count`, `.info`,
        # etc. We treat them as no-ops on the receiver so chains don't
        # crash and the surrounding `.output(...)` still produces a rule.
        return receiver, receiver_kind, None

    # ── Rule emission ────────────────────────────────────────────────────

    def _build_rule_from_chain(
        self,
        expr_before_output: LayerExpr,
        output_args:        list[_Arg],
    ) -> Optional[ParsedRule]:
        """Materialise a ParsedRule from the chain that ended in ``.output``.

        Searches the existing chain for the last check produced before
        ``output`` was called. If there's no check (the deck called
        ``.output`` on a raw layer for a custom rule), we still emit a
        ParsedRule with an empty Constraint so the rule code is at least
        catalogued.
        """
        code = _arg_string(output_args, 0) or ""
        title = _arg_string(output_args, 1) or code
        # The receiver of `.output(...)` is the "last derived layer or check
        # result"; we look for a check that was attached during _apply_method.
        check_node = getattr(expr_before_output, "_lithos_check", None)
        # In our flat model we don't tag receivers; the chain entry walker
        # in _parse_statement_expr finds the check via the chain. Here we
        # only need a Constraint container.
        constraint: Optional[Constraint]
        if check_node is None:
            # Fall back: attach the receiver as a no-op constraint so the
            # rule code is at least catalogued.
            constraint = Constraint(branches=[], deck_dialect="klayout")
        else:
            constraint = Constraint(
                branches      = [ConstraintBranch(check=check_node)],
                deck_dialect  = "klayout",
            )
        aliases = [(code, "foundry_code")]
        if title and title != code:
            aliases.append((title, "deck_rulecheck"))
        return ParsedRule(
            code       = code,
            title      = title,
            aliases    = aliases,
            constraint = constraint,
            deck_block = "",
        )


# ── Pure helpers ────────────────────────────────────────────────────────────

def _first_number(args: list[_Arg]) -> float:
    for a in args:
        if a.number is not None:
            return a.number
    return 0.0


def _nth_number(args: list[_Arg], n: int) -> Optional[float]:
    nums = [a.number for a in args if a.number is not None]
    if n < len(nums):
        return nums[n]
    return None


def _first_layer(args: list[_Arg]) -> Optional[LayerExpr]:
    for a in args:
        if a.layer is not None:
            return a.layer
    return None


def _arg_string(args: list[_Arg], n: int) -> Optional[str]:
    strings = [a.string for a in args if a.string is not None]
    if n < len(strings):
        return strings[n]
    return None


def _unquote(s: str) -> str:
    """Strip surrounding quotes (either ``"`` or ``'``) and unescape minimal."""
    inner = s[1:-1]
    return (
        inner.replace(r'\n', "\n")
             .replace(r'\t', "\t")
             .replace(r'\\', "\\")
             .replace(r'\"', '"')
             .replace(r"\'", "'")
    )


# ── Driver: parse_klayout_drc ───────────────────────────────────────────────
#
# The class-based parser above keeps a per-statement check via the chain,
# but rebuilding the Constraint at .output time needs access to the check
# that was attached to the chain. To make that work cleanly we run a
# second, simpler pass that walks the source line-by-line at statement
# granularity and tracks the last produced check before the .output sink.

class _DriverParser(_Parser):
    """Concrete driver that tracks last-produced check per statement."""

    def parse_deck(self) -> list[ParsedRule]:                       # type: ignore[override]
        rules: list[ParsedRule] = []
        while self._peek().kind != "EOF":
            if self._peek().kind == "IDENT" and self._peek(1).kind == "EQ":
                self._parse_assignment()
                continue
            rule = self._parse_statement_with_check_tracking()
            if rule is not None:
                rules.append(rule)
        return rules

    def _parse_statement_with_check_tracking(self) -> Optional[ParsedRule]:
        receiver, kind = self._parse_atom_with_kind()
        last_check: Optional[object] = None
        rule_args:  Optional[list[_Arg]] = None
        while self._accept("DOT"):
            method_tok = self._expect("IDENT")
            method = method_tok.text
            if method in ("um", "um2"):
                continue
            args: list[_Arg] = []
            if self._accept("LPAREN"):
                args = self._parse_args()
                self._expect("RPAREN")
            if method == "output":
                rule_args = args
                break
            receiver, kind, produced = self._apply_method(
                receiver, kind, method, args, method_tok,
            )
            if produced is not None:
                last_check = produced

        if rule_args is None:
            return None
        code  = _arg_string(rule_args, 0) or ""
        title = _arg_string(rule_args, 1) or code
        if last_check is None:
            constraint = Constraint(branches=[], deck_dialect="klayout")
        else:
            constraint = Constraint(
                branches     = [ConstraintBranch(check=last_check)],   # type: ignore[arg-type]
                deck_dialect = "klayout",
            )
        aliases = [(code, "foundry_code")]
        if title and title != code:
            aliases.append((title, "deck_rulecheck"))
        return ParsedRule(
            code       = code,
            title      = title,
            aliases    = aliases,
            constraint = constraint,
            deck_block = "",
        )


def parse_klayout_drc(src: str) -> list[ParsedRule]:
    """Parse a KLayout DRC Ruby source string and return its rules.

    Only ``expr.output("code", "desc")`` chains produce :class:`ParsedRule`
    entries; top-level assignments just populate the binding table so
    layer names in the rules are recoverable.
    """
    toks = _tokenize(src)
    return _DriverParser(toks, src).parse_deck()
