"""lithos_ingest.parsers.svrf — Calibre SVRF rule-deck parser (tolerant).

Projects a useful subset of Calibre SVRF (Standard Verification Rule Format)
into :class:`lithos_core.ir.Constraint`. Designed to consume **real foundry
decks** (TSMC, GF, IHP, …), which use the conventional shape::

    NAME { @ Human-readable description, possibly multi-line
            @ A continuation line.
        <one or more SVRF statements>
    }

…plus a tolerant scan over the bits in between: ``#IFDEF`` / ``#IFNDEF`` /
``#ENDIF`` / ``#DEFINE`` directives, ``VARIABLE name "values"`` declarations,
and any other non-rule top-level constructs are skipped harmlessly.

Two block syntaxes are recognised:

    NAME { @ desc body }            — TSMC-style; most common in foundry decks
    RULECHECK "name" { body }       — Mentor / synthesized-deck style

Inside each block:

* All ``@ <text>``-introduced lines are concatenated as the rule's
  description (title).
* The parser attempts to extract a structured :class:`Constraint` from
  the first check-shaped statement it recognises. Recognised forms:

    INT  <expr> <cmp> <num> [modifiers...]           — same-layer spacing
    EXT  <expr> [<expr>] <cmp> <num> [modifiers...]  — same/cross-layer spacing
    ENC  <inner> <outer> <cmp> <num> [modifiers...]  — enclosure
    WIDTH  <expr> <cmp> <num> [modifiers...]         — width
    LENGTH <expr> <cmp> <num> [modifiers...]         — length / EOL
    AREA   <expr> <cmp> <num>                        — polygon area

  ``EXTERNAL``/``INTERNAL``/``ENCLOSURE`` are accepted as long-form aliases
  of ``EXT``/``INT``/``ENC``.

* Other body statements (derived-layer assignments, ``COPY``, ``CUT``, ...)
  are scanned past — when the parser hits something it can't structure, it
  skips to the rule's closing ``}`` and still emits a :class:`ParsedRule`
  with at least the code + title so the LLM ingestion + DRC alias resolver
  have something to work with.

Comparator inversion: SVRF measurement ops emit *violations*, not pass
conditions. ``EXT m2 < 0.14`` means "violation if spacing < 0.14", i.e. the
rule "spacing >= 0.14". The IR stores the rule comparator, not the
violation comparator — :func:`_invert_comparator` flips it on the way in.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional

from lithos_core.ir import (
    AreaCheck,
    CheckExpr,
    Constraint,
    ConstraintBranch,
    EnclosureCheck,
    ExistenceCheck,
    LayerBool,
    LayerExpr,
    LayerRef,
    LayerSelect,
    LayerSize,
    SpacingCheck,
    WidthCheck,
)

from lithos_ingest.parsers.types import ParsedRule


class SVRFParseError(SyntaxError):
    """Raised when the parser hits something genuinely unrecoverable.

    By design, most "weird SVRF" gets skipped silently — the parser recovers
    to the next rule block and keeps going. This exception fires only for
    structural problems (unterminated string, runaway brace nesting, lexer
    chokes on an illegal character).
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
    (?P<COMMENT>  // [^\n]* )
  | (?P<DIRECTIVE> \# [A-Za-z_][A-Za-z0-9_]* (?:[^\n]*)? )   # full preprocessor line
  | (?P<WS>       [\ \t]+ )
  | (?P<NL>       \n )
  | (?P<STRING>   "(?:[^"\\]|\\.)*" )
  | (?P<NUMBER>   -?\d+(?:\.\d+)?(?:[eE][-+]?\d+)? )
  | (?P<AT>       @ [^\n]* )                                # description line
  | (?P<LE>       <= )
  | (?P<GE>       >= )
  | (?P<NE>       != )
  | (?P<LT>       < )
  | (?P<GT>       > )
  | (?P<EQ>       = )
  | (?P<LBRACE>   \{ )
  | (?P<RBRACE>   \} )
  | (?P<LPAREN>   \( )
  | (?P<RPAREN>   \) )
  | (?P<IDENT>    [A-Za-z_][A-Za-z0-9_.]* )
  | (?P<COLON>    : )
  | (?P<COMMA>    , )
  | (?P<SEMI>     ; )
    """,
    re.VERBOSE,
)


def _tokenize(src: str) -> list[_Tok]:
    toks: list[_Tok] = []
    line, line_start = 1, 0
    i = 0
    n = len(src)
    while i < n:
        m = _TOKEN_RE.match(src, i)
        if m is None:
            # Tolerant lexer: any single unrecognised character becomes a
            # one-off "UNK" token. The parser ignores it during recovery.
            toks.append(_Tok("UNK", src[i], line, i - line_start + 1))
            i += 1
            continue
        kind = m.lastgroup
        text = m.group()
        col  = i - line_start + 1
        if kind == "WS" or kind == "COMMENT" or kind == "DIRECTIVE":
            pass
        elif kind == "NL":
            line += 1
            line_start = m.end()
        else:
            toks.append(_Tok(kind, text, line, col))
        i = m.end()
    toks.append(_Tok("EOF", "", line, 0))
    return toks


# ── Vocabularies ────────────────────────────────────────────────────────────

# Long-form keywords get their canonical short forms.
_CHECK_ALIASES = {
    "EXTERNAL":  "EXT",
    "INTERNAL":  "INT",
    "ENCLOSURE": "ENC",
}

_CHECK_KEYWORDS = {"EXT", "INT", "ENC", "WIDTH", "LENGTH", "AREA"}

_BOOL_KEYWORDS = {"AND", "OR", "XOR", "NOT"}

_SELECT_KEYWORDS = {
    "INSIDE":   "inside",
    "OUTSIDE":  "outside",
    "INTERACT": "interact",
    "TOUCH":    "touch",
    "ENCLOSE":  "enclose",
    "COVERS":   "covers",
}

# Tokens that trail a check expression as scalar / paired modifiers. We
# consume them silently and capture as opaque strings on the check's
# ``modifiers`` list (only SpacingCheck uses that field today; other
# checks just discard them). Adding new ones doesn't change parser shape.
_MODIFIER_KEYWORDS = {
    "ABUT", "SINGULAR", "REGION", "OPPOSITE", "PARALLEL", "PROJECTING",
    "TOUCH",                          # also appears as a select op
    "SQUARE", "EUCLIDIAN", "ORTHOGONAL", "INTRA_POLYGON", "WITH_ALL",
    "INSIDE_BY", "OUTSIDE_BY", "BY",
    "EDGE", "WITH", "AGAINST",
    "ANGLE", "RUN_LENGTH", "WHOLE",
}

# Top-level keywords whose lines we skip (statements that don't define rules).
_TOP_LEVEL_SKIP_KEYWORDS = {
    "VARIABLE",   "INCLUDE", "PRECISION",
    "LAYER",      "LAYOUT",
    "DRC", "RESULTS", "DATABASE", "RUN", "REPORT",
    "PORT",       "TEXT",
}

_COMPARATOR_BY_TOK = {
    "LT": "<", "LE": "<=", "GT": ">", "GE": ">=", "EQ": "=", "NE": "!=",
}


def _canonical_keyword(text: str) -> str:
    """Return the canonical (short) form of a check keyword if applicable."""
    upper = text.upper()
    return _CHECK_ALIASES.get(upper, upper)


def _invert_comparator(op: str) -> str:
    """Flip a violation comparator into the corresponding rule comparator.

    See module docstring on why: SVRF measurements emit on threshold
    failure, so ``< 0.14`` (violation) means ``>= 0.14`` (rule).
    """
    return {
        "<":  ">=",
        "<=": ">",
        ">":  "<=",
        ">=": "<",
        "=":  "!=",
        "!=": "=",
    }[op]


# ── Parser ──────────────────────────────────────────────────────────────────

class _Parser:
    def __init__(self, toks: list[_Tok], source: str):
        self.toks = toks
        self.source = source
        self.i = 0

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

    def _at_ident(self, *names: str) -> bool:
        t = self._peek()
        if t.kind != "IDENT":
            return False
        upper = t.text.upper()
        return upper in names or _canonical_keyword(upper) in names

    def _skip_to_brace_close(self, depth_already: int = 1) -> None:
        """Consume tokens until the matching ``}``. Used for error recovery."""
        depth = depth_already
        while depth > 0 and self._peek().kind != "EOF":
            t = self._bump()
            if t.kind == "LBRACE":
                depth += 1
            elif t.kind == "RBRACE":
                depth -= 1

    def _block_text(self, start_line: int, end_line: int) -> str:
        lines = self.source.splitlines()
        return "\n".join(lines[start_line - 1: end_line])

    # ── Top level ────────────────────────────────────────────────────────

    def parse_deck(self) -> list[ParsedRule]:
        rules: list[ParsedRule] = []
        while self._peek().kind != "EOF":
            if self._try_parse_rulecheck_form(rules):
                continue
            if self._try_parse_bare_block(rules):
                continue
            if self._try_skip_top_level_statement():
                continue
            # Unknown top-level token — skip and resync to next line/brace.
            self._bump()
        return rules

    # ── Block form recognisers ───────────────────────────────────────────

    def _try_parse_rulecheck_form(self, out: list[ParsedRule]) -> bool:
        """Match ``RULECHECK "name" { body }`` — Mentor-style synthesized form."""
        if not self._at_ident("RULECHECK"):
            return False
        save = self.i
        self._bump()                              # consume RULECHECK
        title_tok = self._accept("STRING")
        if title_tok is None or not self._accept("LBRACE"):
            self.i = save
            return False
        start_line = title_tok.line
        title = _unquote(title_tok.text)
        rule = self._parse_block_body(
            title=title,
            description_extra="",
            start_line=start_line,
            quoted_title=True,
        )
        out.append(rule)
        return True

    def _try_parse_bare_block(self, out: list[ParsedRule]) -> bool:
        """Match ``NAME { @ desc body }`` — TSMC-style real-deck form."""
        save = self.i
        if self._peek().kind != "IDENT":
            return False
        name_tok = self._bump()
        if not self._accept("LBRACE"):
            self.i = save
            return False
        start_line = name_tok.line
        # Description is whatever ``@ ...`` lines appear next (zero or more).
        desc_parts: list[str] = []
        while self._peek().kind == "AT":
            t = self._bump()
            # The AT token text is ``@ <rest of line>``; strip the @ and ws.
            desc_parts.append(t.text[1:].strip())
        description = " ".join(p for p in desc_parts if p)
        rule = self._parse_block_body(
            title=name_tok.text,
            description_extra=description,
            start_line=start_line,
            quoted_title=False,
            bare_code=name_tok.text,
        )
        out.append(rule)
        return True

    def _try_skip_top_level_statement(self) -> bool:
        """Skip whole-line top-level constructs we don't model
        (``VARIABLE x "..."``, ``INCLUDE``, etc.). Returns True if we ate
        anything."""
        if self._at_ident(*_TOP_LEVEL_SKIP_KEYWORDS):
            # Eat to end of the logical statement: anything up to either
            # the next IDENT-at-line-start (start of next statement) or EOF.
            anchor_line = self._peek().line
            while self._peek().kind != "EOF" and self._peek().line == anchor_line:
                self._bump()
            return True
        return False

    # ── Block body ───────────────────────────────────────────────────────

    def _parse_block_body(
        self,
        title:             str,
        description_extra: str,
        start_line:        int,
        quoted_title:      bool,
        bare_code:         Optional[str] = None,
    ) -> ParsedRule:
        """Parse statements inside an open ``{`` block, terminating at ``}``.

        Strategy: try to parse each statement as a known check or as an
        assignment. On failure, skip to the closing brace and emit a rule
        with whatever description we have plus an empty constraint.
        """
        derived: dict[str, LayerExpr] = {}
        branches: list[ConstraintBranch] = []
        body_desc_extra: list[str] = []
        # Bare layer expressions found in the body — used as the fallback
        # ExistenceCheck target when no numeric check was extracted. Many
        # foundry rules express violations as "this layer set must be
        # empty" via a bare expression (e.g. ``PP AND NP`` for implant
        # exclusivity); we capture the last such expression as the check.
        bare_exprs: list[LayerExpr] = []

        while True:
            t = self._peek()
            if t.kind == "EOF":
                break
            if t.kind == "RBRACE":
                self._bump()
                break
            if t.kind == "AT":
                # Additional description inside the body (rare but happens).
                self._bump()
                body_desc_extra.append(t.text[1:].strip())
                continue

            # Try assignment first: IDENT '=' expr
            if t.kind == "IDENT" and self._peek(1).kind == "EQ":
                ok = self._try_parse_inner_assignment(derived)
                if not ok:
                    # Skip just the failing line and keep looking for a
                    # check elsewhere in the body — TSMC decks routinely
                    # define multiple intermediate layers before the final
                    # check, and one unparseable assignment shouldn't lose
                    # the structured constraint.
                    self._skip_to_next_line_in_block()
                continue

            # Try check (prefix-keyword form).
            check = self._try_parse_check()
            if check is not None:
                branches.append(ConstraintBranch(predicate=[], check=check))
                continue

            # Try postfix forms (e.g. ``OD AREA < 0.202``).
            check = self._try_parse_postfix_check()
            if check is not None:
                branches.append(ConstraintBranch(predicate=[], check=check))
                continue

            # Try bare layer expression — the "violation set" idiom.
            be = self._try_parse_bare_expression()
            if be is not None:
                bare_exprs.append(be)
                continue

            # Last-resort recovery.
            recovered = self._skip_unknown_statement()
            if not recovered:
                self._skip_to_brace_close()
                break

        # If we found no numeric check but did harvest bare expressions,
        # emit the last one as an ExistenceCheck (semantics: this layer
        # set must be empty for the design to be DRC-clean).
        if not branches and bare_exprs:
            branches.append(ConstraintBranch(
                predicate=[],
                check=ExistenceCheck(target=bare_exprs[-1], must_be_empty=True),
            ))

        end_line = self.toks[max(0, self.i - 1)].line
        full_desc = (
            (description_extra + (" " + " ".join(body_desc_extra)
                                  if body_desc_extra else ""))
            .strip()
        )

        # Title resolution: for bare-name blocks, prefer the code; fall back
        # to whatever description we accumulated.
        if bare_code:
            code = bare_code
            display_title = full_desc or bare_code
        else:
            display_title = title
            code = _extract_code(title) or title

        constraint = Constraint(
            derived_layers = derived,
            branches       = branches,
            deck_dialect   = "svrf",
            raw_deck_text  = self._block_text(start_line, end_line),
        )

        aliases: list[tuple[str, str]] = []
        if code:
            aliases.append((code, "foundry_code"))
        if quoted_title and title and title != code:
            aliases.append((title, "deck_rulecheck"))
        elif not quoted_title and bare_code and display_title \
                and display_title != bare_code:
            aliases.append((display_title, "deck_rulecheck"))

        return ParsedRule(
            code       = code,
            title      = display_title or code,
            aliases    = aliases,
            constraint = constraint,
            deck_block = constraint.raw_deck_text or "",
        )

    def _try_parse_inner_assignment(
        self, derived: dict[str, LayerExpr],
    ) -> bool:
        """Match ``NAME = expr``. Returns False on parse failure."""
        save = self.i
        name = self._bump().text
        self._bump()                              # EQ
        try:
            expr = self._parse_expr()
        except SVRFParseError:
            self.i = save
            return False
        derived[name] = expr
        return True

    def _skip_to_next_line_in_block(self) -> None:
        """Skip remaining tokens on the current source line, but never
        past the block's closing brace."""
        anchor_line = self._peek().line
        while self._peek().kind not in ("EOF", "RBRACE") \
                and self._peek().line == anchor_line:
            self._bump()

    def _skip_unknown_statement(self) -> bool:
        """Skip one unrecognised statement inside a block body.

        Heuristic: consume tokens until we either land on an RBRACE, an
        AT, or an IDENT whose next token is ``=`` (looks like the next
        assignment) or whose upper-case form is a check keyword. The
        current line's tokens are also a natural stopping point.
        """
        anchor_line = self._peek().line
        consumed = False
        while True:
            t = self._peek()
            if t.kind in ("EOF", "RBRACE", "AT"):
                return consumed
            # Reaching a line break: stop here unless we haven't consumed
            # anything (then take one token so we don't loop forever).
            if t.line != anchor_line:
                if not consumed:
                    self._bump()
                    return True
                return True
            # Hit something check-like — break out so the main loop tries
            # parsing it as a check.
            if t.kind == "IDENT" and _canonical_keyword(t.text) in _CHECK_KEYWORDS:
                return consumed
            # Possible assignment lookahead.
            if t.kind == "IDENT" and self._peek(1).kind == "EQ" and consumed:
                return True
            self._bump()
            consumed = True

    # ── Checks ───────────────────────────────────────────────────────────

    def _try_parse_check(self) -> Optional[CheckExpr]:
        """Try to parse one check statement at the current position.

        Returns ``None`` (leaving position untouched) when the leading
        token isn't a recognised check keyword.
        """
        t = self._peek()
        if t.kind != "IDENT":
            return None
        kw = _canonical_keyword(t.text)
        if kw not in _CHECK_KEYWORDS:
            return None
        save = self.i
        try:
            self._bump()
            if kw == "WIDTH":
                return self._parse_width_check()
            if kw == "LENGTH":
                return self._parse_length_check()
            if kw == "AREA":
                return self._parse_area_check()
            if kw == "EXT":
                return self._parse_spacing_check(kind="external")
            if kw == "INT":
                return self._parse_spacing_check(kind="internal")
            if kw == "ENC":
                return self._parse_enclosure_check()
        except SVRFParseError:
            self.i = save
            return None
        return None

    def _try_parse_bare_expression(self) -> Optional[LayerExpr]:
        """Try to parse the line as a bare layer expression.

        Used as the ExistenceCheck fallback: if the body has no numeric
        check but ends with one or more bare expressions, the *last* such
        expression is interpreted as "this set must be empty".

        We accept the expression iff it parses cleanly and the next token
        is one that ends the statement (newline / closing brace / EOF /
        another AT description / a known modifier we can consume).
        Otherwise we restore position so the unknown-skip path can run.
        """
        save = self.i
        anchor_line = self._peek().line
        try:
            expr = self._parse_expr()
        except SVRFParseError:
            self.i = save
            return None
        # Eat trailing modifier tail (ABUT < 90, SINGULAR REGION, ...).
        self._consume_modifiers()
        nxt = self._peek()
        clean_end = (
            nxt.kind in ("RBRACE", "EOF", "AT")
            or nxt.line != anchor_line
        )
        if not clean_end:
            self.i = save
            return None
        return expr

    def _try_parse_postfix_check(self) -> Optional[CheckExpr]:
        """Match Calibre's layer-first forms: ``<layer> AREA <cmp> <num>``.

        Calibre lets a layer be on the left of the measurement operator
        (the operator is the "postfix" form). We try this only when the
        prefix-form recogniser declined.
        """
        save = self.i
        try:
            expr = self._parse_expr()
        except SVRFParseError:
            self.i = save
            return None
        t = self._peek()
        if t.kind == "IDENT" and t.text.upper() == "AREA" \
                and self._peek(1).kind in _COMPARATOR_BY_TOK:
            self._bump()
            op, thr = self._parse_comparator_and_threshold()
            self._consume_modifiers()
            return AreaCheck(
                target=expr,
                op=_invert_comparator(op),
                threshold_um2=thr,
            )
        # Not a postfix check — give the tokens back.
        self.i = save
        return None

    def _parse_width_check(self) -> WidthCheck:
        target = self._parse_expr()
        op, thr = self._parse_comparator_and_threshold()
        self._consume_modifiers()
        return WidthCheck(target=target, op=_invert_comparator(op), threshold_um=thr)

    def _parse_length_check(self) -> WidthCheck:
        # IR has no dedicated length check yet; project length as a width
        # with the same target+threshold. The deck text and rule code
        # disambiguate downstream.
        target = self._parse_expr()
        op, thr = self._parse_comparator_and_threshold()
        self._consume_modifiers()
        return WidthCheck(target=target, op=_invert_comparator(op), threshold_um=thr)

    def _parse_area_check(self) -> AreaCheck:
        target = self._parse_expr()
        op, thr = self._parse_comparator_and_threshold()
        self._consume_modifiers()
        return AreaCheck(target=target, op=_invert_comparator(op), threshold_um2=thr)

    def _parse_spacing_check(self, *, kind: str):
        """Parse ``INT`` or ``EXT`` followed by 1 or 2 layer args + comparator.

        Calibre semantics:

        * ``INT L`` (one layer) → *width* check on L. The "internal
          distance" of an edge pair within a single polygon is that
          polygon's width.
        * ``INT L1 L2`` → internal-distance spacing between two layers.
        * ``EXT L`` → same-layer outer spacing.
        * ``EXT L1 L2`` → cross-layer outer spacing.

        So a single-layer ``INT`` becomes a :class:`WidthCheck`; everything
        else becomes a :class:`SpacingCheck`.
        """
        layer_a = self._parse_expr()
        nxt = self._peek()
        layer_b: Optional[LayerExpr] = None
        if nxt.kind == "IDENT" and nxt.text.upper() not in _MODIFIER_KEYWORDS \
                and _canonical_keyword(nxt.text) not in _CHECK_KEYWORDS \
                and _COMPARATOR_BY_TOK.get(nxt.kind) is None:
            save = self.i
            try:
                candidate_b = self._parse_expr()
                if self._peek().kind in _COMPARATOR_BY_TOK:
                    layer_b = candidate_b
                else:
                    self.i = save
            except SVRFParseError:
                self.i = save
        elif nxt.kind == "LPAREN":
            layer_b = self._parse_expr()

        op, thr = self._parse_comparator_and_threshold()
        modifiers = self._consume_modifiers()
        rule_op = _invert_comparator(op)

        if kind == "internal" and layer_b is None:
            return WidthCheck(
                target=layer_a, op=rule_op, threshold_um=thr,
            )
        return SpacingCheck(
            layer_a      = layer_a,
            layer_b      = layer_b,
            op           = rule_op,
            threshold_um = thr,
            modifiers    = modifiers,
        )

    def _parse_enclosure_check(self) -> EnclosureCheck:
        inner = self._parse_expr()
        # Real Calibre is ``ENC inner outer < t`` (no BY keyword between).
        # Synthesized form may use BY; accept either.
        if self._peek().kind == "IDENT" and self._peek().text.upper() == "BY":
            self._bump()
        outer = self._parse_expr()
        op, thr = self._parse_comparator_and_threshold()
        self._consume_modifiers()
        return EnclosureCheck(
            inner        = inner,
            outer        = outer,
            op           = _invert_comparator(op),
            threshold_um = thr,
        )

    def _parse_comparator_and_threshold(self) -> tuple[str, float]:
        t = self._peek()
        if t.kind not in _COMPARATOR_BY_TOK:
            raise SVRFParseError(
                f"expected a comparator (<, <=, >, >=, =, !=) "
                f"at line {t.line}, col {t.col}, got {t.kind} {t.text!r}"
            )
        op = _COMPARATOR_BY_TOK[t.kind]
        self._bump()
        num = self._peek()
        if num.kind != "NUMBER":
            raise SVRFParseError(
                f"expected NUMBER after comparator at line {num.line}, "
                f"col {num.col}; got {num.kind} {num.text!r}"
            )
        self._bump()
        return op, float(num.text)

    def _consume_modifiers(self) -> list[str]:
        """Eat the trailing modifier tail of a check expression.

        Modifiers come after the threshold: ``ABUT < 90``, ``SINGULAR
        REGION``, ``OPPOSITE``, etc. We consume any sequence of known
        modifier idents plus their immediate operands (numbers, simple
        comparator + number pairs) until we land on something check-y.
        """
        captured: list[str] = []
        while True:
            t = self._peek()
            if t.kind != "IDENT" or t.text.upper() not in _MODIFIER_KEYWORDS:
                break
            self._bump()
            piece = t.text.upper()
            # Some modifiers carry an operand: ``ABUT < 90``, ``BY 0.1``.
            nxt = self._peek()
            if nxt.kind in _COMPARATOR_BY_TOK and \
                    self._peek(1).kind == "NUMBER":
                piece += f" {nxt.text} {self._peek(1).text}"
                self._bump(); self._bump()
            elif nxt.kind == "NUMBER":
                piece += f" {nxt.text}"
                self._bump()
            captured.append(piece)
        return captured

    # ── Expressions ──────────────────────────────────────────────────────

    def _parse_expr(self) -> LayerExpr:
        left = self._parse_unary()
        while self._peek().kind == "IDENT" \
                and self._peek().text.upper() in _BOOL_KEYWORDS \
                and self._peek().text.upper() not in _MODIFIER_KEYWORDS:
            op_tok = self._bump()
            right = self._parse_unary()
            if op_tok.text.upper() == "NOT":
                left = LayerBool(
                    op="and",
                    operands=[left, LayerBool(op="not", operands=[right])],
                )
            else:
                left = LayerBool(op=op_tok.text.lower(), operands=[left, right])
        return left

    def _parse_unary(self) -> LayerExpr:
        t = self._peek()
        if t.kind == "IDENT" and t.text.upper() == "NOT":
            self._bump()
            return LayerBool(op="not", operands=[self._parse_atom()])
        if t.kind == "IDENT" and t.text.upper() == "SIZE":
            self._bump()
            operand = self._parse_atom()
            if self._peek().kind == "IDENT" and self._peek().text.upper() == "BY":
                self._bump()
            if self._peek().kind != "NUMBER":
                raise SVRFParseError(
                    f"expected NUMBER after SIZE…BY at line {t.line}, col {t.col}"
                )
            by_tok = self._bump()
            return LayerSize(operand=operand, by_um=float(by_tok.text))
        if t.kind == "IDENT" and t.text.upper() in _SELECT_KEYWORDS:
            self._bump()
            subject = self._parse_atom()
            if self._peek().kind == "IDENT" and self._peek().text.upper() == "BY":
                self._bump()
            reference = self._parse_atom()
            return LayerSelect(
                op        = _SELECT_KEYWORDS[t.text.upper()],
                subject   = subject,
                reference = reference,
            )
        return self._parse_atom()

    def _parse_atom(self) -> LayerExpr:
        t = self._peek()
        if self._accept("LPAREN"):
            inner = self._parse_expr()
            if not self._accept("RPAREN"):
                raise SVRFParseError(
                    f"unbalanced parentheses at line {t.line}, col {t.col}"
                )
            return inner
        if t.kind == "IDENT" and t.text.upper() not in _MODIFIER_KEYWORDS \
                and _canonical_keyword(t.text) not in _CHECK_KEYWORDS:
            self._bump()
            return LayerRef(name=t.text)
        raise SVRFParseError(
            f"expected layer expression at line {t.line}, col {t.col}, "
            f"got {t.kind} {t.text!r}"
        )


# ── Public API ──────────────────────────────────────────────────────────────

def parse_svrf(src: str) -> list[ParsedRule]:
    """Parse an SVRF source string and return its rule blocks.

    Tolerant by design — see module docstring. Returns a :class:`ParsedRule`
    for every block matched, even when the body had constructs the parser
    couldn't structure (in that case the rule's :attr:`Constraint.branches`
    is empty but ``code`` and ``title`` are set).
    """
    toks = _tokenize(src)
    return _Parser(toks, src).parse_deck()


# ── Helpers ─────────────────────────────────────────────────────────────────

_CODE_RE = re.compile(r"^([A-Za-z][A-Za-z0-9._]*?)(?:\s*[:\-]\s*|\s*$)")


def _extract_code(title: str) -> Optional[str]:
    """Extract a foundry rule code from a RULECHECK title like
    ``"M2.S.1: metal2 spacing"``. Accepts ``:`` or ``-`` as the separator.
    """
    m = _CODE_RE.match((title or "").strip())
    if not m:
        return None
    code = m.group(1)
    if "." in code or code.isupper():
        return code
    return None


def _unquote(s: str) -> str:
    """Strip surrounding double-quotes from a STRING token (incl. simple escapes)."""
    inner = s[1:-1]
    return (
        inner
        .replace(r"\n", "\n")
        .replace(r"\t", "\t")
        .replace(r'\"', '"')
        .replace(r"\\", "\\")
    )
