"""lithos_ingest.parsers.svrf — Calibre SVRF rule-deck parser.

Projects a useful subset of Calibre SVRF (Standard Verification Rule
Format) into :class:`lithos_core.ir.Constraint`. Designed for **real
foundry decks**, parameterised by ``VARIABLE`` declarations, and using
the full set of measurement verbs the cell generator cares about.

Pipeline
--------

1. **Lexer** — single-pass regex tokeniser; emits ``IDENT``, ``NUMBER``,
   ``STRING``, ``COMMENT``, ``DIRECTIVE``, comparators, braces, ``@``
   description lines, etc.
2. **Symbol-table pre-scan** — extracts ``VARIABLE NAME <num>`` and
   ``#DEFINE NAME <num>`` from the raw source so a check body like
   ``EXT NWi PPOD < NW_S_5 ABUT < 90`` resolves the threshold to its
   numeric value.
3. **Recursive-descent parser** — one method per production; rule-level
   error recovery skips to the rule's closing ``}`` on a parse failure
   and still emits a :class:`ParsedRule` with code + title.

Top-level grammar
-----------------

::

    deck         := ( declaration | rule_block )*

    declaration  := '#' IDENT REST_OF_LINE              // #DEFINE / #IFDEF / #ENDIF
                  | 'VARIABLE' IDENT (NUMBER | STRING)
                  | top_keyword REST_OF_LINE            // INCLUDE / LAYER MAP / ...

    rule_block   := bare_name '{' body '}'
                  | 'RULECHECK' STRING '{' body '}'

    bare_name    := IDENT (':' IDENT)*                   // e.g. VIA1.R.4:M2

    body         := ( at_line | statement )*

    at_line      := '@' REST_OF_LINE                     // description / title

    statement    := layer_assignment
                  | check
                  | bare_layer_expr                      // ExistenceCheck "must be empty"
                  | unknown_line                         // skip & resume

    layer_assignment := IDENT '=' layer_expr

Check verbs
-----------

::

    check
      | INT      <expr> <cmp> <thr> <mods>               // 1-layer ⇒ WidthCheck
      | INT      <expr> <expr> <cmp> <thr> <mods>        // 2-layer SpacingCheck (internal)
      | EXT      <expr> <expr>? <cmp> <thr> <mods>       // SpacingCheck (external)
      | ENC      <inner> ['BY']? <outer> <cmp> <thr> <mods>
      | WIDTH    <expr> <cmp> <thr> <mods>
      | LENGTH   <expr> <cmp> <thr> <mods>               // projected to WidthCheck
      | AREA     <expr> <cmp> <thr>
      | <expr>   AREA <cmp> <thr>                        // postfix form
      | ANGLE    <expr> ( <cmp> <num> ){0,2}             // ExistenceCheck
      | OFFGRID  <expr> <num>?                           // ExistenceCheck
      | DENSITY  <expr> <region>? ( <cmp> <thr> )?       // DensityCheck

``EXTERNAL`` / ``INTERNAL`` / ``ENCLOSURE`` are long-form aliases of
``EXT`` / ``INT`` / ``ENC``.

Threshold resolution
--------------------

::

    threshold := NUMBER | resolvable_ident

A ``resolvable_ident`` is any ``IDENT`` listed in the deck's pre-scanned
``VARIABLE`` / ``#DEFINE`` table. Identifiers without a numeric
definition raise :class:`SVRFParseError`, which the rule-level recovery
catches and turns into a no-branch :class:`ParsedRule`.

Comparator inversion
--------------------

SVRF measurements emit *violations*, not pass conditions. ``EXT m2 < 0.14``
means *"violation if spacing < 0.14"*, i.e. the rule says *"spacing
>= 0.14"*. The IR stores the rule comparator, not the violation
comparator — :func:`_invert_comparator` flips it on the way in.
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Optional

from lithos_core.ir import (
    AreaCheck,
    CheckExpr,
    Constraint,
    ConstraintBranch,
    DensityCheck,
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
  | (?P<BLOCK_COMMENT> /\* (?:[^*]|\*(?!/))* \*/ )           # /* ... */ block comments
  | (?P<DIRECTIVE> \# [A-Za-z_][A-Za-z0-9_]* (?:[^\n]*)? )   # full preprocessor line
  | (?P<WS>       [\ \t]+ )
  | (?P<NL>       \n )
  | (?P<STRING>   "(?:[^"\\]|\\.)*" )
  | (?P<SQ_STRING> '(?:[^'\\]|\\.)*' )                       # single-quoted name (page 73)
  | (?P<NUMBER>   \d+(?:\.\d+)?(?:[eE][-+]?\d+)? )           # non-negative literal; unary - in expressions
  | (?P<AT>       @ [^\n]* )                                # description line
  | (?P<LE>       <= )
  | (?P<GE>       >= )
  | (?P<NE>       != )
  | (?P<EQEQ>     == )                                       # equality (manual page 66)
  | (?P<LT>       < )
  | (?P<GT>       > )
  | (?P<EQ>       = )
  | (?P<LBRACE>   \{ )
  | (?P<RBRACE>   \} )
  | (?P<LBRACK>   \[ )                                       # bracket (manual page 66)
  | (?P<RBRACK>   \] )
  | (?P<LPAREN>   \( )
  | (?P<RPAREN>   \) )
  | (?P<PLUS>     \+ )                                       # math operators (manual page 72)
  | (?P<MINUS>    - )
  | (?P<STAR>     \* )
  | (?P<SLASH>    / )
  | (?P<CARET>    \^ )
  | (?P<PERCENT>  % )
  | (?P<BANG>     ! )                                        # logical negation (page 67)
  | (?P<TILDE>    ~ )                                        # bitwise complement
  | (?P<IDENT>    [A-Za-z_][A-Za-z0-9_.]* )
  | (?P<COLON>    : )
  | (?P<COMMA>    , )
  | (?P<SEMI>     ; )
    """,
    re.VERBOSE,
)


_DIGIT_PREFIXED_NAME_TAIL = re.compile(r"[A-Za-z_][A-Za-z0-9_.]*")


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
        if kind in ("WS", "COMMENT", "BLOCK_COMMENT", "DIRECTIVE"):
            # Skip whitespace and comments. Block comments may span
            # lines — bump line count for any embedded newlines so
            # line numbers stay accurate in error messages.
            if kind == "BLOCK_COMMENT":
                nl = text.count("\n")
                if nl:
                    line += nl
                    line_start = m.end() - len(text.rsplit("\n", 1)[-1])
        elif kind == "NL":
            line += 1
            line_start = m.end()
        elif kind == "SQ_STRING":
            # Single-quoted name: the manual treats it identically to a
            # double-quoted name for case-sensitivity / identifier
            # purposes. Re-emit as STRING so downstream code that already
            # handles "..." also handles '...'.
            toks.append(_Tok("STRING", text, line, col))
        else:
            # Manual page 67 allows layer/rule names that *start with
            # digits* (e.g. ``25_18V_GATE_W``). The regex tokeniser
            # would otherwise split this into NUMBER("25") + IDENT
            # ("_18V_GATE_W"). When a NUMBER is immediately followed
            # (no whitespace) by an identifier-tail character — AND
            # the NUMBER itself contains no decimal point — glue them
            # into a single IDENT. Decimal NUMBERs like ``0.5`` are
            # never name prefixes; this guard avoids accidentally
            # gluing a threshold to a following keyword that happened
            # to land without whitespace (rare but unsafe).
            if kind == "NUMBER" and "." not in text and "e" not in text \
                    and "E" not in text:
                tail = _DIGIT_PREFIXED_NAME_TAIL.match(src, m.end())
                if tail and tail.start() == m.end():
                    toks.append(_Tok("IDENT", text + tail.group(), line, col))
                    i = tail.end()
                    continue
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

_CHECK_KEYWORDS = {
    "EXT", "INT", "ENC", "WIDTH", "LENGTH", "AREA",
    "ANGLE", "OFFGRID", "DENSITY",
    "ENCLOSE", "COPY",
}

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
_ENCLOSE_SHAPE_KEYWORDS = {
    "RECTANGLE", "POLYGON", "EDGE", "BOX", "CIRCLE",
}

# Math functions supported inside SVRF numeric expressions (manual Table 2-4,
# page 72). Identifier followed by ``(`` and listed here is parsed as a
# function call by :meth:`_Parser._parse_numexpr_atom`.
_MATH_FUNCS: dict[str, "callable"] = {
    "CEIL":  math.ceil,
    "FLOOR": math.floor,
    "TRUNC": math.trunc,
    "SQRT":  math.sqrt,
    "ABS":   abs,
    "EXP":   math.exp,
    "LOG":   math.log,
    "SIN":   math.sin,
    "COS":   math.cos,
    "TAN":   math.tan,
    "MIN":   min,
    "MAX":   max,
}

_MODIFIER_KEYWORDS = {
    "ABUT", "SINGULAR", "REGION", "OPPOSITE", "PARALLEL", "PROJECTING",
    "TOUCH",                          # also appears as a select op
    "SQUARE", "EUCLIDIAN", "ORTHOGONAL", "INTRA_POLYGON", "WITH_ALL",
    "INSIDE_BY", "OUTSIDE_BY", "BY",
    "EDGE", "WITH", "AGAINST",
    "RUN_LENGTH", "WHOLE",
    # NB: ``ANGLE`` is no longer a modifier — it's promoted to a
    # first-class check verb. (``ABUT < 90`` is still recognised as a
    # modifier; the bare ``ANGLE`` keyword introduces an angle check.)
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
        # Pre-scan ``#DEFINE NAME VALUE`` directives once so check
        # bodies that use the variable as a threshold still structure.
        # Real foundry decks declare numeric thresholds up front
        # (``#DEFINE NW_S_5 0.5``) and reference them in rule bodies
        # (``EXT NWi PPOD < NW_S_5 ABUT < 90``).
        self.defines: dict[str, float] = _scan_defines(source)

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
        """Match ``NAME { @ desc body }`` — bare-name real-deck form.

        Also accepts ``NAME:SUFFIX { ... }`` (and chains like
        ``NAME:S1:S2``) because some foundry decks qualify a rule code
        with a per-layer / per-severity suffix joined by ``:`` (e.g.
        ``METAL_RULE:M2``, ``SOME_RULE:ERROR``). The colon is not in
        the IDENT character class, so the lexer splits these into
        separate tokens; we glue them back together here.
        """
        save = self.i
        if self._peek().kind != "IDENT":
            return False
        first_tok = self._bump()
        start_line = first_tok.line
        # Glue ``:IDENT`` continuations onto the code so suffix-tagged
        # rule names survive (else every ``X.Y:M2`` block collapses to
        # code ``M2`` and clobbers every other ``…:M2``).
        name_parts = [first_tok.text]
        while self._peek().kind == "COLON" and self._peek(1).kind == "IDENT":
            self._bump()                              # COLON
            name_parts.append(self._bump().text)
        if not self._accept("LBRACE"):
            self.i = save
            return False
        full_name = ":".join(name_parts)
        # Description is whatever ``@ ...`` lines appear next (zero or more).
        desc_parts: list[str] = []
        while self._peek().kind == "AT":
            t = self._bump()
            # The AT token text is ``@ <rest of line>``; strip the @ and ws.
            desc_parts.append(t.text[1:].strip())
        description = " ".join(p for p in desc_parts if p)
        rule = self._parse_block_body(
            title=full_name,
            description_extra=description,
            start_line=start_line,
            quoted_title=False,
            bare_code=full_name,
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
                    # check elsewhere in the body — real decks routinely
                    # define multiple intermediate layers before the
                    # final check, and one unparseable assignment
                    # shouldn't lose the structured constraint.
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

            # Last-resort recovery. ``_skip_unknown_statement`` returns
            # False when the current token *looks* parseable (e.g. a
            # check keyword) so the main loop can retry it — but we
            # already tried it above and it raised. Force-bump one
            # token so we make progress and don't infinite-loop. If
            # that lands us at the closing brace or EOF, the next
            # iteration breaks naturally.
            recovered = self._skip_unknown_statement()
            if not recovered:
                if self._peek().kind in ("EOF", "RBRACE"):
                    break
                self._bump()

        # If we found no numeric check but did harvest bare expressions,
        # emit the last one as an ExistenceCheck (semantics: this layer
        # set must be empty for the design to be DRC-clean).
        if not branches and bare_exprs:
            branches.append(ConstraintBranch(
                predicate=[],
                check=ExistenceCheck(target=bare_exprs[-1], must_be_empty=True),
            ))

        # If still no branch but we DID record a chain of layer
        # assignments, promote the *last* assigned layer to an
        # ExistenceCheck. Real foundry rules routinely express a
        # violation as the terminal layer of a derivation chain
        # (e.g. via-stack BRANCH1 / BRANCH1HASVIA / BRANCH1EDGE /
        # GOODBRANCH chains where the final ``BAD_REGION = …``
        # carries the violations).
        if not branches and derived:
            last_name = list(derived.keys())[-1]
            branches.append(ConstraintBranch(
                predicate=[],
                check=ExistenceCheck(
                    target=LayerRef(name=last_name), must_be_empty=True,
                ),
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
            if kw == "ANGLE":
                return self._parse_angle_check()
            if kw == "OFFGRID":
                return self._parse_offgrid_check()
            if kw == "DENSITY":
                return self._parse_density_check()
            if kw == "ENCLOSE":
                return self._parse_enclose_check()
            if kw == "COPY":
                return self._parse_copy_check()
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

    # ── ANGLE / OFFGRID / DENSITY (no-numeric-threshold checks) ───────────
    #
    # These verbs introduce real DRC checks that don't fit the
    # threshold-comparator-modifier shape of width / spacing /
    # enclosure. The IR doesn't have dedicated classes for ANGLE and
    # OFFGRID yet — projecting them as :class:`ExistenceCheck` over the
    # target layer captures the rule shape (the layer is the violation
    # set; we mark it as one the deck flags). DENSITY uses the existing
    # :class:`DensityCheck` IR.
    #
    # Shapes recognised:
    #
    #   ANGLE   <layer> ( cmp NUMBER )*           // 1 or 2 comparator bounds
    #   OFFGRID <layer> NUMBER                    // off-grid distance
    #   DENSITY <layer> <region_layer> ( cmp threshold )?  // simple form
    #
    # Anything more elaborate (DENSITY ... WINDOW ... STEP ...) is left
    # to the post-recovery skip path; we recover enough to mark the
    # rule as structured.

    def _parse_angle_check(self) -> ExistenceCheck:
        """``ANGLE <layer> ( cmp NUMBER )*`` — angle-bounds check.

        The IR has no AngleCheck yet; capture the rule's shape via
        :class:`ExistenceCheck` over the target layer. The numeric
        bounds (e.g. ``>0 <45``) are consumed so the parser can advance.
        """
        target = self._parse_expr()
        # Consume up to two ``cmp NUMBER`` pairs (e.g. ``ANGLE L >0 <45``).
        for _ in range(2):
            t = self._peek()
            if t.kind in _COMPARATOR_BY_TOK and self._peek(1).kind == "NUMBER":
                self._bump(); self._bump()
            else:
                break
        self._consume_modifiers()
        return ExistenceCheck(target=target, must_be_empty=True)

    def _parse_offgrid_check(self) -> ExistenceCheck:
        """``OFFGRID <layer> NUMBER`` — off-manufacturing-grid check.

        The trailing number is the grid value in nm or um (foundry-
        dependent); we don't currently model it. Capture as an
        existence check over the layer.
        """
        target = self._parse_expr()
        # Optional trailing numeric grid argument.
        if self._peek().kind == "NUMBER":
            self._bump()
        self._consume_modifiers()
        return ExistenceCheck(target=target, must_be_empty=True)

    def _parse_enclose_check(self) -> ExistenceCheck:
        """``ENCLOSE [RECTANGLE|POLYGON|...] <layer> <args...>`` — shape-enclosure check.

        Distinct from ``ENC`` (which is a foundry-typical enclosure
        distance check). ``ENCLOSE`` is a Calibre verb that flags
        polygons enclosing a specified shape/dimension envelope. Real
        forms look like::

            ENCLOSE RECTANGLE OD_SPACE_BOTH OD_S_1 OD_S_3_L
            ENCLOSE RECTANGLE Y GRID M1_S_2_L+GRID

        We capture the rule shape via :class:`ExistenceCheck` on the
        first layer-like argument; later args are scalar dimensions we
        don't model yet. Trailing modifiers / arithmetic on dimensions
        are absorbed by the unknown-recovery path.
        """
        # Optional shape keyword (RECTANGLE / POLYGON / EDGE / ...) —
        # consume one IDENT if it isn't itself a layer name we know.
        t = self._peek()
        if t.kind == "IDENT" and t.text.upper() in _ENCLOSE_SHAPE_KEYWORDS:
            self._bump()
        target = self._parse_expr()
        # Drain the remainder of the statement (numeric / variable args
        # plus modifiers). We bound this by line so we don't eat the
        # next statement.
        anchor_line = self._peek().line if self._peek().kind != "EOF" else -1
        while self._peek().kind != "EOF" and self._peek().line == anchor_line \
                and self._peek().kind != "RBRACE":
            self._bump()
        return ExistenceCheck(target=target, must_be_empty=True)

    def _parse_copy_check(self) -> ExistenceCheck:
        """``COPY <layer>`` — the layer *is* the violation set.

        Some foundry rules express a deck-wide warning by COPYing a
        whole layer into the violation database (e.g. when an option
        switch is incompatible with the selected process). We project
        these as :class:`ExistenceCheck` over the copied layer.
        """
        target = self._parse_expr()
        # Any trailing tokens on the same line are absorbed by the
        # unknown-skip path; nothing meaningful follows COPY.
        return ExistenceCheck(target=target, must_be_empty=True)

    def _parse_density_check(self) -> CheckExpr:
        """``DENSITY <layer> <region> ( cmp NUMBER )?`` — area-fraction check.

        Real foundry forms range from the trivial::

            DENSITY ODx CHIP_NOT_ODEXC < 0.2

        to elaborate window-and-step variants that drive an assignment
        (``ERR_WIN = DENSITY ... WINDOW ... STEP ... INSIDE OF LAYER ...``).
        We handle the inline-check form here and let the more elaborate
        forms fall through to the skip path; even partial structuring
        materially improves coverage.
        """
        target = self._parse_expr()
        # Optional second layer (region scope) — DENSITY a b cmp t.
        region: Optional[LayerExpr] = None
        nxt = self._peek()
        if nxt.kind == "IDENT" and nxt.text.upper() not in _MODIFIER_KEYWORDS \
                and _COMPARATOR_BY_TOK.get(nxt.kind) is None:
            save = self.i
            try:
                region = self._parse_expr()
            except SVRFParseError:
                self.i = save
                region = None
        # Optional comparator + threshold.
        min_ratio: Optional[float] = None
        max_ratio: Optional[float] = None
        if self._peek().kind in _COMPARATOR_BY_TOK:
            op, thr = self._parse_comparator_and_threshold()
            # Convention: "violation if density < X" means rule wants >= X.
            rule_op = _invert_comparator(op)
            if rule_op in (">=", ">"):
                min_ratio = thr
            elif rule_op in ("<=", "<"):
                max_ratio = thr
        self._consume_modifiers()
        return DensityCheck(
            target    = target,
            window_um = 0.0,                 # window size unknown for simple form
            min_ratio = min_ratio,
            max_ratio = max_ratio,
        )

    def _parse_comparator_and_threshold(self) -> tuple[str, float]:
        """Parse ``<cmp> <numeric_expression>``.

        The numeric expression may be a literal, a VARIABLE-resolved
        identifier, a parenthesised sub-expression, or any combination
        with ``+ - * / ^ %`` operators and the math functions documented
        in Table 2-4 of the SVRF manual (CEIL/FLOOR/ABS/SQRT/MIN/MAX/…).
        See :meth:`_parse_numeric_expression`.

        When the check uses an *interval* constraint (``> a < b``,
        ``>= a < b``, etc.; manual Table 2-2), call
        :meth:`_parse_interval_constraint` instead.
        """
        t = self._peek()
        if t.kind not in _COMPARATOR_BY_TOK:
            raise SVRFParseError(
                f"expected a comparator (<, <=, >, >=, =, !=) "
                f"at line {t.line}, col {t.col}, got {t.kind} {t.text!r}"
            )
        op = _COMPARATOR_BY_TOK[t.kind]
        self._bump()
        thr = self._parse_numeric_expression()
        return op, thr

    def _parse_interval_constraint(self) -> tuple[str, float, str, float]:
        """Parse one of the interval constraint forms documented in
        Table 2-2 (manual page 70).

        Returns ``(low_op, low_val, high_op, high_val)``. When only a
        single comparator is present, ``high_op`` is the empty string
        and ``high_val`` is ``float("nan")``.

        Accepted shapes (with ``a`` and ``b`` numeric expressions)::

            < a              -> ("<",  a, "", nan)
            > a              -> (">",  a, "", nan)
            <= a             -> ("<=", a, "", nan)
            >= a             -> (">=", a, "", nan)
            == a             -> ("==", a, "", nan)
            != a             -> ("!=", a, "", nan)
            >  a  <  b       -> (">",  a, "<",  b)
            >= a  <  b       -> (">=", a, "<",  b)
            >  a  <= b       -> (">",  a, "<=", b)
            >= a  <= b       -> (">=", a, "<=", b)
            <  b  >  a       -> (">",  a, "<",  b)
            <  b  >= a       -> (">=", a, "<",  b)
            <= b  >  a       -> (">",  a, "<=", b)
            <= b  >= a       -> (">=", a, "<=", b)
        """
        op1, val1 = self._parse_comparator_and_threshold()
        # Look ahead for a second comparator forming an interval.
        nxt = self._peek()
        if nxt.kind not in _COMPARATOR_BY_TOK:
            return (op1, val1, "", float("nan"))
        op2, val2 = self._parse_comparator_and_threshold()
        # Normalise so the lower-bound op (>, >=) is returned first.
        if op1 in (">", ">=") and op2 in ("<", "<="):
            return (op1, val1, op2, val2)
        if op1 in ("<", "<=") and op2 in (">", ">="):
            return (op2, val2, op1, val1)
        # Unusual ordering — return as-given.
        return (op1, val1, op2, val2)

    # ── Numeric expressions (manual page 72) ──────────────────────────
    #
    # Implements the operator precedence from Table 2-3:
    #
    #   ()                 grouping                (highest)
    #   + - ! ~            unary
    #   * / ^ %            binary
    #   + -                binary                  (lowest)
    #
    # Plus the math functions in Table 2-4: CEIL FLOOR TRUNC SQRT ABS
    # EXP LOG SIN COS TAN MIN(a,b) MAX(a,b).
    #
    # Identifiers in numeric context are VARIABLE / #DEFINE references
    # resolved via ``self.defines``; an undefined identifier raises
    # SVRFParseError so the rule-level recovery still kicks in.

    def _parse_numeric_expression(self) -> float:
        """Top of the numeric-expression precedence climb (binary +/-)."""
        left = self._parse_numexpr_muldiv()
        while self._peek().kind in ("PLUS", "MINUS"):
            op = self._bump().text
            right = self._parse_numexpr_muldiv()
            left = left + right if op == "+" else left - right
        return left

    def _parse_numexpr_muldiv(self) -> float:
        """Binary *, /, ^, % level."""
        left = self._parse_numexpr_unary()
        while self._peek().kind in ("STAR", "SLASH", "CARET", "PERCENT"):
            op = self._bump().text
            right = self._parse_numexpr_unary()
            if op == "*":
                left = left * right
            elif op == "/":
                left = left / right if right != 0 else float("inf")
            elif op == "^":
                left = left ** right
            else:  # %
                left = left % right if right != 0 else float("nan")
        return left

    def _parse_numexpr_unary(self) -> float:
        """Unary +, -, !, ~ level."""
        t = self._peek()
        if t.kind == "PLUS":
            self._bump()
            return self._parse_numexpr_unary()
        if t.kind == "MINUS":
            self._bump()
            return -self._parse_numexpr_unary()
        if t.kind == "BANG":
            self._bump()
            return 0.0 if self._parse_numexpr_unary() != 0 else 1.0
        if t.kind == "TILDE":
            self._bump()
            val = self._parse_numexpr_unary()
            return 0.0 if val > 0 else 1.0
        return self._parse_numexpr_atom()

    def _parse_numexpr_atom(self) -> float:
        """Atomic: NUMBER, parenthesised expr, VARIABLE reference, function call."""
        t = self._peek()
        if t.kind == "NUMBER":
            self._bump()
            return float(t.text)
        if t.kind == "LPAREN":
            self._bump()
            val = self._parse_numeric_expression()
            if self._peek().kind != "RPAREN":
                tok = self._peek()
                raise SVRFParseError(
                    f"expected ')' to close numeric expression at line "
                    f"{tok.line}, col {tok.col}; got {tok.kind} {tok.text!r}"
                )
            self._bump()
            return val
        if t.kind == "IDENT":
            name = t.text
            # Function call: IDENT '(' expr (',' expr)* ')'
            if name.upper() in _MATH_FUNCS and self._peek(1).kind == "LPAREN":
                self._bump()                              # IDENT
                self._bump()                              # LPAREN
                args: list[float] = [self._parse_numeric_expression()]
                while self._peek().kind == "COMMA":
                    self._bump()
                    args.append(self._parse_numeric_expression())
                if self._peek().kind != "RPAREN":
                    tok = self._peek()
                    raise SVRFParseError(
                        f"expected ')' to close {name}() call at line "
                        f"{tok.line}, col {tok.col}; got {tok.text!r}"
                    )
                self._bump()
                return _MATH_FUNCS[name.upper()](*args)
            # Bare identifier: must resolve via the VARIABLE/#DEFINE table.
            if name in self.defines:
                self._bump()
                return self.defines[name]
            raise SVRFParseError(
                f"unresolved identifier {name!r} in numeric expression at "
                f"line {t.line}, col {t.col} (not declared as a #DEFINE or "
                f"VARIABLE; declare it or use a literal)"
            )
        raise SVRFParseError(
            f"expected numeric atom (NUMBER, '(', IDENT) at line {t.line}, "
            f"col {t.col}; got {t.kind} {t.text!r}"
        )

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

    ``#DEFINE NAME <numeric>`` directives are pre-scanned and used to
    resolve identifier-form thresholds (``EXT NWi PPOD < NW_S_5 ABUT < 90``).
    """
    toks = _tokenize(src)
    return _Parser(toks, src).parse_deck()


# ── #DEFINE pre-scan ────────────────────────────────────────────────────────

# Match ``#DEFINE NAME 0.5`` / ``#DEFINE NAME -0.5e-3`` etc. The
# trailing context is permissive — line comments after the value are
# ignored. Identifier names follow C-style ID rules.
_NUM_RE = r"(-?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?)"

# ``#DEFINE NAME 0.5`` (some decks; rare for numeric thresholds).
#
# Variable / define names follow the manual's name rules (page 67):
# letters, digits, periods, underscores. They MAY start with a digit
# (real foundry decks do this for layer-family rules). We exclude
# pure-numeric "names" downstream so ``#DEFINE 0.5 1`` doesn't pick
# ``0.5`` as the symbol-table key.
_DEFINE_RE = re.compile(
    rf"^\s*\#\s*DEFINE\s+([A-Za-z0-9_][A-Za-z0-9_.]*)\s+{_NUM_RE}\b",
    re.MULTILINE,
)
_VARIABLE_RE = re.compile(
    rf"^\s*VARIABLE\s+([A-Za-z0-9_][A-Za-z0-9_.]*)\s+{_NUM_RE}\b",
    re.MULTILINE,
)


def _scan_defines(src: str) -> dict[str, float]:
    """Build a ``name → float`` symbol table from numeric ``#DEFINE`` /
    ``VARIABLE`` declarations.

    Real foundry decks declare thresholds either way::

        #DEFINE M1_W_1   0.090
        VARIABLE  NW_S_5  0.160

    Boolean-flag ``#DEFINE``s (no value) are skipped silently — they
    never appear as rule thresholds.
    """
    out: dict[str, float] = {}
    for pat in (_DEFINE_RE, _VARIABLE_RE):
        for m in pat.finditer(src):
            name = m.group(1)
            # Skip degenerate "names" that are purely numeric — those
            # are actually number literals, not identifiers.
            if name.replace(".", "").isdigit():
                continue
            try:
                out[name] = float(m.group(2))
            except ValueError:                   # pragma: no cover — defensive
                continue
    return out


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
