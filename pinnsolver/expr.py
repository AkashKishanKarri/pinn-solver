"""Restricted mathematical expression grammar (part of module M2).

User text is tokenised and parsed by a small recursive-descent parser that
builds SymPy objects directly. Nothing the user types is ever passed to
``eval``/``exec``, ``sympy.parse_expr`` or ``lambdify`` (all of which execute
generated Python code).

Grammar (whitespace ignored)::

    expr    := term (('+' | '-') term)*
    term    := unary (('*' | '/') unary | <implicit> power)*
    unary   := ('+' | '-') unary | power
    power   := atom (('^' | '**') unary)?          # right associative
    atom    := NUMBER | NAME | NAME '(' expr ')' | '(' expr ')'

Implicit multiplication is accepted after a number (``2u``, ``2(x+1)``), after
a closing parenthesis (``(1 - u)u``, ``(a)(b)``) and between two names separated
by a space (``k u``). ``xy`` is one name, never ``x*y``, and a name followed by
``(`` is always a function call.

Before tokenising, :func:`normalize_text` maps common typographic characters
(prime, minus sign, multiplication dot, pi, superscript 2/3) to their ASCII forms.

Evaluation of the resulting SymPy trees (numerically, in NumPy or PyTorch) is
done by :func:`evaluate`, a tree walker over a fixed whitelist of node types.
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Callable, Dict, List

import sympy as sp

MAX_EXPR_CHARS = 400
MAX_NODES = 300
MAX_DEPTH = 40
MAX_ABS_NUMBER = 1e8
MAX_CONST_LOG10 = 300.0  # a constant power b**e must satisfy |e| * |log10|b|| <= this (stays inside float range)
MAX_EXPAND_TERMS = 5000  # upper bound on the number of terms an expression may expand into


class ParseError(ValueError):
    """Raised for any input that does not follow the restricted grammar."""


# Allowed functions: name -> SymPy class.
FUNCTIONS: Dict[str, Callable] = {
    "sin": sp.sin,
    "cos": sp.cos,
    "tan": sp.tan,
    "exp": sp.exp,
    "log": sp.log,
    "ln": sp.log,
    "sqrt": sp.sqrt,
    "sinh": sp.sinh,
    "cosh": sp.cosh,
    "tanh": sp.tanh,
    "abs": sp.Abs,
}
CONSTANTS = {"pi": sp.pi, "e": sp.E}
RESERVED = set(FUNCTIONS) | set(CONSTANTS)

_TOKEN_RE = re.compile(
    r"""
    (?P<ws>\s+)
  | (?P<num>(?:\d+\.\d*|\.\d+|\d+)(?:[eE][+-]?\d+)?)
  | (?P<name>[A-Za-z][A-Za-z0-9_]*'*)
  | (?P<op>\*\*|[+\-*/^(),])
    """,
    re.VERBOSE,
)


# Typographic characters that users paste or type instead of the ASCII ones.
_UNICODE = {
    "\u2032": "'", "\u2019": "'", "\u2018": "'", "`": "'", "\u00b4": "'", "\u2033": "''",
    "\u2212": "-", "\u2013": "-", "\u2014": "-",
    "\u00b7": "*", "\u22c5": "*", "\u00d7": "*", "\u2217": "*", "\u00f7": "/",
    "\u03c0": "pi", "\u00b2": "^2", "\u00b3": "^3", "\u00a0": " ",
}


def normalize_text(text: str) -> str:
    """Replace typographic look-alikes (prime, minus, dot, pi, superscripts) by ASCII."""
    if not isinstance(text, str):
        return text
    for a, b in _UNICODE.items():
        if a in text:
            text = text.replace(a, b)
    return text


@dataclass
class Token:
    kind: str  # 'num' | 'name' | 'op' | 'end'
    text: str
    pos: int


def tokenize(text: str) -> List[Token]:
    if not isinstance(text, str):
        raise ParseError("Expression must be a string.")
    text = normalize_text(text)
    if len(text) > MAX_EXPR_CHARS:
        raise ParseError(f"Expression is too long ({len(text)} characters; limit {MAX_EXPR_CHARS}).")
    tokens: List[Token] = []
    pos = 0
    while pos < len(text):
        m = _TOKEN_RE.match(text, pos)
        if not m:
            raise ParseError(f"Unexpected character {text[pos]!r} at position {pos + 1}.")
        kind = m.lastgroup
        if kind != "ws":
            tokens.append(Token(kind, m.group(), pos))
        pos = m.end()
    tokens.append(Token("end", "", len(text)))
    return tokens


class _Parser:
    def __init__(self, text: str, resolve_name: Callable[[str, int], sp.Basic]):
        self.text = text
        self.tokens = tokenize(text)
        self.i = 0
        self.resolve_name = resolve_name
        self.depth = 0

    # -- helpers -------------------------------------------------------
    @property
    def tok(self) -> Token:
        return self.tokens[self.i]

    def advance(self) -> Token:
        t = self.tokens[self.i]
        self.i += 1
        return t

    def expect(self, text: str) -> None:
        if self.tok.text != text:
            found = self.tok.text or "end of input"
            raise ParseError(f"Expected {text!r} at position {self.tok.pos + 1}, found {found!r}.")
        self.advance()

    def _enter(self) -> None:
        self.depth += 1
        if self.depth > MAX_DEPTH:
            raise ParseError("Expression is nested too deeply.")

    def _leave(self) -> None:
        self.depth -= 1

    # -- grammar -------------------------------------------------------
    def parse(self) -> sp.Basic:
        if self.tok.kind == "end":
            raise ParseError("Expression is empty.")
        e = self.expr()
        if self.tok.kind != "end":
            raise ParseError(f"Unexpected {self.tok.text!r} at position {self.tok.pos + 1}.")
        return e

    def expr(self) -> sp.Basic:
        self._enter()
        left = self.term()
        while self.tok.text in ("+", "-"):
            op = self.advance().text
            right = self.term()
            left = left + right if op == "+" else left - right
        self._leave()
        return left

    def term(self) -> sp.Basic:
        self._enter()
        left = self.unary()
        while True:
            if self.tok.text in ("*", "/"):
                op = self.advance().text
                right = self.unary()
                if op == "*":
                    left = left * right
                else:
                    if right == 0:
                        raise ParseError("Division by zero.")
                    left = left / right
            elif self._implicit_ok():
                left = left * self.power()
            else:
                break
        self._leave()
        return left

    def _implicit_ok(self) -> bool:
        # Implicit multiplication: "2u", "2(x+1)", "(1-u)u", "(a)(b)" and "k u".
        # A name directly followed by "(" is never a product (it is a function call), and two
        # names need white space between them because "ku" is a single name.
        prev = self.tokens[self.i - 1] if self.i > 0 else None
        if prev is None:
            return False
        if prev.kind == "num" or prev.text == ")":
            return self.tok.kind == "name" or self.tok.text == "("
        if prev.kind == "name" and prev.text not in FUNCTIONS:
            return self.tok.kind == "name"
        return False

    def unary(self) -> sp.Basic:
        self._enter()
        if self.tok.text in ("+", "-"):
            op = self.advance().text
            v = self.unary()
            self._leave()
            return v if op == "+" else -v
        v = self.power()
        self._leave()
        return v

    def power(self) -> sp.Basic:
        self._enter()
        base = self.atom()
        if self.tok.text in ("^", "**"):
            self.advance()
            exponent = self.unary()
            _check_power(base, exponent)
            base = sp.Pow(base, exponent)
        self._leave()
        return base

    def atom(self) -> sp.Basic:
        t = self.tok
        if t.kind == "num":
            self.advance()
            value = float(t.text)
            if not math.isfinite(value) or abs(value) > MAX_ABS_NUMBER:
                raise ParseError(f"Number {t.text} is out of the allowed range (|v| <= {MAX_ABS_NUMBER:g}).")
            if re.fullmatch(r"\d+", t.text):
                return sp.Integer(int(t.text))
            return sp.Rational(t.text) if "e" not in t.text.lower() else sp.Float(value)
        if t.text == "(":
            self.advance()
            e = self.expr()
            self.expect(")")
            return e
        if t.kind == "name":
            self.advance()
            name = t.text
            if name in FUNCTIONS:
                if self.tok.text != "(":
                    raise ParseError(f"Function {name!r} must be followed by '(' at position {self.tok.pos + 1}.")
                self.advance()
                arg = self.expr()
                if self.tok.text == ",":
                    raise ParseError(f"Function {name!r} takes exactly one argument.")
                self.expect(")")
                return FUNCTIONS[name](arg)
            if name in CONSTANTS:
                return CONSTANTS[name]
            if self.tok.text == "(":
                raise ParseError(
                    f"{name!r} is not an allowed function (allowed: {', '.join(sorted(FUNCTIONS))}). "
                    f"If you mean a product, write {name}*(...)."
                )
            return self.resolve_name(name, t.pos)
        found = t.text or "end of input"
        raise ParseError(f"Unexpected {found!r} at position {t.pos + 1}.")


def parse_expression(text: str, resolve_name: Callable[[str, int], sp.Basic]) -> sp.Basic:
    """Parse ``text`` with the restricted grammar.

    ``resolve_name(name, pos)`` maps every bare identifier to a SymPy object
    or raises :class:`ParseError`; it is how callers whitelist symbols.
    """
    result = _Parser(text, resolve_name).parse()
    check_size(result)
    check_constants(result)
    return result


def check_size(e: sp.Basic) -> None:
    n = sum(1 for _ in sp.preorder_traversal(e))
    if n > MAX_NODES:
        raise ParseError(f"Expression is too complex ({n} nodes; limit {MAX_NODES}).")


_MATH_FUNCS = {
    sp.sin: math.sin, sp.cos: math.cos, sp.tan: math.tan, sp.exp: math.exp, sp.log: math.log,
    sp.sinh: math.sinh, sp.cosh: math.cosh, sp.tanh: math.tanh, sp.Abs: abs,
}


def const_value(node: sp.Basic) -> float:
    """Value of a constant (symbol-free) expression as a finite real float, in plain float arithmetic.

    Raises :class:`ParseError` if the constant is complex, infinite or undefined (``sqrt(-1)``,
    ``exp(1000)``, ``log(0)``). Never calls ``evalf``, which can run for a very long time on towers
    such as ``exp(exp(exp(100)))``.
    """

    def rec(n) -> float:
        if n.is_Number or n in (sp.pi, sp.E):
            return float(n)
        if n.is_Add:
            return math.fsum(rec(a) for a in n.args)
        if n.is_Mul:
            out = 1.0
            for a in n.args:
                out *= rec(a)
            return out
        if n.is_Pow:
            b, e = rec(n.args[0]), rec(n.args[1])
            if b < 0 and e != int(e):
                raise ValueError("root of a negative number")
            return math.pow(b, e)
        f = _MATH_FUNCS.get(n.func)
        if f is not None and len(n.args) == 1:
            return f(rec(n.args[0]))
        raise ValueError("not a real number")

    try:
        value = rec(node)
    except (ValueError, OverflowError, ZeroDivisionError, TypeError):
        value = math.nan
    if not math.isfinite(value):
        raise ParseError(f"The constant {_short(node)} is not a finite real number.")
    return value


def _short(node: sp.Basic) -> str:
    text = sp.sstr(node)
    return text if len(text) <= 60 else text[:57] + "..."


def check_constants(e: sp.Basic) -> None:
    """Reject expressions that contain a constant sub-expression which is not a finite real number."""
    if not e.free_symbols:
        const_value(e)
        return
    for a in e.args:
        check_constants(a)


def _check_power(base: sp.Basic, exponent: sp.Basic) -> None:
    """Refuse powers that SymPy would evaluate into astronomically large exact numbers (e.g. 9^9^9)."""
    if exponent.free_symbols:
        return
    ev = const_value(exponent)
    if base.free_symbols:
        return
    bv = const_value(base)
    if bv == 0:
        if ev < 0:
            raise ParseError("Division by zero.")
        return
    if bv < 0 and ev != int(ev):
        raise ParseError(f"The constant ({_short(base)})^({_short(exponent)}) is not a real number.")
    if abs(ev) * abs(math.log10(abs(bv))) > MAX_CONST_LOG10:
        raise ParseError(f"The constant ({_short(base)})^({_short(exponent)}) is out of the allowed range.")


def expanded_terms(e: sp.Basic, cap: int = 10**9) -> int:
    """Upper bound on the number of terms of ``expand(e)`` (so that the expansion can be refused if huge)."""
    if e.is_Atom:
        return 1
    if e.is_Add:
        return min(cap, sum(expanded_terms(a, cap) for a in e.args))
    if e.is_Mul:
        out = 1
        for a in e.args:
            out = min(cap, out * expanded_terms(a, cap))
        return out
    if e.is_Pow:
        m = expanded_terms(e.args[0], cap)
        inner = max(m, expanded_terms(e.args[1], cap))
        ex = e.args[1]
        if ex.is_Integer and int(ex) > 1 and m > 1:
            n = int(ex)
            return min(cap, math.comb(n + m - 1, m - 1)) if n + m < 5000 else cap
        return inner
    return max([expanded_terms(a, cap) for a in e.args] or [1])


def check_expandable(e: sp.Basic) -> None:
    n = expanded_terms(e)
    if n > MAX_EXPAND_TERMS:
        raise ParseError(f"Expression is too complex: it would expand into about {n:.3g} terms (limit {MAX_EXPAND_TERMS}).")


# ---------------------------------------------------------------------------
# Safe numerical evaluation of SymPy trees (NumPy or PyTorch backend)
# ---------------------------------------------------------------------------

def _np_backend():
    import numpy as np

    return {
        sp.sin: np.sin, sp.cos: np.cos, sp.tan: np.tan, sp.exp: np.exp, sp.log: np.log,
        sp.sinh: np.sinh, sp.cosh: np.cosh, sp.tanh: np.tanh, sp.Abs: np.abs, sp.sign: np.sign,
        "sqrt": np.sqrt, "full_like": lambda ref, v: np.full_like(np.asarray(ref, dtype=float), v),
    }


def _torch_backend():
    import torch

    return {
        sp.sin: torch.sin, sp.cos: torch.cos, sp.tan: torch.tan, sp.exp: torch.exp, sp.log: torch.log,
        sp.sinh: torch.sinh, sp.cosh: torch.cosh, sp.tanh: torch.tanh, sp.Abs: torch.abs, sp.sign: torch.sign,
        "sqrt": torch.sqrt, "full_like": lambda ref, v: torch.full_like(ref, v),
    }


_BACKENDS = {"numpy": _np_backend, "torch": _torch_backend}


def evaluate(e: sp.Basic, env: Dict[sp.Symbol, object], backend: str = "numpy", like=None):
    """Evaluate SymPy expression ``e`` numerically by walking its tree.

    ``env`` maps every free symbol to an array/tensor. ``like`` is an array or
    tensor whose shape is used to broadcast constant results.
    """
    fns = _BACKENDS[backend]()
    if like is None and env:
        like = next(iter(env.values()))

    def rec(node):
        if not node.free_symbols:
            # constant sub-expression, e.g. sinh(1) or pi**2: fold to a float
            if not node.is_Number:
                const_value(node)  # raises for complex/infinite constants without calling evalf
            try:
                value = complex(node.evalf())
            except (TypeError, ValueError) as exc:
                raise ParseError(f"Cannot evaluate constant {node}: {exc}") from None
            if value.imag != 0 or not math.isfinite(value.real):
                raise ParseError(f"Constant {node} is not a finite real number.")
            return value.real
        if node.is_Symbol:
            if node not in env:
                raise ParseError(f"No value supplied for symbol {node}.")
            return env[node]
        if node.is_Number or node in (sp.pi, sp.E):
            return float(node)
        if node.is_Add:
            out = rec(node.args[0])
            for a in node.args[1:]:
                out = out + rec(a)
            return out
        if node.is_Mul:
            out = rec(node.args[0])
            for a in node.args[1:]:
                out = out * rec(a)
            return out
        if node.is_Pow:
            base, ex = node.args
            if ex == sp.Rational(1, 2):
                return fns["sqrt"](rec(base))
            if ex.is_Integer:
                k = int(ex)
                b = rec(base)
                if k >= 0:
                    return b ** k
                return 1.0 / (b ** (-k))
            return rec(base) ** rec(ex)
        f = fns.get(node.func)
        if f is not None and len(node.args) == 1:
            return f(rec(node.args[0]))
        raise ParseError(f"Unsupported operation in expression: {node.func.__name__}.")

    out = rec(e)
    if isinstance(out, float):
        if like is None:
            return out
        return fns["full_like"](like, out)
    return out
