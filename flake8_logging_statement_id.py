"""
Lint rule: Enforce that every Python logging call includes
extra={"statement_id": "<hardcoded-uuid>"}.

Supports auto-fix (--fix) via AST transformation: mutates the call node in the
AST, unparses just that node, and does a targeted replacement in the original
source — preserving all formatting, comments, and whitespace outside the call.

Error codes:
    LOG001 - Logging call missing 'extra' keyword argument          [auto-fixable]
    LOG002 - 'extra' argument is not a dict literal                 [manual fix]
    LOG003 - 'extra' dict missing 'statement_id' key                [auto-fixable]
    LOG004 - 'statement_id' value is not a hardcoded UUID string    [manual fix]
    LOG005 - 'statement_id' value is not a valid UUID format        [auto-fixable]

Usage:
    # Check only (exit code 1 if violations found):
    python flake8_logging_statement_id.py src/**/*.py

    # Auto-fix (modifies files in-place):
    python flake8_logging_statement_id.py --fix src/**/*.py

    # As a flake8 plugin (check only — flake8 has no auto-fix):
    pip install -e .
    flake8 --select=LOG your_project/
"""

from __future__ import annotations

import argparse
import ast
import copy
import glob
import re
import sys
import uuid
from typing import Generator

__version__ = "1.0.0"

UUID_PATTERN = re.compile(
    r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$",
    re.IGNORECASE,
)

LOGGING_METHODS = frozenset(
    {"debug", "info", "warning", "error", "critical", "exception", "log", "fatal"}
)

FIXABLE_CODES = frozenset({"LOG001", "LOG003", "LOG005"})


def _error_code(msg: str) -> str:
    return msg[:6]


# ── AST analysis ────────────────────────────────────────────────────────────


def _is_logging_call(node: ast.Call) -> bool:
    """
    Check if an AST Call node is a logging method invocation.

    Matches: logging.info(...), logger.warning(...), self.logger.error(...),
             log.debug(...), LOG.critical(...)
    """
    func = node.func
    return isinstance(func, ast.Attribute) and func.attr in LOGGING_METHODS


def _check_extra_kwarg(node: ast.Call) -> tuple[str, int, int] | None:
    """
    Validate the 'extra' keyword argument on a logging call.
    Returns (error_message, line, col) if invalid, or None if valid.
    """
    extra_kw = None
    for kw in node.keywords:
        if kw.arg == "extra":
            extra_kw = kw
            break

    if extra_kw is None:
        return (
            "LOG001 Logging call missing 'extra' keyword argument with 'statement_id'",
            node.lineno,
            node.col_offset,
        )

    if not isinstance(extra_kw.value, ast.Dict):
        return (
            "LOG002 'extra' must be a dict literal so 'statement_id' can be statically verified",
            extra_kw.value.lineno,
            extra_kw.value.col_offset,
        )

    dict_node: ast.Dict = extra_kw.value

    statement_id_value = None
    for key, value in zip(dict_node.keys, dict_node.values):
        if key is None:
            continue
        if isinstance(key, ast.Constant) and key.value == "statement_id":
            statement_id_value = value
            break

    if statement_id_value is None:
        return (
            "LOG003 'extra' dict missing required 'statement_id' key",
            extra_kw.value.lineno,
            extra_kw.value.col_offset,
        )

    if not isinstance(statement_id_value, ast.Constant) or not isinstance(
        statement_id_value.value, str
    ):
        return (
            "LOG004 'statement_id' must be a hardcoded UUID string literal",
            statement_id_value.lineno,
            statement_id_value.col_offset,
        )

    if not UUID_PATTERN.match(statement_id_value.value):
        return (
            f"LOG005 'statement_id' value '{statement_id_value.value}' is not a valid UUID "
            f"(expected xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx)",
            statement_id_value.lineno,
            statement_id_value.col_offset,
        )

    return None


# ── Flake8 checker (check-only) ────────────────────────────────────────────


class LoggingStatementIdChecker:
    """Flake8 AST checker that enforces statement_id in logging extra dicts."""

    name = "flake8-logging-statement-id"
    version = __version__

    def __init__(self, tree: ast.AST) -> None:
        self.tree = tree

    def run(self) -> Generator[tuple[int, int, str, type], None, None]:
        for node in ast.walk(self.tree):
            if not isinstance(node, ast.Call):
                continue
            if not _is_logging_call(node):
                continue
            error = _check_extra_kwarg(node)
            if error is not None:
                msg, line, col = error
                yield (line, col, msg, type(self))


# ── AST-based auto-fixer ───────────────────────────────────────────────────


def _make_statement_id_pair(new_uuid: str) -> tuple[ast.Constant, ast.Constant]:
    """Create the AST nodes for a "statement_id": "<uuid>" key-value pair."""
    key_node = ast.Constant(value="statement_id")
    value_node = ast.Constant(value=new_uuid)
    return key_node, value_node


def _fix_call_node(node: ast.Call, error_msg: str) -> ast.Call | None:
    """
    Return a fixed copy of the Call node, or None if not auto-fixable.

    Mutates a deep copy of the node's AST — never the original.
    """
    code = _error_code(error_msg)
    if code not in FIXABLE_CODES:
        return None

    fixed = copy.deepcopy(node)
    new_uuid = str(uuid.uuid4())
    key_node, value_node = _make_statement_id_pair(new_uuid)

    if code == "LOG001":
        # No `extra` kwarg → add extra={"statement_id": "<uuid>"}
        extra_dict = ast.Dict(keys=[key_node], values=[value_node])
        extra_kwarg = ast.keyword(arg="extra", value=extra_dict)
        fixed.keywords.append(extra_kwarg)
        return fixed

    elif code == "LOG003":
        # extra={...} exists but no statement_id → prepend it into the dict
        for kw in fixed.keywords:
            if kw.arg == "extra" and isinstance(kw.value, ast.Dict):
                kw.value.keys.insert(0, key_node)
                kw.value.values.insert(0, value_node)
                return fixed

    elif code == "LOG005":
        # statement_id exists but bad UUID → replace the value
        for kw in fixed.keywords:
            if kw.arg == "extra" and isinstance(kw.value, ast.Dict):
                for i, (k, _) in enumerate(zip(kw.value.keys, kw.value.values)):
                    if isinstance(k, ast.Constant) and k.value == "statement_id":
                        kw.value.values[i] = value_node
                        return fixed

    return None


def _offset(source: str, line: int, col: int) -> int:
    """Convert 1-based line + 0-based col to a 0-based character offset."""
    off = 0
    for i, src_line in enumerate(source.splitlines(keepends=True), start=1):
        if i == line:
            return off + col
        off += len(src_line)
    return off + col


def _get_call_span(source: str, node: ast.Call) -> tuple[int, int] | None:
    """
    Get the source character span [start, end) of an entire Call node,
    using AST end_lineno / end_col_offset. Returns None if position
    information is missing.
    """
    if node.end_lineno is None or node.end_col_offset is None:
        return None
    start = _offset(source, node.lineno, node.col_offset)
    end = _offset(source, node.end_lineno, node.end_col_offset)
    return start, end


# ── File-level operations ──────────────────────────────────────────────────


def check_file(filepath: str) -> list[tuple[int, int, str]]:
    """Check a single Python file (no fixes), returning (line, col, message) list."""
    with open(filepath) as f:
        source = f.read()

    try:
        tree = ast.parse(source, filename=filepath)
    except SyntaxError as e:
        return [(e.lineno or 0, e.offset or 0, f"SyntaxError: {e.msg}")]

    errors = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        if not _is_logging_call(node):
            continue
        result = _check_extra_kwarg(node)
        if result:
            msg, line, col = result
            errors.append((line, col, msg))

    return errors


class _Replacement:
    """A source span to replace: source[start:end] → new_text."""

    __slots__ = ("start", "end", "new_text")

    def __init__(self, start: int, end: int, new_text: str):
        self.start = start
        self.end = end
        self.new_text = new_text


def fix_file(filepath: str) -> tuple[list[tuple[int, int, str]], int]:
    """
    Check a file and apply auto-fixes via AST transformation.

    For each fixable violation:
      1. Deep-copy the Call node
      2. Mutate the copy (add/modify the extra kwarg)
      3. ast.unparse() the fixed copy
      4. Replace the original call's source span with the unparsed result

    Returns (all_errors, num_fixed).
    """
    with open(filepath) as f:
        source = f.read()

    try:
        tree = ast.parse(source, filename=filepath)
    except SyntaxError as e:
        return ([(e.lineno or 0, e.offset or 0, f"SyntaxError: {e.msg}")], 0)

    # Collect violations with their AST nodes
    violations: list[tuple[ast.Call, str, int, int]] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        if not _is_logging_call(node):
            continue
        result = _check_extra_kwarg(node)
        if result:
            msg, line, col = result
            violations.append((node, msg, line, col))

    if not violations:
        return ([], 0)

    # Build replacements
    replacements: list[_Replacement] = []
    errors: list[tuple[int, int, str]] = []
    fixed_count = 0

    for node, msg, line, col in violations:
        fixed_node = _fix_call_node(node, msg)
        if fixed_node is not None:
            # Unparse the fixed AST node to get clean Python source
            new_text = ast.unparse(fixed_node)
            span = _get_call_span(source, node)
            if span is None:
                errors.append((line, col, msg))
                continue
            start, end = span
            replacements.append(_Replacement(start, end, new_text))
            fixed_count += 1
            errors.append((line, col, f"{msg} [fixed]"))
        else:
            errors.append((line, col, msg))

    # Apply replacements in reverse order so offsets stay valid
    if replacements:
        replacements.sort(key=lambda r: r.start, reverse=True)
        result_source = source
        for repl in replacements:
            result_source = (
                result_source[: repl.start] + repl.new_text + result_source[repl.end :]
            )

        # Safety: validate the result parses
        try:
            ast.parse(result_source, filename=filepath)
        except SyntaxError:
            return (
                [(line, col, msg) for _, msg, line, col in violations],
                0,
            )

        with open(filepath, "w") as f:
            f.write(result_source)

    return (errors, fixed_count)


# ── CLI ─────────────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Enforce statement_id UUID in Python logging calls."
    )
    parser.add_argument(
        "files",
        nargs="+",
        help="Python files or glob patterns to check",
    )
    parser.add_argument(
        "--fix",
        action="store_true",
        default=False,
        help="Auto-fix: insert missing statement_id UUIDs in-place",
    )
    args = parser.parse_args()

    exit_code = 0
    total_fixed = 0

    filepaths: list[str] = []
    for pattern in args.files:
        expanded = glob.glob(pattern, recursive=True)
        if expanded:
            filepaths.extend(expanded)
        else:
            filepaths.append(pattern)

    for filepath in filepaths:
        if not filepath.endswith(".py"):
            continue

        if args.fix:
            errors, num_fixed = fix_file(filepath)
            total_fixed += num_fixed
        else:
            errors = check_file(filepath)

        for lineno, col, msg in errors:
            print(f"{filepath}:{lineno}:{col}: {msg}")
            if not args.fix or "[fixed]" not in msg:
                exit_code = 1

    if args.fix and total_fixed > 0:
        print(
            f"\n✅ Auto-fixed {total_fixed} logging call(s) with new statement_id UUIDs."
        )
        exit_code = 1

    sys.exit(exit_code)


if __name__ == "__main__":
    main()
