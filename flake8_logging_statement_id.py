"""
Lint rule: Enforce that every Python logging call includes
extra={"statement_id": "<hardcoded-uuid>"}.

Supports auto-fix (--fix) via AST transformation: mutates the call node in the
AST, unparses just that node, and does a targeted replacement in the original
source — preserving all formatting, comments, and whitespace outside the call.

Error codes:
    LSI001 - Logging call missing 'extra' keyword argument          [auto-fixable]
    LSI002 - 'extra' argument is not a dict literal                 [manual fix]
    LSI003 - 'extra' dict missing 'statement_id' key                [auto-fixable]
    LSI004 - 'statement_id' value is not a hardcoded UUID string    [manual fix]
    LSI005 - 'statement_id' value is not a valid UUID format        [auto-fixable]

A call is only treated as a logging call when its receiver looks like a logger
(see _is_logging_call). Suppress a single call with `# noqa: LSI001` (or a bare
`# noqa`) on any line of the call.

Usage:
    # Check only (exit code 1 if violations found):
    python flake8_logging_statement_id.py src/**/*.py

    # Auto-fix (modifies files in-place):
    python flake8_logging_statement_id.py --fix src/**/*.py

    # Also treat receivers named `audit` or `tracer` as loggers:
    python flake8_logging_statement_id.py --logger-names '^(audit|tracer)$' src/**/*.py

    # As a flake8 plugin (check only — flake8 has no auto-fix):
    pip install -e .
    flake8 --select=LSI your_project/
"""

from __future__ import annotations

import argparse
import ast
import copy
import glob
import io
import re
import sys
import tokenize
import uuid
from typing import Any, Generator

__version__ = "1.0.0"

UUID_PATTERN = re.compile(
    r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$",
    re.IGNORECASE,
)

LOGGING_METHODS = frozenset(
    {"debug", "info", "warning", "error", "critical", "exception", "log", "fatal"}
)

FIXABLE_CODES = frozenset({"LSI001", "LSI003", "LSI005"})

# Functions/methods whose return value is a logger.
LOGGER_FACTORIES = frozenset({"getLogger", "getChild"})

# Receiver names that are assumed to be loggers: `logging`, `log`, `logger`
# (any case), optionally behind a snake_case prefix (`_logger`, `audit_log`),
# or a camelCase name ending in `Log`/`Logger` (`auditLogger`).
# Deliberately does not match `catalog`, `dialog`, `math`, `outcome`, ...
DEFAULT_LOGGER_NAME_PATTERN = re.compile(
    r"(?i:(?:^|_)(?:log|logger|logging))$|[a-z0-9](?:Log|Logger)$"
)

NOQA_PATTERN = re.compile(
    r"#\s*noqa(?::\s*(?P<codes>[A-Z]+[0-9]+(?:[,\s]+[A-Z]+[0-9]+)*))?",
    re.IGNORECASE,
)


def _error_code(msg: str) -> str:
    return msg[:6]


# ── AST analysis ────────────────────────────────────────────────────────────


def _terminal_name(node: ast.expr) -> str | None:
    """Last segment of a (dotted) name: `a.b.c` → "c", `x` → "x", else None."""
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return node.attr
    return None


def _is_logger_factory_call(node: ast.expr | None) -> bool:
    """True for `logging.getLogger(...)`, `getLogger(...)`, `x.getChild(...)`."""
    return isinstance(node, ast.Call) and _terminal_name(node.func) in LOGGER_FACTORIES


def _collect_logger_names(tree: ast.AST) -> set[str]:
    """
    Names bound to a logger anywhere in this file, regardless of what they are
    called: targets of `x = logging.getLogger(...)` / `self.x = ...` and
    aliases from `import logging as x`.
    """
    names: set[str] = set()
    for node in ast.walk(tree):
        targets: list[ast.expr] = []
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name == "logging" and alias.asname:
                    names.add(alias.asname)
        elif isinstance(node, ast.Assign) and _is_logger_factory_call(node.value):
            targets = node.targets
        elif isinstance(node, ast.AnnAssign) and _is_logger_factory_call(node.value):
            targets = [node.target]
        for target in targets:
            name = _terminal_name(target)
            if name is not None:
                names.add(name)
    return names


def _has_message_arg(node: ast.Call, method: str) -> bool:
    """
    A real logging call always passes a message (and a level for `.log`).
    `future.exception()` and friends do not.
    """
    if any(kw.arg == "msg" for kw in node.keywords):
        return True
    if any(isinstance(arg, ast.Starred) for arg in node.args):
        return True
    return len(node.args) >= (2 if method == "log" else 1)


def _is_logging_call(
    node: ast.Call,
    known_loggers: set[str],
    logger_names: re.Pattern[str] | None = None,
) -> bool:
    """
    Check if an AST Call node is a logging method invocation.

    The method name must be a logging method, the call must pass a message,
    and the receiver must look like a logger: either a direct
    `logging.getLogger(...)` call, a name bound to one in this file
    (`known_loggers`), or a name matching DEFAULT_LOGGER_NAME_PATTERN or the
    user-supplied `logger_names` pattern.

    Matches: logging.info(...), logger.warning(...), self.logger.error(...),
             log.debug(...), LOG.critical(...)
    Skips:   retry_state.outcome.exception(), math.log(x), parser.error("...")
    """
    func = node.func
    if not isinstance(func, ast.Attribute) or func.attr not in LOGGING_METHODS:
        return False
    if not _has_message_arg(node, func.attr):
        return False
    if _is_logger_factory_call(func.value):
        return True
    name = _terminal_name(func.value)
    if name is None:
        return False
    if name in known_loggers or DEFAULT_LOGGER_NAME_PATTERN.search(name):
        return True
    return logger_names is not None and logger_names.search(name) is not None


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
            "LSI001 Logging call missing 'extra' keyword argument with 'statement_id'",
            node.lineno,
            node.col_offset,
        )

    if not isinstance(extra_kw.value, ast.Dict):
        return (
            "LSI002 'extra' must be a dict literal so 'statement_id' can be statically verified",
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
            "LSI003 'extra' dict missing required 'statement_id' key",
            extra_kw.value.lineno,
            extra_kw.value.col_offset,
        )

    if not isinstance(statement_id_value, ast.Constant) or not isinstance(
        statement_id_value.value, str
    ):
        return (
            "LSI004 'statement_id' must be a hardcoded UUID string literal",
            statement_id_value.lineno,
            statement_id_value.col_offset,
        )

    if not UUID_PATTERN.match(statement_id_value.value):
        return (
            f"LSI005 'statement_id' value '{statement_id_value.value}' is not a valid UUID "
            f"(expected xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx)",
            statement_id_value.lineno,
            statement_id_value.col_offset,
        )

    return None


def _find_violations(
    tree: ast.AST, logger_names: re.Pattern[str] | None = None
) -> list[tuple[ast.Call, str, int, int]]:
    """All violations in a parsed file as (call_node, message, line, col)."""
    known_loggers = _collect_logger_names(tree)
    violations: list[tuple[ast.Call, str, int, int]] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        if not _is_logging_call(node, known_loggers, logger_names):
            continue
        error = _check_extra_kwarg(node)
        if error is not None:
            msg, line, col = error
            violations.append((node, msg, line, col))
    return violations


# ── `# noqa` handling (standalone mode; flake8 applies its own) ─────────────


def _collect_noqa(source: str) -> dict[int, frozenset[str]]:
    """
    Map line number → codes listed in a `# noqa` comment on that line.
    An empty set means a bare `# noqa`, which suppresses every code.
    """
    noqa: dict[int, frozenset[str]] = {}
    try:
        for tok in tokenize.generate_tokens(io.StringIO(source).readline):
            if tok.type != tokenize.COMMENT:
                continue
            match = NOQA_PATTERN.search(tok.string)
            if match is None:
                continue
            codes = match.group("codes")
            noqa[tok.start[0]] = frozenset(
                code.upper() for code in re.split(r"[,\s]+", codes or "") if code
            )
    except (tokenize.TokenError, SyntaxError):
        pass
    return noqa


def _is_suppressed(node: ast.Call, msg: str, noqa: dict[int, frozenset[str]]) -> bool:
    """True if any line of the call carries a `# noqa` covering this error."""
    code = _error_code(msg)
    for line in range(node.lineno, (node.end_lineno or node.lineno) + 1):
        codes = noqa.get(line)
        if codes is None:
            continue
        if not codes or code.startswith(tuple(codes)):
            return True
    return False


# ── Flake8 checker (check-only) ────────────────────────────────────────────


class LoggingStatementIdChecker:
    """Flake8 AST checker that enforces statement_id in logging extra dicts."""

    name = "flake8-logging-statement-id"
    version = __version__
    logger_names: re.Pattern[str] | None = None

    def __init__(self, tree: ast.AST) -> None:
        self.tree = tree

    @classmethod
    def add_options(cls, option_manager: Any) -> None:
        option_manager.add_option(
            "--logger-names",
            default="",
            parse_from_config=True,
            help="Regex for additional receiver names to treat as loggers",
        )

    @classmethod
    def parse_options(cls, options: Any) -> None:
        cls.logger_names = (
            re.compile(options.logger_names) if options.logger_names else None
        )

    def run(self) -> Generator[tuple[int, int, str, type], None, None]:
        for _, msg, line, col in _find_violations(self.tree, self.logger_names):
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

    if code == "LSI001":
        # No `extra` kwarg → add extra={"statement_id": "<uuid>"}
        extra_dict = ast.Dict(keys=[key_node], values=[value_node])
        extra_kwarg = ast.keyword(arg="extra", value=extra_dict)
        fixed.keywords.append(extra_kwarg)
        return fixed

    elif code == "LSI003":
        # extra={...} exists but no statement_id → prepend it into the dict
        for kw in fixed.keywords:
            if kw.arg == "extra" and isinstance(kw.value, ast.Dict):
                kw.value.keys.insert(0, key_node)
                kw.value.values.insert(0, value_node)
                return fixed

    elif code == "LSI005":
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


def check_file(
    filepath: str, logger_names: re.Pattern[str] | None = None
) -> list[tuple[int, int, str]]:
    """Check a single Python file (no fixes), returning (line, col, message) list."""
    with open(filepath) as f:
        source = f.read()

    try:
        tree = ast.parse(source, filename=filepath)
    except SyntaxError as e:
        return [(e.lineno or 0, e.offset or 0, f"SyntaxError: {e.msg}")]

    noqa = _collect_noqa(source)
    return [
        (line, col, msg)
        for node, msg, line, col in _find_violations(tree, logger_names)
        if not _is_suppressed(node, msg, noqa)
    ]


class _Replacement:
    """A source span to replace: source[start:end] → new_text."""

    __slots__ = ("start", "end", "new_text")

    def __init__(self, start: int, end: int, new_text: str):
        self.start = start
        self.end = end
        self.new_text = new_text


def fix_file(
    filepath: str, logger_names: re.Pattern[str] | None = None
) -> tuple[list[tuple[int, int, str]], int]:
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
    noqa = _collect_noqa(source)
    violations = [
        violation
        for violation in _find_violations(tree, logger_names)
        if not _is_suppressed(violation[0], violation[1], noqa)
    ]

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
    parser.add_argument(
        "--logger-names",
        default=None,
        metavar="REGEX",
        help=(
            "Regex for additional receiver names to treat as loggers, on top of "
            "the built-in log/logger/logging pattern (e.g. '^(audit|tracer)$')"
        ),
    )
    args = parser.parse_args()

    logger_names: re.Pattern[str] | None = None
    if args.logger_names:
        try:
            logger_names = re.compile(args.logger_names)
        except re.error as e:
            parser.error(f"--logger-names is not a valid regex: {e}")

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
            errors, num_fixed = fix_file(filepath, logger_names)
            total_fixed += num_fixed
        else:
            errors = check_file(filepath, logger_names)

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
