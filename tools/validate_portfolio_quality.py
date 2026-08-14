"""Validate readability and implementation depth of learner-facing artifacts."""

from __future__ import annotations

import argparse
import ast
import io
import tokenize
from dataclasses import dataclass
from pathlib import Path

import nbformat

ROOT = Path(__file__).resolve().parents[1]
MAX_CODE_LINE_LENGTH = 100
MAX_NOTEBOOK_CODE_LINE_LENGTH = 88
FORBIDDEN_MARKDOWN_TOKENS = {
    "epsilon-epsilon": r"use `\epsilon-\epsilon_\theta`",
    r"\quads": r"separate `\quad` from the following symbol",
    r"\quade": r"separate `\quad` from the following symbol",
}


@dataclass(frozen=True, slots=True)
class NotebookRule:
    min_code_lines: int
    min_classes: int
    min_callables: int
    require_equation: bool
    require_implementation_reason: bool
    require_stage_explanations: bool


PAPER_RULE = NotebookRule(
    min_code_lines=100,
    min_classes=1,
    min_callables=4,
    require_equation=True,
    require_implementation_reason=True,
    require_stage_explanations=True,
)
FIELD_RULE = NotebookRule(
    min_code_lines=110,
    min_classes=1,
    min_callables=7,
    require_equation=True,
    require_implementation_reason=True,
    require_stage_explanations=True,
)
BASE_RULE = NotebookRule(
    min_code_lines=0,
    min_classes=0,
    min_callables=0,
    require_equation=False,
    require_implementation_reason=False,
    require_stage_explanations=False,
)
EXERCISE_PAPER_RULE = NotebookRule(
    min_code_lines=40,
    min_classes=1,
    min_callables=3,
    require_equation=True,
    require_implementation_reason=True,
    require_stage_explanations=True,
)
EXERCISE_FIELD_RULE = NotebookRule(
    min_code_lines=60,
    min_classes=1,
    min_callables=7,
    require_equation=True,
    require_implementation_reason=True,
    require_stage_explanations=True,
)


@dataclass(frozen=True, slots=True)
class PublicDefinition:
    """A learner-visible class or callable API extracted from a notebook."""

    kind: str
    qualified_name: str
    parameters: tuple[str, ...] = ()


def source_text(cell: nbformat.NotebookNode) -> str:
    source = cell.get("source", "")
    return "".join(source) if isinstance(source, list) else str(source)


def semicolon_statements(source: str) -> list[int]:
    """Return lines using semicolons as Python statement separators."""

    result: list[int] = []
    try:
        tokens = tokenize.generate_tokens(io.StringIO(source).readline)
        for token in tokens:
            if token.type == tokenize.OP and token.string == ";":
                result.append(token.start[0])
    except (IndentationError, tokenize.TokenError):
        # The AST parser reports a more actionable syntax failure later.
        return result
    return result


def notebook_python_metrics(
    path: Path,
) -> tuple[int, int, int, list[str]]:
    notebook = nbformat.read(path, as_version=4)
    code_lines = 0
    class_count = 0
    callable_count = 0
    failures: list[str] = []

    for cell_index, cell in enumerate(notebook.cells, start=1):
        if cell.cell_type != "code":
            continue
        source = source_text(cell)
        lines = source.splitlines()
        code_lines += sum(bool(line.strip()) for line in lines)

        for line_index, line in enumerate(lines, start=1):
            if len(line) > MAX_NOTEBOOK_CODE_LINE_LENGTH:
                failures.append(
                    f"cell {cell_index}, line {line_index}: "
                    f"{len(line)} characters "
                    f"(maximum {MAX_NOTEBOOK_CODE_LINE_LENGTH})"
                )

        for line_index in semicolon_statements(source):
            failures.append(
                f"cell {cell_index}, line {line_index}: semicolon joins multiple Python statements"
            )

        try:
            tree = ast.parse(source)
        except SyntaxError as error:
            failures.append(
                f"cell {cell_index}: Python syntax error at line {error.lineno}: {error.msg}"
            )
            continue

        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                class_count += 1
                if node.end_lineno == node.lineno:
                    failures.append(
                        f"cell {cell_index}, line {node.lineno}: "
                        f"one-line class body `{node.name}`"
                    )
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                callable_count += 1
                if node.end_lineno == node.lineno and not _is_protocol_stub(node):
                    failures.append(
                        f"cell {cell_index}, line {node.lineno}: "
                        f"one-line function body `{node.name}`"
                    )

        if cell.get("outputs"):
            failures.append(f"cell {cell_index}: saved outputs must be cleared")
        if cell.get("execution_count") is not None:
            failures.append(f"cell {cell_index}: execution_count must be cleared")

    return code_lines, class_count, callable_count, failures


def _is_protocol_stub(node: ast.FunctionDef | ast.AsyncFunctionDef) -> bool:
    return (
        len(node.body) == 1
        and isinstance(node.body[0], ast.Expr)
        and isinstance(node.body[0].value, ast.Constant)
        and node.body[0].value.value is Ellipsis
    )


def _is_setup_cell(cell: nbformat.NotebookNode) -> bool:
    tags = set(cell.get("metadata", {}).get("tags", ()))
    return bool(tags & {"setup", "local-data", "synthetic-data"})


def validate_stage_explanations(
    notebook: nbformat.NotebookNode,
) -> list[str]:
    """Require a paper-to-code contract immediately before substantive code."""

    failures: list[str] = []
    required_markers = (
        (
            "paper location",
            ("논문의 어느 부분", "원문 위치", "원 논문", "§", "Eq.", "Figure"),
        ),
        ("tensor shape", ("shape", "모양", "입력", "출력", "스칼라")),
        ("implementation reason", ("구현 이유", "선택 이유", "왜")),
        ("completion evidence", ("완료 증거", "검증", "assert", "metric", "지표")),
    )

    for index, cell in enumerate(notebook.cells):
        if cell.cell_type != "code" or _is_setup_cell(cell):
            continue
        if index == 0 or notebook.cells[index - 1].cell_type != "markdown":
            failures.append(
                f"cell {index + 1}: substantive code needs an immediately "
                "preceding paper-stage markdown cell"
            )
            continue

        explanation = source_text(notebook.cells[index - 1])
        for label, alternatives in required_markers:
            if not any(marker in explanation for marker in alternatives):
                failures.append(
                    f"cell {index + 1}: preceding markdown is missing {label}"
                )
    return failures


def validate_notebook(path: Path, rule: NotebookRule) -> list[str]:
    notebook = nbformat.read(path, as_version=4)
    markdown = "\n".join(
        source_text(cell) for cell in notebook.cells if cell.cell_type == "markdown"
    )
    code_lines, classes, callables, failures = notebook_python_metrics(path)

    for token, guidance in FORBIDDEN_MARKDOWN_TOKENS.items():
        if token in markdown:
            failures.append(f"malformed LaTeX token `{token}`; {guidance}")

    if code_lines < rule.min_code_lines:
        failures.append(
            f"only {code_lines} non-empty code lines; expected at least {rule.min_code_lines}"
        )
    if classes < rule.min_classes:
        failures.append(f"only {classes} classes; expected at least {rule.min_classes}")
    if callables < rule.min_callables:
        failures.append(
            f"only {callables} functions/methods; expected at least {rule.min_callables}"
        )

    if rule.require_equation:
        if markdown.count("$$") < 2:
            failures.append("missing a block LaTeX equation (`$$ ... $$`)")
        for label, alternatives in (
            ("equation role", ("수식", "objective", "loss")),
            ("code mapping", ("코드", "Task")),
            ("tensor shape", ("shape", "텐서 모양", "입력/출력")),
            ("paper location", ("원 논문", "원문 위치", "Eq.")),
        ):
            if not any(text in markdown for text in alternatives):
                failures.append(f"markdown does not explain {label}")

    if rule.require_implementation_reason and not any(
        phrase in markdown
        for phrase in (
            "구현 이유",
            "선택 이유",
            "택한 이유",
            "왜 이렇게",
            "왜 이 방식",
        )
    ):
        failures.append("markdown does not explain why the implementation was chosen")

    if rule.require_stage_explanations:
        failures.extend(validate_stage_explanations(notebook))

    return failures


def solution_notebooks(track: str) -> list[tuple[Path, NotebookRule]]:
    items: list[tuple[Path, NotebookRule]] = []
    if track in {"all", "paper"}:
        directory = ROOT / "notebooks" / "paper_reproductions" / "solutions"
        items.extend((path, PAPER_RULE) for path in sorted(directory.glob("*.ipynb")))
    if track in {"all", "field"}:
        directory = ROOT / "notebooks" / "field_reproductions"
        items.extend(
            (path, FIELD_RULE) for path in sorted(directory.glob("*/solutions/*.ipynb"))
        )
    if track in {"all", "base"}:
        directory = ROOT / "notebooks" / "solutions"
        items.extend((path, BASE_RULE) for path in sorted(directory.glob("*.ipynb")))
    return items


def learner_notebooks(track: str) -> list[tuple[Path, NotebookRule]]:
    """Return both exercises and solutions with role-appropriate depth rules."""

    items = solution_notebooks(track)
    if track in {"all", "paper"}:
        directory = ROOT / "notebooks" / "paper_reproductions" / "exercises"
        items.extend(
            (path, EXERCISE_PAPER_RULE) for path in sorted(directory.glob("*.ipynb"))
        )
    if track in {"all", "field"}:
        directory = ROOT / "notebooks" / "field_reproductions"
        items.extend(
            (path, EXERCISE_FIELD_RULE)
            for path in sorted(directory.glob("*/exercises/*.ipynb"))
        )
    if track in {"all", "base"}:
        directory = ROOT / "notebooks" / "exercises"
        items.extend((path, BASE_RULE) for path in sorted(directory.glob("*.ipynb")))
    return items


def curriculum_pairs(track: str) -> list[tuple[Path, Path]]:
    """Return exercise/solution pairs for paper-focused curricula."""

    pairs: list[tuple[Path, Path]] = []
    if track in {"all", "paper"}:
        root = ROOT / "notebooks" / "paper_reproductions"
        for solution in sorted((root / "solutions").glob("*.ipynb")):
            pairs.append((root / "exercises" / solution.name, solution))
    if track in {"all", "field"}:
        root = ROOT / "notebooks" / "field_reproductions"
        for solution in sorted(root.glob("*/solutions/*.ipynb")):
            exercise = solution.parent.parent / "exercises" / solution.name
            pairs.append((exercise, solution))
    return pairs


def _parameter_names(arguments: ast.arguments) -> tuple[str, ...]:
    names = [argument.arg for argument in arguments.posonlyargs]
    names.extend(argument.arg for argument in arguments.args)
    if arguments.vararg is not None:
        names.append(f"*{arguments.vararg.arg}")
    elif arguments.kwonlyargs:
        names.append("*")
    names.extend(argument.arg for argument in arguments.kwonlyargs)
    if arguments.kwarg is not None:
        names.append(f"**{arguments.kwarg.arg}")
    return tuple(names)


class _PublicDefinitionVisitor(ast.NodeVisitor):
    def __init__(self) -> None:
        self.scope: list[str] = []
        self.definitions: dict[tuple[str, str], PublicDefinition] = {}

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        qualified_name = ".".join([*self.scope, node.name])
        definition = PublicDefinition("class", qualified_name)
        self.definitions[(definition.kind, qualified_name)] = definition
        self.scope.append(node.name)
        self.generic_visit(node)
        self.scope.pop()

    def _visit_function(
        self,
        node: ast.FunctionDef | ast.AsyncFunctionDef,
    ) -> None:
        qualified_name = ".".join([*self.scope, node.name])
        if not node.name.startswith("_") or node.name == "__init__":
            definition = PublicDefinition(
                "function",
                qualified_name,
                _parameter_names(node.args),
            )
            self.definitions[(definition.kind, qualified_name)] = definition
        self.scope.append(node.name)
        self.generic_visit(node)
        self.scope.pop()

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self._visit_function(node)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        self._visit_function(node)


def public_definitions(path: Path) -> dict[tuple[str, str], PublicDefinition]:
    visitor = _PublicDefinitionVisitor()
    notebook = nbformat.read(path, as_version=4)
    for cell in notebook.cells:
        if cell.cell_type != "code":
            continue
        try:
            visitor.visit(ast.parse(source_text(cell)))
        except SyntaxError:
            # A learner TODO can be intentionally incomplete. Missing public APIs
            # are reported by the pair comparison below.
            continue
    return visitor.definitions


def _notebook_tree(path: Path) -> ast.Module:
    notebook = nbformat.read(path, as_version=4)
    body: list[ast.stmt] = []
    for cell in notebook.cells:
        if cell.cell_type != "code":
            continue
        body.extend(ast.parse(source_text(cell)).body)
    return ast.Module(body=body, type_ignores=[])


def _constructed_class(expression: ast.expr, classes: set[str]) -> str | None:
    if isinstance(expression, ast.Call):
        if isinstance(expression.func, ast.Name) and expression.func.id in classes:
            return expression.func.id
        if isinstance(expression.func, ast.Attribute):
            return _constructed_class(expression.func.value, classes)
    return None


class _OrchestrationCallVisitor(ast.NodeVisitor):
    def __init__(
        self,
        instances: dict[str, str],
        ignored_methods: set[str],
    ) -> None:
        self.instances = instances
        self.ignored_methods = ignored_methods
        self.loop_depth = 0
        classes = set(instances.values())
        self.meaningful: dict[str, set[str]] = {name: set() for name in classes}
        self.training_path: dict[str, set[str]] = {name: set() for name in classes}

    def _visit_loop(self, node: ast.For | ast.AsyncFor | ast.While) -> None:
        self.loop_depth += 1
        self.generic_visit(node)
        self.loop_depth -= 1

    def visit_For(self, node: ast.For) -> None:
        self._visit_loop(node)

    def visit_AsyncFor(self, node: ast.AsyncFor) -> None:
        self._visit_loop(node)

    def visit_While(self, node: ast.While) -> None:
        self._visit_loop(node)

    def visit_Call(self, node: ast.Call) -> None:
        if isinstance(node.func, ast.Name) and node.func.id in self.instances:
            class_name = self.instances[node.func.id]
            self.meaningful[class_name].add("__call__")

        if isinstance(node.func, ast.Attribute):
            owner = node.func.value
            if isinstance(owner, ast.Name) and owner.id in self.instances:
                class_name = self.instances[owner.id]
                method = node.func.attr
                if method not in self.ignored_methods:
                    self.meaningful[class_name].add(method)
                if method == "fit" or (
                    self.loop_depth > 0
                    and method in {"update", "training_step", "optimize"}
                ):
                    self.training_path[class_name].add(method)
        self.generic_visit(node)


def validate_orchestration_class_usage(path: Path) -> list[str]:
    """Reject portfolio Lab/Agent classes that are only configuration props."""

    tree = _notebook_tree(path)
    classes = {
        node.name
        for node in ast.walk(tree)
        if isinstance(node, ast.ClassDef)
        and (node.name.endswith("Lab") or node.name.endswith("Agent"))
    }
    if not classes:
        return []

    instances: dict[str, str] = {}
    for node in ast.walk(tree):
        targets: list[ast.expr] = []
        value: ast.expr | None = None
        if isinstance(node, ast.Assign):
            targets = list(node.targets)
            value = node.value
        elif isinstance(node, ast.AnnAssign):
            targets = [node.target]
            value = node.value
        if value is None:
            continue
        class_name = _constructed_class(value, classes)
        if class_name is None:
            continue
        for target in targets:
            if isinstance(target, ast.Name):
                instances[target.id] = class_name

    ignored_methods = {
        "config",
        "eval",
        "load_state_dict",
        "named_parameters",
        "parameters",
        "state_dict",
        "to",
        "train",
        "zero_grad",
    }
    call_visitor = _OrchestrationCallVisitor(instances, ignored_methods)
    call_visitor.visit(tree)

    failures: list[str] = []
    for class_name in sorted(classes):
        class_instances = [
            name for name, owner in instances.items() if owner == class_name
        ]
        if not class_instances:
            failures.append(f"orchestration class `{class_name}` is never instantiated")
        elif not call_visitor.meaningful[class_name]:
            failures.append(
                f"orchestration class `{class_name}` is instantiated but its "
                "learning/evaluation methods are never called"
            )
        elif not call_visitor.training_path[class_name]:
            failures.append(
                f"orchestration class `{class_name}` is not the main training "
                "path: call `update`/`training_step` inside the training loop "
                "or call an explicit `fit` method"
            )
        elif "evaluate" not in call_visitor.meaningful[class_name]:
            failures.append(
                f"orchestration class `{class_name}` never runs its `evaluate` method"
            )
    return failures


def validate_curriculum_pair(exercise: Path, solution: Path) -> list[str]:
    """Validate that learners see the same public API used by the solution."""

    exercise_definitions = public_definitions(exercise)
    solution_definitions = public_definitions(solution)
    failures: list[str] = []

    for key, solution_definition in sorted(solution_definitions.items()):
        exercise_definition = exercise_definitions.get(key)
        if exercise_definition is None:
            failures.append(
                "exercise is missing public "
                f"{solution_definition.kind} `{solution_definition.qualified_name}`"
            )
            continue
        if exercise_definition.parameters != solution_definition.parameters:
            failures.append(
                f"signature mismatch for `{solution_definition.qualified_name}`: "
                f"exercise {exercise_definition.parameters}, "
                f"solution {solution_definition.parameters}"
            )

    if "field_reproductions" in solution.parts:
        failures.extend(validate_orchestration_class_usage(solution))
    return failures


def validate_projects() -> list[tuple[Path, list[str]]]:
    results: list[tuple[Path, list[str]]] = []
    paths = sorted((ROOT / "projects" / "nlp").glob("*/starter.py"))
    paths.extend(sorted((ROOT / "projects" / "nlp").glob("*/solution.py")))
    for path in paths:
        source = path.read_text(encoding="utf-8")
        failures: list[str] = []
        for line_number, line in enumerate(source.splitlines(), start=1):
            if len(line) > MAX_CODE_LINE_LENGTH:
                failures.append(
                    f"line {line_number}: {len(line)} characters (maximum {MAX_CODE_LINE_LENGTH})"
                )
        for line_number in semicolon_statements(source):
            failures.append(
                f"line {line_number}: semicolon joins multiple Python statements"
            )
        tree = ast.parse(source)
        class_nodes = [
            node for node in ast.walk(tree) if isinstance(node, ast.ClassDef)
        ]
        classes = len(class_nodes)
        for node in class_nodes:
            if node.end_lineno == node.lineno:
                failures.append(
                    f"line {node.lineno}: one-line class body `{node.name}`"
                )
        function_nodes = [
            node
            for node in ast.walk(tree)
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        ]
        callables = len(function_nodes)
        for node in function_nodes:
            if node.end_lineno == node.lineno and not _is_protocol_stub(node):
                failures.append(
                    f"line {node.lineno}: one-line function body `{node.name}`"
                )
        if classes < 1:
            failures.append("missing a project-level orchestration/configuration class")
        if callables < 4:
            failures.append("expected at least four explicit functions/methods")
        results.append((path, failures))
    return results


def _call_name(node: ast.Call) -> str:
    if isinstance(node.func, ast.Name):
        return node.func.id
    if isinstance(node.func, ast.Attribute):
        return node.func.attr
    return ""


def embedded_code_sources(tree: ast.AST) -> list[tuple[int, str]]:
    """Extract notebook Python stored in curriculum helper calls."""

    sources: list[tuple[int, str]] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        name = _call_name(node)
        candidates: list[ast.expr] = []
        if name in {"code", "_raw_code"}:
            candidates = list(node.args[:2])
        elif name in {"shared_code", "_raw_shared_code"}:
            candidates = list(node.args[:1])
        elif (
            name == "_cell"
            and len(node.args) >= 3
            and isinstance(node.args[0], ast.Constant)
            and node.args[0].value == "code"
        ):
            candidates = list(node.args[1:3])

        for candidate in candidates:
            if isinstance(candidate, ast.Constant) and isinstance(candidate.value, str):
                sources.append((candidate.lineno, candidate.value))
    return sources


def validate_spec_sources(track: str) -> list[tuple[Path, list[str]]]:
    """Check that source-of-truth specs are readable before notebook generation."""

    paths: list[Path] = []
    if track in {"all", "paper"}:
        paths.extend(sorted((ROOT / "tools" / "paper_curriculum").glob("*_specs.py")))
    if track in {"all", "field"}:
        paths.extend(sorted((ROOT / "tools" / "field_curriculum").glob("*_specs.py")))

    results: list[tuple[Path, list[str]]] = []
    for path in paths:
        source = path.read_text(encoding="utf-8")
        failures: list[str] = []
        for line_number, line in enumerate(source.splitlines(), start=1):
            if len(line) > MAX_CODE_LINE_LENGTH:
                failures.append(
                    f"source line {line_number}: {len(line)} characters "
                    f"(maximum {MAX_CODE_LINE_LENGTH})"
                )

        tree = ast.parse(source)
        for literal_line, embedded in embedded_code_sources(tree):
            for embedded_line in semicolon_statements(embedded):
                failures.append(
                    f"embedded code near source line {literal_line}, "
                    f"line {embedded_line}: semicolon joins statements"
                )
            try:
                embedded_tree = ast.parse(embedded)
            except SyntaxError:
                # Some exercise cells intentionally contain an incomplete expression.
                continue
            for node in ast.walk(embedded_tree):
                if isinstance(node, ast.ClassDef) and node.end_lineno == node.lineno:
                    failures.append(
                        f"embedded code near source line {literal_line}: "
                        f"one-line class `{node.name}`"
                    )
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    if node.end_lineno == node.lineno and not _is_protocol_stub(node):
                        failures.append(
                            f"embedded code near source line {literal_line}: "
                            f"one-line function `{node.name}`"
                        )
        results.append((path, failures))
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--track",
        choices=("all", "paper", "field", "base", "projects"),
        default="all",
    )
    args = parser.parse_args()

    checked = 0
    failed = 0
    if args.track != "projects":
        for path, rule in learner_notebooks(args.track):
            checked += 1
            failures = validate_notebook(path, rule)
            if failures:
                failed += 1
                print(f"INVALID  {path.relative_to(ROOT)}")
                for failure in failures:
                    print(f"  - {failure}")
            else:
                print(f"VALID    {path.relative_to(ROOT)}")

        for path, failures in validate_spec_sources(args.track):
            checked += 1
            if failures:
                failed += 1
                print(f"INVALID  {path.relative_to(ROOT)}")
                for failure in failures:
                    print(f"  - {failure}")
            else:
                print(f"VALID    {path.relative_to(ROOT)}")

        for exercise, solution in curriculum_pairs(args.track):
            checked += 1
            failures = validate_curriculum_pair(exercise, solution)
            pair_label = solution.relative_to(ROOT)
            if failures:
                failed += 1
                print(f"INVALID  API pair for {pair_label}")
                for failure in failures:
                    print(f"  - {failure}")
            else:
                print(f"VALID    API pair for {pair_label}")

    if args.track in {"all", "projects"}:
        for path, failures in validate_projects():
            checked += 1
            if failures:
                failed += 1
                print(f"INVALID  {path.relative_to(ROOT)}")
                for failure in failures:
                    print(f"  - {failure}")
            else:
                print(f"VALID    {path.relative_to(ROOT)}")

    if failed:
        raise SystemExit(
            f"FAIL: {failed}/{checked} portfolio artifacts violate the standard"
        )
    print(f"PASS: {checked} portfolio artifacts satisfy the readability/depth standard")


if __name__ == "__main__":
    main()
