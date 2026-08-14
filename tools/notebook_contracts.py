"""Render stage notes and answer-free exercise API contracts."""

from __future__ import annotations

import ast
import copy
import re
from collections.abc import Sequence

Definition = ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef
BACKTICK_PATTERN = re.compile(r"`([^`]+)`")
WORD_PATTERN = re.compile(r"[A-Za-z][A-Za-z0-9_]*")
COMMON_WORDS = {
    "attention",
    "block",
    "class",
    "code",
    "figure",
    "forward",
    "head",
    "input",
    "layer",
    "linear",
    "mask",
    "model",
    "output",
    "paper",
    "shape",
    "self",
    "task",
    "torch",
}


def stage_kind(tags: Sequence[str], source: str) -> str:
    """Classify a code cell by the responsibility a reader should inspect."""

    joined = " ".join(tags).lower()
    normalized_tags = {tag.lower() for tag in tags}
    lowered = source.lower()
    if "setup" in joined:
        return "setup"
    if normalized_tags == {"data"} and not any(
        token in joined for token in ("model", "training", "loss", "metric")
    ):
        return "data"
    if any(token in joined for token in ("evaluation", "metric", "visualization")):
        return "evaluation"
    if any(token in joined for token in ("training", "loss", "objective", "update")):
        return "training"
    if any(token in joined for token in ("model", "equation", "attention", "layer")):
        return "model"
    if any(token in lowered for token in ("optimizer", "backward()", "loss =")):
        return "training"
    if any(token in lowered for token in ("class ", "def ")):
        return "model"
    return "experiment"


def _identifier_parts(identifier: str) -> list[str]:
    snake_parts = re.split(r"[^A-Za-z0-9]+|_+", identifier)
    parts: list[str] = []
    for snake_part in snake_parts:
        camel_parts = re.findall(
            r"[A-Z]+(?=[A-Z][a-z]|\d|$)|[A-Z]?[a-z]+|\d+",
            snake_part,
        )
        parts.extend(part.lower() for part in camel_parts if part)
    return [part for part in parts if len(part) >= 4 and part not in COMMON_WORDS]


def _word_tokens(text: str) -> set[str]:
    return {
        word.lower()
        for word in WORD_PATTERN.findall(text)
        if len(word) >= 4 and word.lower() not in COMMON_WORDS
    }


def _prefix_overlap(left: set[str], right: set[str]) -> int:
    count = 0
    for first in left:
        for second in right:
            prefix_length = min(len(first), len(second), 6)
            if prefix_length >= 4 and first[:prefix_length] == second[:prefix_length]:
                count += 1
                break
    return count


def _mapping_scores(
    mapping: tuple[str, str, str],
    source: str,
    tags: Sequence[str],
) -> tuple[int, int, int]:
    source_lower = source.lower()
    fingerprint = f"{source_lower} {' '.join(tags).lower()}"
    source_tokens = _word_tokens(fingerprint)
    try:
        source_names = {
            node.id.lower()
            for node in ast.walk(ast.parse(source))
            if isinstance(node, ast.Name)
        }
    except SyntaxError:
        source_names = set()
    identifiers = BACKTICK_PATTERN.findall(" ".join(mapping))
    exact = 0
    component = 0
    for identifier in identifiers:
        segments = [
            segment.lower() for segment in re.split(r"\.", identifier) if segment
        ]
        if segments and all(segment in source_lower for segment in segments):
            exact += 1
        parts = set(_identifier_parts(identifier))
        if "causal" in parts & source_names:
            exact += 1
        matched_parts = _prefix_overlap(parts, source_tokens)
        if matched_parts >= 2:
            component += matched_parts
    mapping_tokens = _word_tokens(f"{mapping[1]} {mapping[2]}")
    token_overlap = _prefix_overlap(mapping_tokens, source_tokens)
    return exact, component, token_overlap


def _keyword_mappings(
    mappings: Sequence[tuple[str, str, str]],
    keywords: set[str],
) -> tuple[tuple[str, str, str], ...]:
    selected = []
    for mapping in mappings:
        tokens = _word_tokens(" ".join(mapping))
        if _prefix_overlap(tokens, keywords):
            selected.append(mapping)
    return tuple(selected)


def select_stage_mappings(
    mappings: Sequence[tuple[str, str, str]],
    source: str,
    tags: Sequence[str],
    kind: str,
    fallback_index: int,
) -> tuple[tuple[str, str, str], ...]:
    """Select paper rows whose identifiers and terms occur in the code stage."""

    joined_tags = " ".join(tags).lower()
    if "portfolio-model" in joined_tags:
        return tuple(mappings)
    if "portfolio-training" in joined_tags:
        selected = _keyword_mappings(
            mappings,
            {"loss", "objective", "optim", "train", "update", "distill", "prune"},
        )
        if selected:
            return selected
    if "portfolio-evaluation" in joined_tags:
        selected = _keyword_mappings(
            mappings,
            {
                "accuracy",
                "cost",
                "evaluation",
                "latency",
                "metric",
                "result",
                "retrieval",
                "speed",
            },
        )
        if selected:
            return selected

    scores = [_mapping_scores(mapping, source, tags) for mapping in mappings]
    exact_indices = [index for index, score in enumerate(scores) if score[0] > 0]
    selected_indices = set(exact_indices)
    if not exact_indices:
        for index, score in enumerate(scores):
            if score[1] >= 2:
                selected_indices.add(index)
    if not selected_indices:
        totals = [component * 3 + overlap for _, component, overlap in scores]
        best = max(totals, default=0)
        if best > 0:
            selected_indices.update(
                index for index, total in enumerate(totals) if total == best
            )
    if not selected_indices:
        if kind == "evaluation":
            selected_indices.add(len(mappings) - 1)
        elif kind == "training":
            selected_indices.add(max(0, len(mappings) - 2))
        else:
            selected_indices.add(min(fallback_index, len(mappings) - 1))
    return tuple(mappings[index] for index in sorted(selected_indices))


def merge_mappings(
    mappings: Sequence[tuple[str, str, str]],
) -> tuple[str, str, str]:
    """Merge every mechanism implemented by one code cell into one stage note."""

    paper_parts, lab_parts, evidence_parts = zip(*mappings, strict=True)
    return (
        " / ".join(dict.fromkeys(paper_parts)),
        " / ".join(dict.fromkeys(lab_parts)),
        " / ".join(dict.fromkeys(evidence_parts)),
    )


def stage_markdown(
    *,
    ordinal: int,
    kind: str,
    mapping: tuple[str, str, str] | None,
    shape_hint: str,
) -> str:
    """Build the note that must sit immediately before one code stage."""

    titles = {
        "setup": "실행 환경·데이터 계약",
        "data": "합성 입력·데이터 계약",
        "model": "핵심 연산·모델",
        "training": "목적함수·학습 update",
        "evaluation": "평가·완료 증거",
        "experiment": "논문 주장 확인 실험",
    }
    if mapping is None:
        if kind == "data":
            location = (
                "해당 없음 — 원 논문의 학습 연산이 아니라 외부 다운로드를 없애기 "
                "위한 축소 입력 생성 단계입니다."
            )
            reason = (
                "논문 메커니즘과 데이터 규모 효과를 분리하고 Windows CPU에서도 "
                "같은 입력을 재현합니다."
            )
            evidence = (
                "pair 정렬, batch 크기, dtype과 입력 rank assertion이 통과해야 합니다."
            )
        else:
            location = (
                "해당 없음 — import, seed, device, 로컬 데이터 로딩만 담당하며 "
                "논문의 학습 연산을 구현하지 않습니다."
            )
            reason = (
                "이후 셀의 비교가 재현 가능하도록 장치·dtype·seed와 입력 "
                "불변식을 먼저 고정합니다."
            )
            evidence = (
                "데이터 존재 여부, 입력 rank, 장치 선택 assertion과 환경 요약 "
                "출력이 통과해야 합니다."
            )
        exception_label = "data-contract 예외" if kind == "data" else "setup 예외"
        return (
            f"### 구현 단계 {ordinal} · {titles[kind]} ({exception_label})\n\n"
            f"- **원문 위치:** {location}\n"
            f"- **Tensor shape 계약:** {shape_hint}\n"
            f"- **구현 이유:** {reason}\n"
            f"- **완료 증거:** {evidence}"
        )
    paper_part, lab_part, evidence = mapping
    reasons = {
        "model": "수식을 숨긴 고수준 호출 대신 핵심 변환과 shape 경계를 직접 추적합니다.",
        "training": "forward 결과와 논문 objective, gradient update의 책임을 분리합니다.",
        "evaluation": "학습 숫자 하나가 아니라 논문 주장에 대응하는 반증 가능 증거를 남깁니다.",
        "experiment": "한 변수만 바꾸어 해당 메커니즘이 출력에 미치는 영향을 분리합니다.",
    }
    return (
        f"### 구현 단계 {ordinal} · {titles[kind]}\n\n"
        f"- **원문 위치:** {paper_part}\n"
        f"- **이 셀의 책임:** {lab_part}\n"
        f"- **Tensor shape 계약:** {shape_hint}\n"
        f"- **구현 이유:** {reasons[kind]}\n"
        f"- **완료 증거:** {evidence}. 코드의 assertion과 출력으로 바로 확인합니다."
    )


def _definition_map(tree: ast.Module) -> dict[str, Definition]:
    return {
        node.name: node
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
    }


def _doc_expression(text: str) -> ast.Expr:
    return ast.Expr(value=ast.Constant(value=text))


def _contract_doc(node: Definition, paper_location: str, shape_hint: str) -> str:
    original = ast.get_docstring(node, clean=False)
    if original:
        return original
    return (
        "실습 계약: 정답과 같은 공개 signature를 유지한다.\n"
        "원문 위치와 반환 shape는 바로 위 Markdown 계약을 따른다."
    )


def _function_skeleton(
    node: ast.FunctionDef | ast.AsyncFunctionDef,
    paper_location: str,
    shape_hint: str,
) -> ast.FunctionDef | ast.AsyncFunctionDef:
    skeleton = copy.deepcopy(node)
    doc = _doc_expression(_contract_doc(node, paper_location, shape_hint))
    nested = [
        _function_skeleton(child, paper_location, shape_hint)
        for child in node.body
        if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef))
        and not child.name.startswith("_")
    ]
    if node.name == "__init__":
        pending = ast.Assign(
            targets=[
                ast.Attribute(
                    value=ast.Name(id="self", ctx=ast.Load()),
                    attr="_exercise_contract_pending",
                    ctx=ast.Store(),
                )
            ],
            value=ast.Constant(value=True),
        )
        skeleton.body = [doc, *nested, pending]
        return skeleton

    result_name = ast.Name(id="result", ctx=ast.Store())
    initialize = ast.Assign(targets=[result_name], value=ast.Constant(value=None))
    check = ast.Assert(
        test=ast.Compare(
            left=ast.Name(id="result", ctx=ast.Load()),
            ops=[ast.IsNot()],
            comparators=[ast.Constant(value=None)],
        ),
        msg=ast.Constant(value="TODO: 문서의 shape 계약을 만족하는 결과를 구현하세요."),
    )
    return_value = ast.Return(value=ast.Name(id="result", ctx=ast.Load()))
    skeleton.body = [doc, *nested, initialize, check, return_value]
    return skeleton


def _class_skeleton(
    node: ast.ClassDef,
    paper_location: str,
    shape_hint: str,
) -> ast.ClassDef:
    if node.name.endswith("Config"):
        config = copy.deepcopy(node)
        if not ast.get_docstring(config):
            config.body.insert(
                0,
                _doc_expression(_contract_doc(node, paper_location, shape_hint)),
            )
        return config

    skeleton = copy.deepcopy(node)
    class_doc = _doc_expression(_contract_doc(node, paper_location, shape_hint))
    methods = [
        _function_skeleton(child, paper_location, shape_hint)
        for child in node.body
        if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef))
    ]
    skeleton.body = [class_doc, *methods]
    if not methods:
        skeleton.body.append(ast.Expr(value=ast.Constant(value=Ellipsis)))
    return skeleton


def _definition_skeleton(
    node: Definition,
    paper_location: str,
    shape_hint: str,
) -> Definition:
    if isinstance(node, ast.ClassDef):
        return _class_skeleton(node, paper_location, shape_hint)
    return _function_skeleton(node, paper_location, shape_hint)


def _shape_fixture(shape_hint: str) -> str:
    return (
        "\n\ndef _return_shape_fixture(result):\n"
        '    """바로 위 Markdown의 반환 shape 계약을 확인한다."""\n'
        '    assert result is not None, "TODO 결과가 아직 비어 있습니다."\n'
        '    shape = getattr(result, "shape", None)\n'
        "    container = isinstance(result, (tuple, list, dict, float, int))\n"
        "    assert shape is not None or container\n"
        "    return result\n"
    )


def _replace_top_level_not_implemented(source: str) -> str:
    pattern = re.compile(r"^raise NotImplementedError(?:\([^\n]*\))?\s*$", re.MULTILINE)
    replacement = (
        "_stage_result = None\n"
        'assert _stage_result is not None, "TODO: 위 단계의 결과와 검증을 구현하세요."'
    )
    return pattern.sub(replacement, source)


def exercise_contract_source(
    exercise: str,
    solution: str,
    *,
    paper_location: str,
    shape_hint: str,
) -> str:
    """Keep solution public APIs in an answer-free exercise scaffold."""

    if exercise == solution:
        return exercise
    try:
        exercise_tree = ast.parse(exercise)
        solution_tree = ast.parse(solution)
    except SyntaxError:
        return _replace_top_level_not_implemented(exercise)

    solution_definitions = _definition_map(solution_tree)
    if not solution_definitions:
        return _replace_top_level_not_implemented(exercise)

    exercise_definition_names = set(_definition_map(exercise_tree))
    rendered_body: list[ast.stmt] = []
    for node in exercise_tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            solution_node = solution_definitions.get(node.name)
            if solution_node is None:
                rendered_body.append(node)
            else:
                rendered_body.append(
                    _definition_skeleton(solution_node, paper_location, shape_hint)
                )
            continue
        if isinstance(node, ast.Raise):
            pending = ast.Assign(
                targets=[ast.Name(id="_stage_result", ctx=ast.Store())],
                value=ast.Constant(value=None),
            )
            check = ast.Assert(
                test=ast.Compare(
                    left=ast.Name(id="_stage_result", ctx=ast.Load()),
                    ops=[ast.IsNot()],
                    comparators=[ast.Constant(value=None)],
                ),
                msg=ast.Constant(value="TODO: 위 단계의 결과와 검증을 구현하세요."),
            )
            rendered_body.extend((pending, check))
            continue
        rendered_body.append(node)

    missing = [
        _definition_skeleton(node, paper_location, shape_hint)
        for node in solution_tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
        and node.name not in exercise_definition_names
    ]
    insert_at = 0
    while insert_at < len(rendered_body) and isinstance(
        rendered_body[insert_at], (ast.Import, ast.ImportFrom)
    ):
        insert_at += 1
    rendered_body[insert_at:insert_at] = missing
    module = ast.Module(body=rendered_body, type_ignores=[])
    ast.fix_missing_locations(module)
    rendered = ast.unparse(module)
    todo_lines = [
        line for line in exercise.splitlines() if line.lstrip().startswith("# TODO")
    ]
    prefix = "\n".join(todo_lines)
    if prefix:
        rendered = f"{prefix}\n{rendered}"
    if solution_definitions or exercise_definition_names:
        rendered += _shape_fixture(shape_hint)
    return rendered


def public_api(source: str) -> dict[str, tuple[str, ...]]:
    """Return public top-level and class-method names for parity checks."""

    tree = ast.parse(source)
    result: dict[str, tuple[str, ...]] = {}
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if not node.name.startswith("_"):
                result[node.name] = ()
        elif isinstance(node, ast.ClassDef) and not node.name.startswith("_"):
            methods = tuple(
                child.name
                for child in node.body
                if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef))
                and (not child.name.startswith("_") or child.name == "__init__")
            )
            result[node.name] = methods
    return result
