"""Create exercise/solution pairs from the source 00-12 curriculum notebooks.

This generator replaces the matching files under ``notebooks/exercises`` and
``notebooks/solutions``.  Learners must copy any answers they want to keep
before rebuilding; otherwise their edits in an exercise notebook are lost.
"""

from __future__ import annotations

import copy
from pathlib import Path

import nbformat
from nbformat.v4 import new_markdown_cell
from notebook_api_explanations import annotate_notebook

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOKS = ROOT / "notebooks"
EXERCISES = NOTEBOOKS / "exercises"
SOLUTIONS = NOTEBOOKS / "solutions"


def previous_heading(cells: list, cell_index: int) -> str:
    for cell in reversed(cells[:cell_index]):
        if cell.cell_type != "markdown":
            continue
        for line in cell.source.splitlines():
            if line.lstrip().startswith("#"):
                return line.lstrip("# ").strip()
    return "핵심 구현"


def choose_targets(notebook) -> list[int]:
    candidates: list[int] = []
    preferred: list[int] = []
    for index, cell in enumerate(notebook.cells):
        if cell.cell_type != "code" or index < 3:
            continue
        source = cell.source.strip()
        nonempty_lines = [line for line in source.splitlines() if line.strip()]
        if len(nonempty_lines) < 4:
            continue
        candidates.append(index)
        if "TODO" in source or "def " in source or "class " in source:
            preferred.append(index)

    desired = 3 if len(candidates) < 10 else 4
    selected: list[int] = []
    for index in preferred:
        if index not in selected:
            selected.append(index)
        if len(selected) == desired:
            return sorted(selected)

    if candidates:
        positions = [
            round(i * (len(candidates) - 1) / max(desired - 1, 1))
            for i in range(desired)
        ]
        for position in positions:
            index = candidates[position]
            if index not in selected:
                selected.append(index)
    return sorted(selected[:desired])


def clear_outputs(notebook) -> None:
    for cell in notebook.cells:
        if cell.cell_type == "code":
            cell.outputs = []
            cell.execution_count = None


def build_pair(source_path: Path) -> tuple[Path, Path, int]:
    original = nbformat.read(source_path, as_version=4)
    clear_outputs(original)

    # Choose challenges and capture their lesson headings before API-only
    # Markdown is inserted.  Otherwise an API signature could accidentally
    # become the TODO title when two code cells were originally consecutive.
    target_specs: list[tuple[int, str]] = []
    for original_index in choose_targets(original):
        code_ordinal = (
            sum(
                cell.cell_type == "code"
                for cell in original.cells[: original_index + 1]
            )
            - 1
        )
        target_specs.append(
            (code_ordinal, previous_heading(original.cells, original_index))
        )

    # API notes must be derived from the intact source code before the exercise
    # cells are replaced with TODO scaffolds.  Deep-copying this annotated
    # in-memory notebook gives both sides identical explanations; it does not
    # modify the root source notebook on disk.  The writes below still replace
    # learner-edited exercise files, as documented in the module warning.
    annotate_notebook(original)

    exercise = copy.deepcopy(original)
    solution = copy.deepcopy(original)
    number = source_path.name[:2]

    exercise.cells.insert(
        0,
        new_markdown_cell(
            f"""> 🟧 **실습 노트북 {number} · 빈 코드 창**
>
> 오른쪽의 같은 이름 **정답 노트북**과 셀 위치를 맞춰 보세요. 먼저 정답을 가리고 직접
> 작성한 뒤 막힐 때 한 줄씩 비교하는 방식을 권장합니다. `NotImplementedError`가 있는 셀을
> 완성하면 됩니다.""",
            metadata={"tags": ["paired-banner", "exercise"]},
        ),
    )
    solution.cells.insert(
        0,
        new_markdown_cell(
            f"""> 🟩 **정답 노트북 {number} · 참고 코드 창**
>
> 왼쪽 실습 노트북과 같은 셀 순서입니다. 결과만 복사하기보다 각 shape, 자료형, 손실값과
> 설계 이유를 먼저 예측한 뒤 필요한 부분만 확인하세요.""",
            metadata={"tags": ["paired-banner", "solution"]},
        ),
    )

    # Banner insertion shifts every original cell by one in both notebooks.
    code_indices = [
        index for index, cell in enumerate(original.cells) if cell.cell_type == "code"
    ]
    for challenge_number, (code_ordinal, title) in enumerate(
        target_specs,
        start=1,
    ):
        original_index = code_indices[code_ordinal]
        index = original_index + 1
        exercise.cells[index].source = f"""# TODO {challenge_number}: {title}
# 오른쪽 정답 창의 같은 셀을 참고하되, 먼저 아래 구현을 직접 작성하세요.
# 1) 입력과 출력의 shape/dtype을 종이에 적습니다.
# 2) 최소 구현을 작성하고 Shift+Enter로 실행합니다.
# 3) 정답과 비교한 뒤 값 하나를 바꾸고 결과를 설명합니다.
raise NotImplementedError("{number}번 실습의 TODO {challenge_number}을 완성하세요")"""
        exercise.cells[index].metadata["tags"] = ["exercise", "answer-hidden"]
        exercise.cells[index].metadata["paired_cell"] = index
        solution.cells[index].metadata["tags"] = ["solution", "reference-answer"]
        solution.cells[index].metadata["paired_cell"] = index

    exercise.metadata["paired_learning"] = {
        "role": "exercise",
        "pair": source_path.name,
        "hidden_cell_count": len(target_specs),
    }
    solution.metadata["paired_learning"] = {
        "role": "solution",
        "pair": source_path.name,
        "hidden_cell_count": len(target_specs),
    }

    EXERCISES.mkdir(parents=True, exist_ok=True)
    SOLUTIONS.mkdir(parents=True, exist_ok=True)
    exercise_path = EXERCISES / source_path.name
    solution_path = SOLUTIONS / source_path.name
    nbformat.write(exercise, exercise_path)
    nbformat.write(solution, solution_path)
    return exercise_path, solution_path, len(target_specs)


def main() -> None:
    paths = sorted(NOTEBOOKS.glob("[0-1][0-9]_*.ipynb"))
    paths = [path for path in paths if int(path.name[:2]) <= 12]
    if len(paths) != 13:
        raise AssertionError(
            f"00~12 원본 노트북 13개가 필요합니다: {len(paths)}개 발견"
        )

    for path in paths:
        exercise_path, solution_path, target_count = build_pair(path)
        print(
            f"PAIRED {path.name}: hidden={target_count}, "
            f"exercise={exercise_path.relative_to(ROOT)}, "
            f"solution={solution_path.relative_to(ROOT)}"
        )


if __name__ == "__main__":
    main()
