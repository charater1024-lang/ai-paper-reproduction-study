"""Add unmistakable exercise/solution role banners to every paired notebook."""

from __future__ import annotations

from pathlib import Path

import nbformat
from nbformat.v4 import new_code_cell

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOKS = ROOT / "notebooks"
MARKER = "<!-- paired-role-banner -->"
ARTIFACT_FILENAMES = (
    "notebook_text_classifier.pt",
    "notebook_10_tiny_lm.pt",
    "rnn_sequence_classifier.pt",
    "ai_lab16_best_tiny_transformer.pt",
    "tiny_shapes_cnn.pt",
    "ai_lab21_gpu_resume.pt",
)
HARDWARE_CELL_TAG = "hardware-profile-cell"


def normalize_artifact_names(source: str, role: str) -> str:
    for filename in ARTIFACT_FILENAMES:
        stem, suffix = filename.rsplit(".", maxsplit=1)
        source = source.replace(f"{stem}_exercise.{suffix}", filename)
        source = source.replace(f"{stem}_solution.{suffix}", filename)
        source = source.replace(filename, f"{stem}_{role}.{suffix}")
    return source


def hardware_profile_cell():
    return new_code_cell(
        """# 이 컴퓨터를 감지해 안전한 시작값을 확인합니다. 각 실습은 더 작은 값으로도 동작합니다.
from llm_engineering_lab.hardware import HardwareProfile, recommend_training_profile

LAB_HARDWARE = HardwareProfile.detect()
LAB_PROFILE = recommend_training_profile(LAB_HARDWARE)
print("hardware:", LAB_HARDWARE)
print("recommended profile:", LAB_PROFILE)""",
        metadata={"tags": [HARDWARE_CELL_TAG, "keep-synchronized"]},
    )


def normalized_code(source: str) -> str:
    for filename in ARTIFACT_FILENAMES:
        stem, suffix = filename.rsplit(".", maxsplit=1)
        source = source.replace(f"{stem}_exercise.{suffix}", filename)
        source = source.replace(f"{stem}_solution.{suffix}", filename)
    return "\n".join(
        line
        for line in source.splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    )


def update_banner(path: Path, role: str) -> bool:
    notebook = nbformat.read(path, as_version=4)
    if not notebook.cells or notebook.cells[0].cell_type != "markdown":
        raise AssertionError(f"{path}: 첫 셀이 Markdown이어야 합니다")
    changed = False
    if MARKER not in notebook.cells[0].source:
        expected_emoji = "🟧" if role == "exercise" else "🟩"
        if expected_emoji in notebook.cells[0].source:
            notebook.cells[0].source = f"{MARKER}\n" + notebook.cells[0].source
        elif role == "exercise":
            banner = (
                f"{MARKER}\n> 🟧 **실습 코드 · 왼쪽 창** — `TODO`/빈 구현을 직접 완성하세요. "
                "정답을 보기 전에 입력·출력 shape를 먼저 예측하세요.\n\n"
            )
            notebook.cells[0].source = banner + notebook.cells[0].source
        else:
            banner = (
                f"{MARKER}\n> 🟩 **정답 코드 · 오른쪽 창** — 왼쪽과 같은 셀 위치의 참고 구현입니다. "
                "필요한 한 줄만 확인하고 다시 실습 창으로 돌아가세요.\n\n"
            )
            notebook.cells[0].source = banner + notebook.cells[0].source
        changed = True

    number = int(path.name[:2])
    if number >= 13 and not any(
        HARDWARE_CELL_TAG in cell.metadata.get("tags", []) for cell in notebook.cells
    ):
        notebook.cells.insert(1, hardware_profile_cell())
        changed = True

    for cell in notebook.cells:
        normalized_source = normalize_artifact_names(cell.source, role)
        if normalized_source != cell.source:
            cell.source = normalized_source
            changed = True

    notebook.metadata.setdefault("paired_learning", {})["role"] = role
    nbformat.write(notebook, path)
    return changed


def main() -> None:
    changed = 0
    for role, directory_name in (("exercise", "exercises"), ("solution", "solutions")):
        for path in sorted((NOTEBOOKS / directory_name).glob("[0-9][0-9]_*.ipynb")):
            changed += int(update_banner(path, role))

    for exercise_path in sorted((NOTEBOOKS / "exercises").glob("[0-9][0-9]_*.ipynb")):
        solution_path = NOTEBOOKS / "solutions" / exercise_path.name
        exercise = nbformat.read(exercise_path, as_version=4)
        solution = nbformat.read(solution_path, as_version=4)
        if len(exercise.cells) != len(solution.cells):
            raise AssertionError(f"{exercise_path.name}: 셀 수가 다릅니다")
        for exercise_cell, solution_cell in zip(
            exercise.cells, solution.cells, strict=True
        ):
            solution_cell.id = exercise_cell.id
        meaningful_differences = sum(
            exercise_cell.cell_type == "code"
            and normalized_code(exercise_cell.source)
            != normalized_code(solution_cell.source)
            for exercise_cell, solution_cell in zip(
                exercise.cells, solution.cells, strict=True
            )
        )
        exercise.metadata.setdefault("paired_learning", {})["hidden_cell_count"] = (
            meaningful_differences
        )
        solution.metadata.setdefault("paired_learning", {})["hidden_cell_count"] = (
            meaningful_differences
        )
        nbformat.write(exercise, exercise_path)
        nbformat.write(solution, solution_path)

    print(f"role/profile/artifact normalization complete: changed notebooks={changed}")


if __name__ == "__main__":
    main()
