"""Build 70 paired notebooks: ten landmark papers in each of seven AI fields."""

from __future__ import annotations

import argparse
from pathlib import Path

import nbformat
from field_curriculum.common import FIELD_ORDER, FieldPaperSpec, clean
from notebook_api_explanations import append_api_notes
from notebook_style import format_notebooks

ROOT = Path(__file__).resolve().parents[1]
TRACK = ROOT / "notebooks" / "field_reproductions"
ROLE_MARKER = "<!-- field-paper-role-banner -->"


def load_specs() -> list[FieldPaperSpec]:
    from field_curriculum.compression_specs import SPECS as compression
    from field_curriculum.generative_specs import SPECS as generative
    from field_curriculum.graph_specs import SPECS as graph
    from field_curriculum.multimodal_specs import SPECS as multimodal
    from field_curriculum.nlp_specs import SPECS as nlp
    from field_curriculum.rl_specs import SPECS as rl
    from field_curriculum.vision_specs import SPECS as vision

    specs = [*vision, *nlp, *generative, *rl, *graph, *multimodal, *compression]
    expected_fields = set(FIELD_ORDER)
    actual_fields = {spec.field_id for spec in specs}
    if actual_fields != expected_fields:
        raise ValueError(
            f"field ids differ: expected={expected_fields}, actual={actual_fields}"
        )
    for field_id in FIELD_ORDER:
        field_specs = sorted(
            (spec for spec in specs if spec.field_id == field_id),
            key=lambda spec: spec.number,
        )
        numbers = [spec.number for spec in field_specs]
        if numbers != list(range(10)):
            raise ValueError(f"{field_id}: paper numbers must be 00..09, got {numbers}")
        if len({spec.filename for spec in field_specs}) != 10:
            raise ValueError(f"{field_id}: filenames must be unique")
    return sorted(
        specs, key=lambda spec: (FIELD_ORDER.index(spec.field_id), spec.number)
    )


def role_banner(spec: FieldPaperSpec, role: str) -> str:
    guidance = (
        "> 🟦 **실습본** — TODO를 직접 구현하고 assertion을 통과시키세요. "
        "정답본은 같은 위치의 필요한 부분만 확인합니다."
        if role == "exercise"
        else "> 🟩 **정답·해설본** — 같은 셀 위치의 실행 가능한 참고 구현입니다. "
        "복사하기 전에 수식과 tensor shape를 설명하세요."
    )
    return clean(
        f"""
        {ROLE_MARKER}
        {guidance}

        # {spec.field_title} · {spec.display_title}

        ## {spec.paper_title}

        - **원 논문:** [{spec.paper_title}]({spec.primary_url})
        - **저자 / 발표:** {spec.authors} · {spec.venue} ({spec.year})
        - **난이도 / 예상 시간:** {spec.difficulty} · 약 {spec.expected_minutes}분
        - **선수 지식:** {spec.prerequisites}
        - **로컬 학습 데이터:** `{spec.dataset_file}`

        이 자료는 핵심 메커니즘을 확인하는 **하드웨어 인식 교육용 미니 재현**입니다.
        PyTorch 학습은 CUDA → Apple MPS → CPU 순으로 자동 선택하고, NumPy·환경 simulation 같은
        CPU 연산은 그대로 유지합니다. 원 논문의 전체 데이터·모델·학습 예산·최종 benchmark를
        재현했다는 뜻은 아닙니다.
        """
    )


def scope_cell(spec: FieldPaperSpec) -> str:
    return clean(
        f"""
        ## 재현 범위

        **보존하는 핵심:** {spec.reproduction_goal}

        **축소하거나 생략한 부분:** {spec.original_scale}

        완료 기준은 정답과 문자가 같은 코드가 아니라, TODO 구현·불변식 assertion·작은 metric 또는
        시각화·마지막 해석을 모두 마치는 것입니다.
        """
    )


def mapping_cell(spec: FieldPaperSpec) -> str:
    rows = "\n".join(
        f"| {paper_part} | {lab_part} | {evidence} |"
        for paper_part, lab_part, evidence in spec.mappings
    )
    return clean(
        f"""
        ## 원 논문 ↔ 실습 지도

        절·수식·그림 번호는 위 primary source를 기준으로 합니다. 판본에 따라 페이지가 달라질 수
        있으므로 절 제목과 식 번호를 함께 확인하세요.

        | 원 논문의 위치 | 대응 코드·실험 | 확인할 증거 |
        |---|---|---|
        {rows}
        """
    )


def reflection_cell(spec: FieldPaperSpec, role: str) -> str:
    if role == "exercise":
        return clean(
            """
            ## 실험 기록

            1. 핵심 수식의 기호와 코드 변수 이름을 대응시키세요.
            2. 어떤 assertion이 가장 중요한 구현 오류를 잡는지 설명하세요.
            3. 이 미니 데이터로 검증할 수 없는 원 논문의 주장을 하나 적으세요.
            4. seed를 유지하고 변수 하나만 바꾼 뒤 metric 전후를 기록하세요.
            """
        )
    return clean(
        f"""
        ## 정답 해석 가이드

        핵심은 최종 숫자를 외우는 것이 아니라 **원문 위치 → 수식 → 코드 → 반증 가능한 증거**를
        연결하는 것입니다. 이 노트북은 다음 원 규모 조건을 재현하지 않았습니다:

        > {spec.original_scale}

        다음 실험에서는 데이터 크기, 모델 폭, 학습 step 중 하나만 바꾸고 동일 seed에서 비교하세요.
        """
    )


def hidden_count(spec: FieldPaperSpec) -> int:
    return sum(
        cell.cell_type == "code" and cell.exercise != cell.solution
        for cell in spec.cells
    )


def build_notebook(spec: FieldPaperSpec, role: str) -> nbformat.NotebookNode:
    sources: list[tuple[str, str, tuple[str, ...]]] = [
        ("markdown", role_banner(spec, role), ("field-paper-header", role)),
        ("markdown", scope_cell(spec), ("scope",)),
        ("markdown", mapping_cell(spec), ("paper-map",)),
    ]
    api_seen: set[str] = set()
    for cell in spec.cells:
        if cell.cell_type == "code":
            if not sources or sources[-1][0] != "markdown":
                raise ValueError(
                    f"{spec.field_id}/{spec.number:02d}: every code cell must "
                    "have an immediately preceding Markdown explanation"
                )
            markdown_type, markdown_source, markdown_tags = sources[-1]
            sources[-1] = (
                markdown_type,
                append_api_notes(markdown_source, cell.solution, api_seen),
                markdown_tags,
            )
        sources.append(
            (
                cell.cell_type,
                cell.exercise if role == "exercise" else cell.solution,
                cell.tags,
            )
        )
    sources.append(("markdown", reflection_cell(spec, role), ("reflection", role)))
    cells: list[nbformat.NotebookNode] = []
    for index, (cell_type, source, tags) in enumerate(sources, start=1):
        cell_id = f"{spec.field_id[:10]}-{spec.number:02d}-cell-{index:02d}"
        if cell_type == "markdown":
            cell = nbformat.v4.new_markdown_cell(source, id=cell_id)
        else:
            cell = nbformat.v4.new_code_cell(source, id=cell_id)
        if tags:
            cell.metadata["tags"] = list(tags)
        cells.append(cell)
    counterpart_role = "solutions" if role == "exercise" else "exercises"
    notebook = nbformat.v4.new_notebook(cells=cells)
    notebook.metadata.update(
        {
            "kernelspec": {
                "display_name": "Python (AI Engineering Lab)",
                "language": "python",
                "name": "ai-engineering-lab",
            },
            "language_info": {"name": "python", "version": "3.12"},
            "field_reproduction": {
                "field_id": spec.field_id,
                "field_title": spec.field_title,
                "number": f"{spec.number:02d}",
                "role": role,
                "paper_title": spec.paper_title,
                "year": spec.year,
                "primary_url": spec.primary_url,
                "dataset_file": spec.dataset_file,
                "scope": "hardware-aware-mini-reproduction",
                "device_policy": "auto:cuda-mps-cpu",
                "device_override_env": "AI_LAB_DEVICE",
                "expected_minutes": spec.expected_minutes,
                "hidden_cell_count": hidden_count(spec),
                "counterpart": f"../{counterpart_role}/{spec.filename}",
            },
        }
    )
    return notebook


def write_pair(spec: FieldPaperSpec) -> None:
    written_paths: list[Path] = []
    for role, directory_name in (("exercise", "exercises"), ("solution", "solutions")):
        directory = TRACK / spec.field_id / directory_name
        directory.mkdir(parents=True, exist_ok=True)
        notebook = build_notebook(spec, role)
        nbformat.validate(notebook)
        destination = directory / spec.filename
        nbformat.write(notebook, destination)
        written_paths.append(destination)
    format_notebooks(written_paths)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--field", choices=FIELD_ORDER)
    parser.add_argument(
        "--only", help="paper number 00..09; requires or applies within --field"
    )
    args = parser.parse_args()
    specs = load_specs()
    if args.field:
        specs = [spec for spec in specs if spec.field_id == args.field]
    if args.only is not None:
        number = args.only.strip().zfill(2)
        specs = [spec for spec in specs if f"{spec.number:02d}" == number]
    if not specs:
        raise SystemExit("no matching field-paper specs")
    for spec in specs:
        write_pair(spec)
        print(f"BUILT {spec.field_id}/{spec.filename} (tasks={hidden_count(spec)})")
    print(f"PASS: built {len(specs)} field-paper pairs")


if __name__ == "__main__":
    main()
