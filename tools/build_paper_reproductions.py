"""Generate paired exercise/solution notebooks for 20 landmark AI papers."""

from __future__ import annotations

import argparse
from pathlib import Path

import nbformat
from notebook_contracts import (
    exercise_contract_source,
    merge_mappings,
    select_stage_mappings,
    stage_kind,
    stage_markdown,
)
from notebook_style import format_notebooks
from paper_curriculum.common import PaperSpec, clean
from paper_curriculum.portfolio_notes import portfolio_note, portfolio_shape

ROOT = Path(__file__).resolve().parents[1]
TRACK = ROOT / "notebooks" / "paper_reproductions"
EXERCISES = TRACK / "exercises"
SOLUTIONS = TRACK / "solutions"
ROLE_MARKER = "<!-- paper-paired-role-banner -->"

# Each entry is (AST/name/tag fingerprint, exact mapping row indices).  The plan
# prevents a generic ordinal fallback from drifting when one cell implements
# several mechanisms from non-adjacent paper sections.
PAPER_STAGE_PLANS: dict[int, tuple[tuple[str, tuple[int, ...]], ...]] = {
    0: (
        ("lenet5mini", (0, 1, 2)),
        ("trace_lenet_features", (0, 1, 2)),
        ("optimizer", (3,)),
    ),
    1: (
        ("paper_lrn", (1,)),
        ("pool_input", (2,)),
        ("alexnetmini", (0, 1, 2, 3)),
        ("alexnet.eval", (3,)),
        ("alexnet.train", (3,)),
    ),
    2: (
        ("doubleconv", (0, 1)),
        ("unet_logits", (0, 1)),
        ("paper_border_weight", (3,)),
        ("positive_weight", (2,)),
    ),
    3: (
        ("basicblock", (0, 1, 2)),
        ("identity_block", (0, 1)),
        ("miniresnet", (2,)),
        ("input_gradient_norm", (3,)),
        ("res_optimizer", (0, 3)),
    ),
    4: (
        ("paper_dropout", (0, 1, 2)),
        ("paperdropout", (0, 1, 2, 3)),
        ("dropout_optimizer", (3,)),
        ("grid_axis", (3,)),
    ),
    5: (
        ("manual_batch_norm", (0, 1)),
        ("runningbatchnorm1d", (0, 1, 2)),
        ("scaledmlp", (3,)),
        ("plt.figure", (3,)),
    ),
    6: (
        ("make_skipgram_pairs", (0, 2)),
        ("sgns", (1,)),
        ("torch.manual_seed", (1, 2)),
        ("inference_mode", (3,)),
    ),
    7: (
        ("build_seq2seq_batch", (0, 3)),
        ("seq2seq", (1,)),
        ("decoder_inputs", (0,)),
        ("greedy_decode", (2,)),
        ("per_sequence", (2,)),
    ),
    8: (
        ("additiveattention", (0, 1, 2)),
        ("optimizer", (0, 1, 2, 3)),
        ("inference_mode", (3,)),
        ("plt.subplots", (3,)),
    ),
    9: (
        ("sinusoidal_positions", (3,)),
        ("scaled_dot_product_attention", (0,)),
        ("multiheadselfattention", (0, 1, 2)),
        ("positionwisefeedforward", (1, 2, 5, 6)),
        ("attention_entropy", (2, 4)),
    ),
    10: (
        ("content", (0, 5)),
        ("tinygpt", (1, 2)),
        ("pretrain_optimizer", (0,)),
        ("finetune_optimizer", (3, 4)),
        ("plt.subplots", (0, 3, 4)),
    ),
    11: (
        ("selected_positions", (1, 2)),
        ("tinybert", (0, 4, 5)),
        ("optimizer", (1, 2, 3)),
        ("context_a", (4,)),
    ),
    12: (
        ("reparameterize", (1, 2)),
        ("tinyvae", (1, 2)),
        ("vae_objective", (0,)),
        ("vae.eval", (0, 2)),
    ),
    13: (
        ("generator", (0,)),
        ("gan_step", (1,)),
        ("train_rng", (1,)),
        ("rbf_mmd", (2,)),
    ),
    14: (
        ("replaybuffer", (2,)),
        ("qnetwork", (0, 1)),
        ("epsilon_greedy", (0, 1, 2)),
        ("evaluation_returns", (2,)),
    ),
    15: (
        ("normalize_adjacency", (1,)),
        ("graphconvolution", (0, 2)),
        ("optimizer", (2,)),
        ("draw_graph", (2,)),
    ),
    16: (
        ("make_views", (0,)),
        ("nt_xent", (1, 2)),
        ("tinyencoder", (0,)),
        ("contrastive_update", (2,)),
        ("for parameter", (3,)),
    ),
    17: (
        ("patchify", (0,)),
        ("vitselfattention", (0, 1, 2, 3)),
        ("tokens_no_pos", (4,)),
        ("generator =", (3,)),
    ),
    18: (
        ("q_sample", (0, 1)),
        ("half = 256", (0, 1)),
        ("noisepredictor", (2,)),
        ("predictor =", (3,)),
        ("reverse_step", (2, 4)),
    ),
    19: (
        ("tinyclip", (0, 1)),
        ("clip_loss", (2,)),
        ("retrieval_metrics", (3,)),
        ("zero_shot_predict", (4,)),
    ),
}


def paper_stage_mapping(
    spec: PaperSpec,
    source: str,
    tags: tuple[str, ...],
    kind: str,
    stage_index: int,
) -> tuple[str, str, str]:
    """Return the manually audited paper rows for one semantic code stage."""

    plan = PAPER_STAGE_PLANS.get(spec.number)
    if plan is not None and stage_index < len(plan):
        marker, indices = plan[stage_index]
        fingerprint = f"{source} {' '.join(tags)}".lower()
        if marker not in fingerprint:
            raise ValueError(
                f"paper {spec.number:02d} stage {stage_index + 1}: "
                f"expected semantic marker {marker!r}"
            )
        return merge_mappings(tuple(spec.mappings[index] for index in indices))
    selected = select_stage_mappings(
        spec.mappings,
        source,
        tags,
        kind,
        stage_index,
    )
    return merge_mappings(selected)


def load_specs() -> list[PaperSpec]:
    from paper_curriculum.generative_specs import SPECS as generative_specs
    from paper_curriculum.modern_specs import SPECS as modern_specs
    from paper_curriculum.nlp_specs import SPECS as nlp_specs
    from paper_curriculum.vision_specs import SPECS as vision_specs

    specs = [
        *vision_specs,
        *nlp_specs,
        *generative_specs,
        *modern_specs,
    ]
    specs.sort(key=lambda item: item.number)
    numbers = [spec.number for spec in specs]
    if numbers != list(range(20)):
        raise ValueError(f"paper numbers must be 00..19, got {numbers}")
    filenames = [spec.filename for spec in specs]
    if len(filenames) != len(set(filenames)):
        raise ValueError("paper notebook filenames must be unique")
    return specs


def role_banner(spec: PaperSpec, role: str) -> str:
    if role == "exercise":
        role_text = (
            "> 🟦 **실습 코드 · 왼쪽 창** — TODO를 먼저 직접 구현하세요. "
            "막히면 오른쪽 정답의 같은 셀에서 필요한 한 줄만 확인합니다."
        )
    else:
        role_text = (
            "> 🟩 **정답 코드 · 오른쪽 창** — 왼쪽과 같은 셀 위치의 실행 가능한 참고 구현입니다. "
            "복사하기 전에 구현 이유와 텐서 shape를 설명해 보세요."
        )
    return clean(
        f"""
        {ROLE_MARKER}
        {role_text}

        # {spec.display_title} — {spec.paper_title}

        - **원 논문:** [{spec.paper_title}]({spec.primary_url})
        - **저자 / 발표:** {spec.authors} · {spec.venue} ({spec.year})
        - **난이도 / 예상 시간:** {spec.difficulty} · 약 {spec.expected_minutes}분
        - **선수 지식:** {spec.prerequisites}

        이 노트북은 논문 결과 수치를 그대로 재현하는 대규모 benchmark가 아니라,
        핵심 수식과 구조가 실제로 어떻게 동작하는지 확인하는 **하드웨어 인식 미니 재현**입니다.
        PyTorch 학습은 CUDA → Apple MPS → CPU 순으로 자동 선택하며, 작은 NumPy·Python 실험은
        전송 비용을 피하기 위해 CPU에서 실행될 수 있습니다.
        """
    )


def scope_cell(spec: PaperSpec) -> str:
    return clean(
        f"""
        ## 재현 범위와 완료 기준

        **이 노트북이 재현하는 것**

        {spec.reproduction_goal}

        **원 논문과 다른 것**

        {spec.original_scale}

        완료하려면 TODO를 구현하고, 검증 셀의 `assert`를 모두 통과시키고, 마지막 관찰 질문에
        자신의 말로 답하세요. 수치가 논문과 다르더라도 핵심 불변식과 비교 방향이 맞으면 성공입니다.
        """
    )


def mapping_cell(spec: PaperSpec) -> str:
    rows = "\n".join(
        f"| {paper_part} | {notebook_part} | {evidence} |"
        for paper_part, notebook_part, evidence in spec.mappings
    )
    return clean(
        f"""
        ## 논문 ↔ 노트북 지도

        절 번호와 수식 번호는 위 링크의 대표 공개본을 기준으로 합니다. 판본에 따라 페이지는 달라질 수
        있으므로 **절 제목과 수식 번호**를 함께 사용하세요.

        | 원 논문의 위치 | 이 노트북의 대응 실습 | 확인할 증거 |
        |---|---|---|
        {rows}

        > 읽기 순서: 먼저 해당 절을 훑고 → 코드를 예측해 작성하고 → 출력이 논문의 주장과 왜
        > 연결되는지 한 문장으로 설명합니다.
        """
    )


def implementation_reason_cell(spec: PaperSpec) -> str:
    """Explain why this notebook uses its particular reduced implementation."""

    paper_part, notebook_part, evidence = spec.mappings[0]
    return clean(
        f"""
        ## 구현 선택 이유

        - **원문에서 보존한 부분:** {paper_part}
        - **노트북 구현:** {notebook_part}
        - **이렇게 구현한 이유:** 완성된 고수준 model을 한 줄로 호출하면
          논문의 핵심 shape·mask·loss·상태 변화가 숨겨집니다. 따라서 이 계산을
          이름이 드러나는 class와 함수로 나누고 중간 tensor를 assertion으로
          검사합니다.
        - **실패를 잡는 증거:** {evidence}를 출력 크기만 아니라
          구현 불변식으로 사용합니다.
        - **축소 선택:** {spec.original_scale}는 기본 실행에서 생략합니다.
          이는 Windows CPU에서도 반복 실험이 가능하게 하면서 핵심 메커니즘은
          유지하기 위한 선택입니다.
        """
    )


def reflection_cell(spec: PaperSpec, role: str) -> str:
    if role == "exercise":
        return clean(
            """
            ## 마무리 관찰 기록

            아래 세 문장을 직접 완성하세요.

            1. 이 논문의 핵심 연산은 `_____`이고, 코드에서는 `_____`에 해당한다.
            2. 원 규모 실험 대신 미니 재현을 사용했기 때문에 확인할 수 없는 주장은 `_____`이다.
            3. 한 가지 변수를 바꾸어 다시 실행한다면 `_____`를 바꾸고 `_____`를 측정하겠다.

            **추가 실험:** seed는 유지한 채 한 변수만 바꾸고, 변경 전후의 metric을 표로 남기세요.
            """
        )
    return clean(
        f"""
        ## 마무리 해설과 다음 실험

        1. 핵심 연산과 구현 위치를 원 논문의 용어로 다시 연결해 설명하세요.
        2. 이 미니 실험은 `{spec.original_scale}`를 재현하지 않으므로 논문의 최종 성능을 검증한 것이 아닙니다.
        3. 다음에는 데이터 크기, 모델 폭, 학습 횟수 중 **하나만** 바꾸고 같은 seed에서 metric을 비교하세요.

        정답의 수치 자체보다, shape·확률 정규화·mask·gradient 같은 불변식을 먼저 확인하는 습관이
        논문 구현에서 가장 강한 디버깅 도구입니다.
        """
    )


def notebook_cells(spec: PaperSpec, role: str) -> list[nbformat.NotebookNode]:
    sources: list[tuple[str, str, tuple[str, ...]]] = [
        ("markdown", role_banner(spec, role), ("paper-header", role)),
        ("markdown", scope_cell(spec), ("paper-scope",)),
        ("markdown", mapping_cell(spec), ("paper-map",)),
        ("markdown", portfolio_note(spec.number), ("equation-to-code",)),
        ("markdown", implementation_reason_cell(spec), ("implementation-reason",)),
    ]
    code_ordinal = 0
    substantive_index = 0
    shape_hint = portfolio_shape(spec.number)
    for cell in spec.cells:
        if cell.cell_type == "code":
            code_ordinal += 1
            kind = stage_kind(cell.tags, cell.solution)
            mapping = None
            if kind not in {"setup", "data"}:
                mapping = paper_stage_mapping(
                    spec,
                    cell.solution,
                    cell.tags,
                    kind,
                    substantive_index,
                )
                substantive_index += 1
            sources.append(
                (
                    "markdown",
                    stage_markdown(
                        ordinal=code_ordinal,
                        kind=kind,
                        mapping=mapping,
                        shape_hint=shape_hint,
                    ),
                    ("stage-contract", kind),
                )
            )
        if role == "exercise" and cell.cell_type == "code":
            paper_location = mapping[0] if mapping is not None else "setup 예외"
            source = exercise_contract_source(
                cell.exercise,
                cell.solution,
                paper_location=paper_location,
                shape_hint=shape_hint,
            )
        else:
            source = cell.exercise if role == "exercise" else cell.solution
        sources.append((cell.cell_type, source, cell.tags))
    expected_stages = len(PAPER_STAGE_PLANS[spec.number])
    if substantive_index != expected_stages:
        raise ValueError(
            f"paper {spec.number:02d}: expected {expected_stages} semantic stages, "
            f"found {substantive_index}"
        )
    sources.append(("markdown", reflection_cell(spec, role), ("reflection", role)))

    cells: list[nbformat.NotebookNode] = []
    for index, (cell_type, source, tags) in enumerate(sources, start=1):
        cell_id = f"paper-{spec.number:02d}-cell-{index:02d}"
        if cell_type == "markdown":
            cell = nbformat.v4.new_markdown_cell(source, id=cell_id)
        else:
            cell = nbformat.v4.new_code_cell(source, id=cell_id)
        if tags:
            cell.metadata["tags"] = list(tags)
        cells.append(cell)
    return cells


def hidden_code_count(spec: PaperSpec) -> int:
    return sum(
        cell.cell_type == "code" and cell.exercise != cell.solution
        for cell in spec.cells
    )


def build_notebook(spec: PaperSpec, role: str) -> nbformat.NotebookNode:
    counterpart = (
        f"../{'solutions' if role == 'exercise' else 'exercises'}/{spec.filename}"
    )
    notebook = nbformat.v4.new_notebook(cells=notebook_cells(spec, role))
    notebook.metadata.update(
        {
            "kernelspec": {
                "display_name": "Python (AI Engineering Lab)",
                "language": "python",
                "name": "ai-engineering-lab",
            },
            "language_info": {"name": "python", "version": "3.12"},
            "paper_reproduction": {
                "number": f"{spec.number:02d}",
                "role": role,
                "paper_title": spec.paper_title,
                "year": spec.year,
                "primary_url": spec.primary_url,
                "counterpart": counterpart,
                "expected_minutes": spec.expected_minutes,
                "scope": "hardware-aware-mini-reproduction",
                "device_policy": "auto:cuda-mps-cpu",
                "device_override_env": "AI_LAB_DEVICE",
                "hidden_cell_count": hidden_code_count(spec),
            },
        }
    )
    return notebook


def write_pair(spec: PaperSpec) -> None:
    written_paths: list[Path] = []
    for role, directory in (("exercise", EXERCISES), ("solution", SOLUTIONS)):
        directory.mkdir(parents=True, exist_ok=True)
        notebook = build_notebook(spec, role)
        nbformat.validate(notebook)
        destination = directory / spec.filename
        nbformat.write(notebook, destination)
        written_paths.append(destination)
    format_notebooks(written_paths)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--only", help="one paper number, for example 09")
    args = parser.parse_args()
    specs = load_specs()
    if args.only is not None:
        selected = args.only.strip().zfill(2)
        specs = [spec for spec in specs if f"{spec.number:02d}" == selected]
        if not specs:
            raise SystemExit(f"unknown paper number: {args.only}")
    for spec in specs:
        write_pair(spec)
        print(f"BUILT {spec.filename} (hidden={hidden_code_count(spec)})")
    print(f"PASS: built {len(specs)} paper notebook pairs")


if __name__ == "__main__":
    main()
