"""Add per-stage explanations and learner API contracts to selected field specs."""

from __future__ import annotations

from dataclasses import replace

from notebook_contracts import (
    exercise_contract_source,
    merge_mappings,
    select_stage_mappings,
    stage_kind,
    stage_markdown,
)

from .common import CellSpec, FieldPaperSpec

GRAPH_SHAPES = {
    0: "중심/문맥 id [P] → embedding [P,D] → node logits [P,N]",
    1: "편향 walk [W,L] → 중심/문맥 id [P] → embedding [P,D]",
    2: "A [N,N], X [N,D] → hidden [N,H] → class logits [N,K]",
    3: "sampled A [N,N], X [N,D] → aggregated hidden [N,H]",
    4: "A [N,N], X [N,D] → attention [N,N] → hidden [N,H]",
    5: "A [N,N], X [N,D] → μ/logσ² [N,Z] → link score [N,N]",
    6: "relation A [R,N,N], X [N,D] → relation messages [N,H]",
    7: "A [N,N], X [N,D] → graph embedding [G,H] → logits [G,K]",
    8: "interaction [U,I] → item graph [I,I] → user/item embedding [U/I,D]",
    9: "interaction [U,I] → E_u [U,D], E_i [I,D] → pair score [B]",
}

MULTIMODAL_SHAPES = {
    0: "sequence [B,T,D] → context/future [B,H] → contrastive logits [B,B]",
    1: "두 view [B,C,H,W] → q/k [B,D] → queue logits [B,1+Q]",
    2: "두 view [B,C,H,W] → z₁/z₂ [B,D] → similarity [2B,2B]",
    3: "두 view [B,C,H,W] → online/target [B,D] → prediction [B,D]",
    4: "global/local views [B,C,H,W] → student/teacher logits [B,K]",
    5: "image [B,C,H,W] → patches [B,N,P] → reconstruction [B,N,P]",
    6: "image/text [B,Dᵢ]/[B,T] → embeddings [B,D] → logits [B,B]",
    7: "noisy image/text batch [B,*] → embeddings [B,D] → logits [B,B]",
    8: "image/text [B,*] → filtered pairs [B',*] → three objective scalars",
    9: "visual [B,Nᵥ,D] + text [B,T,D] → conditioned hidden [B,T,D]",
}

COMPRESSION_SHAPES = {
    0: "features [B,D] + teacher logits [B,K] → student logits [B,K]",
    1: "input [B,D] → teacher/student hint [B,Hₜ/Hₛ] → logits [B,K]",
    2: "feature [B,C,H,W] → attention map [B,H,W] → normalized [B,HW]",
    3: "tokens [B,T,D] → hidden/attention [B,T,H]/[B,T,T] → logits [B,K]",
    4: "weight [O,I] → sparse/quantized symbols [O,I] + compression ratio scalar",
    5: "features [B,D] + mask [O,I] → logits [B,K]와 sparsity scalar",
    6: "float tensor [B,D] ↔ uint8/int32 tensor [B,D]와 dequantized [B,D]",
    7: "feature map [B,C,H,W] → expanded/depthwise/projected [B,C',H',W']",
    8: "feature map [B,C,H,W] → split/shuffle [B,C,H,W] → logits [B,K]",
    9: "features [B,D] + width scalar → subnet logits [B,K]와 latency scalar",
}


SHAPE_HINTS = {
    "graph_recommendation": GRAPH_SHAPES,
    "self_supervised_multimodal": MULTIMODAL_SHAPES,
    "distillation_compression": COMPRESSION_SHAPES,
}


GRAPH_STAGE_PLANS: dict[
    int,
    tuple[tuple[str, tuple[int, ...]], ...],
] = {
    0: (
        ("def random_walk", (0,)),
        ("def skipgram_pairs", (1,)),
        ("input_embed =", (2,)),
        ("similarity =", (3,)),
        ("deepwalkportfoliomodel", (0, 1, 2)),
        ("fit_portfolio_model", (2,)),
        ("evaluate_portfolio_model", (3,)),
    ),
    1: (
        ("def transition_probs", (0,)),
        ("def biased_walk", (1, 2)),
        ("def shortest_distances", (3,)),
        ("node2vecportfoliomodel", (0, 1, 2)),
        ("fit_portfolio_model", (2,)),
        ("evaluate_portfolio_model", (3,)),
    ),
    2: (
        ("def normalize_adjacency", (1,)),
        ("class gcnlayer", (0, 2)),
        ("model =", (2, 3)),
        ("gcnportfoliomodel", (0, 1, 2, 3)),
        ("fit_portfolio_model", (3,)),
        ("evaluate_portfolio_model", (3,)),
    ),
    3: (
        ("def sample_neighbors", (0,)),
        ("class meansage", (1, 2)),
        ("classifier =", (3,)),
        ("graphsageportfoliomodel", (0, 1, 2, 3)),
        ("fit_portfolio_model", (1, 2)),
        ("evaluate_portfolio_model", (3,)),
    ),
    6: (
        ("relation_adjacency =", (0, 1)),
        ("def relation_normalize", (1,)),
        ("def basis_weights", (2,)),
        ("def rgcn_layer", (0,)),
        ("changed =", (3,)),
        ("rgcnportfoliomodel", (0, 1, 2)),
        ("fit_portfolio_model", (0, 1, 2)),
        ("evaluate_portfolio_model", (2, 3)),
    ),
    7: (
        ("def gin_aggregate", (1,)),
        ("degree =", (0, 3)),
        ("mlps =", (1, 2)),
        ("ginportfoliomodel", (1, 2)),
        ("fit_portfolio_model", (1,)),
        ("evaluate_portfolio_model", (2,)),
    ),
    8: (
        ("user_item =", (0,)),
        ("def importance_neighbors", (0,)),
        ("class pinsageconv", (1, 2)),
        ("pos =", (3,)),
        ("pinsageportfoliomodel", (0, 1, 2, 3)),
        ("fit_portfolio_model", (3,)),
        ("evaluate_portfolio_model", (3,)),
    ),
    9: (
        ("user_item =", (3,)),
        ("def bipartite_normalized", (0,)),
        ("def lightgcn_embeddings", (1,)),
        ("nu, ni =", (2, 3)),
        ("with torch.no_grad():", (3,)),
        ("lightgcnportfoliomodel", (0, 1, 2, 3)),
        ("fit_portfolio_model", (3,)),
        ("evaluate_portfolio_model", (3,)),
    ),
}


COMPRESSION_STAGE_PLANS: dict[
    int,
    tuple[tuple[str, tuple[int, ...]], ...],
] = {
    0: (
        ("def soft_targets", (0,)),
        ("def kd_loss", (1, 2)),
        ("student =", (2, 3)),
        ("knowledgedistillationportfoliomodel", (0, 1, 2, 3)),
        ("fit_portfolio_model", (1, 2)),
        ("evaluate_portfolio_model", (3,)),
    ),
    1: (
        ("class fitstudent", (0,)),
        ("hint_optimizer", (0, 1)),
        ("optimizer =", (1, 2, 3)),
        ("fitnetsportfoliomodel", (0, 1, 2, 3)),
        ("fit_portfolio_model", (0, 1, 2)),
        ("evaluate_portfolio_model", (3,)),
    ),
    2: (
        ("def attention_map", (0, 1)),
        ("smooth =", (2,)),
        ("optimizer =", (2, 3)),
        ("attentiontransferportfoliomodel", (0, 1, 2, 3)),
        ("fit_portfolio_model", (2, 3)),
        ("evaluate_portfolio_model", (3,)),
    ),
    3: (
        ("class tinystudent", (1,)),
        ("student_attention", (2,)),
        ("sequence_labels", (0, 1, 2, 3)),
        ("tinybertportfoliomodel", (0, 1, 2, 3)),
        ("fit_portfolio_model", (0, 1, 2, 3)),
        ("evaluate_portfolio_model", (3,)),
    ),
    4: (
        ("def magnitude_prune", (1,)),
        ("values =", (2,)),
        ("symbols =", (3,)),
        ("deepcompressionportfoliomodel", (0, 1, 2, 3)),
        ("fit_portfolio_model", (0, 1, 2, 3)),
        ("evaluate_portfolio_model", (0, 1, 2, 3)),
    ),
    5: (
        ("class ticketnet", (0, 1)),
        ("ticket =", (0, 1, 2)),
        ("with torch.no_grad():", (3,)),
        ("lotteryticketportfoliomodel", (0, 1, 2, 3)),
        ("fit_portfolio_model", (0, 1, 2)),
        ("evaluate_portfolio_model", (3,)),
    ),
    6: (
        ("def affine_quantize", (0, 2)),
        ("pytorch cuda", (1,)),
        ("def fake_quant", (3,)),
        ("integerquantizationportfoliomodel", (0, 1, 2, 3)),
        ("fit_portfolio_model", (3,)),
        ("evaluate_portfolio_model", (0, 1, 2, 3)),
    ),
    7: (
        ("channels =", (0,)),
        ("class invertedresidual", (1, 2)),
        ("class mobilemini", (0, 3)),
        ("mobilenetv2portfoliomodel", (0, 1, 2, 3)),
        ("fit_portfolio_model", (3,)),
        ("evaluate_portfolio_model", (0, 2, 3)),
    ),
    8: (
        ("def channel_shuffle", (0, 2)),
        ("class shuffleunit", (1, 2)),
        ("class shufflemini", (3,)),
        ("shufflenetv2portfoliomodel", (0, 1, 2, 3)),
        ("fit_portfolio_model", (1, 2)),
        ("evaluate_portfolio_model", (3,)),
    ),
    9: (
        ("class widthsupernet", (1,)),
        ("optimizer =", (2,)),
        ("budget_ms", (0, 3)),
        ("onceforallportfoliomodel", (0, 1, 2, 3)),
        ("fit_portfolio_model", (0, 1, 2)),
        ("select_subnet_under_budget", (3,)),
    ),
}


MULTIMODAL_STAGE_PLANS: dict[
    int,
    tuple[tuple[str, tuple[int, ...]], ...],
] = {
    0: (
        ("cpcmodel", (0, 1)),
        ("cpc_loss", (2, 3)),
        ("cpc =", (3,)),
        ("cpcportfoliomodel", (0, 1, 2, 3)),
        ("fit_portfolio_model", (3,)),
        ("evaluate_portfolio_model", (3,)),
    ),
    1: (
        ("image_view", (0,)),
        ("momentum_update", (2,)),
        ("moco_objective", (1,)),
        ("query_encoder", (0, 1, 2, 3, 4)),
        ("mocoportfoliomodel", (0, 1, 2, 3, 4)),
        ("fit_portfolio_model", (1, 2, 3, 4)),
        ("evaluate_portfolio_model", (4,)),
    ),
    2: (
        ("make_views", (0,)),
        ("nt_xent", (1, 2)),
        ("encoder =", (0, 1, 2, 3)),
        ("simclrportfoliomodel", (0, 1, 2)),
        ("fit_portfolio_model", (2,)),
        ("evaluate_portfolio_model", (2,)),
    ),
    3: (
        ("byol_view", (0,)),
        ("ema_update", (1,)),
        ("normalized_mse", (2,)),
        ("online, target", (0, 1, 2, 3)),
        ("byolportfoliomodel", (0, 1, 2, 3)),
        ("fit_portfolio_model", (1, 3)),
        ("evaluate_portfolio_model", (2, 3)),
    ),
    4: (
        ("dino_net", (0,)),
        ("dino_distributions", (1, 3)),
        ("distill_ce", (1, 2)),
        ("student =", (0, 1, 2, 3)),
        ("dinoportfoliomodel", (0, 1, 2, 3)),
        ("fit_portfolio_model", (2, 3)),
        ("evaluate_portfolio_model", (0,)),
    ),
    5: (
        ("patchify", (0,)),
        ("random_mask", (1,)),
        ("tinyselfattention", (2, 3)),
        ("mae =", (4,)),
        ("maeportfoliomodel", (1, 2, 3, 4)),
        ("fit_portfolio_model", (1, 4)),
        ("evaluate_portfolio_model", (1, 4)),
    ),
    6: (
        ("tinyclip", (0, 1)),
        ("clip_loss", (1,)),
        ("clip =", (1,)),
        ("classes =", (2, 3, 4)),
        ("clipportfoliomodel", (0, 1)),
        ("fit_portfolio_model", (1,)),
        ("evaluate_portfolio_model", (2, 3, 4)),
    ),
    7: (
        ("noisy_captions", (0,)),
        ("tinyalign", (1, 2, 3)),
        ("align =", (0, 1, 2, 3, 4)),
        ("alignportfoliomodel", (0, 1, 2, 3)),
        ("fit_portfolio_model", (1, 2)),
        ("evaluate_portfolio_model", (0, 4)),
    ),
    8: (
        ("capfilt", (3,)),
        ("tinyblip", (0, 1, 2)),
        ("blip =", (0, 1, 2, 3)),
        ("blipportfoliomodel", (0, 1, 2)),
        ("fit_portfolio_model", (0,)),
        ("evaluate_portfolio_model", (1, 2, 3)),
    ),
    9: (
        ("explicitcrossattention", (0,)),
        ("gatedcrossattention", (1,)),
        ("vocab =", (0, 1, 2)),
        ("conditioned_loss", (2, 3)),
        ("flamingoportfoliomodel", (0, 1, 2, 3)),
        ("fit_portfolio_model", (2, 3)),
        ("evaluate_portfolio_model", (2, 3)),
    ),
}


def _planned_mapping(
    spec: FieldPaperSpec,
    source: str,
    tags: tuple[str, ...],
    kind: str,
    stage_index: int,
) -> tuple[str, str, str]:
    field_plans = {
        "distillation_compression": COMPRESSION_STAGE_PLANS,
        "graph_recommendation": GRAPH_STAGE_PLANS,
        "self_supervised_multimodal": MULTIMODAL_STAGE_PLANS,
    }
    plans = field_plans.get(spec.field_id)
    plan = plans.get(spec.number) if plans is not None else None
    if plan is not None and stage_index < len(plan):
        marker, indices = plan[stage_index]
        fingerprint = f"{source} {' '.join(tags)}".lower()
        if marker not in fingerprint:
            raise ValueError(
                f"{spec.field_id}/{spec.number:02d} stage {stage_index + 1}: "
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


def _shape_hint(spec: FieldPaperSpec) -> str:
    return SHAPE_HINTS[spec.field_id][spec.number]


def enrich_field_spec(spec: FieldPaperSpec) -> FieldPaperSpec:
    """Insert one precise note before every code cell and preserve exercise APIs."""

    cells: list[CellSpec] = []
    code_ordinal = 0
    substantive_index = 0
    shape_hint = _shape_hint(spec)
    for cell in spec.cells:
        if cell.cell_type != "code":
            cells.append(cell)
            continue

        code_ordinal += 1
        kind = stage_kind(cell.tags, cell.solution)
        mapping = None
        if kind not in {"setup", "data"}:
            mapping = _planned_mapping(
                spec,
                cell.solution,
                cell.tags,
                kind,
                substantive_index,
            )
            substantive_index += 1
        note = stage_markdown(
            ordinal=code_ordinal,
            kind=kind,
            mapping=mapping,
            shape_hint=shape_hint,
        )
        cells.append(
            CellSpec(
                cell_type="markdown",
                exercise=note,
                solution=note,
                tags=("stage-contract", kind),
            )
        )
        paper_location = mapping[0] if mapping is not None else "setup 예외"
        exercise = exercise_contract_source(
            cell.exercise,
            cell.solution,
            paper_location=paper_location,
            shape_hint=shape_hint,
        )
        cells.append(replace(cell, exercise=exercise))
    field_plans = {
        "distillation_compression": COMPRESSION_STAGE_PLANS,
        "graph_recommendation": GRAPH_STAGE_PLANS,
        "self_supervised_multimodal": MULTIMODAL_STAGE_PLANS,
    }
    plans = field_plans.get(spec.field_id)
    if plans is not None and spec.number in plans:
        expected = len(plans[spec.number])
        if substantive_index != expected:
            raise ValueError(
                f"{spec.field_id}/{spec.number:02d}: expected {expected} "
                f"semantic stages, found {substantive_index}"
            )
    return replace(spec, cells=tuple(cells))


def enrich_field_specs(
    specs: tuple[FieldPaperSpec, ...],
) -> tuple[FieldPaperSpec, ...]:
    """Apply the portfolio stage contract to a complete ten-paper field."""

    return tuple(enrich_field_spec(spec) for spec in specs)
