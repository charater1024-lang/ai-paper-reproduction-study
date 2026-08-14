"""Regression tests for learner-facing code readability and implementation depth."""

from __future__ import annotations

import ast
import re
from pathlib import Path

import nbformat

from tools.validate_portfolio_quality import (
    curriculum_pairs,
    learner_notebooks,
    validate_curriculum_pair,
    validate_notebook,
    validate_projects,
    validate_spec_sources,
)


def test_all_solution_notebooks_meet_portfolio_contract() -> None:
    failures: list[str] = []

    for path, rule in learner_notebooks("all"):
        for failure in validate_notebook(path, rule):
            failures.append(f"{path}: {failure}")

    assert not failures, "\n".join(failures)


def test_nlp_starters_and_solutions_meet_portfolio_contract() -> None:
    failures = [
        f"{path}: {failure}"
        for path, project_failures in validate_projects()
        for failure in project_failures
    ]

    assert not failures, "\n".join(failures)


def test_curriculum_source_specs_are_human_readable() -> None:
    failures = [
        f"{path}: {failure}"
        for path, source_failures in validate_spec_sources("all")
        for failure in source_failures
    ]

    assert not failures, "\n".join(failures)


def test_exercises_preserve_solution_public_apis_and_use_models() -> None:
    failures = [
        f"{solution}: {failure}"
        for exercise, solution in curriculum_pairs("all")
        for failure in validate_curriculum_pair(exercise, solution)
    ]

    assert not failures, "\n".join(failures)


def test_field_catalog_does_not_cut_descriptions_mid_word() -> None:
    catalog = Path("notebooks/field_reproductions/README.md").read_text(
        encoding="utf-8"
    )
    mid_word_ellipsis = re.findall(r"[^\s|]…", catalog)

    assert not mid_word_ellipsis, (
        "field catalog descriptions must use complete words before an ellipsis"
    )


def _dotted_name(node: ast.expr) -> str:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        prefix = _dotted_name(node.value)
        return f"{prefix}.{node.attr}" if prefix else node.attr
    return ""


def _notebook_code(path: str) -> str:
    notebook = nbformat.read(path, as_version=4)
    return "\n".join(cell.source for cell in notebook.cells if cell.cell_type == "code")


def test_core_transformer_labs_expose_attention_instead_of_hiding_it() -> None:
    notebook_paths = (
        "notebooks/paper_reproductions/solutions/09_attention_is_all_you_need.ipynb",
        "notebooks/paper_reproductions/solutions/10_gpt1_generative_pretraining.ipynb",
        "notebooks/paper_reproductions/solutions/11_bert_pretraining.ipynb",
        "notebooks/paper_reproductions/solutions/17_vision_transformer.ipynb",
        "notebooks/field_reproductions/nlp_llm/solutions/03_attention_is_all_you_need.ipynb",
        "notebooks/field_reproductions/nlp_llm/solutions/04_gpt1.ipynb",
        "notebooks/field_reproductions/nlp_llm/solutions/05_bert.ipynb",
        "notebooks/field_reproductions/nlp_llm/solutions/06_t5.ipynb",
        "notebooks/field_reproductions/vision/solutions/08_vision_transformer.ipynb",
        "notebooks/field_reproductions/vision/solutions/09_detr.ipynb",
    )
    forbidden = {
        "nn.MultiheadAttention",
        "nn.Transformer",
        "nn.TransformerDecoder",
        "nn.TransformerDecoderLayer",
        "nn.TransformerEncoder",
        "nn.TransformerEncoderLayer",
    }
    failures: list[str] = []

    for notebook_path in notebook_paths:
        notebook = nbformat.read(notebook_path, as_version=4)
        for index, cell in enumerate(notebook.cells, start=1):
            if cell.cell_type != "code":
                continue
            tree = ast.parse(cell.source)
            for node in ast.walk(tree):
                if not isinstance(node, ast.Call):
                    continue
                call_name = _dotted_name(node.func)
                if call_name in forbidden:
                    failures.append(
                        f"{notebook_path}, cell {index}: hidden core `{call_name}`"
                    )

    assert not failures, "\n".join(failures)


def _definition_source(path: str, name: str) -> str:
    source = _notebook_code(path)
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if node.name == name:
                segment = ast.get_source_segment(source, node)
                assert segment is not None
                return segment
    raise AssertionError(f"{path}: missing definition `{name}`")


def test_t5_relative_bias_changes_attention_and_receives_gradients() -> None:
    path = "notebooks/field_reproductions/nlp_llm/solutions/06_t5.ipynb"
    source = _notebook_code(path)

    assert "scores = scores + relative_bias[None]" in source
    assert "encoder_relative_bias.embedding.weight.grad" in source
    assert "decoder_relative_bias.embedding.weight.grad" in source


def test_multimodal_portfolio_paths_use_their_claimed_state() -> None:
    root = "notebooks/field_reproductions/self_supervised_multimodal/solutions"
    required_fragments = {
        "01_moco.ipynb": (
            "model.enqueue(keys)",
            "queue_pointer",
        ),
        "05_mae.ipynb": (
            "self.mask_token",
            "scatter",
        ),
        "06_clip.ipynb": (
            "self.logit_scale = nn.Parameter",
            "model.logit_scale.grad",
        ),
        "08_blip.ipynb": (
            "negative_pair",
            "image_embedding.unsqueeze(0)",
        ),
        "09_flamingo.ipynb": (
            "parameter.requires_grad_(False)",
            "trainable_names",
        ),
    }
    failures: list[str] = []

    for filename, fragments in required_fragments.items():
        path = f"{root}/{filename}"
        source = _notebook_code(path)
        for fragment in fragments:
            if fragment not in source:
                failures.append(f"{path}: missing `{fragment}`")

    assert not failures, "\n".join(failures)


def test_graph_portfolio_paths_use_paper_specific_mechanisms() -> None:
    root = "notebooks/field_reproductions/graph_recommendation/solutions"
    required_fragments = {
        "00_deepwalk.ipynb": ("random_walk", "skipgram_pairs"),
        "01_node2vec.ipynb": ("transition_probs", "biased_walk"),
        "02_gcn.ipynb": ("self.input_layer", "self.output_layer"),
        "03_graphsage.ipynb": (
            "sample_neighbors",
            "portfolio_train_neighbors",
            "portfolio_inference_neighbors",
        ),
        "06_rgcn.ipynb": ("self.bases", "self.coefficients"),
        "07_gin.ipynb": (
            "self.epsilons",
            "def graph_readout",
            "(1 + epsilon) * x",
        ),
        "08_pinsage.ipynb": ("importance_neighbors", "recall_at_"),
        "09_lightgcn.ipynb": ("train_ui", "heldout", "recall_at_"),
    }
    failures: list[str] = []

    for filename, fragments in required_fragments.items():
        path = f"{root}/{filename}"
        source = _notebook_code(path)
        for fragment in fragments:
            if fragment not in source:
                failures.append(f"{path}: missing `{fragment}`")

    assert not failures, "\n".join(failures)


def test_compression_portfolio_paths_execute_full_pipelines() -> None:
    root = "notebooks/field_reproductions/distillation_compression/solutions"
    required_fragments = {
        "01_fitnets.ipynb": ("hint_optimizer", "F.kl_div"),
        "02_attention_transfer.ipynb": (
            "teacher_feature",
            "model.teacher.requires_grad_(False)",
        ),
        "03_tinybert.ipynb": (
            "hidden_loss",
            "attention_loss",
            "F.kl_div",
        ),
        "04_deep_compression.ipynb": (
            "self.centroids",
            "index_entropy_bits",
        ),
        "05_lottery_ticket.ipynb": ("initial_state", "fresh_optimizer"),
        "06_integer_quantization.ipynb": (
            "torch.int32",
            "integer_accumulator",
        ),
        "08_shufflenet_v2.ipynb": ("perf_counter", "latency_ms"),
        "09_once_for_all.ipynb": ("select_subnet_under_budget", "feasible"),
    }
    failures: list[str] = []

    for filename, fragments in required_fragments.items():
        path = f"{root}/{filename}"
        source = _notebook_code(path)
        for fragment in fragments:
            if fragment not in source:
                failures.append(f"{path}: missing `{fragment}`")

    assert not failures, "\n".join(failures)


def test_dqn_and_diffusion_paths_execute_paper_specific_updates() -> None:
    required_fragments = {
        "notebooks/paper_reproductions/solutions/14_dqn.ipynb": (
            "target_network",
            "hard_sync_target",
            "next_q = target_network",
        ),
        "notebooks/field_reproductions/generative/solutions/07_ddpm.ipynb": (
            "reverse_step",
            "def sample",
            "trajectory",
        ),
        "notebooks/field_reproductions/generative/solutions/08_score_sde.ipynb": (
            "reverse_euler_maruyama",
            "trajectory",
        ),
        "notebooks/field_reproductions/generative/solutions/09_latent_diffusion.ipynb": (
            "reverse_latent_step",
            "def sample",
            "trajectory",
        ),
    }
    failures: list[str] = []

    for path, fragments in required_fragments.items():
        source = _notebook_code(path)
        for fragment in fragments:
            if fragment not in source:
                failures.append(f"{path}: missing `{fragment}`")

    assert not failures, "\n".join(failures)


def test_graphsage_keeps_training_and_unseen_inference_disjoint() -> None:
    path = "notebooks/field_reproductions/graph_recommendation/solutions/03_graphsage.ipynb"
    source = _notebook_code(path)

    assert "train_adjacency = adjacency[train_index][:, train_index]" in source
    assert "portfolio_train_neighbors" in source
    assert "portfolio_inference_neighbors" in source
    assert "unseen_test_accuracy" in source


def test_alphazero_agent_owns_puct_visits_and_training_targets() -> None:
    path = (
        "notebooks/field_reproductions/reinforcement_learning/solutions/"
        "09_alphazero.ipynb"
    )
    source = _notebook_code(path)

    assert "class AlphaZeroSearchAgent" in source
    assert "def select_action" in source
    assert "visit_counts" in source
    assert "target_pi" in source
    assert "mean_visit_total" in source
    assert "def root_search" not in source


def test_nlp_evaluations_keep_training_rows_and_labels_disjoint() -> None:
    root = "notebooks/field_reproductions/nlp_llm/solutions"
    split_notebooks = (
        "00_word2vec_sgns.ipynb",
        "01_seq2seq.ipynb",
        "02_bahdanau_attention.ipynb",
        "04_gpt1.ipynb",
        "05_bert.ipynb",
        "06_t5.ipynb",
    )
    failures: list[str] = []

    for filename in split_notebooks:
        path = f"{root}/{filename}"
        if "torch.isin" not in _notebook_code(path):
            failures.append(f"{path}: missing an explicit disjoint-row assertion")

    gpt3_path = f"{root}/07_gpt3_few_shot.ipynb"
    gpt3_source = _notebook_code(gpt3_path)
    for fragment in (
        "demonstration_labels",
        "zero_shot_prior",
        'assert not hasattr(model, "labels")',
    ):
        if fragment not in gpt3_source:
            failures.append(f"{gpt3_path}: missing `{fragment}`")

    assert not failures, "\n".join(failures)


def test_pix2pix_detr_and_dino_use_non_leaking_core_paths() -> None:
    pix2pix_path = "notebooks/field_reproductions/generative/solutions/05_pix2pix.ipynb"
    pix2pix_source = _notebook_code(pix2pix_path)
    for fragment in (
        "class TinyUNetGenerator",
        "class SpatialPatchDiscriminator",
        '"patch_grid": (2, 2)',
        "spatial_train_ids",
        "spatial_eval_ids",
    ):
        assert fragment in pix2pix_source

    detr_path = "notebooks/field_reproductions/vision/solutions/09_detr.ipynb"
    detr_evaluation = _definition_source(detr_path, "evaluate_detr")
    assert "no_object_probability" in detr_evaluation
    assert "match_single_target" not in detr_evaluation

    dino_path = (
        "notebooks/field_reproductions/self_supervised_multimodal/solutions/"
        "04_dino.ipynb"
    )
    dino_loss = _definition_source(dino_path, "compute_loss")
    for fragment in (
        "student_first",
        "student_second",
        "teacher_first",
        "teacher_second",
    ):
        assert fragment in dino_loss
