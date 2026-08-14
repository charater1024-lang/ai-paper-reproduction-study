"""Integrity and task-shape checks for the seven local paper-field datasets."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "field_curriculum"


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    digest.update(path.read_bytes())
    return digest.hexdigest()


def test_manifest_checksums_and_license() -> None:
    manifest = json.loads((DATA / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["license"] == "CC0-1.0"
    assert manifest["seed"] == 20260814
    assert len(manifest["files"]) == 7
    for entry in manifest["files"]:
        path = DATA / entry["path"]
        assert path.stat().st_size == entry["bytes"]
        assert file_hash(path) == entry["sha256"]


def test_vision_and_generative_shapes() -> None:
    with np.load(DATA / "vision_shapes.npz", allow_pickle=False) as data:
        assert data["images"].shape == (480, 1, 16, 16)
        assert data["masks"].shape == (480, 16, 16)
        assert data["boxes"].shape == (480, 4)
        assert np.logical_and(data["boxes"] >= 0, data["boxes"] <= 1).all()
        assert set(data["train_idx"]).isdisjoint(set(data["test_idx"]))
    with np.load(DATA / "generative_samples.npz", allow_pickle=False) as data:
        assert data["points"].shape == data["domain_a"].shape == data["domain_b"].shape
        assert data["images"].shape == (300, 1, 8, 8)


def test_nlp_and_rl_task_contracts() -> None:
    nlp = json.loads((DATA / "nlp_corpus.json").read_text(encoding="utf-8"))
    assert len(nlp["sentences"]) == len(nlp["sentiment_labels"]) == 360
    assert len(nlp["translation_pairs"]) == 60
    assert {query["answer_id"] for query in nlp["queries"]} <= {
        document["id"] for document in nlp["documents"]
    }
    rl = json.loads((DATA / "gridworld.json").read_text(encoding="utf-8"))
    assert rl["start_state"] == 0 and rl["terminal_state"] == rl["size"] - 1
    assert len(rl["transitions"]) == 2 * rl["size"]
    assert all(0 <= row["next_state"] < rl["size"] for row in rl["transitions"])


def test_graph_and_multimodal_shapes() -> None:
    with np.load(DATA / "graph_recommendation.npz", allow_pickle=False) as data:
        adjacency = data["adjacency"]
        assert adjacency.shape == (120, 120)
        assert np.array_equal(adjacency, adjacency.T)
        assert data["features"].shape[0] == len(data["labels"]) == 120
        assert data["user_item"].shape == (40, 60)
        assert data["relation_adjacency"].shape == (3, 12, 12)
    with np.load(DATA / "multimodal_pairs.npz", allow_pickle=False) as data:
        count = len(data["labels"])
        assert data["images"].shape == (count, 1, 8, 8)
        assert (
            data["image_features"].shape[0] == data["text_features"].shape[0] == count
        )
        assert np.array_equal(data["tokens"], data["token_ids"])
        assert np.array_equal(data["class_ids"], data["labels"])


def test_compression_shapes_and_teacher_contracts() -> None:
    with np.load(DATA / "compression_bench.npz", allow_pickle=False) as data:
        assert data["features"].shape == (512, 32)
        assert data["teacher_logits"].shape == (512, 4)
        assert data["teacher_weights"].shape == (32, 4)
        assert data["images"].shape == (320, 1, 16, 16)
        assert data["teacher_hidden"].shape == (256, 6, 24)
        attention = data["teacher_attention"]
        assert attention.shape == (256, 6, 6)
        assert np.allclose(attention.sum(axis=-1), 1, atol=1e-5)
        assert data["candidate_configs"].shape == (8, 4)
        assert set(data["train_idx"]).isdisjoint(set(data["test_idx"]))
