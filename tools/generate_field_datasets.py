"""Generate deterministic, license-safe datasets for the seven paper fields."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "data" / "field_curriculum"
SEED = 20260814


def write_npz(name: str, **arrays: np.ndarray) -> Path:
    path = OUTPUT / name
    np.savez_compressed(path, **arrays)
    return path


def vision_data(rng: np.random.Generator) -> Path:
    count, size, classes = 480, 16, 4
    images = np.zeros((count, 1, size, size), dtype=np.float32)
    masks = np.zeros((count, size, size), dtype=np.float32)
    boxes = np.zeros((count, 4), dtype=np.float32)
    labels = np.arange(count, dtype=np.int64) % classes
    for index, label in enumerate(labels):
        canvas = rng.normal(0.03, 0.025, (size, size)).astype(np.float32)
        x = int(rng.integers(2, 9))
        y = int(rng.integers(2, 9))
        width = int(rng.integers(4, 7))
        height = int(rng.integers(4, 7))
        yy, xx = np.mgrid[:size, :size]
        if label == 0:  # filled rectangle
            shape = (xx >= x) & (xx < x + width) & (yy >= y) & (yy < y + height)
        elif label == 1:  # ellipse
            shape = ((xx - (x + width / 2)) / (width / 2)) ** 2 + (
                (yy - (y + height / 2)) / (height / 2)
            ) ** 2 <= 1
        elif label == 2:  # diagonal cross
            shape = (
                (
                    (np.abs((xx - x) - (yy - y)) <= 1)
                    | (np.abs((xx - x) + (yy - y) - (width - 1)) <= 1)
                )
                & (xx >= x)
                & (xx < x + width)
                & (yy >= y)
                & (yy < y + height)
            )
        else:  # hollow rectangle
            outer = (xx >= x) & (xx < x + width) & (yy >= y) & (yy < y + height)
            inner = (xx > x) & (xx < x + width - 1) & (yy > y) & (yy < y + height - 1)
            shape = outer & ~inner
        canvas[shape] += float(rng.uniform(0.75, 1.0))
        images[index, 0] = np.clip(canvas, 0, 1)
        masks[index] = shape
        boxes[index] = np.array([x, y, x + width, y + height], dtype=np.float32) / size
    permutation = rng.permutation(count)
    return write_npz(
        "vision_shapes.npz",
        images=images,
        labels=labels,
        masks=masks,
        boxes=boxes,
        train_idx=permutation[:384],
        test_idx=permutation[384:],
        class_names=np.array(["rectangle", "ellipse", "cross", "frame"]),
    )


def nlp_data(rng: np.random.Generator) -> Path:
    subjects = ["robot", "student", "doctor", "artist", "model", "agent"]
    verbs = ["builds", "studies", "explains", "draws", "predicts", "plans"]
    objects = ["bridge", "paper", "answer", "image", "token", "route"]
    sentiments = ["helpful", "clear", "safe", "confusing", "risky", "wrong"]
    sentences: list[str] = []
    labels: list[int] = []
    for index in range(360):
        positive = index % 2 == 0
        adjective = sentiments[index % 3] if positive else sentiments[3 + index % 3]
        sentence = (
            f"{subjects[index % len(subjects)]} {verbs[(index * 5) % len(verbs)]} "
            f"{adjective} {objects[(index * 7) % len(objects)]}"
        )
        sentences.append(sentence)
        labels.append(int(positive))
    permutation = rng.permutation(len(sentences))
    sentences = [sentences[index] for index in permutation]
    labels = [labels[index] for index in permutation]
    documents = [
        {
            "id": "attention",
            "text": (
                "Scaled dot product attention divides logits by the square root "
                "of key dimension."
            ),
        },
        {
            "id": "bert",
            "text": (
                "BERT pretrains a bidirectional Transformer encoder with masked "
                "language modeling."
            ),
        },
        {
            "id": "gpt",
            "text": (
                "GPT uses a causal language model objective and then adapts the "
                "pretrained decoder."
            ),
        },
        {
            "id": "rag",
            "text": "Retrieval augmented generation combines retrieved passages with a generator.",
        },
        {
            "id": "lora",
            "text": "LoRA freezes base weights and learns low rank update matrices.",
        },
    ]
    queries = [
        {"query": "What scales attention logits?", "answer_id": "attention"},
        {"query": "Which model uses masked language modeling?", "answer_id": "bert"},
        {"query": "What does LoRA train?", "answer_id": "lora"},
    ]
    payload = {
        "seed": SEED,
        "sentences": sentences,
        "sentiment_labels": labels,
        "translation_pairs": [
            {"source": f"{a} {b} {c}", "target": f"{c} {b} {a}"}
            for a, b, c in zip(subjects * 10, verbs * 10, objects * 10, strict=True)
        ],
        "documents": documents,
        "queries": queries,
        "license": "CC0-1.0 synthetic",
    }
    path = OUTPUT / "nlp_corpus.json"
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def generative_data(rng: np.random.Generator) -> Path:
    centers = np.array([[-1.5, -0.3], [1.5, -0.3], [0.0, 1.4]], dtype=np.float32)
    labels = np.arange(600, dtype=np.int64) % len(centers)
    points = centers[labels] + rng.normal(0, 0.25, (600, 2)).astype(np.float32)
    images = np.zeros((300, 1, 8, 8), dtype=np.float32)
    image_labels = np.arange(300, dtype=np.int64) % 3
    for index, label in enumerate(image_labels):
        if label == 0:
            images[index, 0, :, 2:4] = 1
        elif label == 1:
            images[index, 0, 2:4, :] = 1
        else:
            np.fill_diagonal(images[index, 0], 1)
        images[index] = np.clip(images[index] + rng.normal(0, 0.07, (1, 8, 8)), 0, 1)
    domain_a = points
    domain_b = points @ np.array(
        [[0.0, -1.0], [1.0, 0.0]], dtype=np.float32
    ) + np.array([0.4, -0.2], dtype=np.float32)
    return write_npz(
        "generative_samples.npz",
        points=points,
        point_labels=labels,
        images=images,
        image_labels=image_labels,
        domain_a=domain_a,
        domain_b=domain_b,
    )


def gridworld_data(rng: np.random.Generator) -> Path:
    transitions: list[dict[str, int | float | bool]] = []
    size = 7
    for state in range(size):
        for action in (0, 1):
            next_state = max(0, state - 1) if action == 0 else min(size - 1, state + 1)
            done = next_state == size - 1
            transitions.append(
                {
                    "state": state,
                    "action": action,
                    "reward": 1.0 if done else -0.01,
                    "next_state": next_state,
                    "done": done,
                }
            )
    payload = {
        "seed": SEED,
        "size": size,
        "start_state": 0,
        "terminal_state": size - 1,
        "gamma": 0.95,
        "transitions": transitions,
        "bandit_means": [-0.2, 0.0, 0.25, 0.7],
        "offline_action_noise": rng.normal(0, 0.05, 128).round(5).tolist(),
        "license": "CC0-1.0 synthetic",
    }
    path = OUTPUT / "gridworld.json"
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def graph_data(rng: np.random.Generator) -> Path:
    nodes, communities = 120, 3
    labels = np.arange(nodes, dtype=np.int64) % communities
    adjacency = np.zeros((nodes, nodes), dtype=np.float32)
    for left in range(nodes):
        for right in range(left + 1, nodes):
            probability = 0.25 if labels[left] == labels[right] else 0.025
            if rng.random() < probability:
                adjacency[left, right] = adjacency[right, left] = 1
    features = np.eye(communities, dtype=np.float32)[labels] @ rng.normal(
        0, 1, (communities, 8)
    ).astype(np.float32)
    features += rng.normal(0, 0.2, features.shape).astype(np.float32)
    user_item = np.zeros((40, 60), dtype=np.float32)
    for user in range(40):
        preferred = user % communities
        for item in range(60):
            if item % communities == preferred and rng.random() < 0.55:
                user_item[user, item] = 1
            elif rng.random() < 0.035:
                user_item[user, item] = 1
    train_mask = np.zeros(nodes, dtype=bool)
    test_mask = np.zeros(nodes, dtype=bool)
    for community in range(communities):
        indices = np.where(labels == community)[0]
        train_mask[indices[:12]] = True
        test_mask[indices[12:]] = True
    relation_adjacency = np.zeros((3, 12, 12), dtype=np.float32)
    for relation in range(3):
        for node in range(12):
            relation_adjacency[relation, node, (node + relation + 1) % 12] = 1
    relation_features = rng.normal(0, 1, (12, 6)).astype(np.float32)
    return write_npz(
        "graph_recommendation.npz",
        adjacency=adjacency,
        features=features,
        labels=labels,
        train_mask=train_mask,
        test_mask=test_mask,
        user_item=user_item,
        relation_adjacency=relation_adjacency,
        relation_features=relation_features,
    )


def multimodal_data(rng: np.random.Generator) -> Path:
    classes, per_class = 6, 40
    count = classes * per_class
    labels = np.arange(count, dtype=np.int64) % classes
    latent_centers = rng.normal(0, 1, (classes, 6)).astype(np.float32)
    latent = latent_centers[labels] + rng.normal(0, 0.15, (count, 6)).astype(np.float32)
    image_map = rng.normal(0, 1, (6, 12)).astype(np.float32)
    text_map = rng.normal(0, 1, (6, 10)).astype(np.float32)
    images = np.zeros((count, 1, 8, 8), dtype=np.float32)
    for index, label in enumerate(labels):
        row, col = divmod(int(label), 3)
        images[index, 0, 1 + row * 3 : 3 + row * 3, 1 + col * 2 : 3 + col * 2] = 1
        images[index] = np.clip(images[index] + rng.normal(0, 0.06, (1, 8, 8)), 0, 1)
    permutation = rng.permutation(count)
    token_ids = np.stack([labels + offset for offset in range(6)], axis=1).astype(
        np.int64
    )
    sequences = latent[:, None, :4] + rng.normal(0, 0.05, (count, 8, 4)).astype(
        np.float32
    )
    return write_npz(
        "multimodal_pairs.npz",
        images=images,
        image_features=latent @ image_map,
        text_features=latent @ text_map,
        token_ids=token_ids,
        tokens=token_ids,
        labels=labels,
        class_ids=labels,
        sequences=sequences,
        class_image_features=latent_centers @ image_map,
        class_text_features=latent_centers @ text_map,
        train_idx=permutation[:192],
        test_idx=permutation[192:],
    )


def compression_data(rng: np.random.Generator) -> Path:
    """Small teacher/student, calibration, image, and supernet fixtures."""
    count, dimensions, classes = 512, 32, 4
    prototypes = rng.normal(0, 1, (classes, dimensions)).astype(np.float32)
    labels = np.arange(count, dtype=np.int64) % classes
    features = prototypes[labels] + rng.normal(0, 0.55, (count, dimensions)).astype(
        np.float32
    )
    teacher_weights = prototypes.T.astype(np.float32)
    teacher_logits = (features @ teacher_weights / np.sqrt(dimensions)).astype(
        np.float32
    )
    teacher_logits += rng.normal(0, 0.04, teacher_logits.shape).astype(np.float32)

    image_count, size = 320, 16
    image_labels = np.arange(image_count, dtype=np.int64) % classes
    images = np.zeros((image_count, 1, size, size), dtype=np.float32)
    for index, label in enumerate(image_labels):
        canvas = rng.normal(0.02, 0.025, (size, size)).astype(np.float32)
        offset = int(rng.integers(-1, 2))
        if label == 0:
            canvas[3 + offset : 13 + offset, 3:6] += 0.9
        elif label == 1:
            canvas[3:6, 3 + offset : 13 + offset] += 0.9
        elif label == 2:
            for diagonal in range(3, 13):
                canvas[diagonal, np.clip(diagonal + offset, 0, size - 1)] += 0.9
        else:
            canvas[4:12, 4:12] += 0.8
            canvas[6:10, 6:10] -= 0.75
        images[index, 0] = np.clip(canvas, 0, 1)

    sequence_count, tokens, hidden = 256, 6, 16
    token_embeddings = rng.normal(0, 1, (sequence_count, tokens, hidden)).astype(
        np.float32
    )
    hidden_projection = rng.normal(0, 0.3, (hidden, 24)).astype(np.float32)
    teacher_hidden = np.tanh(token_embeddings @ hidden_projection).astype(np.float32)
    attention_logits = (
        token_embeddings @ np.swapaxes(token_embeddings, 1, 2) / np.sqrt(hidden)
    )
    attention_logits -= attention_logits.max(axis=-1, keepdims=True)
    teacher_attention = np.exp(attention_logits)
    teacher_attention /= teacher_attention.sum(axis=-1, keepdims=True)

    dense_weights = rng.normal(0, 0.2, (64, 32)).astype(np.float32)
    dense_weights[np.abs(dense_weights) < 0.08] *= 0.1
    candidate_configs = np.array(
        [
            [8, 2, 3, 0.45],
            [8, 3, 5, 0.70],
            [12, 2, 3, 0.78],
            [12, 3, 5, 1.15],
            [16, 2, 3, 1.22],
            [16, 4, 5, 2.30],
            [24, 3, 3, 2.75],
            [24, 4, 5, 4.10],
        ],
        dtype=np.float32,
    )
    feature_split = rng.permutation(count)
    image_split = rng.permutation(image_count)
    return write_npz(
        "compression_bench.npz",
        features=features,
        labels=labels,
        teacher_logits=teacher_logits,
        teacher_weights=teacher_weights,
        calibration_activations=features[:128],
        dense_weights=dense_weights,
        train_idx=feature_split[:384],
        test_idx=feature_split[384:],
        images=images,
        image_labels=image_labels,
        image_train_idx=image_split[:256],
        image_test_idx=image_split[256:],
        token_embeddings=token_embeddings,
        teacher_hidden=teacher_hidden,
        teacher_attention=teacher_attention.astype(np.float32),
        candidate_configs=candidate_configs,
    )


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)
    paths = [
        vision_data(rng),
        nlp_data(rng),
        generative_data(rng),
        gridworld_data(rng),
        graph_data(rng),
        multimodal_data(rng),
        compression_data(rng),
    ]
    manifest = {
        "schema_version": 1,
        "seed": SEED,
        "license": "CC0-1.0",
        "provenance": "Fully synthetic; generated locally by tools/generate_field_datasets.py",
        "files": [
            {"path": path.name, "bytes": path.stat().st_size, "sha256": sha256(path)}
            for path in paths
        ],
    }
    (OUTPUT / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    for entry in manifest["files"]:
        print(
            f"WROTE {entry['path']} ({entry['bytes']} bytes, sha256={entry['sha256'][:12]}…)"
        )
    print("PASS: seven deterministic field datasets")


if __name__ == "__main__":
    main()
