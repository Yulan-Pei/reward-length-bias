"""Length-bias experiments on Anthropic HH-RLHF helpful-base.

The script uses only NumPy and the Python standard library. It trains pairwise
bag-of-words reward models and evaluates explicit negative, zero, and positive
length coefficients without requiring a GPU.
"""
from __future__ import annotations

import argparse
import gzip
import json
import math
import random
import re
from collections import Counter
from pathlib import Path

import numpy as np


TOKEN_PATTERN = re.compile(r"[a-z]+(?:'[a-z]+)?|\d+", re.IGNORECASE)
ALPHAS = [-2.0, -1.0, -0.5, -0.25, 0.0, 0.25, 0.5, 1.0, 2.0]


def load_pairs(path: Path) -> list[dict[str, str]]:
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle]


def response_only(transcript: str) -> str:
    marker = "\n\nAssistant:"
    return transcript.rsplit(marker, 1)[-1].strip()


def tokens(text: str) -> list[str]:
    return TOKEN_PATTERN.findall(text.lower())


def split_pairs(rows: list[dict[str, str]], train_size: int, val_size: int, seed: int):
    shuffled = list(rows)
    random.Random(seed).shuffle(shuffled)
    selected = shuffled[: train_size + val_size]
    train = selected[:train_size]
    validation = selected[train_size:]
    return train, validation


def build_vocabulary(rows: list[dict[str, str]], vocab_size: int):
    frequency = Counter()
    document_frequency = Counter()
    documents = []
    for row in rows:
        for field in ("chosen", "rejected"):
            answer_tokens = tokens(response_only(row[field]))
            documents.append(answer_tokens)
            frequency.update(answer_tokens)
            document_frequency.update(set(answer_tokens))

    vocabulary_tokens = [
        token for token, _ in frequency.most_common()
        if len(token) > 1
    ][:vocab_size]
    vocabulary = {token: index for index, token in enumerate(vocabulary_tokens)}
    document_count = len(documents)
    idf = np.array([
        math.log((1 + document_count) / (1 + document_frequency[token])) + 1
        for token in vocabulary_tokens
    ], dtype=np.float32)
    return vocabulary, idf


def vectorize(text: str, vocabulary: dict[str, int], idf: np.ndarray) -> np.ndarray:
    vector = np.zeros(len(vocabulary), dtype=np.float32)
    counts = Counter(tokens(text))
    for token, count in counts.items():
        index = vocabulary.get(token)
        if index is not None:
            vector[index] = math.log1p(count) * idf[index]
    norm = np.linalg.norm(vector)
    if norm:
        vector /= norm
    return vector


def make_arrays(rows, vocabulary, idf, length_scale=None):
    chosen_vectors = []
    rejected_vectors = []
    length_differences = []
    for row in rows:
        chosen = response_only(row["chosen"])
        rejected = response_only(row["rejected"])
        chosen_vectors.append(vectorize(chosen, vocabulary, idf))
        rejected_vectors.append(vectorize(rejected, vocabulary, idf))
        length_differences.append(len(chosen) - len(rejected))

    chosen_matrix = np.stack(chosen_vectors)
    rejected_matrix = np.stack(rejected_vectors)
    content = chosen_matrix - rejected_matrix
    raw_length = np.asarray(length_differences, dtype=np.float32)
    if length_scale is None:
        length_scale = float(np.std(raw_length)) or 1.0
    normalized_length = raw_length / length_scale
    return chosen_matrix, rejected_matrix, content, normalized_length, raw_length, length_scale


def train_pairwise(features: np.ndarray, seed: int, epochs=25, batch_size=256, learning_rate=0.08):
    rng = np.random.default_rng(seed)
    weights = np.zeros(features.shape[1], dtype=np.float32)
    losses = []
    for _ in range(epochs):
        order = rng.permutation(len(features))
        for start in range(0, len(features), batch_size):
            batch = features[order[start:start + batch_size]]
            margins = np.clip(batch @ weights, -30, 30)
            probabilities = 1 / (1 + np.exp(margins))
            gradient = -(batch.T @ probabilities) / len(batch) + 1e-4 * weights
            weights -= learning_rate * gradient
        epoch_margins = np.clip(features @ weights, -30, 30)
        losses.append(float(np.logaddexp(0, -epoch_margins).mean()))
    return weights, losses


def train_mlp(chosen: np.ndarray, rejected: np.ndarray, seed: int, hidden_size=32,
              epochs=12, batch_size=256, learning_rate=0.04):
    rng = np.random.default_rng(seed)
    input_size = chosen.shape[1]
    first = rng.normal(0, math.sqrt(2 / input_size), (input_size, hidden_size)).astype(np.float32)
    bias = np.zeros(hidden_size, dtype=np.float32)
    second = rng.normal(0, math.sqrt(2 / hidden_size), hidden_size).astype(np.float32)
    losses = []

    for _ in range(epochs):
        order = rng.permutation(len(chosen))
        for start in range(0, len(chosen), batch_size):
            index = order[start:start + batch_size]
            chosen_batch = chosen[index]
            rejected_batch = rejected[index]
            chosen_pre = chosen_batch @ first + bias
            rejected_pre = rejected_batch @ first + bias
            chosen_hidden = np.maximum(chosen_pre, 0)
            rejected_hidden = np.maximum(rejected_pre, 0)
            hidden_difference = chosen_hidden - rejected_hidden
            margins = np.clip(hidden_difference @ second, -30, 30)
            margin_gradient = -1 / (1 + np.exp(margins)) / len(index)

            second_gradient = hidden_difference.T @ margin_gradient + 1e-4 * second
            chosen_hidden_gradient = margin_gradient[:, None] * second[None, :]
            rejected_hidden_gradient = -chosen_hidden_gradient
            chosen_pre_gradient = chosen_hidden_gradient * (chosen_pre > 0)
            rejected_pre_gradient = rejected_hidden_gradient * (rejected_pre > 0)
            first_gradient = (
                chosen_batch.T @ chosen_pre_gradient
                + rejected_batch.T @ rejected_pre_gradient
                + 1e-4 * first
            )
            bias_gradient = chosen_pre_gradient.sum(axis=0) + rejected_pre_gradient.sum(axis=0)

            second -= learning_rate * second_gradient
            first -= learning_rate * first_gradient
            bias -= learning_rate * bias_gradient

        epoch_margins = mlp_margins(chosen, rejected, (first, bias, second))
        losses.append(float(np.logaddexp(0, -epoch_margins).mean()))
    return (first, bias, second), losses


def mlp_margins(chosen: np.ndarray, rejected: np.ndarray, parameters) -> np.ndarray:
    first, bias, second = parameters
    chosen_hidden = np.maximum(chosen @ first + bias, 0)
    rejected_hidden = np.maximum(rejected @ first + bias, 0)
    return (chosen_hidden - rejected_hidden) @ second


def balanced_indices(raw_length: np.ndarray, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    positive = np.flatnonzero(raw_length > 0)
    negative = np.flatnonzero(raw_length < 0)
    ties = np.flatnonzero(raw_length == 0)
    count = min(len(positive), len(negative))
    selected = np.concatenate([
        rng.choice(positive, count, replace=False),
        rng.choice(negative, count, replace=False),
        ties,
    ])
    rng.shuffle(selected)
    return selected


def accuracy(margins: np.ndarray) -> float:
    return float(((margins > 0) + 0.5 * (margins == 0)).mean())


def metrics(margins: np.ndarray, raw_length: np.ndarray) -> dict[str, float | int]:
    chosen_long = raw_length > 0
    rejected_long = raw_length < 0
    longer_selected = np.where(raw_length > 0, margins > 0, margins < 0)
    result = {
        "n": len(margins),
        "accuracy": round(accuracy(margins), 4),
        "chosen_long_n": int(chosen_long.sum()),
        "rejected_long_n": int(rejected_long.sum()),
        "chosen_long_accuracy": round(accuracy(margins[chosen_long]), 4),
        "rejected_long_accuracy": round(accuracy(margins[rejected_long]), 4),
        "longer_selected_rate": round(float(longer_selected[raw_length != 0].mean()), 4),
    }
    result["length_slice_gap"] = round(
        abs(result["chosen_long_accuracy"] - result["rejected_long_accuracy"]), 4
    )
    return result


def render_report(summary: dict) -> str:
    rows = []
    for experiment in summary["fixed_length_sweep"]:
        test = experiment["test"]
        rows.append(
            f"| {experiment['alpha']:+.2f} | {test['accuracy']:.4f} | "
            f"{test['chosen_long_accuracy']:.4f} | {test['rejected_long_accuracy']:.4f} | "
            f"{test['length_slice_gap']:.4f} | {test['longer_selected_rate']:.4f} |"
        )

    learned_rows = []
    for name in ("joint", "balanced_resampling"):
        experiment = summary["learned_models"][name]
        test = experiment["test"]
        learned_rows.append(
            f"| {name} | {experiment['length_coefficient']:+.4f} | "
            f"{test['accuracy']:.4f} | {test['chosen_long_accuracy']:.4f} | "
            f"{test['rejected_long_accuracy']:.4f} | {test['length_slice_gap']:.4f} |"
        )

    content_rows = []
    for name in ("content_only", "mlp_content_only"):
        test = summary[name]["test"]
        content_rows.append(
            f"| {name} | {test['accuracy']:.4f} | {test['chosen_long_accuracy']:.4f} | "
            f"{test['rejected_long_accuracy']:.4f} | {test['length_slice_gap']:.4f} | "
            f"{test['longer_selected_rate']:.4f} |"
        )

    return "\n".join([
        "# HH-RLHF 长度偏差实验报告",
        "",
        "数据来自 Anthropic HH-RLHF `helpful-base`。长度系数 alpha 作用于标准化后的",
        "`len(chosen) - len(rejected)`。alpha 为负时偏向短回答，为零时忽略显式长度，",
        "为正时偏向长回答。alpha 由验证集评测，测试集只用于最终报告。",
        "",
        "## 数据",
        "",
        f"- 训练偏好对：{summary['data']['train_n']}",
        f"- 验证偏好对：{summary['data']['validation_n']}",
        f"- 测试偏好对：{summary['data']['test_n']}",
        f"- 训练集中 chosen 更长比例：{summary['data']['train_chosen_long_rate']:.4f}",
        f"- 测试集中 chosen 更长比例：{summary['data']['test_chosen_long_rate']:.4f}",
        "",
        "## 固定长度系数消融",
        "",
        "内容分数由 TF-IDF 词袋偏好模型给出，alpha 是额外的长度校准项。",
        "",
        "![长度系数消融](length_sweep.svg)",
        "",
        "| alpha | Accuracy | Chosen-long acc | Rejected-long acc | Slice gap | Longer selected |",
        "|---:|---:|---:|---:|---:|---:|",
        *rows,
        "",
        f"验证集选择的 alpha：**{summary['selected_alpha']:+.2f}**",
        "",
        "## 线性与非线性内容模型",
        "",
        "| Model | Accuracy | Chosen-long acc | Rejected-long acc | Slice gap | Longer selected |",
        "|---|---:|---:|---:|---:|---:|",
        *content_rows,
        "",
        "## 学习长度系数与重采样",
        "",
        "| Model | Learned alpha | Accuracy | Chosen-long acc | Rejected-long acc | Slice gap |",
        "|---|---:|---:|---:|---:|---:|",
        *learned_rows,
        "",
        "## 解释口径",
        "",
        "- Accuracy 衡量 chosen 是否获得更高奖励。",
        "- Slice gap 衡量 chosen 更长与 rejected 更长两组的准确率差距，越小越平衡。",
        "- Longer selected 衡量模型选择较长回答的频率，它不是质量指标，需要结合 Accuracy 看。",
        "- 固定 alpha 消融隔离了长度项的因果影响。重采样实验检验数据层面的长度平衡。",
        "",
        "## 来源",
        "",
        "- 数据：https://huggingface.co/datasets/Anthropic/hh-rlhf",
        "- 论文：https://arxiv.org/abs/2204.05862",
    ])


def write_sweep_svg(summary: dict, path: Path):
    width, height = 900, 500
    left, right, top, bottom = 75, 35, 45, 70
    plot_width = width - left - right
    plot_height = height - top - bottom
    experiments = summary["fixed_length_sweep"]
    alphas = [item["alpha"] for item in experiments]
    series = {
        "Overall accuracy": ("#2563eb", [item["test"]["accuracy"] for item in experiments]),
        "Chosen longer": ("#16a34a", [item["test"]["chosen_long_accuracy"] for item in experiments]),
        "Rejected longer": ("#dc2626", [item["test"]["rejected_long_accuracy"] for item in experiments]),
    }

    def x_position(value):
        return left + (value - min(alphas)) / (max(alphas) - min(alphas)) * plot_width

    def y_position(value):
        return top + (1 - value) * plot_height

    elements = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        '<style>text{font-family:Arial,sans-serif;fill:#111827}.grid{stroke:#e5e7eb}.axis{stroke:#374151;stroke-width:1.4}</style>',
        '<text x="450" y="26" text-anchor="middle" font-size="18" font-weight="bold">HH-RLHF: fixed length coefficient sweep</text>',
    ]
    for value in (0, 0.25, 0.5, 0.75, 1):
        y = y_position(value)
        elements.append(f'<line class="grid" x1="{left}" y1="{y:.1f}" x2="{width-right}" y2="{y:.1f}"/>')
        elements.append(f'<text x="{left-12}" y="{y+5:.1f}" text-anchor="end" font-size="12">{value:.2f}</text>')
    elements.extend([
        f'<line class="axis" x1="{left}" y1="{top}" x2="{left}" y2="{height-bottom}"/>',
        f'<line class="axis" x1="{left}" y1="{height-bottom}" x2="{width-right}" y2="{height-bottom}"/>',
    ])
    for alpha in alphas:
        x = x_position(alpha)
        elements.append(f'<text x="{x:.1f}" y="{height-bottom+23}" text-anchor="middle" font-size="12">{alpha:+g}</text>')
    elements.append(f'<text x="{width/2}" y="{height-18}" text-anchor="middle" font-size="14">Length coefficient alpha</text>')
    elements.append(f'<text x="18" y="{height/2}" text-anchor="middle" font-size="14" transform="rotate(-90 18 {height/2})">Pairwise accuracy</text>')

    for index, (name, (color, values)) in enumerate(series.items()):
        points = " ".join(f"{x_position(a):.1f},{y_position(v):.1f}" for a, v in zip(alphas, values))
        elements.append(f'<polyline points="{points}" fill="none" stroke="{color}" stroke-width="3"/>')
        for alpha, value in zip(alphas, values):
            elements.append(f'<circle cx="{x_position(alpha):.1f}" cy="{y_position(value):.1f}" r="4" fill="{color}"/>')
        legend_x = 585 + index * 105
        elements.append(f'<line x1="{legend_x}" y1="42" x2="{legend_x+22}" y2="42" stroke="{color}" stroke-width="3"/>')
        elements.append(f'<text x="{legend_x+27}" y="46" font-size="11">{name}</text>')
    elements.append('</svg>')
    path.write_text("\n".join(elements) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type=Path, default=Path("data/helpful-base/train.jsonl.gz"))
    parser.add_argument("--test", type=Path, default=Path("data/helpful-base/test.jsonl.gz"))
    parser.add_argument("--out", type=Path, default=Path("reports/hh_rlhf"))
    parser.add_argument("--train-size", type=int, default=8000)
    parser.add_argument("--val-size", type=int, default=2000)
    parser.add_argument("--vocab-size", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    source_train = load_pairs(args.train)
    test_rows = load_pairs(args.test)
    train_rows, validation_rows = split_pairs(
        source_train, args.train_size, args.val_size, args.seed
    )
    vocabulary, idf = build_vocabulary(train_rows, args.vocab_size)
    train_chosen, train_rejected, train_x, train_z, train_raw, length_scale = make_arrays(
        train_rows, vocabulary, idf
    )
    val_chosen, val_rejected, val_x, val_z, val_raw, _ = make_arrays(
        validation_rows, vocabulary, idf, length_scale
    )
    test_chosen, test_rejected, test_x, test_z, test_raw, _ = make_arrays(
        test_rows, vocabulary, idf, length_scale
    )

    content_weights, content_losses = train_pairwise(train_x, args.seed)
    val_content = val_x @ content_weights
    test_content = test_x @ content_weights

    sweep = []
    for alpha in ALPHAS:
        sweep.append({
            "alpha": alpha,
            "validation": metrics(val_content + alpha * val_z, val_raw),
            "test": metrics(test_content + alpha * test_z, test_raw),
        })
    selected = max(
        sweep,
        key=lambda item: (
            item["validation"]["accuracy"],
            -item["validation"]["length_slice_gap"],
            -abs(item["alpha"]),
        ),
    )["alpha"]

    joint_features = np.column_stack([train_x, train_z])
    joint_weights, joint_losses = train_pairwise(joint_features, args.seed)
    balanced = balanced_indices(train_raw, args.seed)
    balanced_weights, balanced_losses = train_pairwise(joint_features[balanced], args.seed)
    mlp_parameters, mlp_losses = train_mlp(train_chosen, train_rejected, args.seed)
    val_mlp = mlp_margins(val_chosen, val_rejected, mlp_parameters)
    test_mlp = mlp_margins(test_chosen, test_rejected, mlp_parameters)

    def learned_result(weights, losses):
        return {
            "length_coefficient": float(weights[-1]),
            "final_train_loss": round(losses[-1], 6),
            "validation": metrics(val_x @ weights[:-1] + weights[-1] * val_z, val_raw),
            "test": metrics(test_x @ weights[:-1] + weights[-1] * test_z, test_raw),
        }

    summary = {
        "data": {
            "dataset": "Anthropic/hh-rlhf helpful-base",
            "train_n": len(train_rows),
            "validation_n": len(validation_rows),
            "test_n": len(test_rows),
            "vocab_size": len(vocabulary),
            "length_scale_characters": round(length_scale, 4),
            "train_chosen_long_rate": round(float((train_raw > 0).mean()), 4),
            "test_chosen_long_rate": round(float((test_raw > 0).mean()), 4),
            "balanced_train_n": len(balanced),
        },
        "content_only": {
            "final_train_loss": round(content_losses[-1], 6),
            "validation": metrics(val_content, val_raw),
            "test": metrics(test_content, test_raw),
        },
        "mlp_content_only": {
            "hidden_size": len(mlp_parameters[-1]),
            "final_train_loss": round(mlp_losses[-1], 6),
            "validation": metrics(val_mlp, val_raw),
            "test": metrics(test_mlp, test_raw),
        },
        "fixed_length_sweep": sweep,
        "selected_alpha": selected,
        "learned_models": {
            "joint": learned_result(joint_weights, joint_losses),
            "balanced_resampling": learned_result(balanced_weights, balanced_losses),
        },
    }

    zero = next(item for item in sweep if item["alpha"] == 0)
    assert zero["test"] == summary["content_only"]["test"]
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "results.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (args.out / "report.md").write_text(render_report(summary) + "\n", encoding="utf-8")
    write_sweep_svg(summary, args.out / "length_sweep.svg")
    if not args.quiet:
        print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

