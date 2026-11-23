"""Frozen MiniLM encoder plus trainable pairwise reward head on HH-RLHF."""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

os.environ.setdefault("USE_TF", "0")

import numpy as np
from sentence_transformers import SentenceTransformer

from real_data_experiment import (
    ALPHAS,
    accuracy,
    balanced_indices,
    load_pairs,
    metrics,
    response_only,
    split_pairs,
    train_pairwise,
)


SEEDS = [11, 22, 33, 44, 55]


def prompt_and_response(transcript: str) -> str:
    marker = "\n\nAssistant:"
    prompt, answer = transcript.rsplit(marker, 1)
    latest_human = prompt.rsplit("\n\nHuman:", 1)[-1].strip()
    return f"Human: {latest_human}\nAssistant: {answer.strip()}"


def encode_rows(model, rows, batch_size: int):
    chosen_texts = [prompt_and_response(row["chosen"]) for row in rows]
    rejected_texts = [prompt_and_response(row["rejected"]) for row in rows]
    texts = chosen_texts + rejected_texts
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    ).astype(np.float32)
    middle = len(rows)
    chosen = embeddings[:middle]
    rejected = embeddings[middle:]
    raw_length = np.asarray([
        len(response_only(row["chosen"])) - len(response_only(row["rejected"]))
        for row in rows
    ], dtype=np.float32)
    return chosen, rejected, raw_length


def load_or_encode(model, splits, cache_path: Path, batch_size: int):
    if cache_path.exists():
        cached = np.load(cache_path)
        return {
            name: (cached[f"{name}_chosen"], cached[f"{name}_rejected"], cached[f"{name}_length"])
            for name in splits
        }
    arrays = {}
    payload = {}
    for name, rows in splits.items():
        chosen, rejected, raw_length = encode_rows(model, rows, batch_size)
        arrays[name] = (chosen, rejected, raw_length)
        payload[f"{name}_chosen"] = chosen
        payload[f"{name}_rejected"] = rejected
        payload[f"{name}_length"] = raw_length
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache_path, **payload)
    return arrays


def bootstrap_interval(margins: np.ndarray, seed: int, repetitions=3000):
    rng = np.random.default_rng(seed)
    values = np.empty(repetitions, dtype=np.float32)
    for index in range(repetitions):
        sample = rng.integers(0, len(margins), len(margins))
        values[index] = accuracy(margins[sample])
    return [round(float(value), 4) for value in np.quantile(values, [0.025, 0.975])]


def paired_bootstrap_delta(first: np.ndarray, second: np.ndarray, seed: int, repetitions=3000):
    rng = np.random.default_rng(seed)
    values = np.empty(repetitions, dtype=np.float32)
    for index in range(repetitions):
        sample = rng.integers(0, len(first), len(first))
        values[index] = accuracy(first[sample]) - accuracy(second[sample])
    return [round(float(value), 4) for value in np.quantile(values, [0.025, 0.975])]


def seed_summary(values):
    return {
        "mean": round(float(np.mean(values)), 4),
        "std": round(float(np.std(values, ddof=1)), 4),
        "values": [round(float(value), 4) for value in values],
    }


def render_report(result: dict) -> str:
    sweep_rows = []
    for item in result["fixed_length_sweep"]:
        test = item["test"]
        sweep_rows.append(
            f"| {item['alpha']:+.2f} | {test['accuracy']:.4f} | "
            f"{test['chosen_long_accuracy']:.4f} | {test['rejected_long_accuracy']:.4f} | "
            f"{test['length_slice_gap']:.4f} |"
        )
    seed_values = ", ".join(f"{value:.4f}" for value in result["five_seed_accuracy"]["values"])
    return "\n".join([
        "# Frozen MiniLM Reward Model 实验",
        "",
        "本实验使用 `sentence-transformers/all-MiniLM-L6-v2` 将真实 prompt-response",
        "编码为 384 维向量，冻结 Transformer 参数，只训练一个标量 Reward Head。",
        "训练目标为 Bradley-Terry pairwise loss。",
        "",
        "## 模型与数据",
        "",
        f"- Encoder 参数量：{result['model']['parameter_count']:,}",
        f"- Reward Head 可训练参数：{result['model']['trainable_head_parameters']}",
        f"- 最大 token 长度：{result['model']['max_sequence_length']}",
        f"- 训练/验证/测试：{result['data']['train_n']} / {result['data']['validation_n']} / {result['data']['test_n']}",
        f"- CPU 总运行时间：{result['runtime_seconds']:.1f} 秒",
        "",
        "## 五次随机种子",
        "",
        f"- 测试准确率：{result['five_seed_accuracy']['mean']:.4f} ± {result['five_seed_accuracy']['std']:.4f}",
        f"- 各次结果：{seed_values}",
        f"- Ensemble 测试准确率：{result['ensemble']['test']['accuracy']:.4f}",
        f"- 95% bootstrap CI：{result['ensemble']['accuracy_ci_95']}",
        "",
        "## 固定长度系数消融",
        "",
        "| alpha | Accuracy | Chosen-long | Rejected-long | Slice gap |",
        "|---:|---:|---:|---:|---:|",
        *sweep_rows,
        "",
        f"验证集选择 alpha：**{result['selected_alpha']:+.2f}**",
        f"相对 alpha=0 的测试准确率变化：{result['selected_vs_zero']['delta']:+.4f}",
        f"变化的 paired bootstrap 95% CI：{result['selected_vs_zero']['delta_ci_95']}",
        "",
        "## 长度平衡重采样",
        "",
        f"- 原始训练的长度系数：{result['joint_head']['length_coefficient']:+.4f}",
        f"- 平衡训练的长度系数：{result['balanced_head']['length_coefficient']:+.4f}",
        f"- 原始测试准确率：{result['joint_head']['test']['accuracy']:.4f}",
        f"- 平衡测试准确率：{result['balanced_head']['test']['accuracy']:.4f}",
        f"- 原始/平衡 slice gap：{result['joint_head']['test']['length_slice_gap']:.4f} / {result['balanced_head']['test']['length_slice_gap']:.4f}",
        "",
        "## 能力边界",
        "",
        "Encoder 使用预训练 Transformer 并保持冻结。项目实际训练的是 Reward Head，",
        "因此它证明了真实 Transformer 表示上的偏好建模与长度诊断能力，",
        "尚未覆盖端到端微调、LoRA、多卡训练或策略优化。",
    ])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type=Path, default=Path("data/helpful-base/train.jsonl.gz"))
    parser.add_argument("--test", type=Path, default=Path("data/helpful-base/test.jsonl.gz"))
    parser.add_argument("--out", type=Path, default=Path("reports/transformer"))
    parser.add_argument("--cache", type=Path, default=Path("data/embeddings/minilm_hh_rlhf.npz"))
    parser.add_argument("--model", default="sentence-transformers/all-MiniLM-L6-v2")
    parser.add_argument("--train-size", type=int, default=8000)
    parser.add_argument("--val-size", type=int, default=2000)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--local-files-only", action="store_true")
    args = parser.parse_args()
    start = time.perf_counter()

    source_train = load_pairs(args.train)
    test_rows = load_pairs(args.test)
    train_rows, validation_rows = split_pairs(
        source_train, args.train_size, args.val_size, args.seed
    )
    model = SentenceTransformer(
        args.model, device="cpu", local_files_only=args.local_files_only
    )
    model.max_seq_length = 256
    splits = {"train": train_rows, "validation": validation_rows, "test": test_rows}
    arrays = load_or_encode(model, splits, args.cache, args.batch_size)
    train_chosen, train_rejected, train_raw = arrays["train"]
    val_chosen, val_rejected, val_raw = arrays["validation"]
    test_chosen, test_rejected, test_raw = arrays["test"]
    train_x = train_chosen - train_rejected
    val_x = val_chosen - val_rejected
    test_x = test_chosen - test_rejected
    length_scale = float(np.std(train_raw)) or 1.0
    train_z, val_z, test_z = train_raw / length_scale, val_raw / length_scale, test_raw / length_scale

    validation_margins = []
    test_margins = []
    seed_accuracies = []
    for seed in SEEDS:
        weights, _ = train_pairwise(train_x, seed, epochs=40, learning_rate=0.06)
        validation_margins.append(val_x @ weights)
        margin = test_x @ weights
        test_margins.append(margin)
        seed_accuracies.append(accuracy(margin))
    validation_content = np.mean(validation_margins, axis=0)
    test_content = np.mean(test_margins, axis=0)

    sweep = []
    for alpha in ALPHAS:
        sweep.append({
            "alpha": alpha,
            "validation": metrics(validation_content + alpha * val_z, val_raw),
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
    selected_test = test_content + selected * test_z
    zero_test = test_content

    joint_train = np.column_stack([train_x, train_z])
    joint_weights, _ = train_pairwise(joint_train, args.seed, epochs=40, learning_rate=0.06)
    balanced = balanced_indices(train_raw, args.seed)
    balanced_weights, _ = train_pairwise(joint_train[balanced], args.seed, epochs=40, learning_rate=0.06)

    def head_result(weights):
        return {
            "length_coefficient": round(float(weights[-1]), 6),
            "validation": metrics(val_x @ weights[:-1] + weights[-1] * val_z, val_raw),
            "test": metrics(test_x @ weights[:-1] + weights[-1] * test_z, test_raw),
        }

    result = {
        "model": {
            "name": args.model,
            "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
            "embedding_dimension": model.get_sentence_embedding_dimension(),
            "max_sequence_length": model.max_seq_length,
            "encoder_frozen": True,
            "trainable_head_parameters": model.get_sentence_embedding_dimension(),
        },
        "data": {
            "dataset": "Anthropic/hh-rlhf helpful-base",
            "train_n": len(train_rows),
            "validation_n": len(validation_rows),
            "test_n": len(test_rows),
            "length_scale_characters": round(length_scale, 4),
        },
        "five_seed_accuracy": seed_summary(seed_accuracies),
        "ensemble": {
            "test": metrics(test_content, test_raw),
            "accuracy_ci_95": bootstrap_interval(test_content, args.seed),
        },
        "fixed_length_sweep": sweep,
        "selected_alpha": selected,
        "selected_vs_zero": {
            "delta": round(accuracy(selected_test) - accuracy(zero_test), 4),
            "delta_ci_95": paired_bootstrap_delta(selected_test, zero_test, args.seed),
        },
        "joint_head": head_result(joint_weights),
        "balanced_head": head_result(balanced_weights),
        "runtime_seconds": round(time.perf_counter() - start, 2),
    }
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "results.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (args.out / "report.md").write_text(render_report(result) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
