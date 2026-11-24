"""Repeat Nectar head training across three data splits using cached embeddings."""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from nectar_listwise_experiment import LOSSES, group_metrics, train_head


SPLIT_SEEDS = (17, 42, 101)
TRAINING_SEEDS = (11, 22, 33)
TUNED_STRENGTH = {"pairwise_bt": 0.001, "top1_pl": 0.005, "full_pl": 0.001}


def summarize(values):
    return {"mean": float(np.mean(values)), "std_across_splits": float(np.std(values, ddof=1)),
            "split_values": [float(value) for value in values]}


def render_report(result):
    rows = []
    for method in result["methods"]:
        metrics = method["summary"]
        rows.append(
            f"| {method['loss']} | {'tuned ODIN' if method['odin'] else 'single'} | "
            f"{metrics['top1_accuracy']['mean']:.4f} ± {metrics['top1_accuracy']['std_across_splits']:.4f} | "
            f"{metrics['ndcg_at_7']['mean']:.4f} ± {metrics['ndcg_at_7']['std_across_splits']:.4f} | "
            f"{metrics['reward_length_correlation']['mean']:+.4f} ± {metrics['reward_length_correlation']['std_across_splits']:.4f} | "
            f"{metrics['conditional_top1_gap']['mean']:+.4f} ± {metrics['conditional_top1_gap']['std_across_splits']:.4f} |"
        )
    return "\n".join([
        "# Nectar 三数据划分稳定性", "",
        "将缓存中的 5,000 个七候选 prompt 重新随机划分三次；每个划分使用 3,000/1,000/1,000",
        "训练/验证/测试组，并对每个方法运行三个优化 seed。下表先在每个划分内对优化 seed 求均值，",
        "再报告三个划分均值 ± 划分间标准差。", "",
        "| Loss | Head | Top-1 | NDCG@7 | Reward-length r | Conditional gap |",
        "|---|---|---:|---:|---:|---:|", *rows, "",
        "tuned ODIN 的 λ 固定为原始验证集选出的 0.001/0.005/0.001，而没有在每个新划分上重新选参。",
        "因此该实验检验已选超参数跨划分迁移是否稳定，不是三次独立的嵌套超参数搜索。", "",
        "三个划分仍来自同一个 5,000 组样本池，不能替代新的外部数据集；但它比只改变模型初始化",
        "更直接地覆盖了数据抽样不确定性。",
    ])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", type=Path, default=Path("data/embeddings/minilm_nectar_7way.npz"))
    parser.add_argument("--out", type=Path, default=Path("reports/nectar_listwise"))
    args = parser.parse_args()
    start = time.perf_counter()
    cached = np.load(args.cache)
    embeddings = np.concatenate([cached[f"{name}_embeddings"] for name in ("train", "validation", "test")])
    lengths = np.concatenate([cached[f"{name}_lengths"] for name in ("train", "validation", "test")])
    methods = []
    for loss in LOSSES:
        for odin in (False, True):
            splits = []
            for split_seed in SPLIT_SEEDS:
                order = np.random.default_rng(split_seed).permutation(len(embeddings))
                train_indices = order[:3000]
                test_indices = order[4000:5000]
                run_metrics = []
                for training_seed in TRAINING_SEEDS:
                    strength = TUNED_STRENGTH[loss] if odin else 0.0
                    trained = train_head(
                        embeddings[train_indices], lengths[train_indices], loss,
                        training_seed, odin, epochs=35, learning_rate=0.03,
                        lambda_length=strength, lambda_orthogonal=strength, batch_size=128,
                    )
                    scores = embeddings[test_indices] @ trained["quality"]
                    run_metrics.append(group_metrics(scores, lengths[test_indices])[0])
                split_metrics = {
                    key: float(np.mean([metrics[key] for metrics in run_metrics]))
                    for key in run_metrics[0]
                    if run_metrics[0][key] is not None
                }
                splits.append({"split_seed": split_seed, "runs": run_metrics, "metrics": split_metrics})
                print(loss, "ODIN" if odin else "single", split_seed,
                      split_metrics["ndcg_at_7"], split_metrics["reward_length_correlation"])
            summary = {
                key: summarize([split["metrics"][key] for split in splits])
                for key in splits[0]["metrics"]
            }
            methods.append({"loss": loss, "odin": odin,
                            "strength": TUNED_STRENGTH[loss] if odin else 0.0,
                            "splits": splits, "summary": summary})
    result = {
        "split_seeds": list(SPLIT_SEEDS), "training_seeds": list(TRAINING_SEEDS),
        "pool_size": len(embeddings), "train_n": 3000, "validation_n": 1000, "test_n": 1000,
        "methods": methods, "runtime_seconds": time.perf_counter() - start,
    }
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "split_stability.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    (args.out / "split_stability.md").write_text(render_report(result), encoding="utf-8")


if __name__ == "__main__":
    main()
