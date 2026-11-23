"""Paired statistical comparison of TF-IDF and frozen MiniLM reward heads."""
import json
import math
from pathlib import Path

import numpy as np

from real_data_experiment import (
    SEEDS,
    accuracy,
    bootstrap_interval,
    build_vocabulary,
    load_pairs,
    make_arrays,
    paired_bootstrap_delta,
    split_pairs,
    train_pairwise,
)


def ensemble_margins(features: np.ndarray, seeds=SEEDS):
    return np.mean([
        features @ train_pairwise(features, seed)[0]
        for seed in seeds
    ], axis=0)


def exact_mcnemar(first_correct: np.ndarray, second_correct: np.ndarray):
    first_only = int((first_correct & ~second_correct).sum())
    second_only = int((~first_correct & second_correct).sum())
    discordant = first_only + second_only
    if discordant == 0:
        p_value = 1.0
    else:
        tail = sum(math.comb(discordant, index) for index in range(min(first_only, second_only) + 1))
        p_value = min(1.0, 2 * tail / (2 ** discordant))
    return {
        "tfidf_only_correct": first_only,
        "minilm_only_correct": second_only,
        "discordant": discordant,
        "two_sided_exact_p": p_value,
    }


def main():
    root = Path(__file__).resolve().parent
    train_rows, _ = split_pairs(
        load_pairs(root / "data/helpful-base/train.jsonl.gz"), 8000, 2000, 42
    )
    test_rows = load_pairs(root / "data/helpful-base/test.jsonl.gz")
    vocabulary, idf = build_vocabulary(train_rows, 1000)
    _, _, train_x, _, _, length_scale = make_arrays(train_rows, vocabulary, idf)
    _, _, test_x, _, _, _ = make_arrays(test_rows, vocabulary, idf, length_scale)

    tfidf_test = np.mean([
        test_x @ train_pairwise(train_x, seed)[0]
        for seed in SEEDS
    ], axis=0)

    cached = np.load(root / "data/embeddings/minilm_hh_rlhf.npz")
    transformer_train_x = cached["train_chosen"] - cached["train_rejected"]
    transformer_test_x = cached["test_chosen"] - cached["test_rejected"]
    minilm_test = np.mean([
        transformer_test_x @ train_pairwise(
            transformer_train_x, seed, epochs=40, learning_rate=0.06
        )[0]
        for seed in SEEDS
    ], axis=0)

    tfidf_correct = tfidf_test > 0
    minilm_correct = minilm_test > 0
    result = {
        "n": len(test_rows),
        "tfidf_accuracy": round(accuracy(tfidf_test), 4),
        "tfidf_ci_95": bootstrap_interval(tfidf_test, 42),
        "minilm_accuracy": round(accuracy(minilm_test), 4),
        "minilm_ci_95": bootstrap_interval(minilm_test, 43),
        "minilm_minus_tfidf": round(accuracy(minilm_test) - accuracy(tfidf_test), 4),
        "paired_delta_ci_95": paired_bootstrap_delta(minilm_test, tfidf_test, 44),
        "both_correct": int((tfidf_correct & minilm_correct).sum()),
        "both_wrong": int((~tfidf_correct & ~minilm_correct).sum()),
        "mcnemar": exact_mcnemar(tfidf_correct, minilm_correct),
    }
    output = root / "reports/model_comparison"
    output.mkdir(parents=True, exist_ok=True)
    (output / "results.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    report = "\n".join([
        "# TF-IDF 与 Frozen MiniLM 配对比较",
        "",
        f"- 测试偏好对：{result['n']}",
        f"- TF-IDF：{result['tfidf_accuracy']:.4f}，95% CI {result['tfidf_ci_95']}",
        f"- MiniLM：{result['minilm_accuracy']:.4f}，95% CI {result['minilm_ci_95']}",
        f"- MiniLM - TF-IDF：{result['minilm_minus_tfidf']:+.4f}",
        f"- 配对 bootstrap 95% CI：{result['paired_delta_ci_95']}",
        f"- 仅 TF-IDF 正确：{result['mcnemar']['tfidf_only_correct']}",
        f"- 仅 MiniLM 正确：{result['mcnemar']['minilm_only_correct']}",
        f"- McNemar 双侧精确检验 p：{result['mcnemar']['two_sided_exact_p']:.6f}",
        "",
        "配对检验使用完全相同的测试样本，适合判断模型差异是否超出样本波动。",
    ])
    (output / "report.md").write_text(report + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
