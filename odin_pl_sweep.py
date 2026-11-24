"""Validation-only ODIN regularization sweep followed by one held-out test evaluation."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from nectar_listwise_experiment import LOSSES, SEEDS, group_metrics, summarize_runs, train_head


CANDIDATES = (0.001, 0.005, 0.01, 0.02, 0.05, 0.1, 1.0)
TUNING_SEEDS = (11, 22, 33)
MAX_ABS_CORRELATION = 0.40


def fit_and_score(train_x, train_lengths, eval_x, eval_lengths, kind, seed, strength):
    trained = train_head(
        train_x, train_lengths, kind, seed, True,
        epochs=35, learning_rate=0.03,
        lambda_length=strength, lambda_orthogonal=strength, batch_size=128,
    )
    quality = eval_x @ trained["quality"]
    metrics, _ = group_metrics(quality, eval_lengths)
    total = quality + eval_x @ trained["length"]
    return {
        "seed": seed,
        "metrics": metrics,
        "training_total_reward_metrics": group_metrics(total, eval_lengths)[0],
    }


def render_report(result, baseline):
    lines = [
        "# ODIN × PL 正则强度消融", "",
        "ODIN 原论文使用强度 1。本实验对 CPU 线性头不直接假设该尺度可迁移，",
        "只在验证集上扫描强度；长度相关与正交惩罚使用同一个系数。", "",
        f"选择规则：先要求质量头 `|reward-length r| ≤ {result['selection']['max_abs_correlation']:.2f}`，",
        "再在满足条件的候选中选择验证集 NDCG@7 最高者。测试集只在选择完成后评测。", "",
        "## 验证集扫描（三个 seed）", "",
        "| Loss | λ | Top-1 | NDCG@7 | Reward-length r |",
        "|---|---:|---:|---:|---:|",
    ]
    for loss in LOSSES:
        for item in result["validation_sweep"][loss]:
            s = item["summary"]
            lines.append(f"| {loss} | {item['strength']:.3f} | {s['top1_accuracy']['mean']:.4f} | "
                         f"{s['ndcg_at_7']['mean']:.4f} | {s['reward_length_correlation']['mean']:+.4f} |")
    lines += ["", "## 选定强度的测试集结果（五个 seed）", "",
              "| Loss | λ | Top-1 | Pairwise | NDCG@7 | Reward-length r | Longest selected |",
              "|---|---:|---:|---:|---:|---:|---:|"]
    for item in result["test"]:
        s = item["summary"]
        lines.append(f"| {item['loss']} | {item['strength']:.3f} | {s['top1_accuracy']['mean']:.4f} | "
                     f"{s['pairwise_accuracy']['mean']:.4f} | {s['ndcg_at_7']['mean']:.4f} | "
                     f"{s['reward_length_correlation']['mean']:+.4f} | {s['longest_selected_rate']['mean']:.4f} |")
    lines += ["", "## 条件切片：tuned ODIN 如何缩小 gap", "",
              "| Loss | Head | 最佳即最长 | 最佳非最长 | Gap |",
              "|---|---|---:|---:|---:|"]
    baseline_single = {item["loss"]: item for item in baseline["methods"] if not item["odin"]}
    for item in result["test"]:
        loss, tuned = item["loss"], item["summary"]
        base = baseline_single[loss]["summary"]
        lines.append(f"| {loss} | single | {base['top1_when_best_longest']['mean']:.4f} | "
                     f"{base['top1_when_best_not_longest']['mean']:.4f} | {base['conditional_top1_gap']['mean']:+.4f} |")
        lines.append(f"| {loss} | tuned ODIN | {tuned['top1_when_best_longest']['mean']:.4f} | "
                     f"{tuned['top1_when_best_not_longest']['mean']:.4f} | {tuned['conditional_top1_gap']['mean']:+.4f} |")
    lines += ["", "小 λ 的 tuned ODIN 不再像 λ=1 那样完全向下拉平，但主要改善仍来自降低‘最佳即最长’切片的优势；",
              "‘最佳非最长’切片只小幅改善。它缓和了模型依赖长度顺风局的程度，没有把困难切片变成容易切片。", "",
              "![NDCG 与长度相关的权衡](tradeoff.svg)", "",
              "该表应与 `report.md` 中的单头基线以及 λ=1 直接迁移结果一起阅读。",
              "若 λ 的轻微变化导致指标突然恶化，应解释为当前小模型上的优化敏感性，而不是隐藏它。"]
    return "\n".join(lines)


def render_tradeoff_svg(result, baseline):
    colors = {"pairwise_bt": "#2563eb", "top1_pl": "#dc2626", "full_pl": "#059669"}
    points = []
    for method in baseline["methods"]:
        summary = method["summary"]
        points.append((method["loss"], "λ=1" if method["odin"] else "single",
                       summary["reward_length_correlation"]["mean"], summary["ndcg_at_7"]["mean"]))
    for item in result["test"]:
        summary = item["summary"]
        points.append((item["loss"], "tuned", summary["reward_length_correlation"]["mean"],
                       summary["ndcg_at_7"]["mean"]))
    width, height, left, top, plot_w, plot_h = 760, 480, 80, 45, 620, 350
    def xy(correlation, ndcg):
        return left + correlation / 0.5 * plot_w, top + (0.81 - ndcg) / 0.08 * plot_h
    svg = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
           '<rect width="100%" height="100%" fill="white"/>',
           '<style>text{font-family:Arial,sans-serif;fill:#172033}.axis{stroke:#64748b;stroke-width:1}.grid{stroke:#e2e8f0;stroke-width:1}</style>',
           f'<text x="{width/2}" y="24" text-anchor="middle" font-size="18" font-weight="bold">Ranking quality vs. length correlation</text>']
    for value in (0.0, 0.1, 0.2, 0.3, 0.4, 0.5):
        x, _ = xy(value, 0.73)
        svg += [f'<line class="grid" x1="{x:.1f}" y1="{top}" x2="{x:.1f}" y2="{top+plot_h}"/>',
                f'<text x="{x:.1f}" y="{top+plot_h+22}" text-anchor="middle" font-size="12">{value:.1f}</text>']
    for value in (0.73, 0.75, 0.77, 0.79, 0.81):
        _, y = xy(0, value)
        svg += [f'<line class="grid" x1="{left}" y1="{y:.1f}" x2="{left+plot_w}" y2="{y:.1f}"/>',
                f'<text x="{left-10}" y="{y+4:.1f}" text-anchor="end" font-size="12">{value:.2f}</text>']
    svg += [f'<line class="axis" x1="{left}" y1="{top+plot_h}" x2="{left+plot_w}" y2="{top+plot_h}"/>',
            f'<line class="axis" x1="{left}" y1="{top}" x2="{left}" y2="{top+plot_h}"/>',
            f'<text x="{left+plot_w/2}" y="{height-22}" text-anchor="middle" font-size="14">Within-prompt reward-length correlation (lower is better)</text>',
            f'<text x="18" y="{top+plot_h/2}" text-anchor="middle" font-size="14" transform="rotate(-90 18 {top+plot_h/2})">NDCG@7 (higher is better)</text>']
    for loss, label, correlation, ndcg in points:
        x, y = xy(correlation, ndcg)
        svg += [f'<circle cx="{x:.1f}" cy="{y:.1f}" r="6" fill="{colors[loss]}"/>',
                f'<text x="{x+8:.1f}" y="{y-8:.1f}" font-size="11">{loss} {label}</text>']
    svg.append('</svg>')
    return "\n".join(svg)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", type=Path, default=Path("data/embeddings/minilm_nectar_7way.npz"))
    parser.add_argument("--out", type=Path, default=Path("reports/nectar_listwise"))
    parser.add_argument("--baseline", type=Path, default=Path("reports/nectar_listwise/results.json"))
    parser.add_argument("--render-only", action="store_true")
    args = parser.parse_args()
    if args.render_only:
        result = json.loads((args.out / "odin_sweep.json").read_text(encoding="utf-8"))
        baseline = json.loads(args.baseline.read_text(encoding="utf-8"))
        (args.out / "odin_sweep.md").write_text(render_report(result, baseline), encoding="utf-8")
        (args.out / "tradeoff.svg").write_text(render_tradeoff_svg(result, baseline), encoding="utf-8")
        return
    cached = np.load(args.cache)
    train_x, train_lengths = cached["train_embeddings"], cached["train_lengths"]
    val_x, val_lengths = cached["validation_embeddings"], cached["validation_lengths"]
    test_x, test_lengths = cached["test_embeddings"], cached["test_lengths"]

    validation_sweep, selected = {}, {}
    for kind in LOSSES:
        items = []
        for strength in CANDIDATES:
            runs = [fit_and_score(train_x, train_lengths, val_x, val_lengths,
                                  kind, seed, strength) for seed in TUNING_SEEDS]
            summary = summarize_runs(runs)
            items.append({"strength": strength, "summary": summary})
            print(kind, strength, summary["ndcg_at_7"]["mean"],
                  summary["reward_length_correlation"]["mean"])
        eligible = [item for item in items
                    if abs(item["summary"]["reward_length_correlation"]["mean"]) <= MAX_ABS_CORRELATION]
        chosen = max(eligible or items, key=lambda item: item["summary"]["ndcg_at_7"]["mean"])
        validation_sweep[kind] = items
        selected[kind] = chosen["strength"]

    test = []
    for kind in LOSSES:
        strength = selected[kind]
        runs = [fit_and_score(train_x, train_lengths, test_x, test_lengths,
                              kind, seed, strength) for seed in SEEDS]
        test.append({"loss": kind, "strength": strength, "runs": runs,
                     "summary": summarize_runs(runs)})
    result = {
        "selection": {"metric": "validation NDCG@7", "max_abs_correlation": MAX_ABS_CORRELATION,
                      "candidates": list(CANDIDATES), "tuning_seeds": list(TUNING_SEEDS),
                      "selected": selected},
        "validation_sweep": validation_sweep,
        "test": test,
    }
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "odin_sweep.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    baseline = json.loads(args.baseline.read_text(encoding="utf-8"))
    (args.out / "odin_sweep.md").write_text(render_report(result, baseline), encoding="utf-8")
    (args.out / "tradeoff.svg").write_text(render_tradeoff_svg(result, baseline), encoding="utf-8")
    print("selected", selected)


if __name__ == "__main__":
    main()
