"""BT, top-1 PL, full PL, and ODIN-style heads on seven-way Nectar rankings."""
from __future__ import annotations

import argparse
import json
import os
import random
import time
from pathlib import Path

os.environ.setdefault("USE_TF", "0")

import numpy as np
import torch
import torch.nn.functional as F
from sentence_transformers import SentenceTransformer


LOSSES = ("pairwise_bt", "top1_pl", "full_pl")
SEEDS = (11, 22, 33, 44, 55)


def load_rows(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def ordered_candidates(row: dict) -> tuple[list[str], np.ndarray]:
    answers = sorted(row["answers"], key=lambda item: item["rank"])
    prompt = row["prompt"].strip()
    texts = [f"{prompt}\n{answer['answer'].strip()}" for answer in answers]
    lengths = np.asarray([len(answer["answer"].strip()) for answer in answers], dtype=np.float32)
    return texts, lengths


def split_rows(rows: list[dict], train_n: int, val_n: int, test_n: int, seed: int):
    if train_n + val_n + test_n > len(rows):
        raise ValueError("requested split is larger than the prepared subset")
    order = list(range(len(rows)))
    random.Random(seed).shuffle(order)
    selected = [rows[index] for index in order[: train_n + val_n + test_n]]
    return {
        "train": selected[:train_n],
        "validation": selected[train_n : train_n + val_n],
        "test": selected[train_n + val_n :],
    }


def load_or_encode(model, splits: dict, cache_path: Path, batch_size: int, rows_per_chunk: int = 100):
    if cache_path.exists():
        cached = np.load(cache_path)
        return {
            name: (cached[f"{name}_embeddings"], cached[f"{name}_lengths"])
            for name in splits
        }
    payload = {}
    result = {}
    chunk_dir = cache_path.with_suffix("")
    chunk_dir.mkdir(parents=True, exist_ok=True)
    for name, rows in splits.items():
        embedding_chunks, length_chunks = [], []
        for start in range(0, len(rows), rows_per_chunk):
            chunk_path = chunk_dir / f"{name}_{start:05d}.npz"
            chunk_rows = rows[start : start + rows_per_chunk]
            if chunk_path.exists():
                chunk = np.load(chunk_path)
                chunk_embeddings, chunk_lengths = chunk["embeddings"], chunk["lengths"]
            else:
                flat_texts, lengths = [], []
                for row in chunk_rows:
                    texts, row_lengths = ordered_candidates(row)
                    flat_texts.extend(texts)
                    lengths.append(row_lengths)
                chunk_embeddings = model.encode(
                    flat_texts,
                    batch_size=batch_size,
                    show_progress_bar=True,
                    convert_to_numpy=True,
                    normalize_embeddings=True,
                ).astype(np.float32).reshape(len(chunk_rows), 7, -1)
                chunk_lengths = np.stack(lengths).astype(np.float32)
                np.savez_compressed(chunk_path, embeddings=chunk_embeddings, lengths=chunk_lengths)
            embedding_chunks.append(chunk_embeddings)
            length_chunks.append(chunk_lengths)
            print(f"encoded {name}: {min(start + rows_per_chunk, len(rows))}/{len(rows)}")
        embeddings = np.concatenate(embedding_chunks)
        length_array = np.concatenate(length_chunks)
        result[name] = (embeddings, length_array)
        payload[f"{name}_embeddings"] = embeddings
        payload[f"{name}_lengths"] = length_array
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache_path, **payload)
    return result


def preference_loss(scores: torch.Tensor, kind: str) -> torch.Tensor:
    """Ranks are already best-to-worst along axis 1."""
    if kind == "pairwise_bt":
        differences = scores[:, :, None] - scores[:, None, :]
        mask = torch.triu(torch.ones(scores.shape[1], scores.shape[1], dtype=torch.bool), diagonal=1)
        return F.softplus(-differences[:, mask]).mean()
    if kind == "top1_pl":
        return (-scores[:, 0] + torch.logsumexp(scores, dim=1)).mean()
    if kind == "full_pl":
        stages = [(-scores[:, index] + torch.logsumexp(scores[:, index:], dim=1))
                  for index in range(scores.shape[1] - 1)]
        return torch.stack(stages, dim=1).mean()
    raise ValueError(f"unknown loss: {kind}")


def centered_correlation(values: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
    """Within-prompt Pearson correlation, pooled after group centering."""
    x = values - values.mean(dim=1, keepdim=True)
    y = lengths - lengths.mean(dim=1, keepdim=True)
    return (x * y).sum() / (torch.sqrt(x.square().sum() * y.square().sum()) + 1e-8)


def ndcg_at_k(scores: np.ndarray) -> np.ndarray:
    relevance = np.arange(7, 0, -1, dtype=np.float32)
    discounts = 1.0 / np.log2(np.arange(2, 9, dtype=np.float32))
    ideal = float(np.sum((2 ** relevance - 1) * discounts))
    order = np.argsort(-scores, axis=1)
    ranked_relevance = relevance[order]
    return np.sum((2 ** ranked_relevance - 1) * discounts, axis=1) / ideal


def group_metrics(scores: np.ndarray, lengths: np.ndarray) -> tuple[dict, dict[str, np.ndarray]]:
    predicted = np.argmax(scores, axis=1)
    top1 = predicted == 0
    pairwise = np.stack([scores[:, i] > scores[:, j] for i in range(7) for j in range(i + 1, 7)], axis=1)
    ties = np.stack([scores[:, i] == scores[:, j] for i in range(7) for j in range(i + 1, 7)], axis=1)
    pairwise_per_group = (pairwise + 0.5 * ties).mean(axis=1)
    ndcg = ndcg_at_k(scores)
    longest = np.argmax(lengths, axis=1)
    best_is_longest = longest == 0
    centered_scores = scores - scores.mean(axis=1, keepdims=True)
    centered_lengths = lengths - lengths.mean(axis=1, keepdims=True)
    denom = np.sqrt(np.sum(centered_scores ** 2) * np.sum(centered_lengths ** 2))
    correlation = float(np.sum(centered_scores * centered_lengths) / denom) if denom else 0.0
    length_selected = predicted == longest
    metrics = {
        "top1_accuracy": float(top1.mean()),
        "pairwise_accuracy": float(pairwise_per_group.mean()),
        "ndcg_at_7": float(ndcg.mean()),
        "reward_length_correlation": correlation,
        "longest_selected_rate": float(length_selected.mean()),
        "best_is_longest_rate": float(best_is_longest.mean()),
        "top1_when_best_longest": float(top1[best_is_longest].mean()) if best_is_longest.any() else None,
        "top1_when_best_not_longest": float(top1[~best_is_longest].mean()) if (~best_is_longest).any() else None,
    }
    per_group = {"top1": top1.astype(np.float32), "pairwise": pairwise_per_group,
                 "ndcg": ndcg, "longest_selected": length_selected.astype(np.float32)}
    return metrics, per_group


def train_head(embeddings: np.ndarray, lengths: np.ndarray, kind: str, seed: int,
               odin: bool, epochs: int, learning_rate: float, lambda_length: float,
               lambda_orthogonal: float, batch_size: int):
    torch.manual_seed(seed)
    x = torch.from_numpy(embeddings)
    length = torch.from_numpy(lengths)
    length = (length - length.mean()) / (length.std() + 1e-6)
    dimension = embeddings.shape[-1]
    quality = torch.nn.Parameter(torch.randn(dimension) * 0.02)
    parameters = [quality]
    length_head = None
    if odin:
        length_head = torch.nn.Parameter(torch.randn(dimension) * 0.02)
        parameters.append(length_head)
    optimizer = torch.optim.AdamW(parameters, lr=learning_rate, weight_decay=1e-3)
    generator = torch.Generator().manual_seed(seed)
    losses = []
    for _ in range(epochs):
        order = torch.randperm(len(x), generator=generator)
        epoch_losses = []
        for start in range(0, len(x), batch_size):
            indices = order[start : start + batch_size]
            batch_x, batch_length = x[indices], length[indices]
            q = batch_x @ quality
            if odin:
                ell = batch_x @ length_head
                total = q + ell
                corr_q = centered_correlation(q, batch_length)
                corr_l = centered_correlation(ell, batch_length)
                cosine = F.cosine_similarity(quality[None, :], length_head[None, :]).squeeze()
                regularizer = lambda_length * (corr_q.abs() - corr_l) + lambda_orthogonal * cosine.square()
            else:
                total = q
                regularizer = torch.zeros((), dtype=total.dtype)
            loss = preference_loss(total, kind) + regularizer
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_losses.append(float(loss.detach()))
        losses.append(float(np.mean(epoch_losses)))
    return {
        "quality": quality.detach().numpy(),
        "length": None if length_head is None else length_head.detach().numpy(),
        "losses": losses,
    }


def bootstrap_ci(values: np.ndarray, seed: int, repetitions: int = 2000):
    rng = np.random.default_rng(seed)
    estimates = np.empty(repetitions, dtype=np.float32)
    for index in range(repetitions):
        sample = rng.integers(0, len(values), len(values))
        estimates[index] = values[sample].mean()
    return [float(x) for x in np.quantile(estimates, [0.025, 0.975])]


def summarize_runs(runs: list[dict]) -> dict:
    summary = {}
    for key in runs[0]["metrics"]:
        values = [run["metrics"][key] for run in runs if run["metrics"][key] is not None]
        summary[key] = {"mean": float(np.mean(values)), "std": float(np.std(values, ddof=1)), "values": values}
    return summary


def render_report(result: dict) -> str:
    rows = []
    for method in result["methods"]:
        summary = method["summary"]
        rows.append(
            f"| {method['loss']} | {'ODIN-Q (λ=1)' if method['odin'] else 'single'} | "
            f"{summary['top1_accuracy']['mean']:.4f} ± {summary['top1_accuracy']['std']:.4f} | "
            f"{summary['pairwise_accuracy']['mean']:.4f} | {summary['ndcg_at_7']['mean']:.4f} | "
            f"{summary['reward_length_correlation']['mean']:+.4f} | {summary['longest_selected_rate']['mean']:.4f} |"
        )
    return "\n".join([
        "# Nectar 七候选排序与长度偏差实验", "",
        "在同一固定 MiniLM 表示、同一数据划分和同一优化器下，比较 pairwise Bradley–Terry、",
        "只监督第一名的 Top-1 Plackett–Luce，以及使用完整七名次序的 Full PL。",
        "ODIN-Q 表示训练时用质量头与长度头之和拟合排序，评测时只使用质量头。", "",
        "## 数据与设置", "",
        f"- 数据：Nectar 七候选 GPT-4 排序子集（{result['data']['train_n']} / {result['data']['validation_n']} / {result['data']['test_n']}）",
        f"- 编码器：{result['model']['name']}（冻结，{result['model']['embedding_dimension']} 维）",
        f"- 随机种子：{result['training']['seeds']}",
        f"- 运行时间：{result['runtime_seconds']:.1f} 秒（CPU）", "",
        "## 测试集结果", "",
        "| Loss | Head | Top-1 | Pairwise | NDCG@7 | Reward-length r | Longest selected |",
        "|---|---|---:|---:|---:|---:|---:|", *rows, "",
        "所有均值来自五个随机种子。`Longest selected` 不是质量指标；它只度量模型是否频繁选择组内最长回答。",
        "`Reward-length r` 先在每个 prompt 内中心化，再跨样本计算 Pearson 相关，避免 prompt 间长度差异混入。", "",
        "## 解释边界", "",
        "这是 CPU 可运行的机制对照：MiniLM 编码器被冻结，只训练线性 reward head。",
        "ODIN 部分是把论文的双头、相关性约束和正交约束推广到 K-wise PL，",
        "不是使用 Vicuna-7B 和 8×A100 的完整论文复现。Nectar 的排序标签来自 GPT-4，",
        "因此指标衡量的是拟合该排序信号的能力，不等同于真实人类偏好。",
        "编码器最多读取 256 token，而长度指标使用完整回答字符数；截断可能改变长回答中的可见信号。",
        "λ=1 是论文强度的直接迁移，不是本模型上的最优值；验证集消融见 `odin_sweep.md`。",
    ])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=Path("data/nectar/nectar_7way.jsonl"))
    parser.add_argument("--out", type=Path, default=Path("reports/nectar_listwise"))
    parser.add_argument("--cache", type=Path, default=Path("data/embeddings/minilm_nectar_7way.npz"))
    parser.add_argument("--model", default="sentence-transformers/all-MiniLM-L6-v2")
    parser.add_argument("--train-size", type=int, default=3000)
    parser.add_argument("--val-size", type=int, default=1000)
    parser.add_argument("--test-size", type=int, default=1000)
    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=35)
    parser.add_argument("--learning-rate", type=float, default=0.03)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--encode-batch-size", type=int, default=64)
    parser.add_argument("--lambda-length", type=float, default=1.0)
    parser.add_argument("--lambda-orthogonal", type=float, default=1.0)
    parser.add_argument("--local-files-only", action="store_true")
    args = parser.parse_args()
    start = time.perf_counter()
    rows = load_rows(args.data)
    splits = split_rows(rows, args.train_size, args.val_size, args.test_size, args.split_seed)
    model = SentenceTransformer(args.model, device="cpu", local_files_only=args.local_files_only)
    model.max_seq_length = 256
    arrays = load_or_encode(model, splits, args.cache, args.encode_batch_size)
    train_x, train_lengths = arrays["train"]
    test_x, test_lengths = arrays["test"]

    methods = []
    for kind in LOSSES:
        for odin in (False, True):
            runs = []
            for seed in SEEDS:
                trained = train_head(train_x, train_lengths, kind, seed, odin, args.epochs,
                                     args.learning_rate, args.lambda_length,
                                     args.lambda_orthogonal, args.batch_size)
                quality_scores = test_x @ trained["quality"]
                metrics, per_group = group_metrics(quality_scores, test_lengths)
                run = {"seed": seed, "metrics": metrics,
                       "top1_ci_95": bootstrap_ci(per_group["top1"], seed),
                       "final_training_loss": trained["losses"][-1]}
                if odin:
                    total_scores = quality_scores + test_x @ trained["length"]
                    run["training_total_reward_metrics"] = group_metrics(total_scores, test_lengths)[0]
                    run["head_cosine"] = float(np.dot(trained["quality"], trained["length"]) /
                                               (np.linalg.norm(trained["quality"]) * np.linalg.norm(trained["length"]) + 1e-12))
                runs.append(run)
            methods.append({"loss": kind, "odin": odin, "runs": runs, "summary": summarize_runs(runs)})
            print(kind, "ODIN" if odin else "single", methods[-1]["summary"]["top1_accuracy"])

    result = {
        "data": {"dataset": "berkeley-nest/Nectar", "candidates_per_prompt": 7,
                 "label_source": "GPT-4 ranking", "train_n": len(splits["train"]),
                 "validation_n": len(splits["validation"]), "test_n": len(splits["test"])},
        "model": {"name": args.model, "embedding_dimension": model.get_sentence_embedding_dimension(),
                  "encoder_frozen": True, "max_sequence_length": model.max_seq_length},
        "training": {"losses": list(LOSSES), "seeds": list(SEEDS), "epochs": args.epochs,
                     "learning_rate": args.learning_rate, "lambda_length": args.lambda_length,
                     "lambda_orthogonal": args.lambda_orthogonal},
        "methods": methods,
        "runtime_seconds": time.perf_counter() - start,
    }
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "results.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    (args.out / "report.md").write_text(render_report(result), encoding="utf-8")
    print(f"wrote {args.out / 'results.json'} and {args.out / 'report.md'}")


if __name__ == "__main__":
    main()
