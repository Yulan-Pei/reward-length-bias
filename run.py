"""CPU pairwise reward modelling on synthetic, evidence-grounded order QA."""
import json
import math
import random
import platform
import time
from pathlib import Path

from micrograd.engine import Value


STATES = ["已签收", "已发货", "已取消", "待发货"]
ROOT = Path(__file__).resolve().parent
LONG = "根据提供的两条订单记录，所查询订单目前的状态为{}。"
SHORT = "{}。"


def make_pairs(split, count, long_fraction, seed):
    rng = random.Random(seed)
    lengths = [i < round(count * long_fraction) for i in range(count)]
    rng.shuffle(lengths)
    rows = []
    for i, good_long in enumerate(lengths):
        target, distractor, unseen = rng.sample(STATES, 3)
        key = f"{split}-{2*i}"
        facts = [(key, target), (f"{split}-{2*i+1}", distractor)]
        rng.shuffle(facts)
        context = "；".join(f"{k}：{s}" for k, s in facts)
        swapped = i % 4 == 0
        wrong = distractor if swapped else unseen
        good, bad = (LONG, SHORT) if good_long else (SHORT, LONG)
        rows.append(dict(prompt=f"{context}。请回答订单 {key} 的状态。", chosen=good.format(target),
                         rejected=bad.format(wrong), chosen_long=good_long,
                         error="entity_swap" if swapped else "unsupported"))
    return rows


def features(prompt, answer):
    # Evidence presence anywhere in the prompt does not establish entity binding.
    supported = any(state in answer and state in prompt for state in STATES)
    return [len(answer) / 20, float(supported)]


def difference(row):
    good = features(row["prompt"], row["chosen"])
    bad = features(row["prompt"], row["rejected"])
    return [a - b for a, b in zip(good, bad)]


def pair_loss(margin):
    # Stable softplus(-margin) with a scalar micrograd backward rule.
    x = margin.data
    out = Value(max(-x, 0) + math.log1p(math.exp(-abs(x))), (margin,), "pair_loss")
    def backward():
        exp_term = math.exp(-abs(x))
        slope = -exp_term / (1 + exp_term) if x >= 0 else -1 / (1 + exp_term)
        margin.grad += slope * out.grad
    out._backward = backward
    return out


def train(rows):
    weights = [Value(0.0), Value(0.0)]
    inputs = [difference(row) for row in rows]
    order = random.Random(42)
    for _ in range(6):
        order.shuffle(inputs)
        for values in inputs:
            for w in weights:
                w.grad = 0
            margin = sum(w * x for w, x in zip(weights, values))
            pair_loss(margin).backward()
            for w in weights:
                w.data -= 0.03 * w.grad
    return [w.data for w in weights]


def evaluate(rows, weights):
    groups = {}
    for row in rows:
        margin = sum(w * x for w, x in zip(weights, difference(row)))
        vote = 0.5 if abs(margin) < 1e-10 else float(margin > 0)
        for group in ("all", row["error"], "chosen_long" if row["chosen_long"] else "rejected_long"):
            groups.setdefault(group, []).append(vote)
    return {k: dict(n=len(v), accuracy=round(sum(v)/len(v), 4)) for k, v in groups.items()}


def main():
    start = time.perf_counter()
    biased = make_pairs("train", 400, 0.95, 7)
    balanced = make_pairs("train", 400, 0.50, 7)
    test = make_pairs("test", 160, 0.50, 13)
    assert [r["prompt"] for r in biased] == [r["prompt"] for r in balanced]
    assert not {r["prompt"] for r in biased}.intersection(r["prompt"] for r in test)
    # Compare the custom gradient to a finite difference at both signs and zero.
    for x in (-2.5, 0, 2.5):
        v = Value(x)
        pair_loss(v).backward()
        numeric = (pair_loss(Value(x+1e-5)).data - pair_loss(Value(x-1e-5)).data) / 2e-5
        assert abs(v.grad - numeric) < 1e-8
    models = {"length_only": [1, 0], "biased": train(biased), "balanced": train(balanced)}
    results = {k: dict(weights=w, slices=evaluate(test, w)) for k, w in models.items()}
    for name, rows in (("biased", biased), ("balanced", balanced), ("test", test)):
        (ROOT / (name+".jsonl")).write_text("".join(json.dumps(r, ensure_ascii=False)+"\n" for r in rows), encoding="utf8")
    report = dict(python=platform.python_version(), platform=platform.platform(), seconds=round(time.perf_counter()-start, 3), models=results)
    (ROOT / "results.json").write_text(json.dumps(report, indent=2), encoding="utf8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

