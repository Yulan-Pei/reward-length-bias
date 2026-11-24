# Reward Length Bias：真实偏好数据上的奖励模型诊断

这个项目研究一个后训练中的实际问题：**奖励模型会不会把“回答更长”误当成“回答更好”？**
实验覆盖 HH-RLHF 二元偏好和 Nectar 七候选排序，并比较 Bradley–Terry、Top-1
Plackett–Luce、Full Plackett–Luce，以及推广到 K-wise 排序的 ODIN 双头去偏方法。

我们在 Anthropic HH-RLHF `helpful-base` 上比较 TF-IDF 线性奖励模型与冻结的 MiniLM 编码器，并显式把长度系数设为负、零、正值。主要结论不是“已经解决偏差”，而是：

- TF-IDF 内容模型测试准确率为 **60.32%**（bootstrap 95% CI `[58.24%, 62.26%]`）。验证集选择 `alpha=+0.25` 后增加 1.10 个百分点，但配对区间 `[-0.59, 2.68]` 个百分点包含 0，不能宣称收益显著。
- 冻结 MiniLM + 线性奖励头达到 **62.23%**；相对 TF-IDF 高 1.91 个百分点，但配对区间 `[-0.34, 4.23]` 个百分点、McNemar `p=0.079`，同样没有越过常用 0.05 显著性阈值。
- 不显式输入长度并不等于没有长度偏差：MiniLM 在 chosen 更长/更短切片上的准确率为 82.52%/31.98%。加入正长度权重会进一步放大差距；负权重会把偏差翻向另一侧，并损害总准确率。
- 平衡重采样降低了学到的长度系数，但没有消除切片差距。因此本项目的结论是“量化、诊断并比较缓解手段”，不是“解决了长度偏差”。
- Nectar 七候选上，BT 与 Full PL 的 NDCG@7 分别为 **0.7984/0.7975**，长度相关分别为 **0.4617/0.4439**。Full PL 只带来轻微的长度相关下降，不能据此声称 PL 自动解决长度偏差。
- ODIN 的 λ=1 直接迁移把质量头长度相关压到约 0.02–0.03，但 NDCG 降到约 0.743；验证集选择较小 λ 后，三种目标的相关下降约 14%–16%，NDCG 只下降 0.001–0.003。去偏存在明确的质量权衡，并对正则强度敏感。

实验使用 8,000/2,000/2,354 个训练、验证、测试偏好对，5 个随机种子，并输出 bootstrap 置信区间、配对检验和可复现报告。

> 边界：MiniLM 编码器是**冻结的**，只训练 384 个参数的线性奖励头。这是真实 Transformer 表征对照，但不是端到端微调，也不是 LoRA。CPU 预算下这样设计是为了保证可复现；后续可在 GPU 上增加 LoRA 作为下一阶段。

## 一键复现

基础 TF-IDF 实验只需 NumPy：

```bash
python -m pip install -r requirements.txt
python run_pipeline.py
```

Transformer 对照需要 Python 3.11、PyTorch 和 Sentence Transformers：

```bash
python -m pip install -r requirements-transformer.txt
python run_transformer_pipeline.py
python compare_models.py
```

Nectar 七候选实验会通过 Hugging Face 官方 rows API 抽取固定 5,000 组样本，
编码 35,000 个回答并缓存嵌入；CPU 首次运行约 20 分钟：

```bash
python run_nectar_pipeline.py
```

在 Windows PowerShell 中，若 Transformers 误加载 TensorFlow，可显式关闭 TensorFlow 后端：

```powershell
$env:USE_TF='0'
python run_transformer_pipeline.py
python compare_models.py
```

首次运行会下载 HH-RLHF 和 `sentence-transformers/all-MiniLM-L6-v2`；之后复用 `data/` 中的本地数据与嵌入缓存。`data/` 不提交到 Git。

## 结果与代码入口

| 内容 | 文件 |
|---|---|
| TF-IDF、MLP、长度扫描与重采样报告 | [`reports/hh_rlhf/report.md`](reports/hh_rlhf/report.md) |
| MiniLM 奖励头报告 | [`reports/transformer/report.md`](reports/transformer/report.md) |
| 两个模型的配对显著性比较 | [`reports/model_comparison/report.md`](reports/model_comparison/report.md) |
| Nectar BT / Top-1 PL / Full PL 主结果 | [`reports/nectar_listwise/report.md`](reports/nectar_listwise/report.md) |
| ODIN × PL 验证集 λ 消融 | [`reports/nectar_listwise/odin_sweep.md`](reports/nectar_listwise/odin_sweep.md) |
| 基础实验主程序 | [`real_data_experiment.py`](real_data_experiment.py) |
| Transformer 主程序 | [`transformer_experiment.py`](transformer_experiment.py) |
| Nectar 数据准备与哈希 | [`prepare_nectar.py`](prepare_nectar.py) |
| K-wise 损失与 ODIN 双头 | [`nectar_listwise_experiment.py`](nectar_listwise_experiment.py) |

核心评分形式为 `margin = r(chosen) - r(rejected)`，训练目标是 Bradley-Terry pairwise loss：

```text
L = log(1 + exp(-margin))
```

`margin > 0` 表示模型把人类偏好的回答排在前面。长度干预使用：

```text
adjusted_margin = content_margin + alpha * normalized_length_difference
```

`alpha < 0` 惩罚更长回答，`alpha = 0` 是内容模型，`alpha > 0` 奖励更长回答。

七候选实验中，候选已按最好到最差排列。Top-1 PL 只监督第一名：

```text
L_top1 = -s_1 + log sum_j exp(s_j)
```

Full PL 对余下列表逐项建模，BT 则把七名次序展开为 21 个二元约束。ODIN 训练时使用
`r_Q + r_L` 拟合上述排序目标，同时让质量头 `r_Q` 与长度低相关、两个头近似正交；
最终只用 `r_Q` 评分。

## 数据与可复现性

二元数据来自论文 [Training a Helpful and Harmless Assistant with Reinforcement Learning from Human Feedback](https://arxiv.org/abs/2204.05862) 发布的 [Anthropic HH-RLHF](https://huggingface.co/datasets/Anthropic/hh-rlhf)。七候选数据来自 [Nectar](https://huggingface.co/datasets/berkeley-nest/Nectar)，K-wise MLE 对照参考 [Starling](https://proceedings.colm.cc/paper_files/paper/2024/hash/24a2b1e2d4c3b20f09eb006d3d610777-Abstract-Conference.html)，双头解耦参考 [ODIN](https://proceedings.mlr.press/v235/chen24bn.html) 及其[官方代码](https://github.com/lichang-chen/odin)。流水线固定数据划分；验证集只用于选择超参数，测试集只做最终报告。

```bash
python -m pytest
python -B run.py
```

CI 会运行 9 个单元测试和无需网络的合成实验。真实数据与 Transformer 实验由于下载量和 CPU 时间较大，不放入每次 CI，但结果 JSON、报告、依赖版本和运行脚本全部提交。

## 仓库结构

```text
prepare_hh_rlhf.py          下载并校验真实数据
real_data_experiment.py     TF-IDF/MLP/长度干预/统计评测
transformer_experiment.py   冻结 MiniLM + pairwise 奖励头
compare_models.py           配对 bootstrap 与 McNemar 检验
run_pipeline.py             基础流水线入口
run_transformer_pipeline.py Transformer 流水线入口
prepare_nectar.py           固定抽取并校验 Nectar 七候选数据
nectar_listwise_experiment.py BT/Top-1 PL/Full PL 与 ODIN-Q
odin_pl_sweep.py            验证集正则扫描与测试集复核
run_nectar_pipeline.py      Nectar 一键流水线
tests/                      单元测试
reports/                    可复现结果与图表
run.py                      早期 micrograd 合成机制实验
```

## 代码来源与许可

本项目实验代码使用 MIT License。`micrograd/` 是 Andrej Karpathy 的 micrograd 快照（版本 `7bc720e`），其原始许可单独保存在 [`MICROGRAD-LICENSE.txt`](MICROGRAD-LICENSE.txt)。

