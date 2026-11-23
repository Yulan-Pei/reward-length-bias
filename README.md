# Reward Length Bias：真实偏好数据上的奖励模型诊断

这个项目研究一个后训练中的实际问题：**奖励模型会不会把“回答更长”误当成“回答更好”？**

我们在 Anthropic HH-RLHF `helpful-base` 上比较 TF-IDF 线性奖励模型与冻结的 MiniLM 编码器，并显式把长度系数设为负、零、正值。主要结论不是“已经解决偏差”，而是：

- TF-IDF 内容模型测试准确率为 **60.32%**（bootstrap 95% CI `[58.24%, 62.26%]`）。验证集选择 `alpha=+0.25` 后增加 1.10 个百分点，但配对区间 `[-0.59, 2.68]` 个百分点包含 0，不能宣称收益显著。
- 冻结 MiniLM + 线性奖励头达到 **62.23%**；相对 TF-IDF 高 1.91 个百分点，但配对区间 `[-0.34, 4.23]` 个百分点、McNemar `p=0.079`，同样没有越过常用 0.05 显著性阈值。
- 不显式输入长度并不等于没有长度偏差：MiniLM 在 chosen 更长/更短切片上的准确率为 82.52%/31.98%。加入正长度权重会进一步放大差距；负权重会把偏差翻向另一侧，并损害总准确率。
- 平衡重采样降低了学到的长度系数，但没有消除切片差距。因此本项目的结论是“量化、诊断并比较缓解手段”，不是“解决了长度偏差”。

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

在本机 Anaconda/Windows 上若 Transformers 误加载 TensorFlow，可先在 PowerShell 执行：

```powershell
$env:USE_TF='0'
& 'D:/anaconda_test/python.exe' run_transformer_pipeline.py
& 'D:/anaconda_test/python.exe' compare_models.py
```

首次运行会下载 HH-RLHF 和 `sentence-transformers/all-MiniLM-L6-v2`；之后复用 `data/` 中的本地数据与嵌入缓存。`data/` 不提交到 Git。

## 结果与代码入口

| 内容 | 文件 |
|---|---|
| TF-IDF、MLP、长度扫描与重采样报告 | [`reports/hh_rlhf/report.md`](reports/hh_rlhf/report.md) |
| MiniLM 奖励头报告 | [`reports/transformer/report.md`](reports/transformer/report.md) |
| 两个模型的配对显著性比较 | [`reports/model_comparison/report.md`](reports/model_comparison/report.md) |
| 基础实验主程序 | [`real_data_experiment.py`](real_data_experiment.py) |
| Transformer 主程序 | [`transformer_experiment.py`](transformer_experiment.py) |
| 完整的新手教程与面试问答 | [`docs/BEGINNER_GUIDE.md`](docs/BEGINNER_GUIDE.md) |

核心评分形式为 `margin = r(chosen) - r(rejected)`，训练目标是 Bradley-Terry pairwise loss：

```text
L = log(1 + exp(-margin))
```

`margin > 0` 表示模型把人类偏好的回答排在前面。长度干预使用：

```text
adjusted_margin = content_margin + alpha * normalized_length_difference
```

`alpha < 0` 惩罚更长回答，`alpha = 0` 是内容模型，`alpha > 0` 奖励更长回答。

## 数据与可复现性

数据来自论文 [Training a Helpful and Harmless Assistant with Reinforcement Learning from Human Feedback](https://arxiv.org/abs/2204.05862) 发布的 [Anthropic HH-RLHF](https://huggingface.co/datasets/Anthropic/hh-rlhf)。流水线校验源文件 SHA-256，固定数据划分；验证集只用于选择超参数，测试集只做最终报告。

```bash
pytest
python -B run.py
```

CI 会运行 5 个单元测试和无需网络的合成实验。真实数据与 Transformer 实验由于下载量和 CPU 时间较大，不放入每次 CI，但结果 JSON、报告、依赖版本和运行脚本全部提交。

## 仓库结构

```text
prepare_hh_rlhf.py          下载并校验真实数据
real_data_experiment.py     TF-IDF/MLP/长度干预/统计评测
transformer_experiment.py   冻结 MiniLM + pairwise 奖励头
compare_models.py           配对 bootstrap 与 McNemar 检验
run_pipeline.py             基础流水线入口
run_transformer_pipeline.py Transformer 流水线入口
tests/                      单元测试
reports/                    可复现结果与图表
docs/BEGINNER_GUIDE.md      从零解释代码、结果和面试回答
run.py                      早期 micrograd 合成机制实验
```

## 代码来源与许可

本项目实验代码使用 MIT License。`micrograd/` 是 Andrej Karpathy 的 micrograd 快照（版本 `7bc720e`），其原始许可单独保存在 [`MICROGRAD-LICENSE.txt`](MICROGRAD-LICENSE.txt)。
