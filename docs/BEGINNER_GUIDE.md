# 从零读懂 Reward Length Bias 项目

这份文档假设你刚读研、会一点 Python，但没有做过奖励模型。目标是让你能自己跑出结果、顺着代码解释实验，并在面试中诚实而有条理地回答追问。

## 1. 项目在做什么

聊天模型对同一个问题生成两个回答。人类更喜欢的记作 `chosen`，另一个记作 `rejected`。奖励模型（Reward Model，RM）给回答一个标量分数，我们希望：

```text
r(chosen) > r(rejected)
```

奖励模型可以给 RLHF、候选重排或数据筛选提供反馈。但是偏好数据中的 chosen 经常更长，模型可能没有学到“正确、有帮助”，只学到“更长通常更好”。这就是长度偏差或长度捷径。

本项目回答五个问题：

1. HH-RLHF 真实数据里是否存在长度相关性？
2. 不把长度作为显式特征，模型是否仍有长度方向差距？
3. 把长度权重设为负、零、正值分别会怎样？
4. 平衡采样能否缓解偏差？
5. Transformer 表征是否显著优于 TF-IDF？

## 2. 为什么它属于后训练

预训练让模型学习语言，后训练让模型更符合人的指令和偏好。RLHF 的简化流程是：

```text
偏好数据 → 训练奖励模型 → 奖励模型评价策略输出 → 优化策略模型
```

本项目集中在奖励模型和 Evaluation。它没有训练聊天大模型，也没有执行 PPO/DPO。面试时应说“奖励模型诊断与去偏实验”，不要说“完成了完整 RLHF 系统”。

## 3. Bradley-Terry 损失

先定义分差：

```text
margin = r(chosen) - r(rejected)
L = log(1 + exp(-margin))
```

`margin` 为正表示排对，为负表示排错。margin 越大，损失越小。模型不必预测绝对质量，只需把同题的回答排序，这叫 pairwise ranking。

在 `real_data_experiment.py` 的 `train_pairwise` 中：`features @ weights` 得到 margin，随后求梯度并更新权重，每轮记录平均损失。

## 4. 三类模型学什么

### TF-IDF 线性模型

TF-IDF 把回答变成词向量，线性模型学习词方向的权重。它快速、可解释，是可靠 baseline；但不理解词序和深层语义，不是大语言模型。

### NumPy MLP

MLP 在 TF-IDF 后加入隐藏层与 ReLU。当前结果 56.88%，低于线性基线，可能因为欠调参、优化不足或输入表征限制。面试时可以说：“我保留失败对照，因为复杂模型并不自动带来提升，实验必须验证而不能想当然。”

### Frozen MiniLM + 奖励头

MiniLM 把“最后一轮用户问题 + 回答”编码为 384 维向量。编码器 2271 万参数全部冻结，只训练 384 参数的线性奖励头。

CPU 下冻结编码器能在数分钟完成并复用嵌入缓存，但不能让语言表征针对偏好任务更新。因此可以说“用真实 Transformer 表征训练 pairwise reward head”，不可以说“微调了 MiniLM”或“做了 LoRA”。LoRA 是下一阶段的 GPU 扩展。

## 5. 长度权重 alpha

```text
adjusted_margin = content_margin + alpha * normalized_length_difference
```

`length_difference = len(chosen)-len(rejected)`，再除以训练集标准差。

- `alpha > 0`：奖励更长回答；
- `alpha = 0`：不显式使用长度；
- `alpha < 0`：惩罚更长回答。

负 alpha 不等于去偏。设得太负会制造“越短越好”的反向捷径。因此项目从 -2 扫到 +2，并同时观察总准确率和两个长度方向切片。

## 6. 为什么分训练、验证、测试集

- 训练集学习模型权重；
- 验证集选择 alpha 等超参数；
- 测试集只在选择完成后报告。

项目固定使用 8,000/2,000/2,354 对。若看测试结果挑 alpha，就发生信息泄漏，最终指标会过于乐观。

## 7. 怎么看结果

先看 Markdown，再看 JSON：

1. `reports/hh_rlhf/report.md`：TF-IDF、MLP、长度扫描、长度差分桶、重采样；
2. `reports/transformer/report.md`：MiniLM 主实验；
3. `reports/model_comparison/report.md`：两个模型的配对显著性比较。

JSON 用于程序读取，能证明报告由流水线自动生成，而不是手抄。

### 核心指标

- `accuracy`：`r(chosen)>r(rejected)` 的比例，平局计 0.5。
- `chosen_long_accuracy`：chosen 更长的切片准确率。
- `rejected_long_accuracy`：chosen 更短的切片准确率。
- `length_slice_gap`：上述两个准确率之差的绝对值。
- `longer_selected_rate`：模型多常选择更长回答。它是行为诊断，不是质量指标。

gap 不能脱离总准确率单独优化：随机猜测器可能 gap 很小，但毫无价值。

### 当前结论

TF-IDF 五个 seed 为 `60.34% ± 0.12%`，ensemble 为 60.32%，bootstrap 95% CI `[58.24%, 62.26%]`。验证集选 `alpha=+0.25` 后点估计提高 1.10 个百分点，但配对 CI `[-0.59, 2.68]` 包含 0。正确说法是“有上升趋势，证据不足以排除随机波动”。

MiniLM 为 62.23%。相对 TF-IDF 高 1.91 个百分点，但配对 CI `[-0.34, 4.23]` 包含 0，McNemar `p=0.079`，仍不能在常用 5% 水平宣称显著胜出。

MiniLM 在 chosen 更长切片为 82.52%，chosen 更短切片仅 31.98%。这说明 alpha=0 不代表没有长度偏差：文本风格、详尽程度等代理特征仍可能间接编码长度。

### 长度差分桶

TF-IDF 报告按 `|len(chosen)-len(rejected)|` 分为 0–50、50–150、150–300、300+ 字符。随着长度差增大，chosen 更长比例与准确率同时上升。这是相关证据，不是严格因果证明；固定 alpha 干预和重采样提供了更强的对照。

## 8. seed、置信区间和显著性

多随机种子用于观察训练随机性；均值旁边的标准差描述不同训练运行之间的波动。

Bootstrap 从测试集有放回抽样 3,000 次，形成准确率的 95% 区间。配对 bootstrap 对两个模型的同一批题一起重采样，适合判断准确率差是否稳定。

McNemar 检验只关注两个模型意见不同的样本。当前只有 TF-IDF 正确 334 对，只有 MiniLM 正确 382 对，双侧精确 `p=0.079`。`p>0.05` 不等于模型完全一样，只表示现有证据不足以拒绝“表现相同”的假设。

## 9. 推荐的代码阅读顺序

1. `response_only`：截取最后一个 Assistant 回答；
2. `build_vocabulary`、`vectorize`：文本变成 TF-IDF；
3. `make_arrays`：形成 chosen/rejected 特征差与长度差；
4. `train_pairwise`：用 BT loss 训练线性头；
5. `metrics`：计算总体和切片指标；
6. `main`：串起数据、训练、选参和报告；
7. `transformer_experiment.py` 的 `prompt_and_response`、`encode_rows`、`load_or_encode`；
8. `compare_models.py`：配对 bootstrap 与 McNemar。

每读一个函数都问：输入是什么、输出是什么、数组形状是什么、有没有偷看测试标签。

## 10. 在你的电脑运行

基础实验：

```powershell
& 'D:/anaconda_test/python.exe' -m pip install -r requirements.txt
& 'D:/anaconda_test/python.exe' run_pipeline.py
```

Transformer 实验：

```powershell
$env:USE_TF='0'
& 'D:/anaconda_test/python.exe' -c "import torch; print(torch.__version__, torch.cuda.is_available())"
& 'D:/anaconda_test/python.exe' run_transformer_pipeline.py
& 'D:/anaconda_test/python.exe' compare_models.py
```

已有模型缓存且希望离线运行：

```powershell
$env:HF_HUB_OFFLINE='1'
& 'D:/anaconda_test/python.exe' transformer_experiment.py --local-files-only
```

测试：

```powershell
& 'D:/anaconda_test/python.exe' -m pytest
& 'D:/anaconda_test/python.exe' -B run.py
```

## 11. 90 秒面试介绍

> 我研究的是 RLHF 奖励模型中的长度捷径。使用 Anthropic HH-RLHF helpful-base 的 8,000/2,000/2,354 个训练、验证、测试偏好对，实现 TF-IDF 线性、MLP 和冻结 MiniLM 的 Bradley-Terry 奖励头。我显式扫描负、零、正长度权重，并比较长度平衡重采样。评测不只看总体准确率，还包括长度方向切片、长度差分桶、5 个随机种子、bootstrap CI 和 McNemar 检验。MiniLM 点估计从 60.32% 提到 62.23%，但配对区间包含 0、p=0.079，因此没有宣称显著提升。更关键的是，alpha=0 时 MiniLM 两个长度切片仍有约 50.5 个百分点差距，说明去掉显式长度也不能消除代理特征。平衡采样降低了长度系数，但没消除偏差。项目边界是冻结编码器，不是 LoRA；下一步会在 GPU 上做可训练 encoder 与长度匹配的反事实数据。

## 12. 高频面试问题

### 为什么不选准确率最高的 alpha 就结束？

必须由验证集选参、测试集确认泛化；小幅提升也可能是样本噪声。项目用配对 bootstrap 检查差值，区间包含 0，所以只报告趋势。

### alpha=0 为什么仍有长度偏差？

删除显式长度数值不会删除文本中的代理。更长回答常包含更多解释、礼貌词和结构，TF-IDF/Transformer 可以利用这些相关特征。

### 负 alpha 是不是去偏？

不是，它可能把“越长越好”变成“越短越好”。目标应是降低无关长度依赖同时保留质量信号，因此要看总体准确率和双向切片的权衡。

### 为什么 MLP 比线性模型差？

可能是优化超参数不足、隐藏层设置不合适或 TF-IDF 表征限制。它说明复杂度不保证收益。当前将其作为负结果，不用来支撑主要结论。

### 这是真实 Reward Model 吗？

数据和 pairwise 目标是真实 RM 形式，MiniLM 也是真实 Transformer；但 encoder 冻结，只训练线性 head，所以是轻量 RM baseline，不是大模型全参数 RM。

### 重采样为什么降低 alpha 却仍有大 gap？

它只平衡 chosen 更长/更短的数量，没有消除长度与写作风格、信息量、题目难度之间的混杂关系。

### 你最重要的工程工作是什么？

固定划分与 seed、校验 SHA-256、防测试泄漏、缓存 Transformer 嵌入、自动生成 JSON/Markdown、单元测试与 CI。这些让结论可复现、可审计。

### 下一步做什么？

有 GPU 后增加 LoRA RM；构造长度匹配的反事实偏好对；按问题类型与质量维度分层；扩大测试集；分别评测“长度不变质量变化”和“质量不变长度变化”。

## 13. 简历表述

可以写：

> 基于 Anthropic HH-RLHF 构建 CPU 可复现的奖励模型长度偏差评测流水线，实现 TF-IDF/MLP 与 Frozen MiniLM pairwise reward head，覆盖负/零/正长度干预及平衡重采样；在 2,354 个测试偏好对上以 5 seeds、bootstrap CI 与 McNemar 检验验证模型差异，发现 MiniLM 点估计提升 1.91pt 但未达 5% 显著水平，并量化 alpha=0 时约 50.5pt 的长度方向切片差距。

不要写“解决了长度偏差”“微调了大模型”“显著提升 1.91pt”或“完成完整 RLHF 上线”。

## 14. 项目能证明什么

它能证明你理解偏好学习、会设计对照、会做统计评测、能构建可复现流水线，也会诚实报告负结果。它还不能证明你做过分布式大模型训练、LoRA/全参 RM 或线上灰度。主动说明边界，再提出下一步方案，会比夸大更可信。
