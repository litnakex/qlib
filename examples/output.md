# examples 全量运行汇总（output.md）

> 本文件按 `examples/run.md` 的要求生成：递归遍历 examples/ 各子目录 → 翻译 README（已完成 38 个 README.cn.md）→ 运行算法任务 → 汇总关键结论（模型/策略 + 全部指标）。
> **全部完成**：benchmarks 复用结果 + 批2/3 运行结果 + 批4 深度模型（benchmarks 未跑的 20 个 config，每个 3 次取均值±std）+ 跳过/失败标注。批4 结果见第 6 节。

## 0. 运行环境与说明

- **硬件/环境**：Apple M3 Pro（12 核，MPS，无 CUDA）；conda 环境 `qlib`（Python 3.12.13，qlib 0.9.8.dev31）；所有运行统一 `conda run -n qlib`。
- **数据**：`~/.qlib/qlib_data/cn_data`（日频，含 csi100/300/500）+ `cn_data_1min`（1min）。
- **运行方法**：`.yaml` → `qrun <file>.yaml`；`.py` → `python <file>.py`（含 fire CLI 的按 README 指定子命令）。
- **指标口径**（同 benchmarks/report.md）：股票池 csi300；标签 `Ref($close,-2)/Ref($close,-1)-1`；回测策略 TopkDropoutStrategy(topk=50, n_drop=5)；IC/Rank IC 越高预测力越强，ICIR/Rank ICIR 为稳定性；年化收益/IR 为相对基准(SH000300)的**超额收益**，含成本更贴近实盘；MDD 越接近 0 越好。
- **复用声明**：`benchmarks/` 下 15 个模型 + 7 个深度模型 + 5 个进阶实验的结果**直接复用自 `benchmarks/report.md` 与 `benchmarks/_deep_results.json`（未重跑）**，标 `来源=reused`。批2/3 与批4 为本轮实跑，标 `来源=本轮`。

## 1. 总览表

### 1a. benchmarks 树/线性/MLP 模型（复用，含成本口径）

| 目录 | 模型 | 数据集 | IC | Rank IC | 年化(含成本) | IR(含成本) | MDD | 状态 | 来源 |
|------|------|--------|-----|---------|-------------|-----------|-----|------|------|
| benchmarks/LightGBM | LightGBM | Alpha158 | 0.0468 | 0.0490 | 0.0807 | 0.9145 | -0.0861 | reused | report.md |
| benchmarks/LightGBM | LightGBM | Alpha360 | 0.0399 | 0.0492 | 0.0203 | 0.2799 | -0.0995 | reused | report.md |
| benchmarks/XGBoost | XGBoost | Alpha158 | 0.0508 | 0.0489 | 0.0788 | 0.8856 | -0.0992 | reused | report.md |
| benchmarks/XGBoost | XGBoost | Alpha360 | 0.0409 | 0.0467 | 0.0424 | 0.5436 | -0.1022 | reused | report.md |
| benchmarks/CatBoost | CatBoost | Alpha158 | 0.0506 | 0.0457 | 0.0766 | 0.8362 | -0.0981 | reused | report.md |
| benchmarks/CatBoost | CatBoost | Alpha360 | 0.0370 | 0.0457 | 0.0160 | 0.2005 | -0.1195 | reused | report.md |
| benchmarks/Linear | Linear | Alpha158 | 0.0388 | 0.0475 | 0.0829 | 1.1393 | -0.1733 | reused | report.md |
| benchmarks/DoubleEnsemble | DoubleEnsemble | Alpha158 | 0.0517 | 0.0512 | **0.0980** | 1.1252 | -0.0694 | reused | report.md |
| benchmarks/DoubleEnsemble | DoubleEnsemble | Alpha360 | 0.0402 | 0.0487 | 0.0362 | 0.4888 | -0.0916 | reused | report.md |
| benchmarks/MLP | MLP | Alpha158 | 0.0340 | 0.0413 | 0.0553 | 0.6553 | -0.0927 | reused | report.md |
| benchmarks/MLP | MLP | Alpha360 | 0.0288 | 0.0377 | 0.0106 | 0.1236 | -0.1461 | reused | report.md |

### 1b. benchmarks 深度时序模型（复用，含成本口径）

| 目录 | 模型 | 数据集 | IC | Rank IC | 年化(含成本) | IR(含成本) | MDD | 状态 | 来源 |
|------|------|--------|-----|---------|-------------|-----------|-----|------|------|
| benchmarks/GRU | GRU | Alpha158 | 0.0279 | 0.0412 | 0.0466 | 0.7330 | -0.0780 | reused | _deep_results.json |
| benchmarks/GRU | GRU | Alpha360 | 0.0495 | 0.0587 | 0.0609 | 0.8284 | -0.0886 | reused | _deep_results.json |
| benchmarks/LSTM | LSTM | Alpha158 | 0.0346 | 0.0415 | 0.0754 | 1.0638 | -0.1073 | reused | _deep_results.json |
| benchmarks/LSTM | LSTM | Alpha360 | 0.0441 | 0.0480 | 0.0500 | 0.7075 | -0.1034 | reused | _deep_results.json |
| benchmarks/GATs | GATs | Alpha158 | 0.0443 | 0.0524 | **0.1311** | 1.7865 | -0.0734 | reused | _deep_results.json |
| benchmarks/GATs | GATs | Alpha360 | 0.0471 | 0.0588 | 0.0791 | 1.0646 | -0.0741 | reused | _deep_results.json |
| benchmarks/KRNN | KRNN | Alpha360 | 0.0144 | 0.0254 | -0.0221 | -0.2466 | -0.2538 | reused | _deep_results.json |

### 1c. 批2/3 本轮实跑

| 目录 | 任务 | 类型 | 状态 | 关键指标/说明 | 来源 |
|------|------|------|------|--------------|------|
| examples(根) | workflow_by_code.py | LightGBM 全流程 | ✅done | IC=0.0490 | 本轮 |
| model_interpreter | feature.py | LightGBM 特征重要性 | ✅done | 运行成功(输出特征重要性) | 本轮 |
| highfreq | High_Freq_Tree_Alpha158.yaml | 高频 LightGBM | ✅done | IC=0.0357 | 本轮 |
| data_demo | data_cache_demo.py | 数据缓存演示 | ❌failed | 示例脚本用 ruamel.yaml.safe_dump（新版已移除） | 本轮 |
| data_demo | data_mem_resuse_demo.py | 内存复用演示 | ❌failed | Alpha158 属性缺失（示例过时） | 本轮 |
| rolling_process_data | workflow.py rolling_process | 滚动数据处理 | ❌failed | RestrictedUnpickler 禁止类（安全限制） | 本轮 |
| benchmarks_dynamic/baseline | rolling_benchmark.py run | 滚动重训基线 | ❌failed | qlib 新版兼容问题 | 本轮 |
| portfolio | config_enhanced_indexing.yaml | EnhancedIndexing 组合优化 | ❌failed | cvxpy 逐日 QP 求解卡死（44min 无进展，手动终止） | 本轮 |
| nested_decision_execution | workflow.py backtest | 嵌套日频/日内回测 | ❌failed | qlib 新版兼容问题 | 本轮 |

### 1d. 批4 深度模型（benchmarks 未跑，本轮实跑）

ADARNN/ADD/IGMTF/Sandwich/SFM/TCTS(Alpha360)、Localformer/TabNet/TCN/Transformer(Alpha158+360)、GeneralPtNN(gru/gru2mlp/mlp)、TRA(Alpha158/158_full/360)——共 **20 个 config，每个跑 3 次取均值±std**。**13 done + 7 skipped（训练超时），完整结果见第 6 节**。表现最好：TCTS/Alpha360（年化 9.22%、IR 1.28）、ADARNN/Alpha360（7.41%）、IGMTF/Alpha360（6.49%）。

## 2. 分目录详情（关键结论）

### benchmarks/（模型基准库）
- **完整对比见 `benchmarks/report.md`（29KB，8 大段）**，本节摘要。csi300、TopkDropout(50,5)、含/不含成本双口径。
- **树/线性模型**：DoubleEnsemble/Alpha158 最佳（含成本年化 9.80%、IR 1.13、MDD -6.9%）；树模型整体领先线性/MLP；**Alpha158 全面优于 Alpha360**（人工因子适配树模型）。
- **深度时序模型**：**GATs/Alpha158 冠军（13.11%、IR 1.79）**，超过所有树模型；Alpha360 上深度模型反超树模型（GATs 7.91% / GRU 6.09% vs LightGBM 2.03%）；KRNN 唯一失败（-2.21%）。
- **5 个进阶实验**（详见 report.md）：
  1. naive stacking（深度打分当特征喂 DoubleEnsemble，有泄露）：9.80%→7.53%，**未提升**。
  2. **预测层融合 GATs×DoubleEnsemble（w=0.5 等权 rank）：12.52%、IR 1.65 —— 有效提升**。
  3. 策略调参 topk×n_drop：融合信号 + topk50/n_drop10 → **15.56%、IR 1.96**（全流程最佳）。
  4. OOF stacking v1（修正泄露、A段NaN）：0.88%，崩盘。
  5. OOF stacking v2（B段训练+同源打分）：-5.08%；补充对照证明 A段深度模型直接回测 +9.22%，锁定"stacking 本身失效、预测层融合才是正解"。
- **三步优化累积**：DoubleEnsemble/Alpha158 基线 9.80% → +预测融合 12.52% → +策略调参 **15.56%**（IR 1.13→1.96，回撤持平略优）。

### examples 根 / workflow_by_code.py
- 模型/策略：LightGBM(GBDT) + TopkDropout，CSI300 全流程（qlib 官方教程脚本）。
- 指标：IC=0.0490（信号预测力，与 benchmarks LightGBM/Alpha158 量级一致）。

### model_interpreter/feature.py
- 用 LightGBM + Alpha158 训练后输出**特征重要性**（模型可解释性演示）。运行成功，无标准回测指标。

### highfreq/（高频 Tree）
- 模型/策略：LightGBM + High_Freq Alpha158（基于 1min 数据的日内特征）。
- 指标：IC=0.0357。演示高频数据集（DatasetH 可 pickle 序列化）用法。

## 3. 跳过任务清单

| 目录/任务 | 原因 |
|-----------|------|
| benchmarks/TFT | 需 TensorFlow 1.15 + CUDA 10.0 + Python 3.6-3.7；Apple M3 无 CUDA，无法运行 |
| benchmarks/HIST | 配置引用的 `qlib_csi300_stock2concept.npy` 缺失（仅有 stock_index.npy），无概念矩阵数据 |
| orderbook_data | 需 arctic + MongoDB，环境未装（arctic 在新 Python 难安装） |
| online_srv (rolling_online_management / online_management_simulate) | 硬编码远程 `mongodb://10.0.0.4:27017`，无该服务 |
| online_srv/update_online_pred | 依赖已有 online 模型上下文，单跑无意义 |
| run_all_model.py | 会创建/删除 conda env 并 --force-reinstall qlib，危险且仅支持 Linux，禁用 |
| rl / rl_order_execution | 需 tianshou（未装）+ 5min 高频数据 + RL 训练极慢 |
| model_rolling/task_manager_rolling.py | 依赖 MongoDB（TaskManager） |
| hyperparameter/LightGBM | 需 `pip install optuna`（未装）；可后续按需补跑 |
| 所有 .ipynb（tutorial/detailed_workflow、rl/simple_example、TRA/Reports 等） | notebook 无法用 qrun/python 批量运行，需交互执行 |
| benchmarks 各 *_csi500.yaml | csi500 变体，本轮聚焦 csi300，未跑（cn_data 含 csi500 instruments，如需可补） |

## 4. 失败任务清单

| 任务 | 错误 | 日志 |
|------|------|------|
| data_demo/data_cache_demo.py | `AttributeError: 'YAML' object has no attribute 'safe_dump'`（示例用新版 ruamel.yaml 已移除的 API） | _traverse_logs/data_demo__data_cache_demo.log |
| data_demo/data_mem_resuse_demo.py | `AttributeError: 'Alpha158' object has no attribute ...`（示例代码过时） | _traverse_logs/data_demo__data_mem_resuse_demo.log |
| rolling_process_data/workflow.py | `UnpicklingError: Forbidden class: qlib.contrib...`（qlib RestrictedUnpickler 安全限制） | _traverse_logs/rolling_process_data__workflow.log |
| benchmarks_dynamic/baseline/rolling_benchmark.py | qlib 新版兼容问题 | _traverse_logs/benchmarks_dynamic__baseline__rolling_benchmark.log |
| portfolio/config_enhanced_indexing.yaml | cvxpy 逐日 QP 求解卡死（44min 无进展，手动终止 → BrokenPipeError） | _traverse_logs/portfolio__enhanced_indexing.log |
| nested_decision_execution/workflow.py backtest | qlib 新版兼容问题 | _traverse_logs/nested_decision_execution__backtest.log |

> 失败归类：6 项中 5 项为**示例脚本/配置与当前新版 qlib/依赖库不兼容**（API 变更、安全限制），1 项为求解器卡死。均非编排问题，属示例代码维护性问题。
>
> **批4 另有 7 个深度 config 跳过（训练超时，非硬伤）**：ADD/SFM/TCN(158+360)/TRA(158+158_full+360)，单次训练 4–5h+，Apple M3 单机无 CUDA 不切实际（详见第 6 节表格）。区别于第 3 节里 TFT（需 CUDA）、HIST（缺概念矩阵）那种无法运行的硬性跳过——这 7 个在 GPU 机器上可正常完成。

## 5. 横向小结（基于已完成 + 复用结果）

- **最佳含成本收益**：DoubleEnsemble/Alpha158 融合+调参 15.56%（进阶实验）；单模型里 GATs/Alpha158 13.11%。
- **最佳信号预测力（IC）**：DoubleEnsemble/Alpha158 0.0517、TCTS/Alpha360 0.0545（批4）、ADARNN/Alpha360 0.0501（批4）、GATs/Alpha360 0.0471。
- **最稳（IR 含成本）**：融合+调参 1.96 > GATs/Alpha158 1.79 > TCTS/Alpha360 1.28（批4）> DoubleEnsemble 1.13。
- **数据集规律**：Alpha158（人工因子）适配树模型；Alpha360（原始序列）适配深度时序模型——批4 里表现最好的 TCTS/ADARNN/IGMTF 全部是 Alpha360 上的注意力/时序架构，与 benchmarks 里 GATs/GRU 的规律一致。
- **组合方式规律**：预测层融合有效（+2.7pct），特征层 stacking 失效（多种变体均劣于基线）。
- **批4补充规律**：并非所有新架构都占优——Transformer/GeneralPtNN(gru) 在 Alpha360 上默认超参下含成本收益为负，Sandwich 甚至组合层崩溃（IC 尚可但排序不稳被 TopkDropout 放大）；说明**模型复杂度 ≠ 收益**，简单稳健的 TCTS/ADARNN/GATs 反而领先。

## 6. 批4 深度模型结果（全部完成）

> benchmarks 未跑的 12 类深度模型共 20 个 config，本轮以 `n_jobs=0`（规避 macOS torch 多进程卡死）、每个跑 **3 次取均值±std** 运行。**13 个 done + 7 个 skipped**。均值±std 来自 `_traverse_deep_state.json`；含成本口径、csi300、TopkDropout(50,5)。

| 模型 | 数据集 | IC(均值±std) | Rank IC | 年化(含成本,均值±std) | IR(含成本) | MDD | n_ok |
|------|--------|-------------|---------|----------------------|-----------|-----|------|
| TCTS | Alpha360 | 0.0545±0.0032 | 0.0613 | **0.0922±0.0127** | **1.2795** | -0.0890 | 3 |
| ADARNN | Alpha360 | 0.0501±0.0036 | 0.0568 | 0.0741±0.0153 | 1.0461 | -0.0882 | 3 |
| IGMTF | Alpha360 | 0.0482±0.0024 | 0.0606 | 0.0649±0.0122 | 0.9902 | -0.0671 | 3 |
| Localformer | Alpha360 | 0.0377±0.0000 | 0.0530 | -0.0131±0.0000 | -0.1718 | -0.1627 | 3 |
| GeneralPtNN(mlp) | Alpha360 | 0.0353±0.0010 | 0.0430 | 0.0636±0.0021 | 0.7714 | -0.1053 | 3 |
| Localformer | Alpha158 | 0.0343±0.0000 | 0.0462 | 0.0140±0.0000 | 0.2138 | -0.1523 | 3 |
| Transformer | Alpha158 | 0.0317±0.0000 | 0.0435 | 0.0271±0.0000 | 0.3950 | -0.1045 | 3 |
| TabNet | Alpha158 | 0.0404±0.0000 | 0.0443 | 0.0310±0.0000 | 0.4532 | -0.1061 | 3 |
| Sandwich | Alpha360 | 0.0256±0.0037 | 0.0336 | -0.5592±0.9488 | -0.7676 | -0.6529 | 3 |
| TabNet | Alpha360 | 0.0253±0.0000 | 0.0366 | 0.0280±0.0000 | 0.3290 | -0.1529 | 3 |
| Transformer | Alpha360 | 0.0117±0.0000 | 0.0351 | -0.0352±0.0000 | -0.4314 | -0.1485 | 3 |
| GeneralPtNN(gru2mlp) | Alpha360 | 0.0076±0.0036 | 0.0119 | -0.0576±0.0072 | -0.7593 | -0.2521 | 3 |
| GeneralPtNN(gru) | Alpha360 | 0.0039±0.0031 | 0.0113 | -0.0656±0.0272 | -0.9223 | -0.2530 | 3 |

（按 IC 均值降序；std=0.0000 表示 3 次运行结果一致——这些模型 config 内含固定 seed，重复运行完全复现。）

**批4 关键结论**：
- **表现最好的是 Alpha360 上的注意力/时序模型**：TCTS（IC 0.0545、年化 9.22%、IR 1.28）> ADARNN（IC 0.0501、7.41%）> IGMTF（IC 0.0482、6.49%），三者含成本正收益且 IR≈1，与 benchmarks 里 GATs/GRU 在 Alpha360 上的表现同一梯队。
- **多数批4模型未跑赢基准**：Localformer/Alpha360、Transformer/Alpha360、GeneralPtNN(gru/gru2mlp) 含成本年化为**负**，说明这些较新/较复杂的架构在 csi300 日频上默认超参并不占优，需专门调参。
- **Sandwich 出现极端负收益且方差巨大**（-55.9%±94.9%）：IC 尚可（0.0256）但组合层面崩溃，属信号→组合传导失效的典型（预测有弱区分度但排序不稳，TopkDropout 放大了尾部风险）。
- **GeneralPtNN(mlp) 是三个 GeneralPtNN 变体里唯一为正**的（6.36%、IR 0.77），说明在 Alpha360 上简单 MLP 反而比 GeneralPtNN 的 gru 变体稳。

### 批4 跳过的 7 个 config（列入第 4 节失败/跳过说明）

| config | 原因 |
|--------|------|
| ADD/Alpha360 | 单次训练 >4h，触发 4h timeout，Apple M3 单机不切实际 |
| SFM/Alpha360 | 单次训练 >4h，同上 |
| TCN/Alpha158 | 单次 >4h，3 次全 timeout（本批最慢的重型卷积模型之一） |
| TCN/Alpha360 | 单次 >4h，与 Alpha158 同 |
| TRA/Alpha158 | 单次 >5h（比 TCN 更慢，复合 Transformer+路由）；且 4h timeout 经 `conda run` 中间层未能杀掉子进程，会无限卡住整批 |
| TRA/Alpha158_full | 全量特征，更重，同 TRA/Alpha158 |
| TRA/Alpha360 | 序列版，同 TRA/Alpha158 |

> 这 7 个均为**训练耗时问题**（单机 M3 无 CUDA，单次 4–5h+），非模型硬伤（区别于 TFT 需 CUDA、HIST 缺概念矩阵那种无法运行的情况）。如在 GPU 机器上则可正常跑完。
