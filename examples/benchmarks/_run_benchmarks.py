"""跑一批 csi300 快模型 workflow，抓取指标，生成 report.md。
逐个运行，失败不中断；每个 run 的完整日志存到 _logs/ 下。
用法: python _run_benchmarks.py
"""
import ast
import re
import subprocess
import sys
import time
from pathlib import Path

BENCH = Path(__file__).resolve().parent
LOGDIR = BENCH / "_logs"
LOGDIR.mkdir(exist_ok=True)

# (模型名, 数据集, 相对配置路径)
JOBS = [
    ("LightGBM", "Alpha158", "LightGBM/workflow_config_lightgbm_Alpha158.yaml"),
    ("LightGBM", "Alpha360", "LightGBM/workflow_config_lightgbm_Alpha360.yaml"),
    ("XGBoost", "Alpha158", "XGBoost/workflow_config_xgboost_Alpha158.yaml"),
    ("XGBoost", "Alpha360", "XGBoost/workflow_config_xgboost_Alpha360.yaml"),
    ("CatBoost", "Alpha158", "CatBoost/workflow_config_catboost_Alpha158.yaml"),
    ("CatBoost", "Alpha360", "CatBoost/workflow_config_catboost_Alpha360.yaml"),
    ("Linear", "Alpha158", "Linear/workflow_config_linear_Alpha158.yaml"),
    ("DoubleEnsemble", "Alpha158", "DoubleEnsemble/workflow_config_doubleensemble_Alpha158.yaml"),
    ("DoubleEnsemble", "Alpha360", "DoubleEnsemble/workflow_config_doubleensemble_Alpha360.yaml"),
    ("MLP", "Alpha158", "MLP/workflow_config_mlp_Alpha158.yaml"),
    ("MLP", "Alpha360", "MLP/workflow_config_mlp_Alpha360.yaml"),
]


def parse_metrics(text):
    """从 qrun 输出里解析信号指标与组合指标。"""
    out = {}
    # 信号指标：逐字段抓，兼容不同模型输出的键集合差异
    # 形如  'IC': np.float64(0.0387..)  或  'Rank IC': np.float32(..)
    for key in ["IC", "ICIR", "Rank IC", "Rank ICIR"]:
        m = re.search(rf"'{re.escape(key)}':\s*np\.float(?:32|64)\((-?[\d.eE+-]+)\)", text)
        if m:
            out[key] = float(m.group(1))
    # 组合指标：三段 risk 表。抓 excess return with/without cost 的 annualized_return / information_ratio / max_drawdown
    for label, key in [
        ("excess return without cost", "wo_cost"),
        ("excess return with cost", "w_cost"),
    ]:
        seg = re.search(
            rf"analysis results of the {label}.*?annualized_return\s+(-?[\d.]+).*?information_ratio\s+(-?[\d.]+).*?max_drawdown\s+(-?[\d.]+)",
            text, re.S,
        )
        if seg:
            out[f"ann_{key}"] = float(seg.group(1))
            out[f"ir_{key}"] = float(seg.group(2))
            out[f"mdd_{key}"] = float(seg.group(3))
    return out


def main():
    results = []
    for i, (model, ds, cfg) in enumerate(JOBS, 1):
        cfg_path = BENCH / cfg
        tag = f"{model}_{ds}"
        log_path = LOGDIR / f"{tag}.log"
        print(f"\n[{i}/{len(JOBS)}] 运行 {tag} ...", flush=True)
        if not cfg_path.exists():
            print(f"  配置不存在，跳过: {cfg_path}")
            results.append((model, ds, {"error": "config missing"}))
            continue
        t0 = time.time()
        proc = subprocess.run(
            ["qrun", str(cfg_path)],
            cwd=str(cfg_path.parent),
            capture_output=True, text=True,
        )
        dt = time.time() - t0
        full = proc.stdout + "\n" + proc.stderr
        log_path.write_text(full)
        if proc.returncode != 0:
            print(f"  失败 (exit {proc.returncode})，耗时 {dt:.0f}s，日志: {log_path.name}")
            results.append((model, ds, {"error": f"exit {proc.returncode}", "sec": dt}))
            continue
        metrics = parse_metrics(full)
        metrics["sec"] = dt
        if "IC" not in metrics:
            print(f"  跑完但未解析到指标 (exit 0)，耗时 {dt:.0f}s，查日志 {log_path.name}")
            metrics["error"] = "no metrics parsed"
        else:
            print(f"  完成，IC={metrics.get('IC'):.4f} 耗时 {dt:.0f}s")
        results.append((model, ds, metrics))
    write_report(results)


def fmt(v, nd=4):
    return f"{v:.{nd}f}" if isinstance(v, (int, float)) else "—"


def write_report(results):
    lines = []
    lines.append("# Qlib Benchmark 对比报告\n")
    lines.append("对同一 csi300 数据集上的多个 CPU 快模型运行完整 qlib workflow（训练→预测→回测），对比信号预测力与回测收益。\n")
    lines.append("- **数据集/股票池**: csi300")
    lines.append("- **标签**: Ref($close,-2)/Ref($close,-1)-1（次日到后日收益）")
    lines.append("- **回测策略**: TopkDropoutStrategy (topk=50, n_drop=5)")
    lines.append("- **指标说明**: IC/Rank IC 越高预测力越强；ICIR/Rank ICIR 是其稳定性；年化收益/信息比率(IR)为超额收益(相对基准)，含成本更贴近实盘；最大回撤(MDD)越接近 0 越好。\n")

    # 信号指标表
    lines.append("## 信号预测指标\n")
    lines.append("| 模型 | 数据集 | IC | ICIR | Rank IC | Rank ICIR | 耗时(s) |")
    lines.append("|------|--------|----|----|---------|-----------|---------|")
    for model, ds, m in results:
        if m.get("error") and "IC" not in m:
            lines.append(f"| {model} | {ds} | ❌ {m['error']} | | | | {fmt(m.get('sec'),0)} |")
        else:
            lines.append(f"| {model} | {ds} | {fmt(m.get('IC'))} | {fmt(m.get('ICIR'))} | {fmt(m.get('Rank IC'))} | {fmt(m.get('Rank ICIR'))} | {fmt(m.get('sec'),0)} |")
    lines.append("")

    # 组合回测指标表（含成本）
    lines.append("## 回测组合指标（超额收益，**含交易成本**）\n")
    lines.append("| 模型 | 数据集 | 年化收益 | 信息比率(IR) | 最大回撤 |")
    lines.append("|------|--------|---------|--------------|----------|")
    for model, ds, m in results:
        if "ann_w_cost" not in m:
            lines.append(f"| {model} | {ds} | — | — | — |")
        else:
            lines.append(f"| {model} | {ds} | {fmt(m.get('ann_w_cost'))} | {fmt(m.get('ir_w_cost'))} | {fmt(m.get('mdd_w_cost'))} |")
    lines.append("")

    # 不含成本
    lines.append("## 回测组合指标（超额收益，不含成本）\n")
    lines.append("| 模型 | 数据集 | 年化收益 | 信息比率(IR) | 最大回撤 |")
    lines.append("|------|--------|---------|--------------|----------|")
    for model, ds, m in results:
        if "ann_wo_cost" not in m:
            lines.append(f"| {model} | {ds} | — | — | — |")
        else:
            lines.append(f"| {model} | {ds} | {fmt(m.get('ann_wo_cost'))} | {fmt(m.get('ir_wo_cost'))} | {fmt(m.get('mdd_wo_cost'))} |")
    lines.append("")

    report = BENCH / "report.md"
    report.write_text("\n".join(lines))
    print(f"\n报告已写入: {report}")


if __name__ == "__main__":
    main()
