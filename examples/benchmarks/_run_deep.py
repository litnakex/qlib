"""跑深度时序模型 (GRU/LSTM/GATs/KRNN)，抓取指标，追加到 report.md。
逐个运行，失败不中断；日志存到 _logs/。在 Apple MPS/CPU 上运行（配置里的 GPU:0 会自动回退）。
增量写盘：每跑完一个模型就把结果存入 _deep_results.json 并刷新 report.md，
即使中途被 kill 也能保留已完成部分。重跑时已完成的模型会跳过（除非 --force）。
"""
import json
import os
import re
import sys
import time
from pathlib import Path
import importlib.util

BENCH = Path(__file__).resolve().parent
LOGDIR = BENCH / "_logs"
LOGDIR.mkdir(exist_ok=True)
RESULTS_JSON = BENCH / "_deep_results.json"

spec = importlib.util.spec_from_file_location("runb", BENCH / "_run_benchmarks.py")
runb = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runb)

import subprocess

JOBS = [
    ("GRU", "Alpha158", "GRU/workflow_config_gru_Alpha158.yaml"),
    ("GRU", "Alpha360", "GRU/workflow_config_gru_Alpha360.yaml"),
    ("LSTM", "Alpha158", "LSTM/workflow_config_lstm_Alpha158.yaml"),
    ("LSTM", "Alpha360", "LSTM/workflow_config_lstm_Alpha360.yaml"),
    ("GATs", "Alpha158", "GATs/workflow_config_gats_Alpha158.yaml"),
    ("GATs", "Alpha360", "GATs/workflow_config_gats_Alpha360.yaml"),
    ("KRNN", "Alpha360", "KRNN/workflow_config_krnn_Alpha360.yaml"),
]


def fmt(v, nd=4):
    return f"{v:.{nd}f}" if isinstance(v, (int, float)) else "—"


def patch_config(cfg_path: Path) -> Path:
    """生成临时配置副本，强制 model.kwargs 里 n_jobs=0（禁用多进程 DataLoader）。
    - 若已有 `n_jobs:` 字段 → 就地替换为 0
    - 若没有 → 在 `class: <Model>` 下方（model kwargs 区）插入一行 n_jobs: 0
    用文本级替换以保持原文件其它内容与缩进不变。
    """
    text = cfg_path.read_text()
    if re.search(r"^\s*n_jobs:\s*\d+", text, re.M):
        patched = re.sub(r"(^\s*)n_jobs:\s*\d+", r"\g<1>n_jobs: 0", text, flags=re.M)
    else:
        # 精确定位 model 段的 kwargs（在 `    model:` 与下一个同级 `    dataset:` 之间）
        lines = text.split("\n")
        out = []
        in_model = False
        inserted = False
        for j, ln in enumerate(lines):
            if re.match(r"^\s{4}model:\s*$", ln):
                in_model = True
            elif re.match(r"^\s{4}\w+:\s*$", ln) and not ln.lstrip().startswith("model"):
                in_model = False  # 离开 model 段（进入 dataset/record 等）
            out.append(ln)
            if in_model and not inserted and re.match(r"^\s{8}kwargs:\s*$", ln):
                nxt = lines[j + 1] if j + 1 < len(lines) else "            x"
                indent = re.match(r"^(\s*)", nxt).group(1)
                out.append(f"{indent}n_jobs: 0")
                inserted = True
        patched = "\n".join(out)
    # GATs 等模型的 model_path 是相对 examples/ 的（如 benchmarks/LSTM/xxx.pkl），
    # 但 qrun 的 cwd 是各模型子目录，会找不到 → 改成绝对路径。
    examples_dir = BENCH.parent  # .../examples
    def _abs_model_path(m):
        rel = m.group(2).strip().strip('"').strip("'")
        p = (examples_dir / rel).resolve()
        return f'{m.group(1)}model_path: "{p}"'
    patched = re.sub(r'(^\s*)model_path:\s*(.+)$', _abs_model_path, patched, flags=re.M)
    tmp = cfg_path.parent / f"_tmp_{cfg_path.name}"
    tmp.write_text(patched)
    return tmp


def load_done():
    if RESULTS_JSON.exists():
        return json.loads(RESULTS_JSON.read_text())
    return {}


def save(done):
    RESULTS_JSON.write_text(json.dumps(done, ensure_ascii=False, indent=2))


def main():
    force = "--force" in sys.argv
    done = {} if force else load_done()
    for i, (model, ds, cfg) in enumerate(JOBS, 1):
        tag = f"{model}_{ds}"
        if tag in done and "IC" in done[tag]:
            print(f"[{i}/{len(JOBS)}] 跳过已完成 {tag} (IC={done[tag]['IC']:.4f})", flush=True)
            continue
        cfg_path = BENCH / cfg
        print(f"[{i}/{len(JOBS)}] 运行 {tag} ...", flush=True)
        # 生成临时配置：强制 n_jobs=0，避免 macOS torch 多进程 DataLoader 共享内存超时卡死
        run_cfg = patch_config(cfg_path)
        t0 = time.time()
        env = dict(os.environ, OMP_NUM_THREADS="4", MKL_NUM_THREADS="4",
                   PYTHONUNBUFFERED="1")
        proc = subprocess.run(["qrun", str(run_cfg)], cwd=str(cfg_path.parent),
                              capture_output=True, text=True, env=env)
        dt = time.time() - t0
        try:
            run_cfg.unlink()
        except OSError:
            pass
        full = proc.stdout + "\n" + proc.stderr
        (LOGDIR / f"{tag}.log").write_text(full)
        if proc.returncode != 0:
            print(f"  失败 exit {proc.returncode}, {dt:.0f}s", flush=True)
            m = {"error": f"exit {proc.returncode}", "sec": dt}
        else:
            m = runb.parse_metrics(full)
            m["sec"] = dt
            if "IC" not in m:
                m["error"] = "no metrics parsed"
                print(f"  跑完但未解析到指标, {dt:.0f}s", flush=True)
            else:
                print(f"  完成 IC={m['IC']:.4f}, {dt:.0f}s", flush=True)
        done[tag] = m
        save(done)                     # 增量写盘：每个模型跑完立即持久化
        append_report(done)            # 每次刷新 report.md 的深度段落
    print("\n全部深度模型处理完毕。")


DEEP_MARKER = "## 深度时序模型（GRU / LSTM / GATs / KRNN）"


def append_report(done):
    # 按 JOBS 顺序组装 results
    results = [(model, ds, done.get(f"{model}_{ds}", {})) for model, ds, _ in JOBS]
    L = []
    L.append("## 深度时序模型（GRU / LSTM / GATs / KRNN）\n")
    L.append("这些是需要 PyTorch 的时序神经网络模型，在 Apple M3 Pro（MPS/CPU）上运行，n_epochs=200 + early_stop=20。理论上更适配 Alpha360（原始价量序列，让网络自动提特征）。\n")
    L.append("### 信号预测指标\n")
    L.append("| 模型 | 数据集 | IC | ICIR | Rank IC | Rank ICIR | 耗时(s) |")
    L.append("|------|--------|----|----|---------|-----------|---------|")
    for model, ds, m in results:
        if "IC" not in m:
            L.append(f"| {model} | {ds} | ❌ {m.get('error','')} | | | | {fmt(m.get('sec'),0)} |")
        else:
            L.append(f"| {model} | {ds} | {fmt(m.get('IC'))} | {fmt(m.get('ICIR'))} | {fmt(m.get('Rank IC'))} | {fmt(m.get('Rank ICIR'))} | {fmt(m.get('sec'),0)} |")
    L.append("")
    L.append("### 回测组合指标（超额收益，**含交易成本**）\n")
    L.append("| 模型 | 数据集 | 年化收益 | 信息比率(IR) | 最大回撤 |")
    L.append("|------|--------|---------|--------------|----------|")
    for model, ds, m in results:
        if "ann_w_cost" not in m:
            L.append(f"| {model} | {ds} | — | — | — |")
        else:
            L.append(f"| {model} | {ds} | {fmt(m['ann_w_cost'])} | {fmt(m['ir_w_cost'])} | {fmt(m['mdd_w_cost'])} |")
    L.append("")
    L.append("### 回测组合指标（超额收益，不含成本）\n")
    L.append("| 模型 | 数据集 | 年化收益 | 信息比率(IR) | 最大回撤 |")
    L.append("|------|--------|---------|--------------|----------|")
    for model, ds, m in results:
        if "ann_wo_cost" not in m:
            L.append(f"| {model} | {ds} | — | — | — |")
        else:
            L.append(f"| {model} | {ds} | {fmt(m['ann_wo_cost'])} | {fmt(m['ir_wo_cost'])} | {fmt(m['mdd_wo_cost'])} |")
    L.append("")

    section = "\n---\n\n" + "\n".join(L)
    report = BENCH / "report.md"
    text = report.read_text()
    idx = text.find(DEEP_MARKER)
    if idx != -1:
        # 已有深度段落 → 连同其上方的 --- 分隔一起替换
        cut = text.rfind("\n---\n", 0, idx)
        text = text[:cut] if cut != -1 else text[:idx].rstrip()
    report.write_text(text.rstrip() + "\n" + section)


if __name__ == "__main__":
    main()
