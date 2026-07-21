"""按 examples/run.md 遍历运行任务的编排器（批2/批3：快任务与轻准备任务）。
- 复用 benchmarks/_run_benchmarks.py 的 parse_metrics 解析 qrun 输出指标。
- 状态写 _traverse_state.json（断点续跑：已 done 的跳过）；日志写 _traverse_logs/。
- 失败不中断：单任务异常记 failed + 原因，继续下一个。
- 深度模型批4 由单独脚本处理，本脚本只跑确定性/轻准备任务。

用法: conda run -n qlib python _traverse_run.py
"""
import json
import os
import subprocess
import time
from pathlib import Path
import importlib.util

EX = Path(__file__).resolve().parent
BENCH = EX / "benchmarks"
LOGDIR = EX / "_traverse_logs"
LOGDIR.mkdir(exist_ok=True)
STATE = EX / "_traverse_state.json"

# 复用 parse_metrics
spec = importlib.util.spec_from_file_location("rb", BENCH / "_run_benchmarks.py")
rb = importlib.util.module_from_spec(spec)
spec.loader.exec_module(rb)

# 任务清单: (job_id, 工作目录(相对EX), 命令(list), 类型)
# 类型: "metrics"=跑完解析IC等指标; "run"=只记运行成功/输出摘要
JOBS = [
    # 批2 — 开箱即跑 / 确定性 / 快
    ("root/workflow_by_code", ".", ["python", "workflow_by_code.py"], "metrics"),
    ("data_demo/data_cache_demo", "data_demo", ["python", "data_cache_demo.py"], "run"),
    ("data_demo/data_mem_resuse_demo", "data_demo", ["python", "data_mem_resuse_demo.py"], "run"),
    ("model_interpreter/feature", "model_interpreter", ["python", "feature.py"], "run"),
    ("rolling_process_data/workflow", "rolling_process_data", ["python", "workflow.py", "rolling_process"], "run"),
    ("benchmarks_dynamic/baseline/rolling_benchmark", "benchmarks_dynamic/baseline",
     ["python", "rolling_benchmark.py", "run"], "run"),
    # 批3 — 轻准备
    ("portfolio/enhanced_indexing", "portfolio", ["qrun", "config_enhanced_indexing.yaml"], "metrics"),
    ("nested_decision_execution/backtest", "nested_decision_execution", ["python", "workflow.py", "backtest"], "run"),
    ("highfreq/tree_alpha158", "highfreq", ["qrun", "workflow_config_High_Freq_Tree_Alpha158.yaml"], "metrics"),
]


def load_state():
    if STATE.exists():
        return json.loads(STATE.read_text())
    return {}


def save_state(s):
    STATE.write_text(json.dumps(s, ensure_ascii=False, indent=2))


def run_job(job_id, cwd_rel, cmd, kind, state):
    log_path = LOGDIR / f"{job_id.replace('/', '__')}.log"
    cwd = EX / cwd_rel
    # 用 experiment_name 隔离 qrun 的 mlflow
    full_cmd = list(cmd)
    if cmd[0] == "qrun":
        full_cmd += ["--experiment_name", f"trav_{job_id.replace('/', '_')}"]
    env = dict(os.environ, OMP_NUM_THREADS="6", PYTHONUNBUFFERED="1")
    print(f"▶ 运行 {job_id}: {' '.join(cmd)} (cwd={cwd_rel})", flush=True)
    t0 = time.time()
    try:
        proc = subprocess.run(full_cmd, cwd=str(cwd), capture_output=True, text=True, env=env, timeout=7200)
    except subprocess.TimeoutExpired:
        state[job_id] = {"status": "failed", "error": "timeout>7200s", "log": str(log_path)}
        save_state(state)
        print(f"  ✗ {job_id} 超时", flush=True)
        return
    dt = time.time() - t0
    full = (proc.stdout or "") + "\n" + (proc.stderr or "")
    log_path.write_text(full)
    rec = {"cmd": " ".join(cmd), "cwd": cwd_rel, "sec": round(dt), "log": str(log_path)}
    if proc.returncode != 0:
        # 取错误尾部
        errtail = "\n".join([l for l in full.splitlines()
                             if any(k in l for k in ("Error", "error", "Exception", "Traceback"))][-3:])
        rec.update(status="failed", error=errtail[:500] or f"exit {proc.returncode}")
        print(f"  ✗ {job_id} 失败 exit {proc.returncode} ({dt:.0f}s)", flush=True)
    elif kind == "metrics":
        m = rb.parse_metrics(full)
        if "IC" in m:
            rec.update(status="done", metrics=m)
            print(f"  ✓ {job_id} IC={m['IC']:.4f} ({dt:.0f}s)", flush=True)
        else:
            rec.update(status="done", note="运行成功但未解析到IC指标")
            print(f"  ✓ {job_id} 运行成功(无标准指标) ({dt:.0f}s)", flush=True)
    else:
        # run 类: 记成功 + stdout 末尾摘要
        tail = "\n".join(proc.stdout.splitlines()[-8:]) if proc.stdout else ""
        rec.update(status="done", note="运行成功", tail=tail[:800])
        print(f"  ✓ {job_id} 运行成功 ({dt:.0f}s)", flush=True)
    state[job_id] = rec
    save_state(state)


def main():
    state = load_state()
    for job_id, cwd_rel, cmd, kind in JOBS:
        if state.get(job_id, {}).get("status") in ("done", "failed"):
            print(f"⏭  跳过已达终态 {job_id}", flush=True)
            continue
        run_job(job_id, cwd_rel, cmd, kind, state)
    print("\n批2/批3 处理完毕。状态见 _traverse_state.json", flush=True)


if __name__ == "__main__":
    main()
