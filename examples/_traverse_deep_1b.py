"""补跑 output.md 第 1b 节的 7 个 benchmarks 深度模型（原为单次复用，现按批4方式跑 3 次取均值±std）。
完全复用 _traverse_deep.py 的逻辑（patch_config/parse_metrics/3次/超时/断点续跑/失败不中断），
仅换独立 state 文件与 config 列表，避免与批4状态冲突。

用法: conda run -n qlib python _traverse_deep_1b.py
"""
import json
import os
import subprocess
import time
import statistics as st
from pathlib import Path
import importlib.util

EX = Path(__file__).resolve().parent
BENCH = EX / "benchmarks"
LOGDIR = EX / "_traverse_logs"
LOGDIR.mkdir(exist_ok=True)
STATE = EX / "_traverse_deep_1b_state.json"

# 复用 patch_config 与 parse_metrics
_rd = importlib.util.spec_from_file_location("rd", BENCH / "_run_deep.py")
rd = importlib.util.module_from_spec(_rd); _rd.loader.exec_module(rd)
parse_metrics = rd.runb.parse_metrics
patch_config = rd.patch_config

N_RUNS = 3
METRIC_KEYS = ["IC", "ICIR", "Rank IC", "Rank ICIR",
               "ann_w_cost", "ir_w_cost", "mdd_w_cost", "ann_wo_cost", "ir_wo_cost", "mdd_wo_cost"]

# 1b 节的 7 个深度模型 config（相对 benchmarks/）
CONFIGS = [
    "GRU/workflow_config_gru_Alpha158.yaml",
    "GRU/workflow_config_gru_Alpha360.yaml",
    "LSTM/workflow_config_lstm_Alpha158.yaml",
    "LSTM/workflow_config_lstm_Alpha360.yaml",
    "GATs/workflow_config_gats_Alpha158.yaml",
    "GATs/workflow_config_gats_Alpha360.yaml",
    "KRNN/workflow_config_krnn_Alpha360.yaml",
]


def load_state():
    return json.loads(STATE.read_text()) if STATE.exists() else {}


def save(s):
    STATE.write_text(json.dumps(s, ensure_ascii=False, indent=2))


def run_once(cfg_rel, job_id, idx):
    cfg_path = BENCH / cfg_rel
    run_cfg = patch_config(cfg_path)
    log = LOGDIR / f"deep1b__{job_id}__run{idx}.log"
    env = dict(os.environ, OMP_NUM_THREADS="4", MKL_NUM_THREADS="4", PYTHONUNBUFFERED="1")
    t0 = time.time()
    try:
        proc = subprocess.run(["qrun", str(run_cfg), "--experiment_name", f"trav_deep1b_{job_id}_{idx}"],
                              cwd=str(cfg_path.parent), capture_output=True, text=True, env=env, timeout=14400)
    except subprocess.TimeoutExpired:
        log.write_text("TIMEOUT>14400s")
        return {"error": "timeout"}
    finally:
        try: run_cfg.unlink()
        except OSError: pass
    dt = time.time() - t0
    full = (proc.stdout or "") + "\n" + (proc.stderr or "")
    log.write_text(full)
    if proc.returncode != 0:
        return {"error": f"exit {proc.returncode}", "sec": round(dt)}
    m = parse_metrics(full)
    if "IC" not in m:
        return {"error": "no metrics", "sec": round(dt)}
    m["sec"] = round(dt)
    return m


def main():
    state = load_state()
    for cfg_rel in CONFIGS:
        job_id = cfg_rel.replace("/", "_").replace("workflow_config_", "").replace(".yaml", "")
        rec = state.get(job_id, {"runs": []})
        ok_runs = [r for r in rec.get("runs", []) if "IC" in r]
        if len(ok_runs) >= N_RUNS:
            print(f"⏭  {job_id} 已有 {len(ok_runs)} 次成功，跳过", flush=True)
            continue
        print(f"▶ {job_id} (已成功 {len(ok_runs)}/{N_RUNS})", flush=True)
        while len([r for r in rec["runs"] if "IC" in r]) < N_RUNS:
            idx = len(rec["runs"]) + 1
            if idx > N_RUNS + 2:
                break
            r = run_once(cfg_rel, job_id, idx)
            rec["runs"].append(r)
            state[job_id] = rec
            save(state)
            tag = f"IC={r['IC']:.4f}" if "IC" in r else r.get("error", "?")
            print(f"  run{idx}: {tag} ({r.get('sec','?')}s)", flush=True)
        ok = [r for r in rec["runs"] if "IC" in r]
        if ok:
            mean = {k: round(st.mean([r[k] for r in ok if k in r]), 4) for k in METRIC_KEYS if any(k in r for r in ok)}
            std = {k: round(st.stdev([r[k] for r in ok if k in r]), 4) if len(ok) >= 2 else 0.0
                   for k in METRIC_KEYS if any(k in r for r in ok)}
            rec.update(status="done", n_ok=len(ok), mean=mean, std=std)
        else:
            rec.update(status="failed", n_ok=0)
        state[job_id] = rec
        save(state)
        print(f"  ✓ {job_id} 完成 n_ok={rec.get('n_ok')}", flush=True)
    print("\n1b 补跑 全部处理完毕。状态见 _traverse_deep_1b_state.json", flush=True)


if __name__ == "__main__":
    main()
