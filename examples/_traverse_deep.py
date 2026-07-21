"""批4：跑 benchmarks 未跑的深度模型，每个 config 跑 3 次取均值±std。
复用 benchmarks/_run_deep.py 的 patch_config(n_jobs=0/绝对model_path) 与 _run_benchmarks.py 的 parse_metrics。
增量写 _traverse_deep_state.json，断点续跑（已达 3 次成功的 config 跳过）。失败不中断。

用法: conda run -n qlib python _traverse_deep.py
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
STATE = EX / "_traverse_deep_state.json"

# 复用 patch_config 与 parse_metrics
_rd = importlib.util.spec_from_file_location("rd", BENCH / "_run_deep.py")
rd = importlib.util.module_from_spec(_rd); _rd.loader.exec_module(rd)
parse_metrics = rd.runb.parse_metrics  # _run_deep 内已加载 _run_benchmarks 为 rd.runb
patch_config = rd.patch_config

N_RUNS = 3
METRIC_KEYS = ["IC", "ICIR", "Rank IC", "Rank ICIR",
               "ann_w_cost", "ir_w_cost", "mdd_w_cost", "ann_wo_cost", "ir_wo_cost", "mdd_wo_cost"]

# benchmarks 未跑的深度模型 config（相对 benchmarks/）
CONFIGS = [
    "ADARNN/workflow_config_adarnn_Alpha360.yaml",
    # "ADD/workflow_config_add_Alpha360.yaml",  # 剔除：单次训练>4h 触发 timeout，本机不切实际，标 skipped
    "IGMTF/workflow_config_igmtf_Alpha360.yaml",
    "Sandwich/workflow_config_sandwich_Alpha360.yaml",
    # "SFM/workflow_config_sfm_Alpha360.yaml",  # 剔除：单次训练>4h 触发 timeout，本机不切实际，标 skipped
    "TCTS/workflow_config_tcts_Alpha360.yaml",
    "Localformer/workflow_config_localformer_Alpha158.yaml",
    "Localformer/workflow_config_localformer_Alpha360.yaml",
    "TabNet/workflow_config_TabNet_Alpha158.yaml",
    "TabNet/workflow_config_TabNet_Alpha360.yaml",
    # "TCN/workflow_config_tcn_Alpha158.yaml",  # 剔除:单次>4h,3次全超时,本机不切实际
    # "TCN/workflow_config_tcn_Alpha360.yaml",  # 剔除:与Alpha158同,单次>4h超时,本机不切实际
    "Transformer/workflow_config_transformer_Alpha158.yaml",
    "Transformer/workflow_config_transformer_Alpha360.yaml",
    "GeneralPtNN/workflow_config_gru.yaml",
    "GeneralPtNN/workflow_config_gru2mlp.yaml",
    "GeneralPtNN/workflow_config_mlp.yaml",
    # "TRA/workflow_config_tra_Alpha158.yaml",  # 剔除:单次训练>5h且timeout经conda run未生效,会卡死整批,本机不切实际
    # "TRA/workflow_config_tra_Alpha158_full.yaml",  # 剔除:单次训练>5h且timeout经conda run未生效,会卡死整批,本机不切实际
    # "TRA/workflow_config_tra_Alpha360.yaml",  # 剔除:单次训练>5h且timeout经conda run未生效,会卡死整批,本机不切实际
]


def load_state():
    return json.loads(STATE.read_text()) if STATE.exists() else {}


def save(s):
    STATE.write_text(json.dumps(s, ensure_ascii=False, indent=2))


def run_once(cfg_rel, job_id, idx):
    cfg_path = BENCH / cfg_rel
    run_cfg = patch_config(cfg_path)  # 生成临时 n_jobs=0 配置
    # TabNet 的构造函数不接受 n_jobs 参数 → 删掉 patch 注入的 n_jobs 行
    if "TabNet" in cfg_rel or "tabnet" in cfg_rel.lower():
        txt = run_cfg.read_text()
        txt = "\n".join(l for l in txt.split("\n") if "n_jobs" not in l)
        run_cfg.write_text(txt)
    log = LOGDIR / f"deep__{job_id}__run{idx}.log"
    env = dict(os.environ, OMP_NUM_THREADS="4", MKL_NUM_THREADS="4", PYTHONUNBUFFERED="1")
    t0 = time.time()
    try:
        proc = subprocess.run(["qrun", str(run_cfg), "--experiment_name", f"trav_deep_{job_id}_{idx}"],
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
            if idx > N_RUNS + 2:  # 最多尝试 N+2 次，避免死循环
                break
            r = run_once(cfg_rel, job_id, idx)
            rec["runs"].append(r)
            state[job_id] = rec
            save(state)
            tag = f"IC={r['IC']:.4f}" if "IC" in r else r.get("error", "?")
            print(f"  run{idx}: {tag} ({r.get('sec','?')}s)", flush=True)
        # 汇总均值±std
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
    print("\n批4 全部处理完毕。状态见 _traverse_deep_state.json", flush=True)


if __name__ == "__main__":
    main()
