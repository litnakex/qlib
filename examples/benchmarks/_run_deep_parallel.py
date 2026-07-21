"""并发跑剩余深度模型（并发度=3），复用 _run_deep.py 的 patch_config / parse_metrics。
每个子任务限制底层线程数（4），3×4≈12 核，避免在 M3 Pro 上过度争抢。
增量写盘：每个模型跑完立即更新 _deep_results.json 并刷新 report.md 深度段落。
已完成的模型（_deep_results.json 里已有 IC）自动跳过。
"""
import json
import os
import subprocess
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import importlib.util

BENCH = Path(__file__).resolve().parent
LOGDIR = BENCH / "_logs"
RESULTS_JSON = BENCH / "_deep_results.json"

spec = importlib.util.spec_from_file_location("rd", BENCH / "_run_deep.py")
rd = importlib.util.module_from_spec(spec)
spec.loader.exec_module(rd)  # 复用 patch_config / append_report / JOBS / parse via runb

CONCURRENCY = 3
THREADS_PER_JOB = "4"

_lock = threading.Lock()


def load_done():
    if RESULTS_JSON.exists():
        return json.loads(RESULTS_JSON.read_text())
    return {}


def run_one(model, ds, cfg):
    cfg_path = BENCH / cfg
    tag = f"{model}_{ds}"
    run_cfg = rd.patch_config(cfg_path)
    env = dict(os.environ, OMP_NUM_THREADS=THREADS_PER_JOB, MKL_NUM_THREADS=THREADS_PER_JOB,
               NUMEXPR_NUM_THREADS=THREADS_PER_JOB, VECLIB_MAXIMUM_THREADS=THREADS_PER_JOB,
               OPENBLAS_NUM_THREADS=THREADS_PER_JOB, PYTHONUNBUFFERED="1")
    t0 = time.time()
    print(f"  ▶ 开始 {tag}", flush=True)
    # 每个模型用独立 experiment_name，避免并发时 MLflow 文件后端创建同名实验冲突
    proc = subprocess.run(["qrun", str(run_cfg), "--experiment_name", f"deep_{tag}"],
                          cwd=str(cfg_path.parent),
                          capture_output=True, text=True, env=env)
    dt = time.time() - t0
    (LOGDIR / f"{tag}.log").write_text(proc.stdout + "\n" + proc.stderr)
    try:
        run_cfg.unlink()
    except OSError:
        pass
    if proc.returncode != 0:
        m = {"error": f"exit {proc.returncode}", "sec": dt}
        print(f"  ✗ {tag} 失败 exit {proc.returncode} ({dt:.0f}s)", flush=True)
    else:
        m = rd.runb.parse_metrics(proc.stdout + "\n" + proc.stderr)
        m["sec"] = dt
        if "IC" not in m:
            m["error"] = "no metrics parsed"
            print(f"  ? {tag} 跑完但无指标 ({dt:.0f}s)", flush=True)
        else:
            print(f"  ✓ {tag} IC={m['IC']:.4f} ({dt:.0f}s)", flush=True)
    # 增量写盘 + 刷新报告（加锁避免并发写冲突）
    with _lock:
        done = load_done()
        done[tag] = m
        RESULTS_JSON.write_text(json.dumps(done, ensure_ascii=False, indent=2))
        rd.append_report(done)
    return tag, m


def main():
    done = load_done()
    todo = [(mo, ds, cfg) for mo, ds, cfg in rd.JOBS
            if f"{mo}_{ds}" not in done or "IC" not in done[f"{mo}_{ds}"]]
    print(f"待跑 {len(todo)} 个，并发度 {CONCURRENCY}：{[f'{m}_{d}' for m,d,_ in todo]}", flush=True)
    with ThreadPoolExecutor(max_workers=CONCURRENCY) as ex:
        futs = [ex.submit(run_one, *job) for job in todo]
        for f in as_completed(futs):
            f.result()
    print("\n全部完成。", flush=True)


if __name__ == "__main__":
    main()
