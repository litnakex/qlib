"""补跑修复：
- CatBoost×2 之前因 Poisson bootstrap (CPU 不支持) 失败 → 生成临时配置改成 Bernoulli 重跑。
- 其余模型直接复用 _logs/ 里已有日志重新解析（含 Linear，之前只是解析器 bug）。
最后重新生成 report.md。
"""
import re
import subprocess
import time
from pathlib import Path

BENCH = Path(__file__).resolve().parent
LOGDIR = BENCH / "_logs"

import importlib.util
spec = importlib.util.spec_from_file_location("runb", BENCH / "_run_benchmarks.py")
runb = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runb)

JOBS = runb.JOBS  # 保持顺序一致
CATBOOST = {"CatBoost_Alpha158", "CatBoost_Alpha360"}


def rerun_catboost(model, ds, cfg):
    cfg_path = BENCH / cfg
    tag = f"{model}_{ds}"
    # 生成临时配置：Poisson -> Bernoulli
    text = cfg_path.read_text()
    patched = text.replace("bootstrap_type: Poisson", "bootstrap_type: Bernoulli")
    tmp = cfg_path.parent / f"_tmp_{cfg_path.name}"
    tmp.write_text(patched)
    print(f"  重跑 {tag} (bootstrap_type: Bernoulli) ...", flush=True)
    t0 = time.time()
    proc = subprocess.run(["qrun", str(tmp)], cwd=str(cfg_path.parent),
                          capture_output=True, text=True)
    dt = time.time() - t0
    full = proc.stdout + "\n" + proc.stderr
    (LOGDIR / f"{tag}.log").write_text(full)
    tmp.unlink(missing_ok=True)
    if proc.returncode != 0:
        print(f"    仍失败 exit {proc.returncode}, {dt:.0f}s")
        return {"error": f"exit {proc.returncode}", "sec": dt}
    m = runb.parse_metrics(full)
    m["sec"] = dt
    m.setdefault("note", "bootstrap_type 由 Poisson 改为 Bernoulli(CPU)")
    print(f"    完成 IC={m.get('IC')}, {dt:.0f}s")
    return m


def main():
    results = []
    for model, ds, cfg in JOBS:
        tag = f"{model}_{ds}"
        if tag in CATBOOST:
            m = rerun_catboost(model, ds, cfg)
        else:
            log = LOGDIR / f"{tag}.log"
            if not log.exists():
                results.append((model, ds, {"error": "log missing"})); continue
            m = runb.parse_metrics(log.read_text())
            if "IC" not in m:
                m["error"] = "no metrics parsed"
        results.append((model, ds, m))
    runb.write_report(results)


if __name__ == "__main__":
    main()
