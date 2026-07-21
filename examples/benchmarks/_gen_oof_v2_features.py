"""OOF stacking v2（B段训练方案）：
A段模型(2008-2011训练)对 2013-2020 全段做同源预测，作为深度打分特征。
之后 DoubleEnsemble 用 train=2013-14 / valid=2015-16 训练，test=2017-20 回测。

关键：train/valid/test 的深度打分全部来自同一个 A段模型 → 分布对齐、无泄露
（A段模型没见过 2013 年之后的任何数据）。

输出 _oof_v2_scores.pkl（列 GRU/LSTM/GATs，覆盖 2013-01 ~ 2020-08）。
"""
import os
from pathlib import Path

import pandas as pd
import ruamel.yaml as yaml

import qlib
from qlib.utils import init_instance_by_config
from qlib.workflow import R

BENCH = Path(__file__).resolve().parent
OUT = BENCH / "_oof_v2_scores.pkl"

# A段模型：train=2008-2011, valid=2012；test 段设为 2013-01~2020-08（用命名段 test 一次性预测全段，避免 slice bug）
A_SEGMENTS = {
    "train": ["2008-01-01", "2011-12-31"],
    "valid": ["2012-01-01", "2012-12-31"],
    "test": ["2013-01-01", "2020-08-01"],
}

MODELS = [
    ("GRU", "GRU/workflow_config_gru_Alpha360.yaml"),
    ("LSTM", "LSTM/workflow_config_lstm_Alpha360.yaml"),
    ("GATs", "GATs/workflow_config_gats_Alpha360.yaml"),
]


def load_yaml(p):
    with open(p) as f:
        return yaml.YAML(typ="safe", pure=True).load(f)


def main():
    qlib.init(provider_uri="~/.qlib/qlib_data/cn_data", region="cn")
    scores = {}
    for name, cfg_rel in MODELS:
        conf = load_yaml(BENCH / cfg_rel)
        task = conf["task"]
        task["model"].setdefault("kwargs", {})
        task["model"]["kwargs"]["n_jobs"] = 0
        mp = task["model"]["kwargs"].get("model_path")
        if mp and not os.path.isabs(mp):
            task["model"]["kwargs"]["model_path"] = str((BENCH.parent / mp).resolve())
        task["dataset"]["kwargs"]["segments"] = A_SEGMENTS

        print(f"[{name}] 构造 A段(2008-2011) model & dataset ...", flush=True)
        model = init_instance_by_config(task["model"])
        dataset = init_instance_by_config(task["dataset"])
        with R.start(experiment_name=f"oof_v2_gen_{name}"):
            print(f"[{name}] 训练 A段模型 ...", flush=True)
            model.fit(dataset)
            print(f"[{name}] 同源预测 2013-2020 全段(命名段 test) ...", flush=True)
            p = model.predict(dataset, segment="test")
            if isinstance(p, pd.DataFrame):
                p = p.iloc[:, 0]
        scores[name] = p.sort_index()
        print(f"[{name}] 预测点数={len(p)}", flush=True)

    df = pd.concat(scores, axis=1)
    df.columns = list(scores.keys())
    df = df.sort_index()
    df = df[~df.index.duplicated(keep="last")]
    df.to_pickle(OUT)
    print(f"\n已保存 OOF v2 特征: {OUT}")
    print("时间范围:", df.index.get_level_values("datetime").min(),
          "→", df.index.get_level_values("datetime").max())
    print("(train=2013-14/valid=2015-16/test=2017-20 的打分全部同源自 A段模型)")


if __name__ == "__main__":
    main()
