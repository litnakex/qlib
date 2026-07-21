"""生成 stacking 特征：用 GRU/LSTM/GATs 在 Alpha360 上训练，
对全时段(2008-2020)预测打分，合并成 3 列特征，存成 pickle 供 DoubleEnsemble 使用。

NOTE(信息泄露): 训练段的预测是 in-sample 的（模型在训练段训练过），
存在乐观偏差。严格 stacking 应用 out-of-fold 预测。这里作为对比实验直接用全时段预测，
并在 report 结论里注明该 caveat。

输出: _stacking_scores.pkl —— DataFrame, MultiIndex(datetime, instrument), 列=[GRU,LSTM,GATs]
"""
import os
from pathlib import Path

import pandas as pd
import ruamel.yaml as yaml

import qlib
from qlib.utils import init_instance_by_config
from qlib.workflow import R

BENCH = Path(__file__).resolve().parent
OUT = BENCH / "_stacking_scores.pkl"

# 全时段：训练/验证/测试都要有预测（DoubleEnsemble 训练时也需要这些特征列）
FULL_SEGMENTS = {
    "train": ["2008-01-01", "2014-12-31"],
    "valid": ["2015-01-01", "2016-12-31"],
    "test": ["2017-01-01", "2020-08-01"],
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
        cfg_path = BENCH / cfg_rel
        conf = load_yaml(cfg_path)
        task = conf["task"]
        # 强制 n_jobs=0 规避 macOS 多进程卡死
        task["model"].setdefault("kwargs", {})
        task["model"]["kwargs"]["n_jobs"] = 0
        # GATs 的 model_path 相对 examples/ → 转绝对
        mp = task["model"]["kwargs"].get("model_path")
        if mp and not os.path.isabs(mp):
            task["model"]["kwargs"]["model_path"] = str((BENCH.parent / mp).resolve())
        # dataset：把 segments 覆盖为全时段
        task["dataset"]["kwargs"]["segments"] = FULL_SEGMENTS

        print(f"[{name}] 构造 model & dataset ...", flush=True)
        model = init_instance_by_config(task["model"])
        dataset = init_instance_by_config(task["dataset"])

        # fit/predict 需要活动的 recorder（qlib pytorch 模型 fit 末尾会调 R.get_recorder 记录指标）
        seg_preds = []
        with R.start(experiment_name=f"stacking_gen_{name}"):
            print(f"[{name}] 训练 ...", flush=True)
            model.fit(dataset)
            # 对每个 segment 预测，拼成全时段
            for seg in ["train", "valid", "test"]:
                print(f"[{name}] 预测 {seg} ...", flush=True)
                p = model.predict(dataset, segment=seg)
                if isinstance(p, pd.DataFrame):
                    p = p.iloc[:, 0]
                seg_preds.append(p)
        full = pd.concat(seg_preds).sort_index()
        full = full[~full.index.duplicated(keep="last")]
        scores[name] = full
        print(f"[{name}] 完成，预测点数={len(full)}", flush=True)

    df = pd.concat(scores, axis=1)  # 列 = MultiIndex? 用 dict 会形成列名
    df.columns = list(scores.keys())  # 展平成 GRU/LSTM/GATs
    df = df.sort_index()
    df.to_pickle(OUT)
    print(f"\n已保存 stacking 特征: {OUT}")
    print(df.describe())
    print("样例:\n", df.head())


if __name__ == "__main__":
    main()
