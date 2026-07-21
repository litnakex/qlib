"""OOF stacking 特征生成(#1，单次 walk-forward 切分)。

训练段(2008-2014)拆成:
  A 段 2008-2012 —— 无更早数据，深度打分置 NaN（gbm 可处理）
  B 段 2013-2014 —— 用 A 段(2008-2012)训练的深度模型 OOF 预测（无泄露）
valid(2015-16)/test(2017-20) —— 复用现有 _stacking_scores.pkl（全训练段模型预测，本就无泄露）

只需新训 3 个"A段模型"预测 B 段。输出 _oof_scores.pkl（列 GRU/LSTM/GATs）。
"""
import os
from pathlib import Path

import pandas as pd
import ruamel.yaml as yaml

import qlib
from qlib.utils import init_instance_by_config
from qlib.workflow import R

BENCH = Path(__file__).resolve().parent
OUT = BENCH / "_oof_scores.pkl"
FULL_PKL = BENCH / "_stacking_scores.pkl"  # 已有的全训练段打分，供 valid/test 复用

# A 段模型的 segments：用标准三段名 train/valid/test（qlib 某些 fetch 路径对标准段名更友好）。
# test 段 = B 段(2013-2014)，即我们要的 OOF 预测区间；predict 时用命名段 "test"。
A_SEGMENTS = {
    "train": ["2008-01-01", "2011-12-31"],
    "valid": ["2012-01-01", "2012-12-31"],
    "test": ["2013-01-01", "2014-12-31"],
}
# B 段的时间范围（训练段后段，用 OOF 打分）
B_START, B_END = "2013-01-01", "2014-12-31"

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
    oof_scores = {}
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
        with R.start(experiment_name=f"oof_gen_{name}"):
            print(f"[{name}] 训练 A段模型 ...", flush=True)
            model.fit(dataset)
            print(f"[{name}] OOF 预测 B段(2013-2014, 命名段 test) ...", flush=True)
            # 用标准命名段 "test"（=B段），避免 date-string slice 触发的 pandas IndexSlice bug
            p = model.predict(dataset, segment="test")
            if isinstance(p, pd.DataFrame):
                p = p.iloc[:, 0]
        oof_scores[name] = p.sort_index()
        print(f"[{name}] B段 OOF 预测点数={len(p)}", flush=True)

    oof_df = pd.concat(oof_scores, axis=1)
    oof_df.columns = list(oof_scores.keys())

    # 组装完整 OOF 版特征：
    #   B段(2013-2014) = OOF 打分
    #   valid/test(2015+) = 复用全训练段打分
    #   A段(2008-2012) = 留空(NaN)，不显式填充
    full = pd.read_pickle(FULL_PKL)
    valid_test = full.loc["2015-01-01":]          # 复用
    b_seg = oof_df.loc[B_START:B_END]             # OOF
    combined = pd.concat([b_seg, valid_test]).sort_index()
    combined = combined[~combined.index.duplicated(keep="last")]
    combined.to_pickle(OUT)
    print(f"\n已保存 OOF 特征: {OUT}")
    print("时间范围:", combined.index.get_level_values("datetime").min(),
          "→", combined.index.get_level_values("datetime").max())
    print("(注: 2008-2012 A段无 OOF 打分，DoubleEnsemble 训练时该段深度特征为 NaN)")
    print(combined.describe())


if __name__ == "__main__":
    main()
