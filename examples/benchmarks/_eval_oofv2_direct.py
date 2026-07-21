"""对照：直接用 OOF v2 的 A段深度模型打分回测（不经 DoubleEnsemble stacking）。
A段模型 = 仅用 2008-2011 训练的 GRU/LSTM/GATs，其对 2013-2020 的同源预测存于 _oof_v2_scores.pkl。
取 test 段(2017-20)，各模型单独回测 + 三者等权 rank 融合回测。无需重训。
"""
import pandas as pd
from pathlib import Path

import qlib
from qlib.backtest import backtest as bt_backtest
from qlib.contrib.evaluate import risk_analysis

BENCH = Path(__file__).resolve().parent
V2_PKL = BENCH / "_oof_v2_scores.pkl"
BENCH_IDX = "SH000300"
# 用一个已知含 label 的 DE run 取 label 算 IC
DE_LABEL = next((BENCH / "DoubleEnsemble/mlruns/348438689375989370").glob("*/artifacts/label.pkl"))


def calc_ic(pred, label):
    df = pd.concat([pred.rename("p"), label.rename("l")], axis=1).dropna()
    ic = df.groupby(level="datetime").apply(lambda x: x["p"].corr(x["l"]))
    ric = df.groupby(level="datetime").apply(lambda x: x["p"].corr(x["l"], method="spearman"))
    return ic.mean(), ric.mean()


def cs_rank(s):
    return s.groupby(level="datetime").rank(pct=True)


def run_bt(score):
    strat = {"class": "TopkDropoutStrategy", "module_path": "qlib.contrib.strategy",
             "kwargs": {"signal": score, "topk": 50, "n_drop": 5}}
    ex = {"class": "SimulatorExecutor", "module_path": "qlib.backtest.executor",
          "kwargs": {"time_per_step": "day", "generate_portfolio_metrics": True}}
    cfg = {"start_time": "2017-01-01", "end_time": "2020-08-01", "account": 100000000,
           "benchmark": BENCH_IDX,
           "exchange_kwargs": {"limit_threshold": 0.095, "deal_price": "close",
                               "open_cost": 0.0005, "close_cost": 0.0015, "min_cost": 5}}
    pmd, _ = bt_backtest(executor=ex, strategy=strat, **cfg)
    rn, _ = pmd["1day"]
    ex_ret = rn["return"] - rn["bench"]
    r_wo = risk_analysis(ex_ret, freq="day")["risk"]
    r_w = risk_analysis(ex_ret - rn["cost"], freq="day")["risk"]
    return r_wo, r_w


def main():
    qlib.init(provider_uri="~/.qlib/qlib_data/cn_data", region="cn")
    v2 = pd.read_pickle(V2_PKL)
    label = pd.read_pickle(DE_LABEL)
    if isinstance(label, pd.DataFrame):
        label = label.iloc[:, 0]

    # test 段
    cols = {}
    for name in ["GRU", "LSTM", "GATs"]:
        cols[name] = v2[name].loc["2017-01-01":"2020-08-01"]
    # 三者等权 rank 融合
    common = cols["GRU"].index
    for c in ["LSTM", "GATs"]:
        common = common.intersection(cols[c].index)
    fused = sum(cs_rank(cols[c].reindex(common)) for c in ["GRU", "LSTM", "GATs"]) / 3
    cols["EqualFuse(GRU+LSTM+GATs)"] = fused

    print("A段模型(仅2008-2011训练)直接回测 — test段2017-2020")
    print("=" * 80)
    for name, sc in cols.items():
        lab = label.reindex(sc.index)
        ic, ric = calc_ic(sc, lab)
        r_wo, r_w = run_bt(sc)
        print(f"{name:26s} IC={ic:.4f} RankIC={ric:.4f} | "
              f"含成本年化={r_w['annualized_return']:.4f} IR={r_w['information_ratio']:.4f} 回撤={r_w['max_drawdown']:.4f} "
              f"| 不含成本年化={r_wo['annualized_return']:.4f}")


if __name__ == "__main__":
    main()
