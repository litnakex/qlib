"""策略层调参(#6)：固定 w=0.5 的 GATs×DoubleEnsemble 融合分数，
网格扫 topk × n_drop，看能否进一步提升含成本收益 / 降回撤。无需重训。
额外记录平均日换手，佐证"降 n_drop 省成本"。
"""
import pandas as pd
from pathlib import Path

import qlib
from qlib.backtest import backtest as bt_backtest
from qlib.contrib.evaluate import risk_analysis

BENCH = Path(__file__).resolve().parent
STACKING_PKL = BENCH / "_stacking_scores.pkl"
DE_RUNS = list((BENCH / "DoubleEnsemble/mlruns/348438689375989370").glob("*/artifacts/pred.pkl"))
BENCH_IDX = "SH000300"

TOPKS = [30, 50, 100]
NDROPS = [2, 5, 10]


def load_series(pkl):
    s = pd.read_pickle(pkl)
    return s.iloc[:, 0] if isinstance(s, pd.DataFrame) else s


def calc_ic(pred, label):
    df = pd.concat([pred.rename("p"), label.rename("l")], axis=1).dropna()
    ic = df.groupby(level="datetime").apply(lambda x: x["p"].corr(x["l"]))
    return ic.mean()


def cs_rank(s):
    return s.groupby(level="datetime").rank(pct=True)


def run_backtest(score, topk, n_drop):
    strat = {"class": "TopkDropoutStrategy", "module_path": "qlib.contrib.strategy",
             "kwargs": {"signal": score, "topk": topk, "n_drop": n_drop}}
    ex = {"class": "SimulatorExecutor", "module_path": "qlib.backtest.executor",
          "kwargs": {"time_per_step": "day", "generate_portfolio_metrics": True}}
    cfg = {"start_time": "2017-01-01", "end_time": "2020-08-01", "account": 100000000,
           "benchmark": BENCH_IDX,
           "exchange_kwargs": {"limit_threshold": 0.095, "deal_price": "close",
                               "open_cost": 0.0005, "close_cost": 0.0015, "min_cost": 5}}
    pmd, _ = bt_backtest(executor=ex, strategy=strat, **cfg)
    rn, _ = pmd["1day"]
    excess = rn["return"] - rn["bench"]
    r_wo = risk_analysis(excess, freq="day")["risk"]
    r_w = risk_analysis(excess - rn["cost"], freq="day")["risk"]
    turnover = rn["turnover"].mean()
    return r_wo, r_w, turnover


def main():
    qlib.init(provider_uri="~/.qlib/qlib_data/cn_data", region="cn")
    # 组装 w=0.5 融合分数
    gats = pd.read_pickle(STACKING_PKL)["GATs"].loc["2017-01-01":"2020-08-01"]
    de_pred = de_label = None
    for run in DE_RUNS:
        p = load_series(run).loc["2017-01-01":"2020-08-01"]
        lab = load_series(run.parent / "label.pkl")
        if 0.045 < calc_ic(p, lab.reindex(p.index)) < 0.058:
            de_pred, de_label = p, lab
    common = de_pred.index.intersection(gats.index)
    fused = 0.5 * cs_rank(de_pred.reindex(common)) + 0.5 * cs_rank(gats.reindex(common))
    lab_c = de_label.reindex(common)

    print(f"融合分数样本点: {len(common)}  (w=0.5, GATs×DoubleEnsemble)")
    print("=" * 90)
    ic = calc_ic(fused, lab_c)
    results = []
    for topk in TOPKS:
        for n_drop in NDROPS:
            r_wo, r_w, turn = run_backtest(fused, topk, n_drop)
            results.append({"topk": topk, "n_drop": n_drop, "IC": ic,
                            "ann_wo": r_wo["annualized_return"], "ir_wo": r_wo["information_ratio"], "mdd_wo": r_wo["max_drawdown"],
                            "ann_w": r_w["annualized_return"], "ir_w": r_w["information_ratio"], "mdd_w": r_w["max_drawdown"],
                            "turnover": turn})
            print(f"topk={topk:3d} n_drop={n_drop:2d} | 含成本年化={r_w['annualized_return']:.4f} "
                  f"IR={r_w['information_ratio']:.4f} 回撤={r_w['max_drawdown']:.4f} 日均换手={turn:.4f}")
    df = pd.DataFrame(results)
    df.to_pickle(BENCH / "_tune_results.pkl")
    best = df.loc[df["ann_w"].idxmax()]
    print("\n最佳(含成本年化): topk=%d n_drop=%d 年化=%.4f IR=%.4f 回撤=%.4f"
          % (best["topk"], best["n_drop"], best["ann_w"], best["ir_w"], best["mdd_w"]))
    print("结果已存 _tune_results.pkl")


if __name__ == "__main__":
    main()
