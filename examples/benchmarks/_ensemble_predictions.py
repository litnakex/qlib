"""预测层融合实验(#2)：把 GATs(Alpha360) 与 DoubleEnsemble(Alpha158) 的 test 段预测融合，
再回测对比。无需重训——直接复用已有预测。

融合方式：
  - rank 融合：各自转横截面百分位(按天)再加权平均，尺度无关，最稳健
  - 试多个权重 w(GATs 占比): 0.0(纯DE) / 0.3 / 0.5 / 0.7 / 1.0(纯GATs)
对每个融合分数：算 IC / Rank IC，并用 qlib backtest 跑 TopkDropout 回测得含/不含成本年化、IR、回撤。
"""
import pandas as pd
import numpy as np
from pathlib import Path

import qlib
from qlib.backtest import backtest as bt_backtest
from qlib.contrib.evaluate import risk_analysis

BENCH = Path(__file__).resolve().parent

# 原始 DoubleEnsemble/Alpha158 的 pred（experiment=workflow）。有两个 run，脚本会自动挑 IC≈0.0517 的那个。
DE_RUNS = list((BENCH / "DoubleEnsemble/mlruns/348438689375989370").glob("*/artifacts/pred.pkl"))
STACKING_PKL = BENCH / "_stacking_scores.pkl"

BENCH_IDX = "SH000300"


def load_series(pkl):
    s = pd.read_pickle(pkl)
    if isinstance(s, pd.DataFrame):
        s = s.iloc[:, 0]
    return s


def calc_ic(pred, label):
    df = pd.concat([pred.rename("p"), label.rename("l")], axis=1).dropna()
    ic = df.groupby(level="datetime").apply(lambda x: x["p"].corr(x["l"]))
    ric = df.groupby(level="datetime").apply(lambda x: x["p"].corr(x["l"], method="spearman"))
    return ic.mean(), ric.mean()


def cs_rank(s):
    # 横截面百分位(按天)，0~1
    return s.groupby(level="datetime").rank(pct=True)


def run_backtest(score):
    """用 TopkDropout(50,5) 对 score 回测，返回 (含成本, 不含成本) 的 risk_analysis dict。"""
    strategy_config = {
        "class": "TopkDropoutStrategy",
        "module_path": "qlib.contrib.strategy",
        "kwargs": {"signal": score, "topk": 50, "n_drop": 5},
    }
    executor_config = {
        "class": "SimulatorExecutor",
        "module_path": "qlib.backtest.executor",
        "kwargs": {"time_per_step": "day", "generate_portfolio_metrics": True},
    }
    backtest_config = {
        "start_time": "2017-01-01", "end_time": "2020-08-01",
        "account": 100000000, "benchmark": BENCH_IDX,
        "exchange_kwargs": {"limit_threshold": 0.095, "deal_price": "close",
                            "open_cost": 0.0005, "close_cost": 0.0015, "min_cost": 5},
    }
    portfolio_metric_dict, _ = bt_backtest(
        executor=executor_config, strategy=strategy_config, **backtest_config
    )
    report_normal, _ = portfolio_metric_dict.get("1day")
    # 超额收益(相对基准)
    ret = report_normal["return"] - report_normal["bench"]
    ret_wo = ret  # without cost
    cost = report_normal["cost"]
    ret_w = ret - cost  # with cost
    r_wo = risk_analysis(ret_wo, freq="day")["risk"]
    r_w = risk_analysis(ret_w, freq="day")["risk"]
    return r_wo, r_w


def main():
    qlib.init(provider_uri="~/.qlib/qlib_data/cn_data", region="cn")

    # 1) 加载 GATs test 预测
    stack = pd.read_pickle(STACKING_PKL)
    gats = stack["GATs"].loc["2017-01-01":"2020-08-01"]

    # 2) 加载 DoubleEnsemble/Alpha158 pred + label，挑 IC≈0.0517 的 run
    de_pred = de_label = None
    for run in DE_RUNS:
        p = load_series(run)
        lab = load_series(run.parent / "label.pkl")
        p_test = p.loc["2017-01-01":"2020-08-01"]
        ic, _ = calc_ic(p_test, lab.reindex(p_test.index))
        print(f"候选 DE run {run.parent.parent.name[:8]} IC={ic:.4f}")
        if 0.045 < ic < 0.058:  # 匹配原始 Alpha158 的 IC 量级
            de_pred, de_label = p_test, lab
    if de_pred is None:
        print("!! 未找到匹配的 DoubleEnsemble/Alpha158 pred，用第一个")
        de_pred = load_series(DE_RUNS[0]).loc["2017-01-01":"2020-08-01"]
        de_label = load_series(DE_RUNS[0].parent / "label.pkl")

    label = de_label.reindex(de_pred.index)

    # 对齐两个预测的共同索引
    common = de_pred.index.intersection(gats.index)
    de_r = cs_rank(de_pred.reindex(common))
    gats_r = cs_rank(gats.reindex(common))
    lab_c = label.reindex(common)

    print(f"\n共同样本点: {len(common)}")
    print("=" * 70)
    results = []
    for w in [0.0, 0.3, 0.5, 0.7, 1.0]:
        fused = (1 - w) * de_r + w * gats_r
        ic, ric = calc_ic(fused, lab_c)
        r_wo, r_w = run_backtest(fused)
        row = {
            "w_GATs": w, "IC": ic, "RankIC": ric,
            "ann_wo": r_wo["annualized_return"], "ir_wo": r_wo["information_ratio"], "mdd_wo": r_wo["max_drawdown"],
            "ann_w": r_w["annualized_return"], "ir_w": r_w["information_ratio"], "mdd_w": r_w["max_drawdown"],
        }
        results.append(row)
        print(f"w(GATs)={w:.1f} | IC={ic:.4f} RankIC={ric:.4f} | "
              f"含成本年化={r_w['annualized_return']:.4f} IR={r_w['information_ratio']:.4f} 回撤={r_w['max_drawdown']:.4f}")

    pd.DataFrame(results).to_pickle(BENCH / "_ensemble_results.pkl")
    print("\n结果已存 _ensemble_results.pkl")


if __name__ == "__main__":
    main()
