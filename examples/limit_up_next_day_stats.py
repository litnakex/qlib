"""Statistics on the day after a limit-up (涨停).

For every stock in ~/.qlib/qlib_data/cn_data, find all trading days T on which the
stock hit the daily price limit (涨停), then compute the fraction of those days for
which the NEXT trading day (T+1) "continued to rise", defined as:

    T+1 open  >= T   close      (opens at or above yesterday's close)
    AND
    T+1 close >  T+1 open       (closes above its own open, i.e. a green candle)

Notes on the data:
- qlib stores forward-adjusted prices (open/close/high/low have been multiplied by
  ``factor``). Cross-day price comparisons are therefore consistent as long as every
  price we compare is adjusted, which is the case here. We do NOT need to un-adjust.
- Limit-up detection uses the ``change`` field (unadjusted daily pct change) rather
  than adjusted prices, because the real price limit (+10% for normal boards,
  +5% for ST) is defined on the raw close-to-close move. We use a threshold slightly
  below 10% to tolerate rounding (an exact 涨停 lands at ~0.0995-0.1001).
"""

import argparse

import numpy as np
import pandas as pd

import qlib
from qlib.constant import REG_CN
from qlib.data import D


# A normal-board 涨停 is +10%. Rounding of the daily close makes the realized
# ``change`` land a hair below/above 0.10, so we accept anything at/above this.
LIMIT_UP_THRESHOLD = 0.095


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--provider_uri",
        default="~/.qlib/qlib_data/cn_data",
        help="qlib data directory",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=LIMIT_UP_THRESHOLD,
        help="min daily 'change' to count as 涨停 (default 0.095 ~= +10%%)",
    )
    args = parser.parse_args()

    qlib.init(provider_uri=args.provider_uri, region=REG_CN)

    instruments = D.instruments(market="all")
    fields = ["$open", "$close", "$change"]

    # Load everything at once; index is (instrument, datetime).
    df = D.features(instruments, fields, freq="day")
    df = df.rename(columns={"$open": "open", "$close": "close", "$change": "change"})

    total_limit_up = 0
    total_continue = 0
    per_stock_records = []

    # Overall counters (over ALL stock-days that have a valid T+1).
    total_valid_days = 0      # every trading day T that has a next day T+1
    total_next_up = 0         # of those, days whose next day "rose"
    total_all_limit_up = 0    # of those, days that were 涨停 (regardless of T+1 outcome)

    # Consecutive-limit-up counters. Anchor day = T (the first limit-up of the streak).
    total_2lu = 0             # streaks where T & T+1 are both 涨停 (and T+2 exists)
    total_2lu_up = 0          #   of those, T+2 "continued to rise"
    total_3lu = 0             # streaks where T, T+1, T+2 are all 涨停 (and T+3 exists)
    total_3lu_up = 0          #   of those, T+3 "continued to rise"

    for inst, g in df.groupby(level="instrument"):
        g = g.droplevel("instrument").sort_index()
        # Next-day open/close aligned onto the current row.
        next_open = g["open"].shift(-1)
        next_close = g["close"].shift(-1)

        is_limit_up = g["change"] >= args.threshold
        # A day T is usable if it has a valid current close and a valid T+1.
        has_next = next_open.notna() & next_close.notna() & g["close"].notna()

        # "次日上涨": same rule as the continue-rise definition, applied to every day.
        next_up = has_next & (next_open >= g["close"]) & (next_close > next_open)

        total_valid_days += int(has_next.sum())
        total_next_up += int((has_next & next_up).sum())
        total_all_limit_up += int((has_next & is_limit_up).sum())

        # ---- Consecutive limit-up streaks (anchored at T) ----
        # Limit-up flags for T, T+1, T+2 (aligned onto row T).
        lu_t = is_limit_up
        lu_t1 = is_limit_up.shift(-1)
        lu_t2 = is_limit_up.shift(-2)
        # Close on the LAST limit-up day, and open/close of the day AFTER it.
        close_t1 = g["close"].shift(-1)   # close of T+1
        open_t2 = g["open"].shift(-2)     # open of T+2
        close_t2 = g["close"].shift(-2)   # close of T+2
        open_t3 = g["open"].shift(-3)     # open of T+3
        close_t3 = g["close"].shift(-3)   # close of T+3

        # 2 consecutive limit-ups (T, T+1), evaluate continuation on T+2.
        two = (lu_t == True) & (lu_t1 == True) & open_t2.notna() & close_t2.notna() & close_t1.notna()
        two_up = two & (open_t2 >= close_t1) & (close_t2 > open_t2)
        total_2lu += int(two.sum())
        total_2lu_up += int(two_up.sum())

        # 3 consecutive limit-ups (T, T+1, T+2), evaluate continuation on T+3.
        # The last limit-up close here is close_t2.
        three = (
            (lu_t == True) & (lu_t1 == True) & (lu_t2 == True)
            & open_t3.notna() & close_t3.notna() & close_t2.notna()
        )
        three_up = three & (open_t3 >= close_t2) & (close_t3 > open_t3)
        total_3lu += int(three.sum())
        total_3lu_up += int(three_up.sum())

        # Limit-up subset (T is 涨停 and has a T+1).
        valid = is_limit_up & has_next
        continued = valid & (next_open >= g["close"]) & (next_close > next_open)

        n_lu = int(valid.sum())
        n_cont = int(continued.sum())
        if n_lu > 0:
            total_limit_up += n_lu
            total_continue += n_cont
            per_stock_records.append(
                {"instrument": inst, "limit_up_days": n_lu, "continue_days": n_cont}
            )

    print("=" * 60)
    print("涨停次日继续上涨统计")
    print("=" * 60)
    print(f"涨停判定阈值 (change >=): {args.threshold:.4f}")
    print(f"覆盖股票数 (至少有一次涨停): {len(per_stock_records)}")
    print(f"涨停总天数 (T, 且存在 T+1): {total_limit_up}")
    print(f"次日继续上涨天数:            {total_continue}")
    if total_limit_up > 0:
        ratio = total_continue / total_limit_up
        print(f"次日继续上涨占比:            {ratio:.4f}  ({ratio * 100:.2f}%)")
    else:
        print("未找到任何涨停样本，请检查数据或阈值。")

    print("\n" + "=" * 60)
    print("全样本统计 (所有股票的所有交易日, 需存在次日 T+1)")
    print("=" * 60)
    print(f"有效交易日总数 (T, 且存在 T+1): {total_valid_days}")
    if total_valid_days > 0:
        r_up = total_next_up / total_valid_days
        r_lu = total_all_limit_up / total_valid_days
        print(f"1. 次日上涨天数: {total_next_up}  占比: {r_up:.4f}  ({r_up * 100:.2f}%)")
        print(f"2. 涨停天数:     {total_all_limit_up}  占比: {r_lu:.4f}  ({r_lu * 100:.2f}%)")

    print("\n" + "=" * 60)
    print("连续涨停后继续上涨统计")
    print("=" * 60)
    print(f"1. 连续2天涨停(T,T+1)的样本数: {total_2lu}")
    if total_2lu > 0:
        r2 = total_2lu_up / total_2lu
        print(f"   其中 T+2 继续上涨: {total_2lu_up}  占比: {r2:.4f}  ({r2 * 100:.2f}%)")
    print(f"2. 连续3天涨停(T,T+1,T+2)的样本数: {total_3lu}")
    if total_3lu > 0:
        r3 = total_3lu_up / total_3lu
        print(f"   其中 T+3 继续上涨: {total_3lu_up}  占比: {r3:.4f}  ({r3 * 100:.2f}%)")

    if per_stock_records:
        per_stock = pd.DataFrame(per_stock_records).set_index("instrument")
        per_stock["continue_ratio"] = per_stock["continue_days"] / per_stock["limit_up_days"]
        out_path = "limit_up_next_day_stats_per_stock.csv"
        per_stock.sort_values("limit_up_days", ascending=False).to_csv(out_path)
        print(f"\n每只股票明细已保存: {out_path}")


if __name__ == "__main__":
    main()
