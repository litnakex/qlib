"""Export all limit-up (涨停) records plus their follow-up windows.

For the whole A-share universe under ~/.qlib/qlib_data/cn_data, find every trading day
T on which a stock hit the daily price limit (涨停, ``$change >= threshold``) and emit a
table containing:

- the limit-up day T itself, and
- the following observation days T+1, T+2, T+3.

Chained extension: if T+1 is itself a limit-up, the window is re-anchored on it, so its
own T+1..T+3 are also emitted. Equivalently, a row is kept iff its distance (in trading
days) to the most recent limit-up at/before it is <= FOLLOW_DAYS. Consecutive limit-ups
therefore keep extending the window, and each (instrument, date) appears at most once
(natural de-duplication).

The full table is written to ``limitup_records.csv``; a human-readable summary with a
preview of the first rows is written to ``limitup_records.md``.

Notes on the data:
- qlib stores forward-adjusted prices (open/high/low/close, and volume divided by
  factor). ``$change`` is the unadjusted daily pct change (raw close-to-close move),
  which is what defines the real +10% limit, so limit-up detection uses ``$change``.
- ``$vwap`` does not exist in this dataset; OHLCV + change are used.
"""

import argparse

import pandas as pd

import qlib
from qlib.constant import REG_CN
from qlib.data import D


# A normal-board 涨停 is +10%; rounding of the daily close makes the realized
# ``change`` land a hair below/above 0.10, so we accept anything at/above this.
LIMIT_UP_THRESHOLD = 0.095

# Number of trading days after a limit-up to emit as follow-up observation rows.
FOLLOW_DAYS = 3

# How many rows of the full table to preview inside the .md file.
MD_PREVIEW_ROWS = 100


def build_records(df, threshold):
    """Return the de-duplicated record table (one row per kept (instrument, date))."""
    parts = []
    for inst, g in df.groupby(level="instrument"):
        g = g.droplevel("instrument").sort_index()
        is_lu = (g["change"] >= threshold).to_numpy()

        # Distance (in trading days) to the most recent limit-up at/before each row.
        # dist == 0 -> the limit-up day itself; 1..FOLLOW_DAYS -> follow-up window.
        dist = []
        d = None  # None until the first limit-up appears
        for lu in is_lu:
            if lu:
                d = 0
            elif d is not None:
                d += 1
            dist.append(d if d is not None else -1)

        sub = g.copy()
        sub["instrument"] = inst
        sub["days_since_limit_up"] = dist
        sub["is_limit_up"] = is_lu
        # Keep the limit-up day and its follow-up window (chained via dist reset).
        sub = sub[(sub["days_since_limit_up"] >= 0) & (sub["days_since_limit_up"] <= FOLLOW_DAYS)]
        if not sub.empty:
            parts.append(sub)

    if not parts:
        return pd.DataFrame()

    out = pd.concat(parts)
    out.index.name = "date"
    out = out.reset_index()
    # date ascending, then instrument lexicographic ascending.
    out = out.sort_values(["date", "instrument"], kind="mergesort").reset_index(drop=True)
    out["date"] = pd.to_datetime(out["date"]).dt.strftime("%Y-%m-%d")

    cols = [
        "instrument",
        "date",
        "is_limit_up",
        "days_since_limit_up",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "change",
    ]
    return out[cols]


def df_to_markdown(df):
    """Render a DataFrame as a GitHub-flavored markdown table (no external deps)."""
    cols = list(df.columns)
    header = "| " + " | ".join(cols) + " |"
    sep = "| " + " | ".join("---" for _ in cols) + " |"
    rows = []
    for _, r in df.iterrows():
        cells = []
        for c in cols:
            v = r[c]
            if c == "volume" and isinstance(v, float):
                cells.append(f"{v:.0f}")
            elif isinstance(v, float):
                cells.append(f"{v:.4f}")
            else:
                cells.append(str(v))
        rows.append("| " + " | ".join(cells) + " |")
    return "\n".join([header, sep] + rows)


def write_markdown(md_path, table, threshold, n_limit_up):
    n_total = len(table)
    preview = table.head(MD_PREVIEW_ROWS)
    lines = []
    lines.append("# 涨停记录及后续窗口明细")
    lines.append("")
    lines.append("## 说明")
    lines.append("")
    lines.append(
        f"- 数据源：`~/.qlib/qlib_data/cn_data`（A股全市场，日频，前复权价格）。"
    )
    lines.append(
        f"- 涨停判定：日涨跌幅 `change >= {threshold:.4f}`（约 +10% 涨停，含四舍五入容差；不含 ST 的 ±5%）。"
    )
    lines.append(
        f"- 对每个涨停日 T，输出 T 及其后 {FOLLOW_DAYS} 个交易日（T+1..T+{FOLLOW_DAYS}）；"
        f"若后续日本身也涨停，则以其为新的 T **链式顺延**窗口。"
    )
    lines.append("- 同一 (股票, 日期) 只出现一行（去重合并）。")
    lines.append("- 排序：日期升序；同一日期内股票代号字典序升序。")
    lines.append("")
    lines.append("### 字段")
    lines.append("")
    lines.append("| 列 | 含义 |")
    lines.append("|---|---|")
    lines.append("| `instrument` | 股票代号 |")
    lines.append("| `date` | 交易日 |")
    lines.append("| `is_limit_up` | 当日是否涨停 |")
    lines.append(
        "| `days_since_limit_up` | 距最近一次涨停的交易日数（0=涨停日本身，1..N=后续观察日） |"
    )
    lines.append("| `open`/`high`/`low`/`close` | 前复权 OHLC |")
    lines.append("| `volume` | 成交量（前复权口径） |")
    lines.append("| `change` | 日涨跌幅 |")
    lines.append("")
    lines.append("## 统计")
    lines.append("")
    lines.append(f"- 涨停日总数：{n_limit_up}")
    lines.append(f"- 表格总行数（含后续观察行）：{n_total}")
    lines.append(f"- 完整数据见同目录 `limitup_records.csv`。")
    lines.append("")
    lines.append(f"## 预览（前 {len(preview)} 行）")
    lines.append("")
    lines.append(df_to_markdown(preview))
    lines.append("")

    with open(md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


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
    parser.add_argument("--csv", default="limitup_records.csv", help="output CSV path")
    parser.add_argument("--md", default="limitup_records.md", help="output markdown path")
    args = parser.parse_args()

    qlib.init(provider_uri=args.provider_uri, region=REG_CN)

    instruments = D.instruments(market="all")
    fields = ["$open", "$high", "$low", "$close", "$volume", "$change"]
    df = D.features(instruments, fields, freq="day")
    df = df.rename(columns={c: c[1:] for c in df.columns})  # strip leading '$'

    table = build_records(df, args.threshold)
    n_limit_up = int(table["is_limit_up"].sum()) if not table.empty else 0

    table.to_csv(args.csv, index=False)
    write_markdown(args.md, table, args.threshold, n_limit_up)

    print(f"涨停日总数: {n_limit_up}")
    print(f"表格总行数: {len(table)}")
    print(f"CSV 已保存: {args.csv}")
    print(f"Markdown 已保存: {args.md}")


if __name__ == "__main__":
    main()
