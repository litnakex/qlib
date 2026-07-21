#  Copyright (c) Microsoft Corporation.
#  Licensed under the MIT License.


import re

import pandas as pd

import qlib
from qlib.constant import REG_CN

from qlib.utils import init_instance_by_config
from qlib.tests.data import GetData
from qlib.tests.config import CSI300_GBDT_TASK


# Human-readable meaning for each Alpha158 feature family. The key is the alphabetic
# prefix of the feature name (e.g. "ROC" for "ROC5"); the trailing number is the
# rolling window / lag in trading days.
FEATURE_GROUP_DESC = {
    # K-bar shape features
    "KMID": "K线实体相对开盘价: (收-开)/开",
    "KLEN": "K线全长相对开盘价: (高-低)/开",
    "KMID2": "K线实体占全长比例: (收-开)/(高-低)",
    "KUP": "上影线相对开盘价: (高-max(开,收))/开",
    "KUP2": "上影线占全长比例: (高-max(开,收))/(高-低)",
    "KLOW": "下影线相对开盘价: (min(开,收)-低)/开",
    "KLOW2": "下影线占全长比例: (min(开,收)-低)/(高-低)",
    "KSFT": "收盘价在当日高低区间的位置: (2收-高-低)/开",
    "KSFT2": "收盘价在当日高低区间的位置(归一化): (2收-高-低)/(高-低)",
    # Normalized OHLCV (window = lag in days, 0 = 当日)
    "OPEN": "N日前开盘价 / 当日收盘价",
    "HIGH": "N日前最高价 / 当日收盘价",
    "LOW": "N日前最低价 / 当日收盘价",
    "CLOSE": "N日前收盘价 / 当日收盘价",
    "VWAP": "N日前成交均价 / 当日收盘价",
    "VOLUME": "N日前成交量 / 当日成交量",
    # Rolling statistics (window = 回看天数)
    "ROC": "N日收益率(动量): N日前收盘价 / 当日收盘价",
    "MA": "N日收盘均价 / 当日收盘价",
    "STD": "N日收盘价标准差 / 当日收盘价 (波动率)",
    "BETA": "N日价格线性回归斜率 / 当日收盘价 (趋势斜率)",
    "RSQR": "N日价格线性回归 R^2 (趋势拟合优度)",
    "RESI": "N日价格线性回归残差 / 当日收盘价",
    "MAX": "N日最高价 / 当日收盘价",
    "MIN": "N日最低价 / 当日收盘价",
    "QTLU": "N日收盘价 80% 分位数 / 当日收盘价",
    "QTLD": "N日收盘价 20% 分位数 / 当日收盘价",
    "RANK": "当日收盘价在过去N日中的百分位排名",
    "RSV": "随机指标 RSV: (收-N日最低)/(N日最高-N日最低)",
    "IMAX": "N日内最高价距今天数 / N (高点新鲜度)",
    "IMIN": "N日内最低价距今天数 / N (低点新鲜度)",
    "IMXD": "N日内(最高价位置-最低价位置)距今差 / N",
    "CORR": "N日收盘价与对数成交量的相关系数",
    "CORD": "N日收盘价变化与成交量变化的相关系数",
    "CNTP": "N日内上涨天数占比",
    "CNTN": "N日内下跌天数占比",
    "CNTD": "N日内(上涨-下跌)天数占比",
    "SUMP": "N日内上涨总幅度占总波动比例 (类似RSI)",
    "SUMN": "N日内下跌总幅度占总波动比例",
    "SUMD": "N日内(上涨-下跌)幅度占总波动比例",
    "VMA": "N日成交量均值 / 当日成交量",
    "VSTD": "N日成交量标准差 / 当日成交量",
    "WVMA": "N日成交量加权的价格波动率",
    "VSUMP": "N日内成交量上升幅度占总变化比例",
    "VSUMN": "N日内成交量下降幅度占总变化比例",
    "VSUMD": "N日内成交量(上升-下降)幅度占总变化比例",
}


def describe_feature(name, expr):
    """Return a (meaning, expression) description for a feature name.

    ``expr`` is the exact qlib expression (the ground-truth definition). ``meaning``
    is a human-readable gloss looked up by the feature's alphabetic prefix, with the
    trailing number (rolling window / lag in trading days) filled in.
    """
    m = re.match(r"^([A-Za-z]+)(\d*)$", name)
    if m:
        prefix, num = m.group(1), m.group(2)
        desc = FEATURE_GROUP_DESC.get(prefix)
        if desc is not None:
            if num != "":
                # substitute the rolling window / lag everywhere N appears
                desc = desc.replace("N日", f"{num}日").replace("/ N", f"/ {num}").replace("过去N", f"过去{num}")
            return desc
    return "(未收录的特征, 见表达式)"


def build_feature_meaning(handler):
    """Map feature name -> (meaning, qlib expression) using the handler config."""
    fields, names = handler.get_feature_config()
    return {name: (describe_feature(name, expr), expr) for name, expr in zip(names, fields)}


if __name__ == "__main__":
    # use default data
    provider_uri = "~/.qlib/qlib_data/cn_data"  # target_dir
    GetData().qlib_data(target_dir=provider_uri, region=REG_CN, exists_skip=True)

    qlib.init(provider_uri=provider_uri, region=REG_CN)

    ###################################
    # train model
    ###################################
    # model initialization
    model = init_instance_by_config(CSI300_GBDT_TASK["model"])
    dataset = init_instance_by_config(CSI300_GBDT_TASK["dataset"])
    model.fit(dataset)

    # get model feature importance
    feature_importance = model.get_feature_importance()

    # build the name -> (meaning, expression) mapping from the dataset's handler
    meaning = build_feature_meaning(dataset.handler)

    # LightGBM names features by column order as "Column_<i>"; recover the real names.
    ordered_names = list(meaning.keys())

    def resolve_name(idx_name):
        m = re.match(r"^Column_(\d+)$", str(idx_name))
        if m:
            i = int(m.group(1))
            if 0 <= i < len(ordered_names):
                return ordered_names[i]
        return str(idx_name)

    rows = []
    for idx_name, imp in feature_importance.sort_values(ascending=False).items():
        real_name = resolve_name(idx_name)
        desc, expr = meaning.get(real_name, ("(未知特征)", "-"))
        rows.append({"feature": real_name, "importance": imp, "meaning": desc, "expression": expr})

    table = pd.DataFrame(rows)

    pd.set_option("display.max_rows", None)
    pd.set_option("display.max_colwidth", None)
    pd.set_option("display.width", 200)

    print("\nfeature importance with meaning (sorted by importance):")
    print(table.to_string(index=False))
