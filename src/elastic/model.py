"""Own-price elasticity of demand, from naive to double machine learning.

log(units) is regressed on log(price) for the product-week panel, step by step:

1. pooled OLS: compares cheap and expensive products, not price changes;
2. product and week fixed effects: only a product's price moving against its
   own usual level, net of common weekly shocks (holidays, panel size);
3. plus promotion controls: weeks on sale are often also weeks on display or in
   the mailer, which lift sales on their own;
4. plus last week's price: a deep discount pulls purchases forward, and the
   week after sells less (pantry loading);
5. double machine learning (partially linear model, Chernozhukov et al. 2018):
   within-product deviations of log units and log price are residualised on
   promotions, lags, seasonality and category with cross-fitted LightGBM,
   folds grouped by product. The elasticity is the slope between residuals.

Standard errors are clustered by product.
"""
from __future__ import annotations

import json
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from sklearn.model_selection import GroupKFold


def load() -> pd.DataFrame:
    d = pd.read_parquet("data/panel.parquet").sort_values(["PRODUCT_ID", "WEEK_NO"])
    g = d.groupby("PRODUCT_ID")
    d["log_display"] = np.log1p(d.display_stores)
    d["log_mailer"] = np.log1p(d.mailer_stores)
    # Previous week's price relative to the product's usual price; 0 when the product
    # was not bought that week (no observed deviation).
    prev_week = g.WEEK_NO.shift(1)
    d["lag_rel_price"] = np.where(prev_week == d.WEEK_NO - 1, g.log_rel_price.shift(1), 0.0)
    d["lag_observed"] = (prev_week == d.WEEK_NO - 1).astype(int)
    d["week_of_year"] = (d.WEEK_NO - 1) % 52
    d["commodity"] = d.COMMODITY_DESC.astype("category")
    d["product"] = d.PRODUCT_ID.astype("category")
    return d.reset_index(drop=True)


def ols(d, formula):
    m = smf.ols(formula, data=d).fit(cov_type="cluster", cov_kwds={"groups": d.PRODUCT_ID.cat.codes
                                                                     if hasattr(d.PRODUCT_ID, "cat") else
                                                                     pd.factorize(d.PRODUCT_ID)[0]})
    return {"elasticity": float(m.params["log_price"]), "se": float(m.bse["log_price"]),
            **({"lag_price": float(m.params["lag_rel_price"]), "lag_se": float(m.bse["lag_rel_price"])}
               if "lag_rel_price" in m.params else {})}


def within(d, cols):
    """Remove product means (the product fixed effect) before machine learning."""
    return d[cols] - d.groupby("PRODUCT_ID")[cols].transform("mean")


def dml(d: pd.DataFrame, folds: int = 5, seed: int = 0) -> dict:
    yw = within(d, ["log_units"])["log_units"].to_numpy()
    tw = within(d, ["log_price"])["log_price"].to_numpy()
    X = d[["log_display", "log_mailer", "lag_rel_price", "lag_observed", "week_of_year", "WEEK_NO", "commodity"]]
    ry, rt = np.zeros(len(d)), np.zeros(len(d))
    params = dict(n_estimators=300, learning_rate=0.05, num_leaves=31, min_child_samples=50,
                  subsample=0.8, subsample_freq=1, random_state=seed, verbose=-1)
    for tr, te in GroupKFold(folds).split(X, groups=d.PRODUCT_ID):
        ry[te] = yw[te] - lgb.LGBMRegressor(**params).fit(X.iloc[tr], yw[tr]).predict(X.iloc[te])
        rt[te] = tw[te] - lgb.LGBMRegressor(**params).fit(X.iloc[tr], tw[tr]).predict(X.iloc[te])
    theta = float(np.sum(rt * ry) / np.sum(rt * rt))
    # Cluster-robust (by product) standard error for the residual-on-residual slope.
    eps = ry - theta * rt
    score = pd.Series(rt * eps).groupby(d.PRODUCT_ID.to_numpy()).sum().to_numpy()
    se = float(np.sqrt(np.sum(score ** 2)) / np.sum(rt * rt))
    return {"elasticity": theta, "se": se, "price_variation_left": float(rt.std() / tw.std())}


def breakeven_margin(e: float, cut: float) -> float | None:
    """Gross margin above which a price cut of `cut` raises profit, given elasticity e.

    Profit before: m (price 1, cost 1 - m). After: (m - cut) * (1 - cut) ** e.
    Returns None when no margin makes the cut pay (demand too inelastic).
    """
    lift = (1 - cut) ** e
    m = cut * lift / (lift - 1) if lift > 1 else None
    return float(m) if m is not None and m <= 1 else None


def main() -> None:
    d = load()
    res = {"rows": int(len(d)), "products": int(d.PRODUCT_ID.nunique())}
    fe = "C(PRODUCT_ID) + C(WEEK_NO)"
    res["pooled"] = ols(d, "log_units ~ log_price")
    res["fixed_effects"] = ols(d, f"log_units ~ log_price + {fe}")
    res["fe_promo"] = ols(d, f"log_units ~ log_price + log_display + log_mailer + {fe}")
    res["fe_promo_lag"] = ols(d, f"log_units ~ log_price + log_display + log_mailer + lag_rel_price + lag_observed + {fe}")
    res["dml"] = dml(d)
    print(json.dumps({k: res[k] for k in ("pooled", "fixed_effects", "fe_promo", "fe_promo_lag", "dml")}, indent=1),
          flush=True)

    by = {}
    for c, sub in d.groupby("COMMODITY_DESC"):
        if sub.PRODUCT_ID.nunique() < 9:
            continue
        sub = sub.copy()
        sub["commodity"] = sub.commodity.cat.remove_unused_categories()
        r = dml(sub)
        r["products"] = int(sub.PRODUCT_ID.nunique())
        r["rows"] = int(len(sub))
        r["revenue_change_price_up_5pct"] = float(1.05 ** (1 + r["elasticity"]) - 1)
        r["breakeven_margin_10pct_cut"] = breakeven_margin(r["elasticity"], 0.10)
        by[c] = r
        print(c, {k: round(v, 3) if isinstance(v, float) else v for k, v in r.items()}, flush=True)
    res["by_commodity"] = by
    Path("reports").mkdir(exist_ok=True)
    Path("reports/elasticity.json").write_text(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
