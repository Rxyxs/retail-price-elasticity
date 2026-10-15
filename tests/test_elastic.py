import numpy as np
import pandas as pd
import pytest

from elastic.model import breakeven_margin, dml


def _panel(true_e=-1.5, promo_lift=0.6, n_prod=60, n_week=80, seed=0):
    """Promotion weeks cut the price AND lift sales on their own, like a real display."""
    rng = np.random.default_rng(seed)
    rows = []
    for p in range(n_prod):
        base = rng.normal(3, 1)
        for w in range(n_week):
            promo = rng.random() < 0.25
            lp = np.log(2.0) - 0.25 * promo + rng.normal(0, 0.05)
            lu = base + true_e * lp + promo_lift * promo + rng.normal(0, 0.2)
            rows.append((p, w, lu, lp, int(promo)))
    d = pd.DataFrame(rows, columns=["PRODUCT_ID", "WEEK_NO", "log_units", "log_price", "promo"])
    d["log_display"] = d.promo * 2.0
    d["log_mailer"] = 0.0
    d["lag_rel_price"] = 0.0
    d["lag_observed"] = 0
    d["week_of_year"] = d.WEEK_NO % 52
    d["commodity"] = pd.Categorical(["x"] * len(d))
    return d


def test_dml_removes_promotion_confounding():
    d = _panel()
    dp = d.log_price - d.groupby("PRODUCT_ID").log_price.transform("mean")
    du = d.log_units - d.groupby("PRODUCT_ID").log_units.transform("mean")
    naive = np.polyfit(dp, du, 1)[0]
    r = dml(d)
    assert naive < -2.5                                   # promotion lift is credited to price
    assert r["elasticity"] == pytest.approx(-1.5, abs=3 * r["se"] + 0.05)


def test_breakeven_margin():
    # e = -2, 10% cut: volume x1.2346, the cut pays above ~52.6% gross margin
    assert breakeven_margin(-2.0, 0.10) == pytest.approx(0.526, abs=0.001)
    # inelastic demand: no margin makes a cut pay
    assert breakeven_margin(-0.5, 0.10) is None
    # at the break-even margin, profit is unchanged
    m = breakeven_margin(-3.0, 0.15)
    assert m == pytest.approx((m - 0.15) * 0.85 ** -3.0)
