"""README figures: the estimator ladder and elasticities by category."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

OUT = Path("docs/img")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    r = json.loads(Path("reports/elasticity.json").read_text())

    steps = [("pooled OLS", "pooled"), ("+ product & week\nfixed effects", "fixed_effects"),
             ("+ display &\nmailer", "fe_promo"), ("+ last week's\nprice", "fe_promo_lag"),
             ("double ML", "dml")]
    fig, ax = plt.subplots(figsize=(8, 4))
    vals = [r[k]["elasticity"] for _, k in steps]
    errs = [1.96 * r[k]["se"] for _, k in steps]
    colors = ["#BAB0AC", "#E15759", "#4E79A7", "#4E79A7", "#59A14F"]
    ax.bar([s for s, _ in steps], vals, yerr=errs, capsize=4, color=colors)
    for i, v in enumerate(vals):
        ax.text(i, v - 0.12, f"{v:.2f}", ha="center", va="top")
    ax.axhline(-1, color="k", lw=0.8, ls=":")
    ax.text(4.45, -1.03, "unit elastic", ha="right", va="top", fontsize=8)
    ax.set_ylim(-2.6, 0.1)
    ax.set_ylabel("own-price elasticity")
    ax.set_title("Each control changes the answer; double ML confirms the last step")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(OUT / "ladder.png", dpi=130)
    plt.close(fig)

    by = sorted(r["by_commodity"].items(), key=lambda kv: kv[1]["elasticity"])
    fig, ax = plt.subplots(figsize=(8, 4.8))
    names = [f"{k.title()} ({v['products']})" for k, v in by]
    e = [v["elasticity"] for _, v in by]
    ci = [1.96 * v["se"] for _, v in by]
    # Grey when the interval includes -1: the data cannot say which side it is on.
    col = ["#BAB0AC" if x - c < -1 < x + c else "#E15759" if x < -1 else "#59A14F" for x, c in zip(e, ci)]
    ax.barh(names, e, xerr=ci, color=col, capsize=3)
    ax.axvline(-1, color="k", lw=0.8, ls=":")
    ax.set_xlabel("own-price elasticity, double ML (95% CI)")
    ax.set_title("Red: elastic, a price rise loses revenue. Green: inelastic. Grey: cannot tell", fontsize=10)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(OUT / "by_category.png", dpi=130)
    plt.close(fig)
    print("figures written")


if __name__ == "__main__":
    main()
