"""Build a product-week panel of units, prices and promotion from dunnhumby.

The Complete Journey follows 2,500 households for 102 weeks, so sales per
product, store and week are very sparse. The panel pools stores: for every
product and week it holds the units the panel bought, the average price paid
(after the loyalty-card discount), the shelf price before it, and the number
of stores where the product was on in-store display or in the weekly mailer.

Only packaged categories sold by the unit are kept; meat, produce, fuel and
coupon items are sold by weight or are not products.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import polars as pl

RAW = Path("data/raw")
DATASET = "frtgnn/dunnhumby-the-complete-journey"
COMMODITIES = [
    "SOFT DRINKS", "FLUID MILK PRODUCTS", "CHEESE", "FROZEN PIZZA", "BAKED BREAD/BUNS/ROLLS",
    "COLD CEREAL", "BEERS/ALES", "BAG SNACKS", "YOGURT", "SOUP", "COOKIES/CONES", "CRACKERS/MISC BKD FD",
]
MIN_WEEKS = 60       # product sold in at least this many of the 102 weeks
MIN_UNITS = 300      # and at least this many units overall


def download(raw: Path = RAW) -> None:
    if (raw / "transaction_data.csv").exists():
        return
    raw.mkdir(parents=True, exist_ok=True)
    kaggle = Path(sys.executable).with_name("kaggle.exe" if sys.platform == "win32" else "kaggle")
    subprocess.run([str(kaggle), "datasets", "download", "-d", DATASET, "-p", str(raw), "--unzip"], check=True)


def build() -> pl.DataFrame:
    download()
    money = {c: pl.Float64 for c in ("SALES_VALUE", "RETAIL_DISC", "COUPON_DISC", "COUPON_MATCH_DISC")}
    t = pl.read_csv(RAW / "transaction_data.csv", schema_overrides=money)
    prod = pl.read_csv(RAW / "product.csv", infer_schema_length=0).with_columns(pl.col("PRODUCT_ID").cast(pl.Int64))
    t = t.join(prod.select("PRODUCT_ID", "COMMODITY_DESC", "BRAND", "MANUFACTURER"), on="PRODUCT_ID")
    t = t.filter(pl.col("COMMODITY_DESC").is_in(COMMODITIES) & (pl.col("QUANTITY") > 0) & (pl.col("SALES_VALUE") > 0))

    w = (t.group_by("PRODUCT_ID", "WEEK_NO")
         .agg(units=pl.col("QUANTITY").sum(),
              paid=pl.col("SALES_VALUE").sum(),
              shelf=(pl.col("SALES_VALUE") - pl.col("RETAIL_DISC")).sum(),
              discounted_share=(pl.col("RETAIL_DISC") < 0).mean(),
              stores=pl.col("STORE_ID").n_unique())
         .with_columns(price=pl.col("paid") / pl.col("units"), shelf_price=pl.col("shelf") / pl.col("units")))

    keep = (w.group_by("PRODUCT_ID").agg(n_weeks=pl.len(), total=pl.col("units").sum())
            .filter((pl.col("n_weeks") >= MIN_WEEKS) & (pl.col("total") >= MIN_UNITS)).select("PRODUCT_ID"))
    w = w.join(keep, on="PRODUCT_ID")

    causal = (pl.scan_csv(RAW / "causal_data.csv", schema_overrides={"display": pl.Utf8, "mailer": pl.Utf8})
              .join(keep.lazy(), left_on="PRODUCT_ID", right_on="PRODUCT_ID")
              .group_by("PRODUCT_ID", "WEEK_NO")
              # causal_data only lists store-weeks with some promotion, so a share of its rows
              # would be a share among promoted stores. Count stores instead.
              .agg(display_stores=(pl.col("display") != "0").sum(), mailer_stores=(pl.col("mailer") != "0").sum())
              .collect())
    w = w.join(causal, on=["PRODUCT_ID", "WEEK_NO"], how="left").with_columns(
        pl.col("display_stores").fill_null(0), pl.col("mailer_stores").fill_null(0))

    # Each product's typical price, to express weekly prices as deviations from it.
    w = w.with_columns(ref_price=pl.col("shelf_price").median().over("PRODUCT_ID"))
    w = w.with_columns(log_units=pl.col("units").log(), log_price=pl.col("price").log(),
                       log_rel_price=(pl.col("price") / pl.col("ref_price")).log())
    meta = prod.select("PRODUCT_ID", "COMMODITY_DESC", "BRAND", "MANUFACTURER")
    return w.join(meta, on="PRODUCT_ID").sort("PRODUCT_ID", "WEEK_NO")


if __name__ == "__main__":
    df = build()
    Path("data").mkdir(exist_ok=True)
    df.write_parquet("data/panel.parquet")
    print(f"{df.height:,} product-weeks, {df['PRODUCT_ID'].n_unique():,} products")
    print(df.group_by("COMMODITY_DESC").agg(pl.col("PRODUCT_ID").n_unique().alias("products"),
                                           pl.len().alias("rows"),
                                           pl.col("log_rel_price").std().alias("price_sd"),
                                           (pl.col("display_stores") > 0).mean().alias("display_weeks"),
                                           (pl.col("mailer_stores") > 0).mean().alias("mailer_weeks"))
          .sort("rows", descending=True).to_pandas().round(3).to_string())
