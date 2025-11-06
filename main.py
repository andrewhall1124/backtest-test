import sf_quant.data as sfd
import sf_quant.optimizer as sfo
import sf_quant.backtester as sfb
import datetime as dt
import polars as pl
import cvxpy as cp
import numpy as np

IC = 0.05

print("Loading data...")
df_data = sfd.load_assets(
    start=dt.date(2024, 1, 1),
    end=dt.date(2024, 12, 31),
    columns=["date", "barrid", "price", "return", "specific_risk", "predicted_beta"],
    in_universe=True
).with_columns(pl.col("return", "specific_risk").truediv(100))

print("Computing signals...")
df_signals = (
    df_data.sort("barrid", "date")
    .with_columns(
        pl.col("return")
        .log1p()
        .rolling_sum(230)
        .shift(22)
        .over("barrid")
        .alias("momentum")
    )
    .with_columns(
        pl.col("momentum")
        .sub(pl.col("momentum").mean())
        .truediv(pl.col("momentum").std())
        .over("date")
        .alias("score")
    )
)

print("Computing alphas...")
df_alphas = df_signals.with_columns(
    pl.lit(IC)
    .mul(pl.col("score"))
    .mul(pl.col("specific_risk"))
    .alias("alpha")
)

print("Applying filters...")
df_filtered = (
    df_alphas
    .filter(
        pl.col('alpha').is_not_null(),
    )
)

class ZeroBeta(sfo.constraints.Constraint):
    def __call__(self, weights: cp.Variable, **kwargs) -> cp.Constraint:
        betas: np.ndarray | None = kwargs.get("betas")
        if betas is None:
            raise ValueError("ZeroBeta requires betas")
        return betas @ weights == 0


print("Computing weights...")
weights = sfb.backtest_parallel(data=df_filtered, constraints=[ZeroBeta()], gamma=2)
weights.write_parquet("weights.parquet")
