import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def compute_returns(pred_df, price_df, top_k=5):
    """
    Build a daily equal-weight portfolio from model predictions and compute
    its daily return series, alongside a universe equal-weight benchmark.

    Args:
        pred_df (pd.DataFrame): columns ['date', 'stock', 'pred_return'] where
            'pred_return' is the model’s score (e.g., predicted next-day return).
        price_df (pd.DataFrame): columns ['date', 'stock', 'return'] where
            'return' is the realized daily % return (e.g., 1.23 for 1.23%).
        top_k (int): number of top-ranked stocks selected each day.

    Returns:
        pd.DataFrame with columns:
            - 'date'
            - 'daily_portfolio_return' (strategy daily return in decimal)
            - 'equal_weight_return'    (universe equal-weight daily return in decimal)
        The output is filtered to the evaluation window [2024-05-04, 2025-01-01].
    """
    # Join predictions with realized returns
    true_returns = price_df[["date", "stock", "return"]].copy()
    merged = pred_df.merge(true_returns, on=["date", "stock"], how="left")

    # For each day, select the top-k stocks by predicted return
    top_df = (
        merged.sort_values(["date", "pred_return"], ascending=False)
              .groupby("date")
              .head(top_k)
    )

    # Equal weights so that the sum of weights per day equals 1
    top_df["weight"] = 1.0 / top_k

    # Convert % return to decimal and compute per-stock contribution
    top_df["daily_portfolio_return"] = top_df["weight"] * (top_df["return"] / 100.0)

    # Aggregate contributions to get daily portfolio returns
    portfolio_daily = (
        top_df.groupby("date")["daily_portfolio_return"].sum().reset_index()
    )

    # Build a universe equal-weight benchmark: average of all stocks’ daily returns
    uni = price_df.copy()
    uni["return_norm"] = uni["return"] / 100.0  # convert to decimal
    avg_daily_return = (
        uni.groupby("date")["return_norm"].mean().reset_index()
           .rename(columns={"return_norm": "equal_weight_return"})
    )

    # Merge strategy returns with the benchmark
    portfolio_daily = portfolio_daily.merge(avg_daily_return, on="date", how="left")

    # Restrict to the evaluation window (based on trading days)
    portfolio_daily = portfolio_daily[
        (portfolio_daily["date"] >= pd.to_datetime("2024-05-04")) &
        (portfolio_daily["date"] <= pd.to_datetime("2025-01-01"))
    ].copy()

    return portfolio_daily


def evaluate_portfolio(portfolio_df, freq="daily", compare=True):
    df = portfolio_df.copy()

    # Accumulated Return (AR)
    df["cumulative_return"] = (1 + df["daily_portfolio_return"]).cumprod()
    total_cum_return = df["cumulative_return"].iloc[-1] - 1

    # Annualization factor
    freq_map = {"daily": 252, "weekly": 52, "monthly": 12}
    periods_per_year = freq_map.get(freq, 252)
    n_periods = len(df)

    # Sharpe Ratio (SR)
    annualized_return = (1 + total_cum_return) ** (periods_per_year / n_periods) - 1
    annualized_vol = df["daily_portfolio_return"].std() * np.sqrt(periods_per_year)
    sharpe_ratio = annualized_return / annualized_vol if annualized_vol != 0 else np.nan

    # Calmar Ratio (CR)
    running_max = df["cumulative_return"].cummax()
    drawdown = 1 - df["cumulative_return"] / running_max
    max_drawdown = drawdown.max()
    calmar_ratio = annualized_return / max_drawdown if max_drawdown != 0 else np.nan

    print("📊 Portfolio Performance")
    print("📈 Accumulated Return (AR): {:.2f}%".format(total_cum_return * 100))
    print("💡 Sharpe Ratio (SR): {:.4f}".format(sharpe_ratio))
    print("📉 Calmar Ratio (CR): {:.4f}".format(calmar_ratio))

    results = {
        "portfolio": {
            "AR": total_cum_return,
            "SR": sharpe_ratio,
            "CR": calmar_ratio,
        }
    }

    # Equal-weight Benchmark
    if compare and "equal_weight_return" in df.columns:
        df["cumulative_eq"] = (1 + df["equal_weight_return"]).cumprod()
        eq_cum_return = df["cumulative_eq"].iloc[-1] - 1
        eq_ann_return = (1 + eq_cum_return) ** (periods_per_year / n_periods) - 1
        eq_vol = df["equal_weight_return"].std() * np.sqrt(periods_per_year)
        eq_sharpe = eq_ann_return / eq_vol if eq_vol != 0 else np.nan
        eq_drawdown = 1 - df["cumulative_eq"] / df["cumulative_eq"].cummax()
        eq_max_drawdown = eq_drawdown.max()
        eq_calmar = eq_ann_return / eq_max_drawdown if eq_max_drawdown != 0 else np.nan

        print("\n📊 Equal-Weight Benchmark")
        print("📈 Accumulated Return (AR): {:.2f}%".format(eq_cum_return * 100))
        print("💡 Sharpe Ratio (SR): {:.4f}".format(eq_sharpe))
        print("📉 Calmar Ratio (CR): {:.4f}".format(eq_calmar))

        results["equal_weight"] = {
            "AR": eq_cum_return,
            "SR": eq_sharpe,
            "CR": eq_calmar,
        }


def plot_returns_comparison_multi(
    models: dict,
    start_date=None,
    end_date=None,
    title="Accumulated Return Comparison (k = 5)",
    benchmark_label="Equal-Weight Benchmark",
):
    """
    Plot cumulative returns for multiple model portfolios, plus a single benchmark
    taken from the first model's equal_weight_return.

    Args:
        models (dict): {"Model Name": dataframe, ...}
                       Each dataframe must have columns:
                       - "date" (datetime-like or parseable)
                       - "daily_portfolio_return" (decimal, e.g., 0.004 for 0.4%)
                       - "equal_weight_return" (decimal) [only required for the first model]
        start_date (str or pd.Timestamp, optional): filter start date
        end_date   (str or pd.Timestamp, optional): filter end date
        title (str): plot title
        benchmark_label (str): legend label for the benchmark
    """

    if not models:
        raise ValueError("`models` dict is empty.")

    # Use the first model's DF as the source for the market benchmark
    first_model_name = next(iter(models))
    first_df = models[first_model_name].copy()

    # Normalize/parse date and sort once
    for name, df in models.items():
        if "date" not in df.columns:
            raise ValueError(f"DataFrame for '{name}' is missing 'date' column.")
        models[name] = df.copy()
        models[name]["date"] = pd.to_datetime(models[name]["date"])
        models[name] = models[name].sort_values("date")

    # Optional date filter
    if start_date is not None:
        for name in models:
            models[name] = models[name][
                models[name]["date"] >= pd.to_datetime(start_date)
            ]
    if end_date is not None:
        for name in models:
            models[name] = models[name][
                models[name]["date"] <= pd.to_datetime(end_date)
            ]

    # Compute cumulative returns for each model
    cum_series = {}
    for name, df in models.items():
        if "daily_portfolio_return" not in df.columns:
            raise ValueError(
                f"DataFrame for '{name}' is missing 'daily_portfolio_return' column."
            )
        df = df.sort_values("date").copy()
        df["cumulative_portfolio"] = (1 + df["daily_portfolio_return"]).cumprod()
        cum_series[name] = (df["date"], df["cumulative_portfolio"])

    # Benchmark from the first model’s equal_weight_return
    if "equal_weight_return" not in first_df.columns:
        raise ValueError(
            f"Benchmark requires 'equal_weight_return' in the first model's DataFrame ('{first_model_name}')."
        )
    bench_df = models[first_model_name].copy()
    bench_df["cumulative_equal_weight"] = (
        1 + bench_df["equal_weight_return"]
    ).cumprod()

    # Plot
    plt.figure(figsize=(12, 5))
    # Models
    for name, (dates, cum_vals) in cum_series.items():
        plt.plot(dates, cum_vals, linewidth=2, label=name)
    # Benchmark
    plt.plot(
        bench_df["date"],
        bench_df["cumulative_equal_weight"],
        linewidth=2,
        linestyle="--",
        label=benchmark_label,
    )

    plt.xlabel("Date")
    plt.ylabel("Accumulated Return")
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
