import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick

from config import TRADING_DAYS, RISK_FREE

# -------------------------------------------------------------------
# Métricas de performance
# -------------------------------------------------------------------
def compute_metrics(returns: pd.Series, label: str) -> dict:
    """
    Calcula métricas anualizadas de um portfólio ou benchmark.
    """
    ret_annual = returns.mean() * TRADING_DAYS
    vol_annual = returns.std() * np.sqrt(TRADING_DAYS)
    sharpe     = (ret_annual - RISK_FREE) / vol_annual

    # Drawdown máximo
    cumulative  = (1 + returns).cumprod()
    rolling_max = cumulative.cummax()
    drawdown    = (cumulative - rolling_max) / rolling_max
    max_dd      = drawdown.min()

    return {
        "label":         label,
        "return":        ret_annual,
        "volatility":    vol_annual,
        "sharpe":        sharpe,
        "max_drawdown":  max_dd,
        "cumulative":    cumulative
    }


def build_portfolio_returns(
    asset_returns: pd.DataFrame,
    weights: np.ndarray
) -> pd.Series:
    """
    Calcula retorno diário de um portfólio dado pesos fixos.
    """
    return asset_returns.dot(weights)


# -------------------------------------------------------------------
# Visualizações
# -------------------------------------------------------------------
def plot_cumulative_comparison(
    portfolios: list,
    save_path: str = "outputs/benchmark_comparison.png"
):
    """
    Plota retorno acumulado dos portfólios vs benchmarks.
    portfolios: lista de dicts com keys 'label' e 'cumulative'
    """
    fig, ax = plt.subplots(figsize=(13, 6))
    fig.patch.set_facecolor("#0f1117")
    ax.set_facecolor("#0f1117")

    styles = {
        "Max Sharpe":    {"color": "#00f5d4", "lw": 2.5, "ls": "-"},
        "Min Volatility":{"color": "#f72585", "lw": 2.5, "ls": "-"},
        "IBOV":          {"color": "#ffd166", "lw": 1.8, "ls": "--"},
        "SPY":           {"color": "#a8dadc", "lw": 1.8, "ls": "--"},
    }

    for p in portfolios:
        style = styles.get(p["label"], {"color": "white", "lw": 1.5, "ls": "-"})
        ax.plot(p["cumulative"].index,
                p["cumulative"].values - 1,
                label=p["label"],
                color=style["color"],
                linewidth=style["lw"],
                linestyle=style["ls"])

    ax.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1, decimals=0))
    ax.tick_params(colors="white")
    for spine in ax.spines.values():
        spine.set_edgecolor("#333333")

    ax.axhline(0, color="white", linewidth=0.5, linestyle=":")
    ax.set_title("Cumulative Return — Portfolio vs Benchmarks (2019–2024)",
                 color="white", fontsize=14, fontweight="bold")
    ax.set_xlabel("Date", color="white", fontsize=11)
    ax.set_ylabel("Cumulative Return", color="white", fontsize=11)
    ax.legend(facecolor="#1a1a2e", labelcolor="white", fontsize=11)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Gráfico salvo em '{save_path}'")


def plot_metrics_table(
    metrics: list,
    save_path: str = "outputs/metrics_table.png"
):
    """
    Plota tabela comparativa de métricas.
    """
    labels = [m["label"]                    for m in metrics]
    rets   = [f'{m["return"]:.1%}'          for m in metrics]
    vols   = [f'{m["volatility"]:.1%}'      for m in metrics]
    shrps  = [f'{m["sharpe"]:.2f}'          for m in metrics]
    dds    = [f'{m["max_drawdown"]:.1%}'    for m in metrics]

    fig, ax = plt.subplots(figsize=(10, 3))
    fig.patch.set_facecolor("#0f1117")
    ax.set_facecolor("#0f1117")
    ax.axis("off")

    table = ax.table(
        cellText=list(zip(labels, rets, vols, shrps, dds)),
        colLabels=["Portfolio", "Annual Return", "Volatility",
                   "Sharpe Ratio", "Max Drawdown"],
        cellLoc="center",
        loc="center"
    )

    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1.2, 2.2)

    # Estilo das células
    for (row, col), cell in table.get_celld().items():
        cell.set_edgecolor("#333333")
        if row == 0:
            cell.set_facecolor("#1a1a2e")
            cell.set_text_props(color="white", fontweight="bold")
        else:
            cell.set_facecolor("#0f1117")
            cell.set_text_props(color="white")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Gráfico salvo em '{save_path}'")


# -------------------------------------------------------------------
# Execução
# -------------------------------------------------------------------
if __name__ == "__main__":
    from data_loader import load_prices, BENCHMARKS
    from returns import compute_returns, compute_annual_stats
    from optimizer import maximize_sharpe, minimize_volatility

    # Carrega dados
    prices        = load_prices()
    bench_prices  = prices[BENCHMARKS]
    asset_prices  = prices.drop(columns=BENCHMARKS)

    asset_returns = compute_returns(asset_prices)
    bench_returns = compute_returns(bench_prices)
    mean_ret, cov = compute_annual_stats(asset_returns)
    tickers       = list(mean_ret.index)

    # Portfólios otimizados
    max_sharpe_port = maximize_sharpe(mean_ret, cov)
    min_vol_port    = minimize_volatility(mean_ret, cov)

    # Retornos diários de cada portfólio
    ms_returns  = build_portfolio_returns(asset_returns, max_sharpe_port["weights"])
    mv_returns  = build_portfolio_returns(asset_returns, min_vol_port["weights"])
    ibov_returns = bench_returns["^BVSP"]
    spy_returns  = bench_returns["SPY"]

    # Métricas
    metrics = [
        compute_metrics(ms_returns,   "Max Sharpe"),
        compute_metrics(mv_returns,   "Min Volatility"),
        compute_metrics(ibov_returns, "IBOV"),
        compute_metrics(spy_returns,  "SPY"),
    ]

    # Imprime resumo
    print(f"\n{'Label':<18} {'Return':>10} {'Vol':>10} {'Sharpe':>10} {'Max DD':>10}")
    print("-" * 60)
    for m in metrics:
        print(f"{m['label']:<18} {m['return']:>10.1%} "
              f"{m['volatility']:>10.1%} {m['sharpe']:>10.2f} "
              f"{m['max_drawdown']:>10.1%}")

    # Gráficos
    plot_cumulative_comparison(metrics)
    plot_metrics_table(metrics)