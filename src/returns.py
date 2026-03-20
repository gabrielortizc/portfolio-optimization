import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick

TRADING_DAYS = 252  # dias úteis no ano

# -------------------------------------------------------------------
# Ativos (sem benchmarks — eles entram só na comparação)
# -------------------------------------------------------------------
PORTFOLIO_TICKERS = [
    "AAPL", "BBAS3.SA", "BRK-B", "ITUB4.SA",
    "JNJ", "JPM", "PETR4.SA", "SPY",
    "VALE3.SA", "WEGE3.SA", "XOM"
]

# -------------------------------------------------------------------
# Funções
# -------------------------------------------------------------------
def compute_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """
    Calcula retornos diários simples.
    Preenche NaNs causados por feriados com 0 (sem variação no dia).
    """
    returns = prices.pct_change(fill_method=None)
    returns = returns.fillna(0)
    return returns


def compute_annual_stats(returns: pd.DataFrame):
    """
    Retorna retorno médio anualizado e matriz de covariância anualizada.
    """
    mean_returns = returns.mean() * TRADING_DAYS
    cov_matrix   = returns.cov() * TRADING_DAYS
    return mean_returns, cov_matrix


def plot_correlation(returns: pd.DataFrame, save_path: str = "outputs/correlation.png"):
    """
    Plota e salva heatmap da matriz de correlação.
    """
    corr = returns.corr()
    tickers = corr.columns.tolist()
    n = len(tickers)

    fig, ax = plt.subplots(figsize=(11, 9))
    fig.patch.set_facecolor("#0f1117")
    ax.set_facecolor("#0f1117")

    # Heatmap manual
    im = ax.imshow(corr.values, cmap="RdYlGn", vmin=-1, vmax=1)

    # Eixos
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(tickers, rotation=45, ha="right", color="white", fontsize=10)
    ax.set_yticklabels(tickers, color="white", fontsize=10)

    # Valores dentro das células
    for i in range(n):
        for j in range(n):
            ax.text(j, i, f"{corr.values[i, j]:.2f}",
                    ha="center", va="center", fontsize=8,
                    color="black" if abs(corr.values[i, j]) < 0.7 else "white")

    # Colorbar
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.yaxis.set_tick_params(color="white")
    plt.setp(cbar.ax.yaxis.get_ticklabels(), color="white")

    ax.set_title("Correlation Matrix — Portfolio Assets", color="white",
                 fontsize=14, fontweight="bold", pad=15)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Gráfico salvo em '{save_path}'")


def plot_cumulative_returns(returns: pd.DataFrame, save_path: str = "outputs/cumulative_returns.png"):
    """
    Plota retorno acumulado de cada ativo ao longo do tempo.
    """
    cumulative = (1 + returns).cumprod() - 1

    fig, ax = plt.subplots(figsize=(13, 6))
    fig.patch.set_facecolor("#0f1117")
    ax.set_facecolor("#0f1117")

    colors = plt.cm.tab20.colors
    for i, col in enumerate(cumulative.columns):
        ax.plot(cumulative.index, cumulative[col], label=col,
                linewidth=1.4, color=colors[i % len(colors)])

    ax.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1, decimals=0))
    ax.tick_params(colors="white")
    ax.xaxis.label.set_color("white")
    ax.yaxis.label.set_color("white")
    for spine in ax.spines.values():
        spine.set_edgecolor("#333333")

    ax.set_title("Cumulative Returns (2019–2024)", color="white",
                 fontsize=14, fontweight="bold")
    ax.legend(fontsize=8, ncol=3, facecolor="#1a1a2e", labelcolor="white",
              framealpha=0.7)
    ax.axhline(0, color="white", linewidth=0.5, linestyle="--")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Gráfico salvo em '{save_path}'")


# -------------------------------------------------------------------
# Teste rápido
# -------------------------------------------------------------------
if __name__ == "__main__":
    from data_loader import load_prices, BENCHMARKS

    prices  = load_prices()

    # Separa portfólio dos benchmarks
    bench   = prices[BENCHMARKS]
    assets  = prices.drop(columns=BENCHMARKS)

    returns           = compute_returns(assets)
    mean_ret, cov_mat = compute_annual_stats(returns)

    print("--- Retorno médio anualizado ---")
    print(mean_ret.sort_values(ascending=False).map("{:.1%}".format))

    print("\n--- Shape da matriz de covariância ---")
    print(cov_mat.shape)

    plot_correlation(returns)
    plot_cumulative_returns(returns)