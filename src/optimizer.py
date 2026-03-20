import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
from scipy.optimize import minimize

from config import TRADING_DAYS, RISK_FREE

# -------------------------------------------------------------------
# Simulação de portfólios aleatórios
# -------------------------------------------------------------------
def simulate_portfolios(
    mean_returns: pd.Series,
    cov_matrix: pd.DataFrame,
    n_portfolios: int = 10_000,
    random_seed: int = 42
) -> pd.DataFrame:
    """
    Gera n_portfolios com pesos aleatórios e calcula
    retorno, volatilidade e Sharpe de cada um.
    """
    np.random.seed(random_seed)
    n_assets = len(mean_returns)
    results  = []

    for _ in range(n_portfolios):
        weights = np.random.random(n_assets)
        weights /= weights.sum()  # normaliza para somar 1

        ret  = np.dot(weights, mean_returns)
        vol  = np.sqrt(weights @ cov_matrix.values @ weights)
        shrp = (ret - RISK_FREE) / vol

        results.append({
            "return":   ret,
            "vol":      vol,
            "sharpe":   shrp,
            "weights":  weights
        })

    return pd.DataFrame(results)


# -------------------------------------------------------------------
# Otimização via scipy
# -------------------------------------------------------------------
def _portfolio_stats(weights, mean_returns, cov_matrix):
    ret  = np.dot(weights, mean_returns)
    vol  = np.sqrt(weights @ cov_matrix.values @ weights)
    shrp = (ret - RISK_FREE) / vol
    return ret, vol, shrp


def maximize_sharpe(mean_returns: pd.Series, cov_matrix: pd.DataFrame) -> dict:
    """Portfólio de máximo Sharpe ratio."""
    n = len(mean_returns)

    def neg_sharpe(w):
        _, _, s = _portfolio_stats(w, mean_returns, cov_matrix)
        return -s

    constraints = {"type": "eq", "fun": lambda w: w.sum() - 1}
    bounds      = tuple((0, 0.4) for _ in range(n))
    w0          = np.ones(n) / n  # pesos iniciais iguais

    result = minimize(neg_sharpe, w0, method="SLSQP",
                      bounds=bounds, constraints=constraints)

    ret, vol, shrp = _portfolio_stats(result.x, mean_returns, cov_matrix)
    return {"weights": result.x, "return": ret, "vol": vol, "sharpe": shrp}


def minimize_volatility(mean_returns: pd.Series, cov_matrix: pd.DataFrame) -> dict:
    """Portfólio de mínima variância."""
    n = len(mean_returns)

    def portfolio_vol(w):
        return np.sqrt(w @ cov_matrix.values @ w)

    constraints = {"type": "eq", "fun": lambda w: w.sum() - 1}
    bounds      = tuple((0, 0.4) for _ in range(n))
    w0          = np.ones(n) / n

    result = minimize(portfolio_vol, w0, method="SLSQP",
                      bounds=bounds, constraints=constraints)

    ret, vol, shrp = _portfolio_stats(result.x, mean_returns, cov_matrix)
    return {"weights": result.x, "return": ret, "vol": vol, "sharpe": shrp}

def compute_efficient_frontier(
    mean_returns: pd.Series,
    cov_matrix: pd.DataFrame,
    n_points: int = 100
) -> pd.DataFrame:
    """Traça a fronteira eficiente otimizando para cada nível de retorno alvo."""
    n      = len(mean_returns)
    bounds = tuple((0, 0.40) for _ in range(n))
    frontier = []

    ret_min = mean_returns.min()
    ret_max = mean_returns.max()

    for target in np.linspace(ret_min, ret_max, n_points):
        constraints = [
            {"type": "eq", "fun": lambda w: w.sum() - 1},
            {"type": "eq", "fun": lambda w, t=target: np.dot(w, mean_returns) - t}
        ]
        result = minimize(
            lambda w: np.sqrt(w @ cov_matrix.values @ w),
            np.ones(n) / n,
            method="SLSQP",
            bounds=bounds,
            constraints=constraints
        )
        if result.success:
            vol = np.sqrt(result.x @ cov_matrix.values @ result.x)
            frontier.append({"return": target, "vol": vol})

    return pd.DataFrame(frontier)
# -------------------------------------------------------------------
# Visualização
# -------------------------------------------------------------------
def plot_efficient_frontier(
    simulated: pd.DataFrame,
    max_sharpe: dict,
    min_vol: dict,
    tickers: list,
    frontier: pd.DataFrame = None,
    save_path: str = "outputs/efficient_frontier.png"
):
    fig, ax = plt.subplots(figsize=(12, 7))
    fig.patch.set_facecolor("#0f1117")
    ax.set_facecolor("#0f1117")

    sc = ax.scatter(
        simulated["vol"], simulated["return"],
        c=simulated["sharpe"], cmap="plasma",
        alpha=0.4, s=8
    )

    # Linha da fronteira eficiente
    if frontier is not None:
        ax.plot(frontier["vol"], frontier["return"],
                color="white", linewidth=2, linestyle="--",
                label="Efficient Frontier", zorder=4)

    ax.scatter(max_sharpe["vol"], max_sharpe["return"],
               marker="*", color="#00f5d4", s=300, zorder=5,
               label=f'Max Sharpe ({max_sharpe["sharpe"]:.2f})')

    ax.scatter(min_vol["vol"], min_vol["return"],
               marker="D", color="#f72585", s=120, zorder=5,
               label=f'Min Volatility ({min_vol["vol"]:.1%})')

    cbar = fig.colorbar(sc, ax=ax, pad=0.02)
    cbar.set_label("Sharpe Ratio", color="white", fontsize=10)
    cbar.ax.yaxis.set_tick_params(color="white")
    plt.setp(cbar.ax.yaxis.get_ticklabels(), color="white")

    ax.xaxis.set_major_formatter(mtick.PercentFormatter(xmax=1, decimals=0))
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1, decimals=0))
    ax.tick_params(colors="white")
    for spine in ax.spines.values():
        spine.set_edgecolor("#333333")

    ax.set_xlabel("Annual Volatility", color="white", fontsize=11)
    ax.set_ylabel("Annual Return", color="white", fontsize=11)
    ax.set_title("Efficient Frontier — Markowitz Optimization",
                 color="white", fontsize=14, fontweight="bold")
    ax.legend(facecolor="#1a1a2e", labelcolor="white", fontsize=10)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Gráfico salvo em '{save_path}'")


def print_allocation(portfolio: dict, tickers: list, label: str):
    """Imprime alocação de forma legível."""
    print(f"\n--- {label} ---")
    print(f"Retorno anual : {portfolio['return']:.1%}")
    print(f"Volatilidade  : {portfolio['vol']:.1%}")
    print(f"Sharpe Ratio  : {portfolio['sharpe']:.2f}")
    print("\nAlocação:")
    for ticker, w in sorted(zip(tickers, portfolio["weights"]),
                             key=lambda x: -x[1]):
        if w > 0.001:
            print(f"  {ticker:<12} {w:.1%}")


# -------------------------------------------------------------------
# Execução
# -------------------------------------------------------------------
if __name__ == "__main__":
    from data_loader import load_prices, BENCHMARKS
    from returns import compute_returns, compute_annual_stats

    prices          = load_prices()
    assets          = prices.drop(columns=BENCHMARKS)
    returns         = compute_returns(assets)
    mean_ret, cov   = compute_annual_stats(returns)
    tickers         = list(mean_ret.index)

    simulated        = simulate_portfolios(mean_ret, cov)
    max_sharpe_port  = maximize_sharpe(mean_ret, cov)
    min_vol_port     = minimize_volatility(mean_ret, cov)
    frontier         = compute_efficient_frontier(mean_ret, cov)

    print_allocation(max_sharpe_port, tickers, "Portfólio de Máximo Sharpe")
    print_allocation(min_vol_port,    tickers, "Portfólio de Mínima Variância")

    plot_efficient_frontier(simulated, max_sharpe_port, min_vol_port,
                            tickers, frontier)