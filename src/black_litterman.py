import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
from optimizer import maximize_sharpe, minimize_volatility, plot_efficient_frontier, simulate_portfolios, compute_efficient_frontier

from config import TRADING_DAYS, RISK_FREE, DELTA, TAU

# -------------------------------------------------------------------
# Prior — retornos implícitos do mercado
# -------------------------------------------------------------------
def compute_market_implied_returns(
    cov_matrix: pd.DataFrame,
    market_weights: np.ndarray,
    delta: float = DELTA
) -> np.ndarray:
    """
    Retornos implícitos pelo equilíbrio de mercado (reverse optimization).
    π = δ * Σ * w_mkt
    """
    return delta * cov_matrix.values @ market_weights


# -------------------------------------------------------------------
# Black-Litterman
# -------------------------------------------------------------------
def black_litterman(
    cov_matrix: pd.DataFrame,
    pi: np.ndarray,
    P: np.ndarray,
    Q: np.ndarray,
    omega: np.ndarray,
    tau: float = TAU
) -> np.ndarray:
    """
    Combina prior (pi) com views (P, Q, omega) e retorna
    vetor de retornos esperados ajustados.

    P      — matriz de views (n_views x n_assets)
    Q      — vetor de retornos esperados das views
    omega  — matriz de incerteza das views (diagonal)
    """
    sigma = cov_matrix.values
    M     = np.linalg.inv(
                np.linalg.inv(tau * sigma) + P.T @ np.linalg.inv(omega) @ P
            )
    mu_bl = M @ (np.linalg.inv(tau * sigma) @ pi + P.T @ np.linalg.inv(omega) @ Q)
    return mu_bl


def build_views(tickers: list) -> tuple:
    """
    Define as views relativas do analista.

    View 1: AAPL supera JNJ em 5% ao ano
    View 2: WEGE3 supera VALE3 em 3% ao ano
    View 3: JPM supera ITUB4 em 2% ao ano
    """
    n = len(tickers)
    t = {ticker: i for i, ticker in enumerate(tickers)}

    P = np.zeros((3, n))
    Q = np.zeros(3)

    # View 1
    P[0, t["AAPL"]]     =  1
    P[0, t["JNJ"]]      = -1
    Q[0]                =  0.05

    # View 2
    P[1, t["WEGE3.SA"]] =  1
    P[1, t["VALE3.SA"]] = -1
    Q[1]                =  0.03

    # View 3
    P[2, t["JPM"]]      =  1
    P[2, t["ITUB4.SA"]] = -1
    Q[2]                =  0.02

    # Incerteza proporcional à variância das views
    omega = np.diag([0.00005, 0.00005, 0.00005])
    return P, Q, omega


# -------------------------------------------------------------------
# Comparação de retornos esperados
# -------------------------------------------------------------------
def plot_expected_returns_comparison(
    tickers: list,
    pi: np.ndarray,
    mu_bl: np.ndarray,
    mean_hist: np.ndarray,
    save_path: str = "outputs/bl_expected_returns.png"
):
    """
    Compara retornos históricos, prior de mercado e BL ajustado.
    """
    x      = np.arange(len(tickers))
    width  = 0.28

    fig, ax = plt.subplots(figsize=(13, 6))
    fig.patch.set_facecolor("#0f1117")
    ax.set_facecolor("#0f1117")

    ax.bar(x - width, mean_hist,  width, label="Historical",     color="#a8dadc", alpha=0.85)
    ax.bar(x,         pi,         width, label="Market Implied",  color="#ffd166", alpha=0.85)
    ax.bar(x + width, mu_bl,      width, label="Black-Litterman", color="#00f5d4", alpha=0.85)

    ax.set_xticks(x)
    ax.set_xticklabels(tickers, rotation=45, ha="right", color="white", fontsize=9)
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1, decimals=0))
    ax.tick_params(colors="white")
    for spine in ax.spines.values():
        spine.set_edgecolor("#333333")

    ax.axhline(0, color="white", linewidth=0.5, linestyle="--")
    ax.set_title("Expected Returns: Historical vs Market Implied vs Black-Litterman",
                 color="white", fontsize=13, fontweight="bold")
    ax.legend(facecolor="#1a1a2e", labelcolor="white", fontsize=10)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Gráfico salvo em '{save_path}'")

def get_market_weights(tickers: list) -> np.ndarray:
    """
    Busca capitalização de mercado via yfinance e retorna pesos normalizados.
    """
    import yfinance as yf

    caps = []
    for ticker in tickers:
        info = yf.Ticker(ticker).info
        cap  = info.get("marketCap", 0)
        caps.append(cap)
        print(f"  {ticker:<12} ${cap/1e9:.1f}B")

    caps = np.array(caps, dtype=float)
    return caps / caps.sum()

# -------------------------------------------------------------------
# Execução
# -------------------------------------------------------------------
if __name__ == "__main__":
    from data_loader import load_prices, BENCHMARKS
    from returns import compute_returns, compute_annual_stats

    # Dados
    prices        = load_prices()
    asset_prices  = prices.drop(columns=BENCHMARKS)
    asset_returns = compute_returns(asset_prices)
    mean_ret, cov = compute_annual_stats(asset_returns)
    tickers       = list(mean_ret.index)
    n             = len(tickers)

    print("Buscando capitalizações de mercado...")
    market_weights = get_market_weights(tickers)

    # Prior
    pi = compute_market_implied_returns(cov, market_weights)

    # Views
    P, Q, omega = build_views(tickers)

    # Black-Litterman
    mu_bl = black_litterman(cov, pi, P, Q, omega)
    mu_bl_series = pd.Series(mu_bl, index=tickers)

    # Otimização com retornos BL
    max_sharpe_bl = maximize_sharpe(mu_bl_series, cov)
    min_vol_bl    = minimize_volatility(mu_bl_series, cov)
    simulated_bl  = simulate_portfolios(mu_bl_series, cov)
    frontier_bl   = compute_efficient_frontier(mu_bl_series, cov)

    # Resultados
    print("\n--- Retornos Esperados: Histórico vs Prior vs BL ---")
    df_compare = pd.DataFrame({
        "Historical":      mean_ret.values,
        "Market Implied":  pi,
        "Black-Litterman": mu_bl
    }, index=tickers)
    print(df_compare.map(lambda x: f"{x:.1%}"))

    print("\n--- Portfólio BL: Máximo Sharpe ---")
    print(f"Retorno: {max_sharpe_bl['return']:.1%} | "
          f"Vol: {max_sharpe_bl['vol']:.1%} | "
          f"Sharpe: {max_sharpe_bl['sharpe']:.2f}")
    for ticker, w in sorted(zip(tickers, max_sharpe_bl["weights"]), key=lambda x: -x[1]):
        if w > 0.001:
            print(f"  {ticker:<12} {w:.1%}")

    # Gráficos
    plot_expected_returns_comparison(tickers, pi, mu_bl, mean_ret.values)
    plot_efficient_frontier(simulated_bl, max_sharpe_bl, min_vol_bl,
                            tickers, frontier_bl,
                            save_path="outputs/bl_efficient_frontier.png")