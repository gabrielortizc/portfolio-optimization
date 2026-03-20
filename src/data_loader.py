import yfinance as yf
import pandas as pd

# -------------------------------------------------------------------
# Ativos
# -------------------------------------------------------------------
TICKERS_BR = ["ITUB4.SA", "PETR4.SA", "WEGE3.SA", "BBAS3.SA", "VALE3.SA"]
TICKERS_US = ["AAPL", "JPM", "XOM", "JNJ", "BRK-B"]
BENCHMARKS  = ["^BVSP", "SPY"]

ALL_TICKERS = TICKERS_BR + TICKERS_US + BENCHMARKS

# -------------------------------------------------------------------
# Funções
# -------------------------------------------------------------------
def download_prices(
    tickers: list = ALL_TICKERS,
    start: str    = "2019-01-01",
    end: str      = "2024-12-31",
    save_path: str = "data/prices.csv"
) -> pd.DataFrame:
    """
    Baixa preços de fechamento ajustados via yfinance e salva em CSV.
    Retorna DataFrame com colunas = tickers, índice = datas.
    """
    print(f"Baixando dados para {len(tickers)} ativos...")

    raw    = yf.download(tickers, start=start, end=end, auto_adjust=True)
    prices = raw["Close"]
    
    prices = prices.dropna(how="all")

    prices.to_csv(save_path)
    print(f"Salvo em '{save_path}' — shape: {prices.shape}")

    return prices


def load_prices(path: str = "data/prices.csv") -> pd.DataFrame:
    """Carrega preços do CSV já baixado."""
    return pd.read_csv(path, index_col=0, parse_dates=True)


# -------------------------------------------------------------------
# Teste rápido
# -------------------------------------------------------------------
if __name__ == "__main__":
    prices = download_prices()

    print("\n--- Últimas 5 linhas ---")
    print(prices.tail())

    print("\n--- Dados faltantes por ativo ---")
    print(prices.isna().sum())