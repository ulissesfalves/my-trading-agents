import yfinance as yf
import pandas as pd
import requests
from bs4 import BeautifulSoup
from datetime import date, timedelta

class BreakoutScanner:
    """
    Scans the market to find stocks breaking out of a consolidation pattern.
    """
    def __init__(self, market='sp500', consolidation_period=20, volume_factor=1.5):
        self.market = market
        self.consolidation_period = consolidation_period
        self.volume_factor = volume_factor
        self.tickers = self.get_tickers()

    def get_tickers(self):
        """
        Gets a list of tickers for the specified market.
        Currently supports S&P 500.
        """
        if self.market == 'sp500':
            # Scrape S&P 500 tickers from Wikipedia
            url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
            response = requests.get(url)
            soup = BeautifulSoup(response.text, 'html.parser')
            table = soup.find('table', {'id': 'constituents'})
            tickers = []
            for row in table.find_all('tr')[1:]:
                ticker = row.find('td').text.strip()
                # Correct for tickers that might have a different format in Yahoo Finance
                ticker = ticker.replace('.', '-')
                tickers.append(ticker)
            return tickers
        else:
            raise ValueError("Unsupported market. Currently only 'sp500' is supported.")

    def scan(self):
        """
        Performs the market scan to find breakout candidates.
        """
        candidates = []
        end_date = date.today()
        start_date = end_date - timedelta(days=self.consolidation_period * 2) # Fetch more data for moving averages

        print(f"Scanning {len(self.tickers)} tickers...")
        for ticker in self.tickers:
            try:
                # Download daily data
                data = yf.download(ticker, start=start_date, end=end_date, progress=False, auto_adjust=True)

                if len(data) < self.consolidation_period:
                    continue

                # Define the consolidation period data
                cons_data = data.iloc[-self.consolidation_period-1:-1]
                
                # Today's data
                latest_data = data.iloc[-1]

                # --- Breakout Logic ---
                # 1. Define consolidation range (highest high in the period)
                consolidation_high = cons_data['High'].max()
                
                # 2. Check for price breakout
                is_breakout = latest_data['Close'] > consolidation_high

                # 3. Check for volume confirmation
                average_volume = cons_data['Volume'].mean()
                is_volume_confirmed = latest_data['Volume'] > average_volume * self.volume_factor

                if is_breakout and is_volume_confirmed:
                    candidates.append(ticker)

            except Exception as e:
                # Silently fail for tickers with data issues
                # print(f"Could not process {ticker}: {e}")
                pass
        
        return candidates

if __name__ == '__main__':
    # Example usage:
    scanner = BreakoutScanner()
    breakout_stocks = scanner.scan()
    print("\nPotential Breakout Stocks Found:")
    print(breakout_stocks)

