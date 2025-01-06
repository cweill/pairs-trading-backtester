import os
from datetime import datetime, timedelta

import pandas as pd
import yfinance as yf


def fetch_and_save_test_data():
    """
    Fetch test data for pairs trading and save to CSV files.
    Each pair is saved in its own CSV file with both stocks' prices.
    """
    # Define pairs to test with
    pairs = [
        ("KO", "PEP"),  # Coca-Cola and Pepsi
        ("JPM", "GS"),  # JP Morgan and Goldman Sachs
        ("CVX", "XOM"),  # Chevron and Exxon Mobil
    ]

    # Create test_data directory if it doesn't exist
    if not os.path.exists("test_data"):
        os.makedirs("test_data")

    # Fetch one year of data for each pair
    end_date = datetime.now()
    start_date = end_date - timedelta(days=365)

    for stock1, stock2 in pairs:
        print(f"Fetching data for {stock1}-{stock2}...")

        try:
            # Download data
            df1 = yf.download(stock1, start=start_date, end=end_date)
            df2 = yf.download(stock2, start=start_date, end=end_date)

            # Print available columns for debugging
            print(f"{stock1} columns: {df1.columns.tolist()}")
            print(f"{stock2} columns: {df2.columns.tolist()}")

            # Get Close prices using multi-index access
            if isinstance(df1.columns, pd.MultiIndex):
                price1 = df1[("Close", stock1)]
                price2 = df2[("Close", stock2)]
            else:
                price1 = df1["Close"]
                price2 = df2["Close"]

            # Create DataFrame with proper index
            pair_data = pd.DataFrame({stock1: price1, stock2: price2}, index=df1.index)

            # Clean data
            pair_data = pair_data.dropna()

            # Calculate and print correlation
            corr = pair_data[stock1].corr(pair_data[stock2])
            print(f"Correlation between {stock1} and {stock2}: {corr:.3f}")
            print(f"Number of trading days: {len(pair_data)}")
            print(f"Date range: {pair_data.index[0]} to {pair_data.index[-1]}\n")

            # Save to CSV
            filename = f"testdata/{stock1}_{stock2}_prices.csv"
            pair_data.to_csv(filename)
            print(f"Saved to {filename}")

        except Exception as e:
            print(f"Error fetching data for {stock1}-{stock2}: {str(e)}")
            raise  # Re-raise the exception to see full traceback during development


if __name__ == "__main__":
    fetch_and_save_test_data()
