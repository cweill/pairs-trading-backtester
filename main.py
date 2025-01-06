from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import statsmodels.api as sm
import streamlit as st
import yfinance as yf
from plotly.subplots import make_subplots


class PairsTrader:
    def __init__(self, stock1, stock2, start_date, end_date, z_score_threshold):
        self.stock1 = stock1
        self.stock2 = stock2
        self.start_date = start_date
        self.end_date = end_date
        self.z_score_threshold = z_score_threshold
        self.position = 0

    def fetch_data(self):
        """Fetch stock data from Yahoo Finance"""
        try:
            # Download complete data
            df1 = yf.download(self.stock1, start=self.start_date, end=self.end_date)
            df2 = yf.download(self.stock2, start=self.start_date, end=self.end_date)

            # Check if data is empty
            if df1.empty or df2.empty:
                raise ValueError(
                    f"No data found for one or both symbols: {self.stock1}, {self.stock2}"
                )

            # Try to get Adjusted Close, fall back to Close if not available
            try:
                price1 = df1["Adj Close"].squeeze()  # Convert to Series if needed
                price2 = df2["Adj Close"].squeeze()  # Convert to Series if needed
            except KeyError:
                price1 = df1["Close"].squeeze()  # Convert to Series if needed
                price2 = df2["Close"].squeeze()  # Convert to Series if needed

            # Ensure we have Series objects
            price1 = pd.Series(price1, index=df1.index)
            price2 = pd.Series(price2, index=df2.index)

            # Reindex both series to handle any missing dates
            all_dates = sorted(set(price1.index) | set(price2.index))
            price1 = price1.reindex(all_dates)
            price2 = price2.reindex(all_dates)

            # Combine and clean data
            df = pd.DataFrame(
                {self.stock1: price1, self.stock2: price2}, index=all_dates
            )

            # Check for sufficient data
            if len(df) < 20:  # Minimum required for z-score calculation
                raise ValueError(
                    "Insufficient data for analysis (minimum 20 data points required)"
                )

            df = df.dropna()

            if df.empty:
                raise ValueError(
                    "No overlapping trading dates found between the two stocks"
                )

            return df

        except Exception as e:
            raise Exception(f"Error fetching data: {str(e)}")

    def calculate_spread(self, df):
        """Calculate the spread between two stocks using linear regression"""
        X = sm.add_constant(df[self.stock1])
        model = sm.OLS(df[self.stock2], X).fit()
        hedge_ratio = model.params[1]
        spread = df[self.stock2] - hedge_ratio * df[self.stock1]
        return spread, hedge_ratio

    def calculate_z_score(self, spread):
        """Calculate z-score of the spread"""
        mean = spread.rolling(window=20).mean()
        std = spread.rolling(window=20).std()
        z_score = (spread - mean) / std
        return z_score

    def generate_signals(self, z_score):
        """Generate trading signals based on z-score"""
        signals = pd.DataFrame(index=z_score.index)
        signals["z_score"] = z_score
        signals["position"] = 0

        # Long spread when z-score is below negative threshold
        signals.loc[z_score < -self.z_score_threshold, "position"] = 1

        # Short spread when z-score is above positive threshold
        signals.loc[z_score > self.z_score_threshold, "position"] = -1

        return signals

    def calculate_returns(self, df, signals, hedge_ratio):
        """Calculate strategy returns"""
        # Calculate daily returns for both stocks
        returns = pd.DataFrame(index=signals.index)
        returns[self.stock1] = df[self.stock1].pct_change()
        returns[self.stock2] = df[self.stock2].pct_change()

        # Calculate spread returns
        spread_returns = returns[self.stock2] - hedge_ratio * returns[self.stock1]

        # Calculate strategy returns
        signals["strategy_returns"] = signals["position"].shift(1) * spread_returns
        signals["cumulative_returns"] = (1 + signals["strategy_returns"]).cumprod()

        return signals

    def calculate_statistics(self, signals):
        """Calculate trading statistics"""
        stats = {}
        returns = signals["strategy_returns"].dropna()

        stats["Total Returns"] = (
            f"{(signals['cumulative_returns'].iloc[-1] - 1) * 100:.2f}%"
        )
        stats["Annual Returns"] = f"{returns.mean() * 252 * 100:.2f}%"
        stats["Annual Volatility"] = f"{returns.std() * np.sqrt(252) * 100:.2f}%"
        stats["Sharpe Ratio"] = (
            f"{(returns.mean() * 252) / (returns.std() * np.sqrt(252)):.2f}"
        )
        stats["Max Drawdown"] = (
            f"{((1 + returns).cumprod().div((1 + returns).cumprod().cummax()) - 1).min() * 100:.2f}%"
        )

        return stats

    def run_backtest(self):
        """Run the complete backtest"""
        df = self.fetch_data()
        spread, hedge_ratio = self.calculate_spread(df)
        z_score = self.calculate_z_score(spread)
        signals = self.generate_signals(z_score)
        signals = self.calculate_returns(df, signals, hedge_ratio)
        stats = self.calculate_statistics(signals)

        return df, spread, z_score, signals, stats


def plot_backtest_results(df, z_score, signals):
    """Create plots for the backtest results"""
    fig = make_subplots(
        rows=3,
        cols=1,
        subplot_titles=("Stock Prices", "Z-Score", "Cumulative Returns"),
        vertical_spacing=0.1,
        row_heights=[0.4, 0.3, 0.3],
    )

    # Plot stock prices
    fig.add_trace(
        go.Scatter(x=df.index, y=df.iloc[:, 0], name=df.columns[0]), row=1, col=1
    )
    fig.add_trace(
        go.Scatter(x=df.index, y=df.iloc[:, 1], name=df.columns[1]), row=1, col=1
    )

    # Plot z-score
    fig.add_trace(go.Scatter(x=z_score.index, y=z_score, name="Z-Score"), row=2, col=1)
    fig.add_hline(y=2, line_dash="dash", line_color="red", row=2, col=1)
    fig.add_hline(y=-2, line_dash="dash", line_color="red", row=2, col=1)

    # Plot cumulative returns
    fig.add_trace(
        go.Scatter(
            x=signals.index, y=signals["cumulative_returns"], name="Cumulative Returns"
        ),
        row=3,
        col=1,
    )

    fig.update_layout(height=900, showlegend=True)
    return fig


def main():
    st.title("Pairs Trading Backtester")

    # Sidebar inputs
    st.sidebar.header("Trading Parameters")

    stock1 = st.sidebar.text_input("Stock 1 Symbol", value="AAPL")
    stock2 = st.sidebar.text_input("Stock 2 Symbol", value="MSFT")

    def get_ytd_days():
        now = datetime.now()
        start_of_year = datetime(now.year, 1, 1)
        return (now - start_of_year).days

    lookback_options = {
        "3 Months": 90,
        "6 Months": 180,
        "YTD": get_ytd_days(),
        "1 Year": 365,
        "2 Years": 730,
        "5 Years": 1825,
        "10 Years": 3650,
    }
    lookback = st.sidebar.selectbox("Lookback Period", list(lookback_options.keys()))

    end_date = datetime.now()
    start_date = end_date - timedelta(days=lookback_options[lookback])

    z_score_threshold = st.sidebar.slider(
        "Z-Score Threshold", min_value=1.0, max_value=3.0, value=2.0, step=0.1
    )

    # Run backtest button
    if st.sidebar.button("Run Backtest"):
        with st.spinner("Running backtest..."):
            try:
                # Initialize and run backtest
                trader = PairsTrader(
                    stock1, stock2, start_date, end_date, z_score_threshold
                )
                df, spread, z_score, signals, stats = trader.run_backtest()

                # Display statistics
                st.header("Trading Statistics")
                st.table(
                    pd.DataFrame(
                        list(stats.items()), columns=["Metric", "Value"]
                    ).set_index("Metric")
                )

                # Display plots
                st.header("Backtest Results")
                fig = plot_backtest_results(df, z_score, signals)
                st.plotly_chart(fig, use_container_width=True)

                # Display correlation
                correlation = df[stock1].corr(df[stock2])
                st.write(
                    f"Correlation between {stock1} and {stock2}: {correlation:.2f}"
                )

            except Exception as e:
                st.error(f"An error occurred: {str(e)}")


if __name__ == "__main__":
    main()
