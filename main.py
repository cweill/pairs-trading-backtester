from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import statsmodels.api as sm
import streamlit as st
import yfinance as yf
from plotly.subplots import make_subplots


class PairsTrader:
    def __init__(
        self, stock1, stock2, start_date, end_date, threshold_params, method="zscore"
    ):
        self.stock1 = stock1
        self.stock2 = stock2
        self.start_date = start_date
        self.end_date = end_date
        self.threshold_params = (
            threshold_params  # Either z_score threshold or half-life period
        )
        self.method = method
        self.position = 0

    def calculate_half_life(self, spread):
        """Calculate the half-life of mean reversion"""
        try:
            # Remove any NaN values
            spread = spread.dropna()

            # Create lag version of spread
            spread_lag = spread.shift(1)
            spread_diff = spread - spread_lag

            # Clean up pairs
            spread_lag = spread_lag.dropna()
            spread_diff = spread_diff.dropna()

            # Add constant to lagged spread
            spread_lag = sm.add_constant(spread_lag)

            # Regression of difference against levels
            model = sm.OLS(spread_diff, spread_lag)
            results = model.fit()

            # Calculate half-life
            half_life = -np.log(2) / results.params[1]

            # If half_life is negative or too large, default to 21 days
            if half_life <= 0 or half_life > 252:  # 252 trading days in a year
                return 21

            return round(abs(half_life))

        except Exception as e:
            print(f"Error calculating half-life: {str(e)}")  # Debug print
            return 21  # Default to 21 days if calculation fails

    def generate_signals_half_life(self, spread, half_life):
        """Generate trading signals based on half-life mean reversion"""
        signals = pd.DataFrame(index=spread.index)

        # Calculate mean and standard deviation using half-life period
        mean = spread.ewm(halflife=half_life).mean()
        std = spread.ewm(halflife=half_life).std()

        # Calculate normalized spread
        normalized_spread = (spread - mean) / std
        signals["normalized_spread"] = normalized_spread
        signals["position"] = 0

        # Generate signals based on standard deviations
        signals.loc[normalized_spread < -2, "position"] = (
            1  # Long position when spread is 2 std below mean
        )
        signals.loc[normalized_spread > 2, "position"] = (
            -1
        )  # Short position when spread is 2 std above mean

        return signals

    def generate_signals_zscore(self, z_score):
        """Generate trading signals based on z-score"""
        """Generate trading signals based on z-score"""
        signals = pd.DataFrame(index=z_score.index)
        signals["z_score"] = z_score
        signals["position"] = 0

        # Long spread when z-score is below negative threshold
        signals.loc[z_score < -self.threshold_params, "position"] = 1

        # Short spread when z-score is above positive threshold
        signals.loc[z_score > self.threshold_params, "position"] = -1

        return signals

    def run_backtest(self):
        """Run the complete backtest"""
        try:
            df = self.fetch_data()
            spread, hedge_ratio, is_reversed = self.calculate_spread(df)

            if self.method == "half-life":  # Changed from 'halflife' to 'half-life'
                half_life = self.calculate_half_life(spread)
                signals = self.generate_signals_half_life(spread, self.threshold_params)
                z_score = signals["normalized_spread"]  # For plotting purposes
            else:
                z_score = self.calculate_z_score(spread)
                signals = self.generate_signals_zscore(z_score)

            signals = self.calculate_returns(df, signals, hedge_ratio, is_reversed)
            stats = self.calculate_statistics(signals)

            return df, spread, z_score, signals, stats
        except Exception as e:
            print(f"Error in run_backtest: {str(e)}")  # Debug print
            raise

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
        # Calculate correlations to determine which stock to use as independent variable
        correlation = df[self.stock1].corr(df[self.stock2])

        # Standardize both price series
        stock1_std = (df[self.stock1] - df[self.stock1].mean()) / df[self.stock1].std()
        stock2_std = (df[self.stock2] - df[self.stock2].mean()) / df[self.stock2].std()

        # Create standardized DataFrame
        df_std = pd.DataFrame({self.stock1: stock1_std, self.stock2: stock2_std})

        # Choose the more volatile stock as dependent variable
        stock1_vol = df[self.stock1].pct_change().std()
        stock2_vol = df[self.stock2].pct_change().std()

        if stock1_vol > stock2_vol:
            X = sm.add_constant(df_std[self.stock2])
            y = df_std[self.stock1]
            is_reversed = True
        else:
            X = sm.add_constant(df_std[self.stock1])
            y = df_std[self.stock2]
            is_reversed = False

        # Fit the model on standardized data
        model = sm.OLS(y, X).fit()
        hedge_ratio = model.params[1]

        # Calculate spread on standardized data
        if is_reversed:
            spread = df_std[self.stock1] - hedge_ratio * df_std[self.stock2]
        else:
            spread = df_std[self.stock2] - hedge_ratio * df_std[self.stock1]

        return spread, hedge_ratio, is_reversed

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

    def calculate_returns(self, df, signals, hedge_ratio, is_reversed):
        """Calculate strategy returns"""
        # Calculate daily returns for both stocks
        returns = pd.DataFrame(index=signals.index)
        returns[self.stock1] = df[self.stock1].pct_change()
        returns[self.stock2] = df[self.stock2].pct_change()

        # Calculate spread returns based on the regression direction
        if is_reversed:
            spread_returns = returns[self.stock1] - hedge_ratio * returns[self.stock2]
        else:
            spread_returns = returns[self.stock2] - hedge_ratio * returns[self.stock1]

        # Calculate strategy returns
        signals["strategy_returns"] = signals["position"].shift(1) * spread_returns
        signals["cumulative_returns"] = (1 + signals["strategy_returns"]).cumprod()

        return signals

    def calculate_statistics(self, signals):
        """Calculate trading statistics"""
        stats = {}
        returns = signals["strategy_returns"].dropna()

        # Handle empty or insufficient data
        if len(returns) < 2:
            return {
                "Total Returns": "N/A",
                "Annual Returns": "N/A",
                "Annual Volatility": "N/A",
                "Sharpe Ratio": "N/A",
                "Max Drawdown": "N/A",
            }

        # Calculate trading days in the sample
        trading_days = len(returns)
        annualization_factor = 252 / trading_days

        # Calculate total return
        total_return = (signals["cumulative_returns"].iloc[-1] - 1) * 100
        stats["Total Returns"] = f"{total_return:.2f}%"

        # Calculate annualized return
        period_return = returns.mean() * trading_days
        annual_return = period_return * annualization_factor
        stats["Annual Returns"] = f"{annual_return * 100:.2f}%"

        # Calculate annualized volatility
        annual_vol = returns.std() * np.sqrt(252)
        stats["Annual Volatility"] = f"{annual_vol * 100:.2f}%"

        # Calculate Sharpe Ratio
        # Using 0 as risk-free rate for simplicity
        if annual_vol != 0:  # Prevent division by zero
            sharpe = annual_return / annual_vol
            stats["Sharpe Ratio"] = f"{sharpe:.2f}"
        else:
            stats["Sharpe Ratio"] = "N/A"

        # Calculate Maximum Drawdown
        cum_returns = (1 + returns).cumprod()
        rolling_max = cum_returns.cummax()
        drawdowns = cum_returns / rolling_max - 1
        max_drawdown = drawdowns.min() * 100
        stats["Max Drawdown"] = f"{max_drawdown:.2f}%"

        # Add number of trades
        position_changes = signals["position"].diff().fillna(0)
        num_trades = (position_changes != 0).sum()
        stats["Number of Trades"] = f"{num_trades}"

        # Add win rate if there are trades
        if num_trades > 0:
            profitable_trades = (returns > 0).sum()
            win_rate = (profitable_trades / num_trades) * 100
            stats["Win Rate"] = f"{win_rate:.2f}%"
        else:
            stats["Win Rate"] = "N/A"

        return stats


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

    # Add method selection
    method = st.sidebar.selectbox(
        "Trading Method",
        ["Z-Score", "Half-life"],
        format_func=lambda x: (
            "Z-Score Threshold" if x == "Z-Score" else "Half-life Mean Reversion"
        ),
    )

    # Dynamic threshold input based on method
    if method == "Z-Score":
        threshold_params = st.sidebar.slider(
            "Z-Score Threshold", min_value=1.0, max_value=3.0, value=2.0, step=0.1
        )
    else:
        threshold_params = st.sidebar.slider(
            "Half-life Period (days)",
            min_value=1,
            max_value=100,
            value=21,
            step=1,
            help="Number of days for half-life calculation",
        )

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

    # Run backtest button
    if st.sidebar.button("Run Backtest"):
        with st.spinner("Running backtest..."):
            try:
                # Initialize and run backtest with selected method
                # Convert method name to lowercase and replace space with hyphen
                method_param = method.lower().replace(" ", "-")
                trader = PairsTrader(
                    stock1, stock2, start_date, end_date, threshold_params, method_param
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

                if method == "Half-life":
                    half_life = trader.calculate_half_life(spread)
                    st.write(f"Calculated Half-life: {half_life:.2f} days")

            except Exception as e:
                st.error(f"An error occurred: {str(e)}")


if __name__ == "__main__":
    main()
