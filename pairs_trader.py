import numpy as np
import pandas as pd
import statsmodels.api as sm
import yfinance as yf


class PairsTrader:
    def __init__(
        self,
        stock1,
        stock2,
        start_date,
        end_date,
        threshold_params,
        method="zscore",
        leverage=1.0,
    ):
        self.stock1 = stock1
        self.stock2 = stock2
        self.start_date = start_date
        self.end_date = end_date
        self.threshold_params = threshold_params
        self.method = method
        self.leverage = leverage
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
        signals = pd.DataFrame(index=z_score.index)
        signals["z_score"] = z_score
        signals["position"] = 0

        # Long spread when z-score is below negative threshold
        signals.loc[z_score < -self.threshold_params, "position"] = 1

        # Short spread when z-score is above positive threshold
        signals.loc[z_score > self.threshold_params, "position"] = -1

        return signals

    def generate_signals_bollinger(self, spread):
        """Generate trading signals using Bollinger Bands"""
        window = self.threshold_params["window"]
        num_std = self.threshold_params["num_std"]

        signals = pd.DataFrame(index=spread.index)
        rolling_mean = spread.rolling(window=window).mean()
        rolling_std = spread.rolling(window=window).std()

        upper_band = rolling_mean + (rolling_std * num_std)
        lower_band = rolling_mean - (rolling_std * num_std)

        signals["position"] = 0
        signals.loc[spread < lower_band, "position"] = 1  # Long when below lower band
        signals.loc[spread > upper_band, "position"] = -1  # Short when above upper band

        return signals

    def generate_signals_rsi(self, spread):
        """Generate trading signals using RSI divergence"""
        period = self.threshold_params["rsi_period"]
        threshold = self.threshold_params["rsi_threshold"]

        signals = pd.DataFrame(index=spread.index)

        # Calculate RSI
        delta = spread.diff()

        # Separate gains and losses
        gains = delta.copy()
        losses = delta.copy()
        gains[gains < 0] = 0
        losses[losses > 0] = 0
        losses = abs(losses)  # Make losses positive

        # Calculate average gains and losses
        avg_gains = gains.rolling(window=period).mean()
        avg_losses = losses.rolling(window=period).mean()

        # Calculate RS and RSI
        rs = avg_gains / avg_losses
        rsi = 100 - (100 / (1 + rs))

        # Handle edge cases
        rsi = rsi.replace([np.inf, -np.inf], np.nan)
        rsi = rsi.fillna(50)  # Fill NaN with neutral value

        # Store RSI for plotting
        signals["rsi"] = rsi
        signals["position"] = 0

        # Generate signals
        signals.loc[rsi < threshold, "position"] = 1  # Long when oversold
        signals.loc[rsi > (100 - threshold), "position"] = -1  # Short when overbought

        return signals

    def generate_signals_kalman(self, spread):
        """Generate trading signals using Kalman Filter"""
        delta = self.threshold_params["delta"]
        R = self.threshold_params["R"]

        signals = pd.DataFrame(index=spread.index)
        signals["position"] = 0  # Initialize position column first
        n = len(spread)

        # Initialize Kalman Filter parameters
        x = spread.iloc[0]  # State estimate
        P = 1.0  # Error estimate
        Q = delta  # Process variance

        filtered_spread = np.zeros(n)
        filtered_spread[0] = x

        # Run Kalman Filter
        for t in range(1, n):
            # Predict
            x = x
            P = P + Q

            # Update
            K = P / (P + R)
            x = x + K * (spread.iloc[t] - x)
            P = (1 - K) * P

            filtered_spread[t] = x

        filtered_spread = pd.Series(filtered_spread, index=spread.index)
        signals["filtered_spread"] = filtered_spread

        # Generate signals based on filtered spread
        signals.loc[filtered_spread < -self.threshold_params["delta"], "position"] = 1
        signals.loc[filtered_spread > self.threshold_params["delta"], "position"] = -1

        # Debug prints
        print(f"First few positions: {signals['position'].head()}")
        print(f"Position value counts: {signals['position'].value_counts()}")

        return signals

    def run_backtest(self):
        """Run the complete backtest"""
        try:
            df = self.fetch_data()
            spread, hedge_ratio, is_reversed = self.calculate_spread(df)

            if self.method == "half-life":
                signals = self.generate_signals_half_life(spread, self.threshold_params)
                indicator = spread  # Use raw spread for visualization
                signals["spread"] = spread  # Store spread for reference
            elif self.method == "bollinger":
                signals = self.generate_signals_bollinger(spread)
                indicator = (
                    spread
                    - spread.rolling(window=self.threshold_params["window"]).mean()
                ) / spread.rolling(window=self.threshold_params["window"]).std()
            elif self.method == "rsi":
                signals = self.generate_signals_rsi(spread)
                indicator = signals["rsi"]  # Use RSI values instead of spread
            elif self.method == "kalman":
                signals = self.generate_signals_kalman(spread)
                indicator = signals["filtered_spread"]
            else:  # Default z-score method
                indicator = self.calculate_z_score(spread)
                signals = self.generate_signals_zscore(indicator)

            signals = self.calculate_returns(df, signals, hedge_ratio, is_reversed)
            stats = self.calculate_statistics(signals)

            return df, spread, indicator, signals, stats
        except Exception as e:
            print(f"Error in run_backtest: {str(e)}")
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
        # Check for NaN values first
        if df.isnull().any().any():
            raise ValueError("Data contains NaN values")

        # Check for minimum required data points
        if len(df) < 2:
            raise ValueError("Insufficient data points")

        # Convert to numpy arrays for faster operations
        stock1_data = df[self.stock1].to_numpy()
        stock2_data = df[self.stock2].to_numpy()

        # Calculate standard deviations for volatility comparison
        stock1_std = np.std(stock1_data, ddof=1)
        stock2_std = np.std(stock2_data, ddof=1)

        # Early exit for invalid data
        if stock1_std == 0 or stock2_std == 0:
            raise ValueError("One or both stocks show no price variation")

        # Choose the more volatile stock as dependent variable
        stock1_vol = np.std(np.diff(stock1_data) / stock1_data[:-1])
        stock2_vol = np.std(np.diff(stock2_data) / stock2_data[:-1])

        if stock1_vol > stock2_vol:
            X = sm.add_constant(df[self.stock2])
            y = df[self.stock1]
            is_reversed = True
        else:
            X = sm.add_constant(df[self.stock1])
            y = df[self.stock2]
            is_reversed = False

        # Fit the model on raw price data
        model = sm.OLS(y, X).fit()
        hedge_ratio = model.params.iloc[1]

        # Calculate spread using raw prices
        if is_reversed:
            spread = df[self.stock1] - hedge_ratio * df[self.stock2]
        else:
            spread = df[self.stock2] - hedge_ratio * df[self.stock1]

        # Handle potential NaN values in spread
        spread = spread.replace([np.inf, -np.inf], np.nan)
        if spread.isna().any():
            raise ValueError("Invalid values detected in spread calculation")

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
        returns[self.stock1] = df[self.stock1].pct_change(fill_method=None)
        returns[self.stock2] = df[self.stock2].pct_change(fill_method=None)

        # Calculate spread returns based on the regression direction
        if is_reversed:
            spread_returns = returns[self.stock1] - hedge_ratio * returns[self.stock2]
        else:
            spread_returns = returns[self.stock2] - hedge_ratio * returns[self.stock1]

        # Calculate strategy returns with explicit position handling
        signals["strategy_returns"] = 0.0
        # Use shifted positions for all returns
        signals["strategy_returns"] = (
            signals["position"].shift(1) * spread_returns * self.leverage
        )

        # Calculate cumulative returns
        signals["cumulative_returns"] = (1 + signals["strategy_returns"]).cumprod()

        return signals

    def calculate_statistics(self, signals):
        """Calculate trading statistics"""
        returns = signals["strategy_returns"].dropna()

        # Calculate number of trades by looking at position changes
        position_changes = signals["position"].diff()

        # A trade occurs when:
        # 1. Position changes from 1 to -1 or vice versa (absolute change of 2)
        # 2. Position changes from 0 to non-zero or vice versa (absolute change of 1)
        num_trades = (abs(position_changes) >= 1).sum()

        # Calculate win rate only if there are trades
        if num_trades > 0:
            # Calculate trade returns by summing returns between position changes
            trade_returns = []
            current_position = signals["position"].iloc[0]
            trade_start = signals.index[0]

            for idx in signals.index[1:]:
                if signals.loc[idx, "position"] != current_position:
                    # Position changed, calculate return for completed trade
                    if trade_start is not None:
                        trade_return = (
                            1 + signals.loc[trade_start:idx, "strategy_returns"]
                        ).prod() - 1
                        trade_returns.append(trade_return)

                    # Start new trade
                    trade_start = idx
                    current_position = signals.loc[idx, "position"]

            # Handle last trade if still open
            if trade_start is not None and trade_start != signals.index[-1]:
                trade_return = (
                    1 + signals.loc[trade_start:, "strategy_returns"]
                ).prod() - 1
                trade_returns.append(trade_return)

            # Calculate win rate from trade returns
            profitable_trades = sum(1 for ret in trade_returns if ret > 0)
            win_rate = profitable_trades / len(trade_returns) if trade_returns else 0.0
        else:
            win_rate = 0.0

        statistics = {
            "Total Returns": (signals["cumulative_returns"].iloc[-1] - 1),
            "Annual Returns": (1 + returns.mean()) ** 252 - 1,
            "Annual Volatility": returns.std() * np.sqrt(252),
            "Sharpe Ratio": (
                (returns.mean() / returns.std()) * np.sqrt(252)
                if returns.std() != 0
                else 0
            ),
            "Max Drawdown": (
                signals["cumulative_returns"] / signals["cumulative_returns"].cummax()
                - 1
            ).min(),
            "Number of Trades": num_trades,
            "Win Rate": win_rate,
        }

        return statistics
