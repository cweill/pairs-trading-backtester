from datetime import datetime, timedelta

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from pairs_trader import PairsTrader


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

    # Create a form in the sidebar
    with st.sidebar:
        st.header("Trading Parameters")

        # Stock inputs
        stock1 = st.text_input("Stock 1 Symbol", value="AAPL")
        stock2 = st.text_input("Stock 2 Symbol", value="MSFT")

        # Method selection
        method = st.selectbox(
            "Trading Method",
            ["Z-Score", "Half-life", "Kalman", "Bollinger", "RSI"],
            format_func=lambda x: {
                "Z-Score": "Z-Score Threshold",
                "Half-life": "Half-life Mean Reversion",
                "Kalman": "Kalman Filter",
                "Bollinger": "Bollinger Bands",
                "RSI": "RSI Divergence",
            }[x],
        )

        # Dynamic threshold input based on method
        if method == "Z-Score":
            threshold_params = st.slider(
                "Z-Score Threshold",
                min_value=1.0,
                max_value=3.0,
                value=1.5,
                step=0.1,
            )
        elif method == "Bollinger":
            threshold_params = {
                "window": st.slider("Window Size", 5, 50, 20),
                "num_std": st.slider(
                    "Number of Standard Deviations", 1.0, 3.0, 2.0, 0.1
                ),
            }
        elif method == "RSI":
            threshold_params = {
                "rsi_period": st.slider("RSI Period", 5, 30, 14),
                "rsi_threshold": st.slider("RSI Threshold", 20, 40, 30),
            }
        elif method == "Kalman":
            threshold_params = {
                "delta": st.slider(
                    "Delta (Kalman measurement noise)", 1e-4, 1e-1, 1e-2, format="%.4f"
                ),
                "R": st.slider(
                    "R (Kalman process noise)", 1e-4, 1e-1, 1e-2, format="%.4f"
                ),
            }
        else:
            threshold_params = st.slider(
                "Half-life Period (days)",
                min_value=1,
                max_value=100,
                value=21,
                step=1,
                help="Number of days for half-life calculation",
            )

        leverage = st.slider(
            "Leverage",
            min_value=1.0,
            max_value=5.0,
            value=1.0,
            step=0.5,
            help="Trading leverage multiplier. Use with caution as it amplifies both gains and losses.",
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
        lookback = st.selectbox("Lookback Period", list(lookback_options.keys()))

        # Create an empty placeholder for the run button
        button_placeholder = st.empty()
        # Add the manual run button at the end of the sidebar
        run_button = button_placeholder.button("Run Backtest")

    # Calculate dates
    end_date = datetime.now()
    start_date = end_date - timedelta(days=lookback_options[lookback])

    # Create a hash of all input parameters
    input_params_hash = hash(
        f"{stock1}{stock2}{method}{str(threshold_params)}{leverage}{lookback}"
    )

    # Store the hash in session state if it doesn't exist
    if "last_params_hash" not in st.session_state:
        st.session_state.last_params_hash = None

    # Run backtest if inputs changed or button pressed
    if input_params_hash != st.session_state.last_params_hash or run_button:
        st.session_state.last_params_hash = input_params_hash
        with st.spinner("Running backtest..."):
            try:
                # Initialize and run backtest with selected method
                method_param = method.lower().replace(" ", "-")
                trader = PairsTrader(
                    stock1,
                    stock2,
                    start_date,
                    end_date,
                    threshold_params,
                    method_param,
                    leverage,
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
