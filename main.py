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
