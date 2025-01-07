from datetime import datetime, timedelta
from urllib.parse import quote, unquote

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from pairs_trader import PairsTrader


def plot_backtest_results(df, indicator, signals, spread, method, threshold_params):
    """Create plots for the backtest results"""
    fig = make_subplots(
        rows=4,
        cols=1,
        subplot_titles=(
            "Stock Prices",
            "Spread",
            {
                "Z-Score": "Z-Score",
                "Half-life": "Normalized Spread",
                "Kalman": "Filtered Spread",
                "Bollinger": "Spread with Bands",
                "RSI": "RSI",
            }[method],
            "Cumulative Returns",
        ),
        vertical_spacing=0.08,
        row_heights=[0.28, 0.22, 0.22, 0.28],
    )

    # Plot stock prices
    fig.add_trace(
        go.Scatter(x=df.index, y=df.iloc[:, 0], name=df.columns[0]), row=1, col=1
    )
    fig.add_trace(
        go.Scatter(x=df.index, y=df.iloc[:, 1], name=df.columns[1]), row=1, col=1
    )

    # Plot spread
    fig.add_trace(go.Scatter(x=signals.index, y=spread, name="Spread"), row=2, col=1)

    # Plot indicator based on method
    if method == "Z-Score":
        fig.add_trace(
            go.Scatter(x=indicator.index, y=indicator, name="Z-Score"), row=3, col=1
        )
        threshold = float(threshold_params)
        fig.add_hline(y=threshold, line_dash="dash", line_color="red", row=3, col=1)
        fig.add_hline(y=-threshold, line_dash="dash", line_color="red", row=3, col=1)
    elif method == "Kalman":
        fig.add_trace(
            go.Scatter(x=indicator.index, y=indicator, name="Filtered Spread"),
            row=3,
            col=1,
        )
        delta = float(threshold_params["delta"])
        fig.add_hline(y=delta, line_dash="dash", line_color="red", row=3, col=1)
        fig.add_hline(y=-delta, line_dash="dash", line_color="red", row=3, col=1)
    elif method == "RSI":
        fig.add_trace(
            go.Scatter(x=indicator.index, y=indicator, name="RSI"), row=3, col=1
        )
        threshold = int(threshold_params["rsi_threshold"])
        fig.add_hline(y=threshold, line_dash="dash", line_color="red", row=3, col=1)
        fig.add_hline(
            y=100 - threshold, line_dash="dash", line_color="red", row=3, col=1
        )
    elif method == "Bollinger":
        fig.add_trace(
            go.Scatter(x=indicator.index, y=indicator, name="Spread"), row=3, col=1
        )
        if "upper_band" in signals.columns and "lower_band" in signals.columns:
            fig.add_trace(
                go.Scatter(
                    x=signals.index,
                    y=signals["upper_band"],
                    name="Upper Band",
                    line=dict(dash="dash"),
                ),
                row=3,
                col=1,
            )
            fig.add_trace(
                go.Scatter(
                    x=signals.index,
                    y=signals["lower_band"],
                    name="Lower Band",
                    line=dict(dash="dash"),
                ),
                row=3,
                col=1,
            )

    # Plot cumulative returns
    fig.add_trace(
        go.Scatter(
            x=signals.index, y=signals["cumulative_returns"], name="Cumulative Returns"
        ),
        row=4,
        col=1,
    )

    fig.update_layout(
        height=1200,
        showlegend=True,
        margin=dict(t=30, b=30),
    )
    return fig


def format_statistics_for_display(statistics):
    """Format statistics dictionary values for display"""
    formatted_stats = {
        "Total Returns": f"{statistics['Total Returns']:.2%}",
        "Annual Returns": f"{statistics['Annual Returns']:.2%}",
        "Annual Volatility": f"{statistics['Annual Volatility']:.2%}",
        "Sharpe Ratio": f"{statistics['Sharpe Ratio']:.2f}",
        "Max Drawdown": f"{statistics['Max Drawdown']:.2%}",
        "Number of Trades": f"{statistics['Number of Trades']}",
        "Win Rate": f"{statistics['Win Rate']:.2%}",
    }
    return formatted_stats


def get_query_params():
    """Get current query parameters"""
    query_params = st.query_params

    # Helper function to get param with default and handle URL encoding
    def get_param(key, default):
        value = query_params.get(key, default)
        if isinstance(value, list):
            value = value[0]
        # URL decode the value if it's a string and not empty
        if isinstance(value, str):
            value = unquote(value)
            # Return default if value is empty
            if not value:
                return default
        return value

    # Define valid methods and lookback periods
    valid_methods = ["Z-Score", "Half-life", "Kalman", "Bollinger", "RSI"]
    valid_lookbacks = [
        "3 Months",
        "6 Months",
        "YTD",
        "1 Year",
        "2 Years",
        "5 Years",
        "10 Years",
    ]

    # Get method from params with case-insensitive matching
    method_param = get_param("method", "Z-Score")
    method = next(
        (m for m in valid_methods if m.lower() == method_param.lower()),
        "Z-Score",  # default if no match
    )

    # Get lookback with validation
    lookback_param = get_param("lookback", "10 Years")
    lookback = next(
        (l for l in valid_lookbacks if l == lookback_param),
        "10 Years",  # default if no match
    )

    # Create threshold dictionary with defaults based on method
    threshold = {}
    if method == "Z-Score":
        threshold["zscore"] = get_param("zscore", "1.5")
    elif method == "Kalman":
        threshold["delta"] = get_param("delta", "0.01")
        threshold["R"] = get_param("R", "0.01")
    elif method == "Bollinger":
        threshold["window"] = get_param("window", "20")
        threshold["num_std"] = get_param("num_std", "2.0")
    elif method == "RSI":
        threshold["rsi_period"] = get_param("rsi_period", "14")
        threshold["rsi_threshold"] = get_param("rsi_threshold", "30")
    else:  # Half-life
        threshold["half_life"] = get_param("half_life", "21")

    return {
        "stock1": get_param("stock1", "AAPL"),
        "stock2": get_param("stock2", "MSFT"),
        "method": method,
        "lookback": lookback,
        "leverage": float(get_param("leverage", "1.0")),
        "threshold": threshold,
    }


def update_query_params(params):
    """Update query parameters in URL"""

    # Helper function to URL encode values
    def encode_param(value):
        return quote(str(value))

    # Ensure method is in correct case
    valid_methods = ["Z-Score", "Half-life", "Kalman", "Bollinger", "RSI"]
    valid_lookbacks = [
        "3 Months",
        "6 Months",
        "YTD",
        "1 Year",
        "2 Years",
        "5 Years",
        "10 Years",
    ]

    method = next(
        (m for m in valid_methods if m.lower() == params["method"].lower()), "Z-Score"
    )

    lookback = next((l for l in valid_lookbacks if l == params["lookback"]), "10 Years")

    # Flatten threshold parameters into main params with URL encoding
    query_params = {
        "stock1": encode_param(params["stock1"]),
        "stock2": encode_param(params["stock2"]),
        "method": encode_param(method),
        "lookback": encode_param(lookback),
        "leverage": encode_param(str(params["leverage"])),
    }

    # Add method-specific threshold parameters with URL encoding
    if method == "Z-Score":
        query_params["zscore"] = encode_param(params["threshold"]["zscore"])
    elif method == "Kalman":
        query_params["delta"] = encode_param(params["threshold"]["delta"])
        query_params["R"] = encode_param(params["threshold"]["R"])
    elif method == "Bollinger":
        query_params["window"] = encode_param(params["threshold"]["window"])
        query_params["num_std"] = encode_param(params["threshold"]["num_std"])
    elif method == "RSI":
        query_params["rsi_period"] = encode_param(params["threshold"]["rsi_period"])
        query_params["rsi_threshold"] = encode_param(
            params["threshold"]["rsi_threshold"]
        )
    else:  # Half-life
        query_params["half_life"] = encode_param(params["threshold"]["half_life"])

    st.query_params.update(query_params)


def get_start_date(lookback):
    """Calculate start date based on lookback period"""
    end_date = datetime.now()
    lookback_options = {
        "3 Months": 90,
        "6 Months": 180,
        "YTD": (end_date - datetime(end_date.year, 1, 1)).days,
        "1 Year": 365,
        "2 Years": 730,
        "5 Years": 1825,
        "10 Years": 3650,
    }
    days = lookback_options.get(lookback, 3650)  # Default to 10 years
    return end_date - timedelta(days=days)


def main():
    st.set_page_config(layout="wide")
    st.title("Pairs Trading Backtester")

    # Create sidebar for inputs
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
                max_value=5.0,
                value=1.5,
                step=0.1,
            )
        elif method == "Kalman":
            threshold_params = {
                "delta": st.select_slider(
                    "Delta (Kalman measurement noise)",
                    options=[1e-4, 1e-3, 1e-2, 1e-1],
                    value=1e-2,
                ),
                "R": st.select_slider(
                    "R (Kalman process noise)",
                    options=[1e-4, 1e-3, 1e-2, 1e-1],
                    value=1e-2,
                ),
            }
        elif method == "Bollinger":
            threshold_params = {
                "window": st.slider("Window Size", 5, 50, value=20),
                "num_std": st.slider(
                    "Number of Standard Deviations",
                    1.0,
                    3.0,
                    value=2.0,
                    step=0.1,
                ),
            }
        elif method == "RSI":
            threshold_params = {
                "rsi_period": st.slider("RSI Period", 5, 30, value=14),
                "rsi_threshold": st.slider("RSI Threshold", 20, 40, value=30),
            }
        else:  # Half-life
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
        lookback = st.selectbox(
            "Lookback Period",
            list(lookback_options.keys()),
            index=len(lookback_options) - 1,  # Default to 10 Years
        )

    # Always run backtest (not just on button click)
    with st.spinner("Running backtest..."):
        try:
            # Initialize and run backtest
            method_param = method.lower().replace(" ", "-")
            end_date = datetime.now()
            start_date = end_date - timedelta(days=lookback_options[lookback])

            trader = PairsTrader(
                stock1,
                stock2,
                start_date,
                end_date,
                threshold_params,
                method_param,
                leverage,
            )
            df, spread, indicator, signals, stats = trader.run_backtest()

            # Create three columns for layout: stats and two chart columns
            col1, col2, col3 = st.columns([1, 2, 2])

            # Left column for statistics and correlation
            with col1:
                st.subheader("Trading Statistics")
                formatted_stats = format_statistics_for_display(stats)
                st.table(formatted_stats)

                # Display correlation
                correlation = df[stock1].corr(df[stock2])
                st.write(
                    f"Correlation between {stock1} and {stock2}: {correlation:.2f}"
                )

                if method == "Half-life":
                    half_life = trader.calculate_half_life(spread)
                    st.write(f"Calculated Half-life: {half_life:.2f} days")

            # Create two separate figures for the charts
            fig1 = make_subplots(
                rows=2,
                cols=1,
                subplot_titles=("Stock Prices", "Spread"),
                vertical_spacing=0.15,
                row_heights=[0.6, 0.4],
            )

            # Plot stock prices
            fig1.add_trace(
                go.Scatter(x=df.index, y=df.iloc[:, 0], name=df.columns[0]),
                row=1,
                col=1,
            )
            fig1.add_trace(
                go.Scatter(x=df.index, y=df.iloc[:, 1], name=df.columns[1]),
                row=1,
                col=1,
            )

            # Plot spread
            fig1.add_trace(
                go.Scatter(x=signals.index, y=spread, name="Spread"), row=2, col=1
            )

            fig1.update_layout(height=600, showlegend=True, margin=dict(t=30, b=30))

            # Create second figure for indicator and returns
            fig2 = make_subplots(
                rows=2,
                cols=1,
                subplot_titles=(
                    {
                        "Z-Score": "Z-Score",
                        "Half-life": "Spread",
                        "Kalman": "Filtered Spread",
                        "Bollinger": "Spread with Bands",
                        "RSI": "RSI",
                    }[method],
                    "Cumulative Returns",
                ),
                vertical_spacing=0.15,
                row_heights=[0.5, 0.5],
            )

            # Plot indicator based on method
            if method == "Z-Score":
                fig2.add_trace(
                    go.Scatter(x=indicator.index, y=indicator, name="Z-Score"),
                    row=1,
                    col=1,
                )
                threshold = float(threshold_params)
                fig2.add_hline(
                    y=threshold, line_dash="dash", line_color="red", row=1, col=1
                )
                fig2.add_hline(
                    y=-threshold, line_dash="dash", line_color="red", row=1, col=1
                )
            elif method == "Kalman":
                fig2.add_trace(
                    go.Scatter(x=indicator.index, y=indicator, name="Filtered Spread"),
                    row=1,
                    col=1,
                )
                delta = float(threshold_params["delta"])
                fig2.add_hline(
                    y=delta, line_dash="dash", line_color="red", row=1, col=1
                )
                fig2.add_hline(
                    y=-delta, line_dash="dash", line_color="red", row=1, col=1
                )
            elif method == "RSI":
                fig2.add_trace(
                    go.Scatter(x=indicator.index, y=indicator, name="RSI"), row=1, col=1
                )
                threshold = int(threshold_params["rsi_threshold"])
                fig2.add_hline(
                    y=threshold, line_dash="dash", line_color="red", row=1, col=1
                )
                fig2.add_hline(
                    y=100 - threshold, line_dash="dash", line_color="red", row=1, col=1
                )
            elif method == "Bollinger":
                fig2.add_trace(
                    go.Scatter(x=indicator.index, y=indicator, name="Spread"),
                    row=1,
                    col=1,
                )
                if "upper_band" in signals.columns and "lower_band" in signals.columns:
                    fig2.add_trace(
                        go.Scatter(
                            x=signals.index,
                            y=signals["upper_band"],
                            name="Upper Band",
                            line=dict(dash="dash"),
                        ),
                        row=1,
                        col=1,
                    )
                    fig2.add_trace(
                        go.Scatter(
                            x=signals.index,
                            y=signals["lower_band"],
                            name="Lower Band",
                            line=dict(dash="dash"),
                        ),
                        row=1,
                        col=1,
                    )
            elif method == "Half-life":
                fig2.add_trace(
                    go.Scatter(x=indicator.index, y=indicator, name="Spread"),
                    row=1,
                    col=1,
                )

            # Plot cumulative returns
            fig2.add_trace(
                go.Scatter(
                    x=signals.index,
                    y=signals["cumulative_returns"],
                    name="Cumulative Returns",
                ),
                row=2,
                col=1,
            )

            fig2.update_layout(height=600, showlegend=True, margin=dict(t=30, b=30))

            # Middle column for first set of charts
            with col2:
                st.subheader("Price & Spread")
                st.plotly_chart(fig1, use_container_width=True)

            # Right column for second set of charts
            with col3:
                st.subheader("Signals & Returns")
                st.plotly_chart(fig2, use_container_width=True)

        except Exception as e:
            st.error(f"An error occurred: {str(e)}")


if __name__ == "__main__":
    main()
