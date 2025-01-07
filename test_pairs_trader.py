import os
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import pytest

from pairs_trader import PairsTrader

# Define test data configurations
TEST_DATA = [
    {
        "pair": ("KO", "PEP"),
        "file": "testdata/KO_PEP_prices.csv",
        "hedge_ratio": 0.682918,
        "spread_mean": 122.63486011555092,
        "spread_std": 5.4291812891269595,
        "correlation": 0.477083,
        "returns": {
            "mean": -0.000234,
            "std": 0.00159382,
            "first_valid_return": 0.0,
            "cumulative_end": 0.9609620754,
        },
        "z_score": {
            "mean": -0.137923008367,
            "std": 1.261495268,
            "autocorr": 0.85671456,
            "valid_count": 232,  # len(df) - 20 for rolling window
        },
        "kalman": {
            "delta": 0.001,  # Process variance
            "R": 0.001,  # Measurement variance
            "filtered_spread": {
                "mean": 123.045693553,
                "std": 3.9125366477085,
                "first_value": 123.72486,
                "last_value": 110.84654282,
                "signal_counts": {
                    "long": 0,  # Number of long signals
                    "short": 251,  # Number of short signals
                    "neutral": 0,  # Number of neutral signals
                },
            },
            "statistics": {
                "Total Returns": -0.835241,
                "Annual Returns": -0.0867,
                "Annual Volatility": 0.4251,
                "Sharpe Ratio": -0.21340445,
                "Max Drawdown": -0.9069902937,
                "Number of Trades": 10,
                "Win Rate": 0.6363636363636364,
            },
        },
    },
    {
        "pair": ("JPM", "GS"),
        "file": "testdata/JPM_GS_prices.csv",
        "hedge_ratio": 3.047697,
        "spread_mean": -154.1184725368894,
        "spread_std": 15.189135208761494,
        "correlation": 0.97600,
        "returns": {
            "mean": -0.00050325332,
            "std": 0.00424301,
            "first_valid_return": 0,
            "cumulative_end": 0.879276,
        },
        "z_score": {
            "mean": -0.164736883,
            "std": 1.2502362187,
            "autocorr": 0.8213976,
            "valid_count": 232,
        },
        "kalman": {
            "delta": 0.001,
            "R": 0.001,
            "filtered_spread": {
                "mean": -153.40719484947576,
                "std": 11.89076653,
                "first_value": -133.6428,
                "last_value": -154.4130054065,
                "signal_counts": {"long": 251, "short": 0, "neutral": 0},
            },
            "statistics": {
                "Total Returns": 3.458113,
                "Annual Returns": 0.2523,
                "Annual Volatility": 0.38668,
                "Sharpe Ratio": 0.5822559,
                "Max Drawdown": -0.705553,
                "Number of Trades": 26,
                "Win Rate": 0.7037037037037037,
            },
        },
    },
    {
        "pair": ("CVX", "XOM"),
        "file": "testdata/CVX_XOM_prices.csv",
        "hedge_ratio": 0.4970833,
        "spread_mean": 95.28237075800826,
        "spread_std": 5.406686835210159,
        "correlation": 0.54244,
        "returns": {
            "mean": -0.00046285,
            "std": 0.00300328,
            "first_valid_return": 0.0,
            "cumulative_end": 0.8896928308,
        },
        "z_score": {
            "mean": 0.09104704236,
            "std": 1.26939483297,
            "autocorr": 0.8523894121,
            "valid_count": 231,
        },
        "kalman": {
            "delta": 0.001,
            "R": 0.001,
            "filtered_spread": {
                "mean": 95.226223,
                "std": 4.4554974,
                "first_value": 95.881971975,
                "last_value": 94.0895771,
                "signal_counts": {"long": 0, "short": 250, "neutral": 0},
            },
            "statistics": {
                "Total Returns": 0.462161,
                "Annual Returns": 0.059119414,
                "Annual Volatility": 0.196737,
                "Sharpe Ratio": 0.291985,
                "Max Drawdown": -0.5844998603,
                "Number of Trades": 9,
                "Win Rate": 1,
            },
        },
    },
    {
        "pair": ("AAPL", "MSFT"),
        "file": "testdata/AAPL_MSFT_prices.csv",
        "hedge_ratio": 0.8137367586847457,
        "spread_mean": -134.58820833103275,
        "spread_std": 21.8588494583802,
        "correlation": 0.5329,
        "returns": {
            "mean": -0.00042302,
            "std": 0.0028099322751453863,
            "first_valid_return": 0,
            "cumulative_end": 0.89872611,
        },
        "z_score": {
            "mean": 0.14646263039689,
            "std": 1.30125976,
            "autocorr": 0.868049345848578,
            "valid_count": 231,
        },
        "kalman": {
            "delta": 0.001,
            "R": 0.001,
            "filtered_spread": {
                "mean": -135.483760629022,
                "std": 19.628204260,
                "first_value": -118.5790031238,
                "last_value": -103.61349680,
                "signal_counts": {"long": 250, "short": 0, "neutral": 0},
            },
            "statistics": {
                "Total Returns": 3.850771,
                "Annual Returns": 0.19874,
                "Annual Volatility": 0.214411,
                "Sharpe Ratio": 0.845764,
                "Max Drawdown": -0.2798129,
                "Number of Trades": 53,
                "Win Rate": 0.8703703703703703,
            },
        },
    },
]


@pytest.fixture(
    params=TEST_DATA,
    ids=lambda x: f"{x['pair'][0]}_{x['pair'][1]}",  # Creates IDs like "KO_PEP", "JPM_GS", etc.
)
def market_data(request):
    """
    Fixture that loads real market data for testing.
    """
    test_config = request.param
    stock1, stock2 = test_config["pair"]
    filename = test_config["file"]

    # Read the CSV file directly
    df = pd.read_csv(filename, parse_dates=["Date"], index_col="Date")
    return df, stock1, stock2, test_config


@pytest.fixture
def trader(market_data):
    """Create a PairsTrader instance using market data"""
    df, stock1, stock2, _ = market_data
    start_date = df.index[0].strftime("%Y-%m-%d")
    end_date = df.index[-1].strftime("%Y-%m-%d")

    return PairsTrader(
        stock1=stock1,
        stock2=stock2,
        start_date=start_date,
        end_date=end_date,
        threshold_params=2.0,
        method="zscore",
    )


def test_calculate_spread(trader, market_data):
    """Test spread calculation using real market data"""
    df, _, _, test_config = market_data
    spread, hedge_ratio, is_reversed = trader.calculate_spread(df)

    # Test for NaN values
    assert not spread.isnull().any(), "Spread contains NaN values"

    # Test spread properties with exact values using pytest.approx()
    # Note: For non-standardized data, we expect the mean to still be close to zero
    # but the standard deviation will be different for each pair
    assert spread.mean() == pytest.approx(
        test_config["spread_mean"], abs=1.0
    ), "Spread mean should be approximately zero"
    assert spread.std() == pytest.approx(
        test_config["spread_std"], abs=1.0
    ), "Spread should match expected standard deviation"

    # Test hedge ratio with exact values (now using raw price ratios)
    assert isinstance(hedge_ratio, float), "Hedge ratio should be a float"
    assert hedge_ratio == pytest.approx(
        test_config["hedge_ratio"], abs=1e-2
    ), f"Hedge ratio for {trader.stock1}-{trader.stock2} should be approximately {test_config['hedge_ratio']}"

    # Test correlation with exact value (correlation remains the same with or without standardization)
    corr = df[trader.stock1].corr(df[trader.stock2])
    assert corr == pytest.approx(
        test_config["correlation"], abs=1e-2
    ), f"Stock pair correlation should be approximately {test_config['correlation']}"


def test_calculate_z_score(trader, market_data):
    """Test z-score calculation using real market data"""
    df, _, _, test_config = market_data
    spread, _, _ = trader.calculate_spread(df)
    z_score = trader.calculate_z_score(spread)

    # Test number of valid z-scores
    valid_z_scores = z_score.dropna()
    assert (
        len(valid_z_scores) == test_config["z_score"]["valid_count"]
    ), "Should have expected number of valid z-scores"

    # Test z-score statistical properties with exact values
    assert z_score.mean() == pytest.approx(
        test_config["z_score"]["mean"], abs=1e-4
    ), "Z-score mean should match expected value"

    assert z_score.std() == pytest.approx(
        test_config["z_score"]["std"], abs=1e-4
    ), "Z-score std should be close to 1"

    # Test for stationarity with exact autocorrelation value
    assert z_score.autocorr() == pytest.approx(
        test_config["z_score"]["autocorr"], abs=1e-4
    ), "Z-score autocorrelation should match expected value"


def test_generate_signals(trader, market_data):
    """Test signal generation using real market data"""
    df, _, _, _ = market_data
    spread, _, _ = trader.calculate_spread(df)
    z_score = trader.calculate_z_score(spread)
    signals = trader.generate_signals_zscore(z_score)

    # Verify signal properties
    assert "position" in signals.columns, "Signals should contain position column"
    assert signals["position"].isin([-1, 0, 1]).all(), "Positions should be -1, 0, or 1"

    # Test position changes reflect threshold crossings
    threshold = trader.threshold_params
    assert (
        signals.loc[z_score > threshold, "position"] == -1
    ).all(), f"Should short when z > {threshold}"
    assert (
        signals.loc[z_score < -threshold, "position"] == 1
    ).all(), f"Should long when z < -{threshold}"
    assert (
        signals.loc[(z_score >= -threshold) & (z_score <= threshold), "position"] == 0
    ).all(), f"No position when |z| <= {threshold}"


def test_calculate_returns(trader, market_data):
    """Test returns calculation using real market data"""
    df, _, _, test_config = market_data
    spread, hedge_ratio, is_reversed = trader.calculate_spread(df)
    z_score = trader.calculate_z_score(spread)
    signals = trader.generate_signals_zscore(z_score)
    signals = trader.calculate_returns(df, signals, hedge_ratio, is_reversed)

    # Find first valid index for returns (after initial NaN values)
    first_valid_idx = signals["strategy_returns"].first_valid_index()
    returns = signals["strategy_returns"].dropna()

    # Test return properties with exact values
    assert returns.mean() == pytest.approx(
        test_config["returns"]["mean"], abs=1e-4
    ), "Mean return should match expected value"

    assert returns.std() == pytest.approx(
        test_config["returns"]["std"], abs=1e-4
    ), "Return volatility should match expected value"

    # Test first valid return
    if first_valid_idx is not None:
        assert signals.loc[first_valid_idx, "strategy_returns"] == pytest.approx(
            test_config["returns"]["first_valid_return"], abs=1e-4
        ), "First valid return should match expected value"

    # Test final cumulative return
    assert signals["cumulative_returns"].iloc[-1] == pytest.approx(
        test_config["returns"]["cumulative_end"], abs=1e-4
    ), "Final cumulative return should match expected value"

    # Verify returns with zero positions are zero
    zero_pos_mask = signals["position"] == 0
    zero_pos_returns = signals.loc[zero_pos_mask, "strategy_returns"].fillna(0)
    assert (
        zero_pos_returns.abs() < 1e-6
    ).all(), "Returns with zero position should be very close to zero"

    # Verify position changes align with returns
    non_zero_returns = signals[signals["strategy_returns"].abs() > 1e-6]
    assert (
        non_zero_returns["position"].shift(1) != 0
    ).all(), "Non-zero returns should only occur after non-zero positions"


def test_error_handling(trader):
    """Test error handling with invalid data"""
    # Test with empty DataFrame
    empty_df = pd.DataFrame(
        columns=[trader.stock1, trader.stock2]
    )  # Use actual stock names
    with pytest.raises(
        ValueError,
        match="Pandas data cast to numpy dtype of object|Insufficient data",  # Accept either error message
    ):
        trader.calculate_spread(empty_df)

    # Test with constant prices
    dates = pd.date_range(start="2023-01-01", periods=100)
    const_df = pd.DataFrame(
        {
            trader.stock1: [100] * 100,
            trader.stock2: [150] * 100,
        },  # Use actual stock names
        index=dates,
    )
    with pytest.raises(Exception):
        trader.calculate_spread(const_df)

    # Test with NaN values
    nan_df = pd.DataFrame(
        {
            trader.stock1: [100, np.nan, 102] * 10,
            trader.stock2: [150, 151, np.nan] * 10,
        },
        index=pd.date_range(start="2023-01-01", periods=30),
    )
    with pytest.raises(Exception):
        trader.calculate_spread(nan_df)


def test_generate_signals_kalman(trader, market_data):
    """Test Kalman filter signal generation using real market data"""
    df, _, _, test_config = market_data

    # Create a new trader instance with Kalman method
    kalman_trader = PairsTrader(
        stock1=trader.stock1,
        stock2=trader.stock2,
        start_date=trader.start_date,
        end_date=trader.end_date,
        threshold_params={
            "delta": test_config["kalman"]["delta"],
            "R": test_config["kalman"]["R"],
        },
        method="kalman",
    )

    # Calculate spread and generate signals
    spread, hedge_ratio, is_reversed = kalman_trader.calculate_spread(df)
    signals = kalman_trader.generate_signals_kalman(spread)

    # Test filtered spread properties
    filtered_spread = signals["filtered_spread"]
    expected = test_config["kalman"]["filtered_spread"]

    assert filtered_spread.mean() == pytest.approx(
        expected["mean"], abs=1e-4
    ), "Filtered spread mean should match expected value"

    assert filtered_spread.std() == pytest.approx(
        expected["std"], abs=1e-4
    ), "Filtered spread std should match expected value"

    assert filtered_spread.iloc[0] == pytest.approx(
        expected["first_value"], abs=1e-4
    ), "First filtered spread value should match expected"

    assert filtered_spread.iloc[-1] == pytest.approx(
        expected["last_value"], abs=1e-4
    ), "Last filtered spread value should match expected"

    # Test signal generation
    assert "position" in signals.columns, "Signals should contain position column"
    assert signals["position"].isin([-1, 0, 1]).all(), "Positions should be -1, 0, or 1"

    # Test signal counts
    signal_counts = {
        "long": (signals["position"] == 1).sum(),
        "short": (signals["position"] == -1).sum(),
        "neutral": (signals["position"] == 0).sum(),
    }

    expected_counts = expected["signal_counts"]
    assert (
        signal_counts["long"] == expected_counts["long"]
    ), f"Expected {expected_counts['long']} long signals, got {signal_counts['long']}"
    assert (
        signal_counts["short"] == expected_counts["short"]
    ), f"Expected {expected_counts['short']} short signals, got {signal_counts['short']}"
    assert (
        signal_counts["neutral"] == expected_counts["neutral"]
    ), f"Expected {expected_counts['neutral']} neutral signals, got {signal_counts['neutral']}"

    # Test signal transitions
    position_changes = signals["position"].diff()
    assert (
        abs(position_changes).max() <= 2
    ), "Position changes should not exceed 2 (from -1 to 1 or vice versa)"


def test_calculate_statistics_kalman(trader, market_data):
    """Test statistics calculation for Kalman filter strategy"""
    df, _, _, test_config = market_data

    # Create a new trader instance with Kalman method
    kalman_trader = PairsTrader(
        stock1=trader.stock1,
        stock2=trader.stock2,
        start_date=trader.start_date,
        end_date=trader.end_date,
        threshold_params={
            "delta": test_config["kalman"]["delta"],
            "R": test_config["kalman"]["R"],
        },
        method="kalman",
    )

    # Run the strategy
    spread, hedge_ratio, is_reversed = kalman_trader.calculate_spread(df)
    signals = kalman_trader.generate_signals_kalman(spread)
    signals = kalman_trader.calculate_returns(df, signals, hedge_ratio, is_reversed)

    # Calculate statistics
    stats = kalman_trader.calculate_statistics(signals)
    expected_stats = test_config["kalman"]["statistics"]

    # Test each statistic matches expected values
    for key, expected_value in expected_stats.items():
        assert stats[key] == pytest.approx(
            expected_value, abs=1e-4
        ), f"{key} should be {expected_value}, got {stats[key]}"


if __name__ == "__main__":
    pytest.main([__file__])
