import os
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import pytest

from pairs_trader import PairsTrader


@pytest.fixture(params=[("KO", "PEP"), ("JPM", "GS"), ("CVX", "XOM"), ("AAPL", "MSFT")])
def market_data(request):
    """
    Fixture that loads real market data for testing.
    """
    stock1, stock2 = request.param
    filename = f"testdata/{stock1}_{stock2}_prices.csv"

    # Read the CSV file directly
    df = pd.read_csv(filename, parse_dates=["Date"], index_col="Date")
    return df, stock1, stock2


@pytest.fixture
def trader(market_data):
    """Create a PairsTrader instance using market data"""
    df, stock1, stock2 = market_data
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
    df, _, _ = market_data
    spread, hedge_ratio, is_reversed = trader.calculate_spread(df)

    # Basic sanity checks
    assert not spread.isnull().any(), "Spread contains NaN values"
    assert abs(spread.mean()) < 0.1, "Spread mean should be close to zero"
    assert spread.std() > 0, "Spread should have positive standard deviation"

    # Verify hedge ratio is reasonable
    assert isinstance(hedge_ratio, float), "Hedge ratio should be a float"
    assert 0.1 < abs(hedge_ratio) < 10.0, "Hedge ratio should be reasonable"

    # Add correlation check with lower threshold
    corr = df[trader.stock1].corr(df[trader.stock2])
    assert abs(corr) > 0.4, "Stock pair should show moderate correlation"


def test_calculate_z_score(trader, market_data):
    """Test z-score calculation using real market data"""
    df, _, _ = market_data
    spread, _, _ = trader.calculate_spread(df)
    z_score = trader.calculate_z_score(spread)

    # Statistical properties checks using actual data characteristics
    assert (
        len(z_score.dropna()) >= len(df) - 20
    ), "Should have z-scores after initial window"
    assert (
        -4.0 < z_score.mean() < 4.0
    ), "Z-score mean should be within reasonable bounds"
    assert 0.5 < z_score.std() < 5.0, "Z-score volatility should be reasonable"

    # Test for stationarity by checking if z-scores mean-revert
    assert z_score.autocorr() < 0.95, "Z-scores should not have unit root"


def test_generate_signals(trader, market_data):
    """Test signal generation using real market data"""
    df, _, _ = market_data
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
    df, _, _ = market_data
    spread, hedge_ratio, is_reversed = trader.calculate_spread(df)
    z_score = trader.calculate_z_score(spread)
    signals = trader.generate_signals_zscore(z_score)
    signals = trader.calculate_returns(df, signals, hedge_ratio, is_reversed)

    # Find first valid index for returns (after initial NaN values)
    first_valid_idx = signals["strategy_returns"].first_valid_index()

    # Verify return calculations with actual data
    assert "strategy_returns" in signals.columns, "Missing strategy returns"
    assert "cumulative_returns" in signals.columns, "Missing cumulative returns"

    # Test return properties
    assert (
        not signals["strategy_returns"].isnull().all()
    ), "Strategy returns should not be all NaN"
    assert signals["strategy_returns"].std() > 0, "Strategy should show some volatility"
    assert (
        abs(signals["strategy_returns"].mean()) < 0.1
    ), "Average daily return should be reasonable"

    # Test cumulative return properties starting from first valid index
    if first_valid_idx is not None:
        assert (
            abs(signals.loc[first_valid_idx, "cumulative_returns"] - 1.0) < 1e-10
        ), "Cumulative returns should start at 1.0"
        assert (
            signals.loc[first_valid_idx:, "cumulative_returns"] >= 0
        ).all(), "Cumulative returns cannot be negative"

    # Verify returns with zero positions are very close to zero or NaN
    zero_pos_mask = signals["position"] == 0
    zero_pos_returns = signals.loc[zero_pos_mask, "strategy_returns"].fillna(0)
    assert (
        zero_pos_returns.abs() < 1e-6
    ).all(), "Returns with zero position should be very close to zero"


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


if __name__ == "__main__":
    pytest.main([__file__])
