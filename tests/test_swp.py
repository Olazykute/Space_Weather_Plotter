"""
Unit tests for the space_weather module.

This module contains tests for the functions in the space_weather module, including:
- fetch_data: Fetches data from the NASA API.
- prepare_dataframe: Prepares the fetched data into a Polars DataFrame.
- plot_data: Plots the data using Matplotlib.
- main: The main function that orchestrates the fetching, preparing, and plotting of data.

The tests use the pytest framework and mock external dependencies such as the requests library.
"""
from unittest import mock
import pytest
import polars as pl
from src.space_weather_plotter.space_weather import (
    fetch_data,
    prepare_dataframe,
    resample_data,
    plot_data,
    main,
)


# Mock requests.get
@mock.patch("src.space_weather_plotter.space_weather.requests.get")
def test_fetch_data_success(mock_get):
    """
    Test fetch_data function for a successful API response.

    This test mocks the requests.get method to simulate a successful API response with status code 200
    and a JSON payload. It verifies that the fetch_data function returns the expected JSON data.
    """
    mock_response = mock.Mock()
    mock_response.status_code = 200
    mock_response.json.return_value = {"key": "value"}
    mock_get.return_value = mock_response

    result = fetch_data("FLR", {"startDate": "2024-01-01", "endDate": "2024-11-23"})
    assert result == {"key": "value"}


@mock.patch("src.space_weather_plotter.space_weather.requests.get")
def test_fetch_data_failure(mock_get):
    """
    Test fetch_data function for a failed API response.

    This test mocks the requests.get method to simulate a failed API response with status code 404
    and a "Not Found" message. It verifies that the fetch_data function returns None in case of failure.
    """
    mock_response = mock.Mock()
    mock_response.status_code = 404
    mock_response.text = "Not Found"
    mock_get.return_value = mock_response

    result = fetch_data("FLR", {"startDate": "2024-01-01", "endDate": "2024-11-23"})
    assert result is None


def test_prepare_dataframe_valid_data():
    """
    Test prepare_dataframe function with valid data.

    This test verifies that the prepare_dataframe function correctly converts a list of dictionaries
    into a Polars DataFrame with the specified column mappings and timestamp column.
    """
    data = [{"beginTime": "2024-01-01T00:00Z", "classType": "M1"}]
    columns_mapping = {"beginTime": "time", "classType": "intensity"}
    df = prepare_dataframe(data, columns_mapping, timestamp_col="time")
    assert not df.is_empty()
    assert df.columns == ["time", "intensity"]


def test_prepare_dataframe_empty_data():
    """
    Test prepare_dataframe function with empty data.

    This test verifies that the prepare_dataframe function returns an empty Polars DataFrame
    when provided with an empty list of dictionaries.
    """
    data = []
    columns_mapping = {"beginTime": "time", "classType": "intensity"}
    df = prepare_dataframe(data, columns_mapping)
    assert df.is_empty()


@mock.patch("src.space_weather_plotter.space_weather.plt.show")
def test_plot_data_valid_df(mock_show):
    """
    Test plot_data function with a valid DataFrame.

    This test verifies that the plot_data function successfully plots data from a valid Polars DataFrame
    without raising any exceptions.
    """
    data = [{"time": "2024-01-01T00:00Z", "intensity": "M1"}]
    columns_mapping = {"beginTime": "time", "classType": "intensity"}
    df = prepare_dataframe(data, columns_mapping, timestamp_col="time")
    plot_data(
        df,
        "time",
        "intensity",
        "Solar Flares",
        "Time",
        "Intensity",
        kind="bar",
        color="orange",
    )
    mock_show.assert_called_once()


@mock.patch("src.space_weather_plotter.space_weather.plt.show")
def test_plot_data_empty_df(mock_show):
    """
    Test plot_data function with an empty DataFrame.

    This test verifies that the plot_data function handles an empty Polars DataFrame without raising any exceptions.
    """
    df = pl.DataFrame()
    plot_data(
        df,
        "time",
        "intensity",
        "Solar Flares",
        "Time",
        "Intensity",
        kind="bar",
        color="orange",
    )
    mock_show.assert_not_called()


def test_resample_data_valid_frequency():
    """
    Test resample_data function with a valid frequency.

    This test verifies that the resample_data function correctly resamples the data when provided with a valid frequency.
    """
    data = {
        "time": [
            "2024-01-01T00:00Z",
            "2024-01-01T01:00Z",
            "2024-01-01T02:00Z",
            "2024-01-02T00:00Z",
        ],
        "intensity": [1, 2, 3, 4],
    }
    df = pl.DataFrame(data)
    resampled_df = resample_data(df, "time", "1d")
    assert resampled_df.shape[0] == 2  # Expecting 2 days of data


def test_resample_data_invalid_frequency():
    """
    Test resample_data function with an invalid frequency.

    This test verifies that the resample_data function raises a ValueError when provided with an invalid frequency.
    """
    data = {
        "time": [
            "2024-01-01T00:00Z",
            "2024-01-01T01:00Z",
            "2024-01-01T02:00Z",
            "2024-01-02T00:00Z",
        ],
        "intensity": [1, 2, 3, 4],
    }
    df = pl.DataFrame(data)
    with pytest.raises(ValueError, match="invalid frequency"):
        resample_data(df, "time", "invalid_frequency")


def test_resample_data_empty_df():
    """
    Test resample_data function with an empty DataFrame.

    This test verifies that the resample_data function handles an empty DataFrame correctly.
    """
    data = {"time": [], "intensity": []}
    df = pl.DataFrame(data)
    with pytest.raises(ValueError, match="empty dataframe"):
        resample_data(df, "time", "1d")


@mock.patch("builtins.input", side_effect=["1", "4"])
@mock.patch("src.space_weather_plotter.space_weather.fetch_data")
@mock.patch("src.space_weather_plotter.space_weather.plot_data")
def test_main(mock_plot_data, mock_fetch_data, mock_input):
    """
    Test main function.

    This test mocks the fetch_data, prepare_dataframe, and plot_data functions to verify that the main function
    orchestrates the fetching, preparing, and plotting of data correctly.
    """
    mock_fetch_data.return_value = [
        {"beginTime": "2024-01-01T00:00Z", "classType": "M1"}
    ]
    main()
    assert mock_plot_data.called
