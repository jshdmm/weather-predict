import pandas as pd
import pytest
import requests
from src.fetch_weather import fetch_weather_data
from src.setup_db import weatherDB

############ TEST 1: Test for correct db insertion ###############
"""
Create Mock Test to see if the fetch_weather_data function correctly inserts data into the database.
"""

# create sample response to mock API json response
SAMPLE_RESPONSE = {
    "hourly": {
        "time": [1640995200, 1640998800],  # 2022-01-01 00:00, 01:00 UTC
        "temperature_2m": [3.4, 3.1],
    }
}

# class to create mock response object (later used by pytest.monkeypatch)
class MockResponse:

    # initialize the mock response with json data and status code
    def __init__(self, json_data: dict, status_code: int) -> None:
        self.json_data = json_data
        self.status_code = status_code

    # method to return the json data
    def json(self) -> dict:
        return self.json_data

# test weather data insertion
def test_fetch_weather_data(tmp_path: pytest.TempPathFactory, monkeypatch: pytest.MonkeyPatch) -> None:

    # replace the requests.get method with a mock that returns the sample response
    monkeypatch.setattr(
        requests, "get",
        lambda url: MockResponse(SAMPLE_RESPONSE, 200)
    )

    dburl = f"sqlite:///{tmp_path}/weather.db"
    fetch_weather_data(dburl, "https://example.invalid/archive")

    df = weatherDB(dburl).get_weather_data()

    # test number of columns
    assert len(df) == 2

    # test entries
    assert set(df["temperature_2m"]) == {3.4, 3.1}

    # test time column is converted to datetime
    assert isinstance(df["time"].iloc[0], pd.Timestamp)






########### TEST 2: Test for failed API request ##########
def test_fetch_weather_data_failed_request(tmp_path: pytest.TempPathFactory, monkeypatch: pytest.MonkeyPatch) -> None:

    # replace the requests.get method with a mock that returns a 404 status code
    monkeypatch.setattr(
        requests, "get",
        lambda url: MockResponse(None, 404)
    )

    # test that no data is inserted into the database when the API request fails
    dburl = f"sqlite:///{tmp_path}/weather.db"
    fetch_weather_data(dburl, "https://example.invalid/archive")

    df = weatherDB(dburl).get_weather_data()

    # test that no data was inserted into the database
    assert len(df) == 0






########### TEST 3: No duplicate entries ##########
def test_fetch_weather_data_avoid_duplicates(tmp_path: pytest.TempPathFactory, monkeypatch: pytest.MonkeyPatch):
    # replace the requests.get method with a mock that returns the sample response
    monkeypatch.setattr(
        requests, "get",
        lambda url: MockResponse(SAMPLE_RESPONSE, 200)
    )

    # call the function twice to simulate two fetches of the same data
    dburl = f"sqlite:///{tmp_path}/weather.db"
    fetch_weather_data(dburl, "https://example.invalid/archive")

    # Call the function again to simulate a second fetch
    fetch_weather_data(dburl, "https://example.invalid/archive")

    # get weatherdata from the DB
    df = weatherDB(dburl).get_weather_data()

    # test that no duplicate entries were inserted into the database
    assert len(df) == 2  # Should still be 2 entries, not 4

