import click
from src.setup_db import weatherDB
from src.features import FEATURE_COLS, add_time_features
import pandas as pd
import requests
from datetime import datetime
from zoneinfo import ZoneInfo
from dateutil.relativedelta import relativedelta




def fetch_weather_data(dburl: str, url: str) -> None:
    """
    This function is meant to keep track of an archive of the weather data as provided by the open-meteo.com archive API endpoint. It keeps track of the last 2 years of available weather data and updates the database accordingly.
    """

    # Initialize the weatherDB instance with the provided database URL
    db = weatherDB(dburl)

    # Send a GET request to the API
    download = requests.get(url)

    # Check if the request was successful
    if download.status_code == 200:
        # Parse the JSON response
        data = download.json()

        # Convert the 'hourly' data to a pandas DataFrame
        df = pd.DataFrame(data['hourly'])

        # Fill any NaN values with empty strings
        df.fillna('', inplace=True)

        # Convert the DataFrame to a list of dictionaries, each representing a record
        data = df.to_dict(orient='records')

        # Insert the data into the database using the add_weather_data method
        db.add_weather_data(data)

        # Print a confirmation message indicating successful insertion
        print("Inserted data from Open Meteo into the database.")
    else:
        # Print an error message if the data retrieval failed
        print("Failed to retrieve data.")


def get_archive(period: int) -> dict:
    """
    This function constructs the URL to fetch weather data from the Open Meteo API for the last 2 years.

    args:
        period (int): The number of days to look back for the weather data. Default is 730 days (2 years).
    returns:
        str: The constructed URL for fetching weather data.

    """

    # get Berlin's timezone and today's date
    berlin = ZoneInfo("Europe/Berlin")
    
    # end point of the archive, e.g. today
    archive_end = datetime.now(berlin).date()

    # archive time window
    archive_start = archive_end - relativedelta(days=period)

    # Construct the URL with the specified parameters
    url = f"https://archive-api.open-meteo.com/v1/era5?latitude=52.4676&longitude=13.4020&timeformat=unixtime&start_date={archive_start}&end_date={archive_end}&hourly=temperature_2m,relativehumidity_2m,rain,snowfall,windspeed_10m,winddirection_10m,soil_temperature_0_to_7cm"

    archive_dict = {
        "url": url,
        "start_date": archive_start,
        "end_date": archive_end,
        "period": period
    }


    return archive_dict


# weather forecast (needs different open meteo API)
def get_forecast(days: int = 7) -> dict:
    """
    Constructs the URL to fetch the upcoming forecast from the Open Meteo
    forecast API. Unlike the archive API, this endpoint lives on a
    different host and uses underscored variable names.

    args:
        days (int): how many days ahead to forecast. Default is 7.
    returns:
        dict: the constructed URL plus the number of days requested.
    """

    url = (
        "https://api.open-meteo.com/v1/forecast"
        "?latitude=52.4676&longitude=13.4020&timeformat=unixtime"
        f"&forecast_days={days}"
        "&hourly=relative_humidity_2m,rain,snowfall,wind_speed_10m,wind_direction_10m"
    )

    return {"url": url, "days": days}


def fetch_forecast_data(dburl: str, url: str, model) -> None:
    """
    Fetches the upcoming forecast, predicts temperature_2m with the given
    (already trained) model, and stores the predictions in the weather_data
    DB. temperature_2m itself is empty for these rows. The next archive fetch will fill in the Open Meteo predictions once that timestamp is in the past, so predictions can
    be compared.
    """

    db = weatherDB(dburl)
    download = requests.get(url)

    # if request was successful, parse the JSON and store the predictions in the database
    if download.status_code == 200:
        data = download.json()
        df = pd.DataFrame(data['hourly'])

        # rename columns so that archive and forecast features match
        df = df.rename(columns={
            "relative_humidity_2m": "relativehumidity_2m",
            "wind_speed_10m": "windspeed_10m",
            "wind_direction_10m": "winddirection_10m",
        })
        df.fillna(0, inplace=True)

        # add time features
        df["time"] = pd.to_datetime(df["time"], unit="s")
        df = add_time_features(df)
        df["predicted_temperature_2m"] = model.predict(df[FEATURE_COLS])

        store_cols = ["time", "relativehumidity_2m", "rain", "snowfall",
                      "windspeed_10m", "winddirection_10m", "predicted_temperature_2m"]
        db.add_weather_data(df[store_cols].to_dict(orient='records'))

        print(f"Stored {len(df)} forecast predictions in the database.")
    else:
        print("Failed to retrieve forecast data.")

        


@click.command()
@click.option('-d', '--dburl', required=True, help='Database URL')
@click.option('-p', '--period', default=730, type=int, help='Archive period in days')
def main(dburl: str, period: int) -> None:

    # get archive URL based on the specified period
    archive_info = get_archive(period)
    fetch_weather_data(dburl, archive_info["url"])

if __name__ == '__main__':
    # Execute the main function when the script is run directly
    main()


