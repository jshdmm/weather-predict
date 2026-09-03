import click
from src.setup_db import weatherDB
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
        "end_date": archive_end
    }


    return archive_dict

        


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
