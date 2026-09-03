import click
from src.fetch_weather import fetch_weather_data, build_archive_url
from src.train import train_model

@click.command()
@click.option('-d', '--dburl', required=True, help='Database URL')
@click.option('-p', '--period', default=730, type=int, help='Archive period in days')
def main(dburl: str, period: int) -> None:
    url = build_archive_url(period)    # get database url for specified period
    fetch_weather_data(dburl, url)    # fetch weather data for specified time period
    train_model(dburl)               # train model on the fetched data

if __name__ == '__main__':
    main()