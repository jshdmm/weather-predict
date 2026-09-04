import click
from src.fetch_weather import fetch_weather_data, get_archive
from src.train import train_model

@click.command()
@click.option('-d', '--dburl', required=True, help='Database URL')
@click.option('-p', '--period', default=730, type=int, help='Archive period in days')
def main(dburl: str, period: int) -> None:
    archive_dict = get_archive(period)
    fetch_weather_data(dburl, archive_dict["url"])
    train_model(dburl, archive_dict, seed=4036018)               # train model on the fetched data

if __name__ == '__main__':
    main()