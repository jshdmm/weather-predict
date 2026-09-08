import shutil
import click
from huggingface_hub import HfApi, hf_hub_download

DB_REPO_ID = "jshdmm/weather-predict-berlin-db"
DB_FILENAME = "weather.db"


def download_weather_db(repo_id: str = DB_REPO_ID, local_path: str = DB_FILENAME) -> None:
    """
    Downloads the shared weather.db from the Hugging Face dataset repo and
    saves it at local_path. If no file has ever been uploaded yet (e.g. the
    very first run), it just leaves local_path untouched so the pipeline
    starts fresh.
    """
    try:
        cached_path = hf_hub_download(repo_id=repo_id, filename=DB_FILENAME, repo_type="dataset")
    except Exception as e:
        print(f"No existing weather.db found on Hugging Face yet ({e}). Starting fresh.")
        return

    shutil.copy(cached_path, local_path)
    print(f"Downloaded weather.db from https://huggingface.co/datasets/{repo_id}")


def upload_weather_db(repo_id: str = DB_REPO_ID, local_path: str = DB_FILENAME) -> None:
    """
    Uploads the local weather.db to the Hugging Face dataset repo, creating
    the repo the first time this is called.
    """
    api = HfApi()
    api.create_repo(repo_id=repo_id, repo_type="dataset", exist_ok=True)
    api.upload_file(
        path_or_fileobj=local_path,
        path_in_repo=DB_FILENAME,
        repo_id=repo_id,
        repo_type="dataset",
    )
    print(f"Uploaded {local_path} to https://huggingface.co/datasets/{repo_id}")


@click.command()
@click.option('-a', '--action', type=click.Choice(['pull', 'push']), required=True,
              help='pull downloads weather.db from Hugging Face, push uploads it')
def main(action: str) -> None:
    if action == 'pull':
        download_weather_db()
    else:
        upload_weather_db()


if __name__ == '__main__':
    main()
