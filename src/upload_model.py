import glob
import os
import click
from huggingface_hub import HfApi

DEFAULT_REPO_ID = "jshdmm/weather-predict-berlin"


def upload_latest_model(repo_id: str = DEFAULT_REPO_ID, results_dir: str = "results") -> str:
    """
    Uploads the most recently trained model (by file mtime) to a Hugging
    Face Hub model repo.

    Authenticates via the HF_TOKEN environment variable (GitHub Actions Secret).
    """
    # Look for most recent model
    candidates = glob.glob(os.path.join(results_dir, "model_*.pkl"))
    if not candidates:
        raise FileNotFoundError(
            f"No model file found in {results_dir}/ -- run training first."
        )
    latest = max(candidates, key=os.path.getmtime)

    # Upload to Hugging Face Hub
    api = HfApi()
    api.upload_file(
        path_or_fileobj=latest,
        path_in_repo="model.pkl",
        repo_id=repo_id,
        repo_type="model",
    )
    print(f"Uploaded {latest} to https://huggingface.co/{repo_id} as model.pkl")
    return latest


@click.command()
@click.option('-r', '--repo-id', default=DEFAULT_REPO_ID, help='Hugging Face Hub model repo id')
def main(repo_id: str) -> None:
    upload_latest_model(repo_id)


if __name__ == '__main__':
    main()
