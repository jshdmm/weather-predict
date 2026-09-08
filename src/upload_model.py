import glob
import os
import json
import click
import joblib
from huggingface_hub import HfApi, hf_hub_download

DEFAULT_REPO_ID = "jshdmm/weather-predict-berlin"


def upload_latest_model(repo_id: str = DEFAULT_REPO_ID, results_dir: str = "results") -> str:
    """
    Uploads the most recently trained model (by file mtime), along with its
    matching results summary, to a Hugging Face Hub model repo.

    Authenticates via the HF_TOKEN environment variable (GitHub Actions Secret).
    """
    # Look for most recent model
    candidates = glob.glob(os.path.join(results_dir, "model_*.pkl"))
    if not candidates:
        raise FileNotFoundError(
            f"No model file found in {results_dir}/ -- run training first."
        )
    latest = max(candidates, key=os.path.getmtime)
    latest_results = latest.replace(".pkl", ".json")

    # Upload to Hugging Face Hub
    api = HfApi()
    api.upload_file(
        path_or_fileobj=latest,
        path_in_repo="model.pkl",
        repo_id=repo_id,
        repo_type="model",
    )
    print(f"Uploaded {latest} to https://huggingface.co/{repo_id} as model.pkl")

    api.upload_file(
        path_or_fileobj=latest_results,
        path_in_repo="results.json",
        repo_id=repo_id,
        repo_type="model",
    )
    print(f"Uploaded {latest_results} to https://huggingface.co/{repo_id} as results.json")

    return latest


def download_latest_model(repo_id: str = DEFAULT_REPO_ID):
    """
    Downloads and loads the most recently uploaded model from the Hugging
    Face Hub model repo.
    """
    model_path = hf_hub_download(repo_id=repo_id, filename="model.pkl")
    return joblib.load(model_path)


def download_latest_results(repo_id: str = DEFAULT_REPO_ID) -> dict:
    """
    Downloads and parses the results summary uploaded alongside the model --
    the counterpart to the results.json half of upload_latest_model.
    """
    results_path = hf_hub_download(repo_id=repo_id, filename="results.json")
    with open(results_path, "r") as f:
        return json.load(f)


@click.command()
@click.option('-r', '--repo-id', default=DEFAULT_REPO_ID, help='Hugging Face Hub model repo id')
def main(repo_id: str) -> None:
    upload_latest_model(repo_id)


if __name__ == '__main__':
    main()
