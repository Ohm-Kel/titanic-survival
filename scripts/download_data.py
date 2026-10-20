"""Download the Kaggle Titanic competition data into data/raw/."""

import logging
import zipfile
from pathlib import Path

from kaggle.api.kaggle_api_extended import KaggleApi

COMPETITION = "titanic"
RAW_DIR = Path("data/raw")

logger = logging.getLogger(__name__)


def main() -> None:
    """Authenticate with Kaggle, download the competition files and unzip them."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    api = KaggleApi()
    api.authenticate()
    api.competition_download_files(COMPETITION, path=RAW_DIR, quiet=True)

    archive = RAW_DIR / f"{COMPETITION}.zip"
    with zipfile.ZipFile(archive) as zf:
        zf.extractall(RAW_DIR)
    archive.unlink()

    for csv_file in sorted(RAW_DIR.glob("*.csv")):
        logger.info("Downloaded %s", csv_file)


if __name__ == "__main__":
    main()