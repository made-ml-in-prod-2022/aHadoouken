from dataclasses import dataclass
from typing import Optional


@dataclass
class DownloadParams:
    download_data: bool
    google_drive_url: str


@dataclass
class DataParams:
    input_data_path: str
    downloading_data_params: DownloadParams
