from dataclasses import dataclass, field


@dataclass
class DownloadParams:
    download_data: bool
    google_drive_url: str


@dataclass
class SplittingParams:
    test_size: float = field(default=0.2)
    random_state: int = field(default=42)


@dataclass
class DataParams:
    input_data_path: str
    downloading_data_params: DownloadParams
    splitting_params: SplittingParams
