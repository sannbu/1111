import dataclasses
from dataclasses import dataclass


@dataclass
class TrainConfig:
    images_dir: str
    labels_dir: str
    out_dir: str
    img_size: int = 320
    timesteps: int = 1000
    beta_start: float = 1e-4
    beta_end: float = 0.02
    batch_size: int = 4
    lr: float = 2e-4
    epochs: int = 100
    num_workers: int = 4
    seed: int = 42
    log_interval: int = 1
    save_interval: int = 1


@dataclass
class SampleConfig:
    ckpt: str
    background_image: str
    label_txt: str
    out_image: str
    img_size: int = 320
    timesteps: int = 1000
    beta_start: float = 1e-4
    beta_end: float = 0.02
    seed: int = 0


def as_dict(cfg):
    """Utility to convert dataclass or argparse Namespace to dict."""
    if dataclasses.is_dataclass(cfg):
        return dataclasses.asdict(cfg)
    return vars(cfg)
