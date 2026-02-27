"""Auto-download SAM / SAM2 checkpoints."""

import os
import urllib.request

from huggingface_hub import hf_hub_download

# SAM v1: model_type -> (url, filename)
# Original .pth checkpoints hosted by Facebook (not on HF Hub).
SAM_CHECKPOINTS: dict[str, tuple[str, str]] = {
    "vit_h": (
        "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth",
        "sam_vit_h_4b8939.pth",
    ),
    "vit_l": (
        "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth",
        "sam_vit_l_0b3195.pth",
    ),
    "vit_b": (
        "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth",
        "sam_vit_b_01ec64.pth",
    ),
}

# SAM2: yaml config name -> (hf_repo, filename)
SAM2_CHECKPOINTS: dict[str, tuple[str, str]] = {
    # SAM 2.0
    "sam2_hiera_t.yaml": ("facebook/sam2-hiera-tiny", "sam2_hiera_tiny.pt"),
    "sam2_hiera_s.yaml": ("facebook/sam2-hiera-small", "sam2_hiera_small.pt"),
    "sam2_hiera_b+.yaml": ("facebook/sam2-hiera-base-plus", "sam2_hiera_base_plus.pt"),
    "sam2_hiera_l.yaml": ("facebook/sam2-hiera-large", "sam2_hiera_large.pt"),
    # SAM 2.1
    "sam2.1_hiera_t.yaml": ("facebook/sam2.1-hiera-tiny", "sam2.1_hiera_tiny.pt"),
    "sam2.1_hiera_s.yaml": ("facebook/sam2.1-hiera-small", "sam2.1_hiera_small.pt"),
    "sam2.1_hiera_b+.yaml": (
        "facebook/sam2.1-hiera-base-plus",
        "sam2.1_hiera_base_plus.pt",
    ),
    "sam2.1_hiera_l.yaml": ("facebook/sam2.1-hiera-large", "sam2.1_hiera_large.pt"),
}

_CACHE_DIR = os.path.join(os.path.expanduser("~"), ".cache", "kovec", "sam")


def ensure_sam_checkpoint(model_type: str) -> str:
    """Return local path to SAM v1 checkpoint, downloading if needed."""
    if model_type not in SAM_CHECKPOINTS:
        raise ValueError(
            f"Unknown SAM model_type '{model_type}'. "
            f"Available: {', '.join(SAM_CHECKPOINTS)}"
        )
    url, filename = SAM_CHECKPOINTS[model_type]
    os.makedirs(_CACHE_DIR, exist_ok=True)
    local_path = os.path.join(_CACHE_DIR, filename)
    if not os.path.exists(local_path):
        print(f"Downloading {filename} ...")
        urllib.request.urlretrieve(url, local_path)
    return local_path


def ensure_sam2_checkpoint(config_name: str) -> str:
    """Return local path to SAM2 checkpoint, downloading from HF if needed."""
    if config_name not in SAM2_CHECKPOINTS:
        raise ValueError(
            f"Unknown SAM2 config '{config_name}'. "
            f"Available: {', '.join(SAM2_CHECKPOINTS)}"
        )
    repo_id, filename = SAM2_CHECKPOINTS[config_name]
    return hf_hub_download(repo_id, filename)
