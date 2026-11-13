# FreeTimeGS++
# 2025-2026 Lucas Yunkyu Lee <lucaslee@postech.ac.kr>, SNU VGI Lab

import os

from imagecodecs import png_encode
from torch import Tensor

type PathLike = str | os.PathLike


def num_workers() -> int:
    # Benchmark: cap decode workers (each holds full-res frames in RAM); results are unaffected.
    if os.environ.get("FTGS_NUM_WORKERS"):
        return int(os.environ["FTGS_NUM_WORKERS"])
    return len(os.sched_getaffinity(0))


def tensor_to_png(img: Tensor) -> bytes:
    img_np = img.detach().mul(255).add(0.5).clamp(0, 255).byte().cpu()
    return bytes(png_encode(img_np))
