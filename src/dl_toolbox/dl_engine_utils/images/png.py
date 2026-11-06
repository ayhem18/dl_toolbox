"""PNG helpers for inspect logging."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from dl_toolbox.base.packaging import require


def read_png_hwc(path: Path) -> np.ndarray:
    matplotlib = require(package="matplotlib")
    image = matplotlib.image.imread(str(path))
    return np.asarray(image)
