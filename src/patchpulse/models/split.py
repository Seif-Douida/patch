"""A labeled set of reviews for training or evaluation. Kept free of pandas: the jobs image's
retrain (Phase 7) builds splits from SQL, without the `ml` group."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class Split:
    ids: list[int]
    texts: list[str]
    labels: NDArray[np.bool_]  # (reviews, aspects)
