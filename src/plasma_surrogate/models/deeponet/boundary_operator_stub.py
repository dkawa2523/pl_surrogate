"""Lightweight boundary-operator stub used by unit tests."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class BoundaryOperatorStub:
    w_log_ne: float = 0.08
    w_te: float = 0.06
    w_en: float = 0.04
    bias: float = 0.0

    def predict_target(self, log_ne: np.ndarray, te: np.ndarray, phi: np.ndarray) -> np.ndarray:
        lne = np.asarray(log_ne, dtype=np.float32)
        tt = np.asarray(te, dtype=np.float32)
        pp = np.asarray(phi, dtype=np.float32)
        if lne.shape != tt.shape or lne.shape != pp.shape:
            raise ValueError(
                f"BoundaryOperatorStub shape mismatch: log_ne={lne.shape}, te={tt.shape}, phi={pp.shape}"
            )
        gy, gx = np.gradient(pp, axis=(-2, -1), edge_order=1)
        e_n = np.sqrt(gx**2 + gy**2).astype(np.float32)
        target = (
            float(self.w_log_ne) * lne
            + float(self.w_te) * tt
            + float(self.w_en) * e_n
            + float(self.bias)
        )
        return np.asarray(target, dtype=np.float32)

    def __call__(self, log_ne: np.ndarray, te: np.ndarray, phi: np.ndarray) -> np.ndarray:
        return self.predict_target(log_ne, te, phi)
