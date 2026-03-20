"""Schema contracts for cond/axis/channel preprocessing."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class CondSchema:
    order: list[str]

    def _lookup(self, payload: dict[str, Any], key: str) -> float:
        cur: Any = payload
        for token in key.split("."):
            if isinstance(cur, dict) and token in cur:
                cur = cur[token]
            else:
                raise KeyError(f"Missing key in cond payload: {key}")
        return float(cur)

    def encode(self, cond_dict: dict[str, Any]) -> np.ndarray:
        return np.array([self._lookup(cond_dict, key) for key in self.order], dtype=np.float32)

    def to_dict(self) -> dict[str, Any]:
        return {"order": self.order}

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "CondSchema":
        return cls(order=list(raw.get("order", [])))


@dataclass
class AxisSchema:
    mode: str = "steady"
    harmonics: int = 1

    def encode(self, value: float = 0.0) -> np.ndarray:
        if self.mode == "steady":
            return np.zeros((0,), dtype=np.float32)
        if self.mode == "time":
            return np.array([float(value)], dtype=np.float32)
        if self.mode == "phase_sincos":
            out: list[float] = []
            for k in range(1, self.harmonics + 1):
                ang = 2.0 * np.pi * k * float(value)
                out.extend([float(np.sin(ang)), float(np.cos(ang))])
            return np.array(out, dtype=np.float32)
        raise ValueError(f"Unknown axis mode: {self.mode}")

    def to_dict(self) -> dict[str, Any]:
        return {"mode": self.mode, "harmonics": self.harmonics}

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "AxisSchema":
        return cls(mode=raw.get("mode", "steady"), harmonics=int(raw.get("harmonics", 1)))


@dataclass
class ChannelSpec:
    name: str
    source: str
    normalize: str = "none"
    role: str = "feature"


@dataclass
class ChannelMap:
    channels: list[ChannelSpec] = field(default_factory=list)

    def names(self) -> list[str]:
        return [c.name for c in self.channels]

    def to_dict(self) -> dict[str, Any]:
        return {
            "channels": [
                {
                    "name": c.name,
                    "source": c.source,
                    "normalize": c.normalize,
                    "role": c.role,
                }
                for c in self.channels
            ]
        }

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "ChannelMap":
        channels = [ChannelSpec(**item) for item in raw.get("channels", [])]
        return cls(channels=channels)
