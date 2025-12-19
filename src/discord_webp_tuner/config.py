from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class SweepConfig:
    q_min: int
    q_max: int
    q_step: int
    method: int = 6
    sharp_yuv: bool = False
    bg_color: str = "808080"
    long_edge: int = 1280
    save_webp: str = "best"  # all|best|none
    jobs: int = 1

    def to_jsonable(self) -> dict[str, Any]:
        d = asdict(self)
        return d

    @property
    def q_values(self) -> list[int]:
        if self.q_step <= 0:
            raise ValueError(f"q_step must be > 0 (got {self.q_step})")
        if self.q_min > self.q_max:
            raise ValueError(f"q_min must be <= q_max (got {self.q_min}..{self.q_max})")
        return list(range(self.q_min, self.q_max + 1, self.q_step))
