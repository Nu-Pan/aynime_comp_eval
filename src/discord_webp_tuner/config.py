from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Iterable


@dataclass(frozen=True)
class TargetSpec:
    name: str
    long_edge: int


@dataclass(frozen=True)
class SweepConfig:
    q_min: int
    q_max: int
    q_step: int
    method: int = 6
    sharp_yuv: bool = False
    bg_color: str = "808080"
    # targetB is intentionally omitted; we only evaluate the native-view target (A).
    targets: tuple[TargetSpec, ...] = (TargetSpec("A", 1280),)
    save_webp: str = "best"  # all|best|none
    jobs: int = 1

    def to_jsonable(self) -> dict[str, Any]:
        d = asdict(self)
        d["targets"] = [asdict(t) for t in self.targets]
        return d

    @property
    def q_values(self) -> list[int]:
        if self.q_step <= 0:
            raise ValueError(f"q_step must be > 0 (got {self.q_step})")
        if self.q_min > self.q_max:
            raise ValueError(f"q_min must be <= q_max (got {self.q_min}..{self.q_max})")
        return list(range(self.q_min, self.q_max + 1, self.q_step))


def parse_target_long_edges(items: Iterable[str] | None) -> tuple[TargetSpec, ...]:
    if not items:
        return (TargetSpec("A", 1280),)

    targets: list[TargetSpec] = []
    for raw in items:
        if "=" not in raw:
            raise ValueError(f"Invalid --target-long-edge value: {raw!r} (expected like A=1080)")
        name, value = raw.split("=", 1)
        name = name.strip()
        try:
            long_edge = int(value)
        except ValueError as e:
            raise ValueError(f"Invalid long edge for {name!r}: {value!r}") from e
        if long_edge <= 0:
            raise ValueError(f"Long edge must be > 0 for {name!r} (got {long_edge})")
        targets.append(TargetSpec(name=name, long_edge=long_edge))

    return tuple(targets)
