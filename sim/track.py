"""
The real track: centre line, inner border, outer border.

WHY THIS EXISTS

The simulator had no track. It had a racing line, and it decided off-track by
measuring distance from that line. Real DeepRacer measures `distance_from_center`
from the CENTRE line and compares it with half the track width.

The difference is not cosmetic. Measured against the real geometry, the racing
line already touches the inner border at 52 of its 226 points, and its greatest
distance from centre is 0.531 m against a half width of 0.533 m. The optimal
line already uses the whole track.

So a corridor of plus or minus 0.6 m FROM THE LINE put the car up to 0.6 m into
the grass at every apex, and the old simulator scored those laps as valid. That
is why a trained policy reported 17.3 s against a racing-line time of 18.147 s:
it was not cutting well, it was driving off the track.

Track files are (N, 6): centre x, centre y, inner x, inner y, outer x, outer y.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent


class Track:
    """Real geometry, with the off-track test DeepRacer actually uses."""

    def __init__(self, npy_path: Path, margin: float = 0.02):
        a = np.load(Path(npy_path))
        if a.ndim != 2 or a.shape[1] < 6:
            raise ValueError(f"{npy_path} is not an (N, 6) track array")
        self.center = a[:, 0:2]
        self.inner = a[:, 2:4]
        self.outer = a[:, 4:6]
        widths = np.linalg.norm(self.inner - self.outer, axis=1)
        self.width = float(np.median(widths))
        self.half = self.width / 2.0
        self._n = len(self.center)

        # CALIBRATION, not a fudge.
        #
        # The reference racing line reaches 0.5414 m from the centre while half
        # the width is 0.5334 m, so a strict centre-outside-half test calls the
        # known-good optimal line off-track at 6 of its 226 points. A reference
        # lap that the evaluator rejects means the EVALUATOR is wrong.
        #
        # Two real causes, both worth the margin:
        #   1. the line file and the track file are discretised separately
        #   2. DeepRacer ends an episode on ALL WHEELS OFF, so the car centre may
        #      sit outside half width while the car is still on the surface
        #
        # This margin is the STRICT reading. The physical all-wheels-off limit is
        # larger, about half width plus half the car width (~0.11 m). Staying
        # strict keeps the bound question honest.
        self.margin = margin
        self.limit = self.half + self.margin

    # -- geometry ---------------------------------------------------------

    def nearest_center_index(self, x: float, y: float, near: int | None = None,
                             window: int = 30) -> int:
        if near is None:
            d = np.sum((self.center - np.array([x, y])) ** 2, axis=1)
            return int(np.argmin(d))
        idxs = [(near + k) % self._n for k in range(-6, window)]
        best, bd = idxs[0], float("inf")
        for i in idxs:
            dx = self.center[i][0] - x
            dy = self.center[i][1] - y
            d = dx * dx + dy * dy
            if d < bd:
                bd, best = d, i
        return best

    def distance_from_center(self, x: float, y: float, near: int | None = None) -> float:
        """Perpendicular distance to the centre line, the DeepRacer definition."""
        i = self.nearest_center_index(x, y, near)
        a = self.center[i]
        b = self.center[(i + 1) % self._n]
        ab = b - a
        L2 = float(ab @ ab) or 1e-12
        t = max(0.0, min(1.0, float((np.array([x, y]) - a) @ ab) / L2))
        proj = a + t * ab
        return float(math.dist((x, y), proj))

    def is_offtrack(self, x: float, y: float, near: int | None = None) -> bool:
        """True when the car has left the track surface."""
        return self.distance_from_center(x, y, near) > self.limit

    def is_left_of_center(self, x: float, y: float, near: int | None = None) -> bool:
        i = self.nearest_center_index(x, y, near)
        a = self.center[i]
        b = self.center[(i + 1) % self._n]
        return float((b[0] - a[0]) * (y - a[1]) - (b[1] - a[1]) * (x - a[0])) > 0

    def length(self) -> float:
        return float(sum(
            math.dist(self.center[i], self.center[(i + 1) % self._n])
            for i in range(self._n)))

    # -- reporting --------------------------------------------------------

    def audit_line(self, line_xy) -> dict:
        """How much room does a racing line leave? Used to catch a bad corridor."""
        d = [self.distance_from_center(p[0], p[1]) for p in line_xy]
        room = [self.half - v for v in d]
        return {
            "points": len(line_xy),
            "max_from_center": max(d),
            "half_width": self.half,
            "min_room_to_edge": min(room),
            "points_within_10cm_of_edge": sum(1 for r in room if r < 0.10),
            "uses_full_track": max(d) > self.half - 0.02,
        }


def load_track(name: str = "2022_april_pro_ccw", margin: float = 0.02) -> Track:
    return Track(ROOT / "data" / "tracks" / f"{name}.npy", margin=margin)
