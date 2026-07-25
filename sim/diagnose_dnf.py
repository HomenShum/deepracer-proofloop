"""Where does the car leave the track, and is the reference line even drivable?"""
import ast
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "sim"))
sys.path.insert(0, str(ROOT / "scorer"))

from track import load_track            # noqa: E402
from track_sim import load_line, load_actions  # noqa: E402
from train import rollout               # noqa: E402


def main():
    tk = load_track("2022_april_pro_ccw")
    line = load_line(ROOT / "data" / "lines" / "optimals_newest_2022_april_pro_ccw.txt")
    acts = load_actions(ROOT / "data" / "lines" / "AS21_newest_2022_april_pro_ccw.txt")

    # 1. Is the REFERENCE LINE itself inside my off-track test?
    d = [tk.distance_from_center(p[0], p[1]) for p in line]
    over = [i for i, v in enumerate(d) if v > tk.limit]
    print(f"half width {tk.half:.4f} m, calibrated limit {tk.limit:.4f} m")
    print(f"racing line max distance from centre {max(d):.4f} m")
    print(f"racing line points my test calls OFF TRACK: {len(over)} of {len(line)}")
    if over:
        print("  -> the reference line FAILS my own off-track test.")
        print("     The test is wrong, not the line. A known-good lap must be drivable.")

    # 2. Where does a sane policy actually die?
    print("\nwhere a middle-of-the-road policy leaves the track:")
    for g in ([8.0, 0.6, 0.0, 1.0, 1.0], [12.0, 0.5, 0.0, 1.5, 1.0]):
        fin, steps, lap, tr = rollout(line, acts, g)
        last = tr[-1]
        print(f"  genome sp={g[1]:.2f} -> {'FINISHED' if fin else 'DNF'} at step {steps}, "
              f"distance from centre {last['distance_from_center']:.4f} m")

    # 3. What threshold would make the reference line exactly drivable?
    print(f"\n  a threshold of {max(d):.4f} m makes the reference line drivable")
    print(f"  geometric half width is {tk.half:.4f} m")
    print(f"  the gap is {max(d) - tk.half:.4f} m")
    print("\n  DeepRacer ends an episode when ALL wheels are off, so the car centre")
    print("  may sit slightly outside half width while the car is still on track.")


if __name__ == "__main__":
    main()
