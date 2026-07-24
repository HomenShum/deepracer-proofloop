"""Why does fixing the crash in v4d produce a slower trained car?

Sweeps a grid of policies, keeps the ones that finish, and asks the only
question that matters for a summed reward:

    Among laps that complete, does this reward function pay MORE for a
    slower lap?

If it does, then maximising it makes the car slower, and training will do
exactly that.
"""
import contextlib
import io
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scorer"))
sys.path.insert(0, str(ROOT / "sim"))

from core import load_reward_function                    # noqa: E402
from track_sim import load_line, load_actions, spearman  # noqa: E402
from train import rollout                                # noqa: E402

line = load_line(ROOT / "data" / "lines" / "optimals_newest_Ross_racing_line.txt")
acts = load_actions(ROOT / "data" / "lines" / "AS21_newest_Ross_racing_line.txt")


def total_reward(fn, trace):
    t = 0.0
    with contextlib.redirect_stdout(io.StringIO()):
        for p in trace:
            try:
                t += float(fn(dict(p)))
            except Exception:
                pass
    return t


# find genomes that actually complete a lap
finishers = []
for la in (4, 6, 8, 10, 12):
    for sp in (0.35, 0.45, 0.55, 0.65, 0.75, 0.85, 0.95):
        for caution in (0.4, 1.0, 1.8):
            g = [float(la), sp, 0.0, caution, 1.0]
            fin, steps, lap, tr = rollout(line, acts, g)
            if fin:
                finishers.append((g, steps, lap, tr))

print(f"{len(finishers)} of 105 policies completed a lap")
if finishers:
    laps = [f[2] for f in finishers]
    print(f"lap times {min(laps):.2f}s to {max(laps):.2f}s\n")

for label, f in [("HUMAN v4d (crashes on ~half the steps)", "_v4d_HUMAN.py"),
                 ("AGENT v4d (crash fixed)", "_v4d_AGENT.py")]:
    fn = load_reward_function(Path(f))
    rewards = [total_reward(fn, t[3]) for t in finishers]
    laps = [t[2] for t in finishers]
    steps = [t[1] for t in finishers]

    # rho against SPEED. Positive means the reward prefers fast laps.
    rho_speed = spearman(rewards, [-l for l in laps])
    # rho against STEP COUNT. Positive means the reward simply pays per step.
    rho_steps = spearman(rewards, [float(s) for s in steps])

    best = max(range(len(rewards)), key=lambda i: rewards[i])
    fastest = min(range(len(laps)), key=lambda i: laps[i])

    print(f"{label}")
    print(f"  rho(total reward, SPEED)      {rho_speed:+.3f}   "
          f"{'prefers fast' if rho_speed > 0 else 'PREFERS SLOW'}")
    print(f"  rho(total reward, STEP COUNT) {rho_steps:+.3f}   "
          f"{'pays per step, so a longer lap earns more' if rho_steps > 0.5 else ''}")
    print(f"  highest-reward lap {laps[best]:.2f}s   fastest available {laps[fastest]:.2f}s   "
          f"cost {(laps[best]-laps[fastest])/laps[fastest]*100:+.1f}%")
    print()
