# DeepRacer ProofLoop

An offline test that asks one question about a reinforcement-learning reward function:

> **Can this function tell a good trajectory from a bad one?**

If it cannot, every training run built on it is measuring nothing. The model will optimise the reward you wrote, not the behaviour you wanted.

This is [Proof-Driven Development](https://github.com/HomenShum/proof-driven-development), Rule 2, applied to a reward function: *the proof must be able to fail.*

---

## What this is not

**This does not predict lap times.** You cannot compute a lap time from a reward function. That needs training and a simulator.

It answers a cheaper question that catches a different class of bug: given a perfect lap and seven degraded laps, does the function rank them correctly?

Cost to run: zero. No AWS account, no training, no simulator.

---

## Result on 12 real reward functions

These are the reward functions from the JPMorgan Palo Alto AWS DeepRacer team, 2023 and 2024 seasons. The best of them took a lap from **23.260 s to 18.121 s**, against a theoretical racing-line optimum of 19.673 s.

**10 of 12 passed every gate. Two failed, for opposite reasons.**

### Failure 1: the oldest function carries no per-step signal at all

`reward_function_type3_opt9.py` returns the identical total, **6742.25**, for all eight trajectories. A perfect lap, a slow lap, an off-track car, and a backwards lap are indistinguishable.

The measured cause, step by step:

| Steps | Reward returned |
| --- | --- |
| 154 of 155 | about **0.002** each |
| The final step, at `progress == 100` | **6741.94** |

So more than 99.9 percent of the total comes from one terminal bonus:

```python
# opt9
if progress == 100:
    finish_reward = max(1e-3, (-REWARD_FOR_FASTEST_TIME /
                        (15*(STANDARD_TIME-FASTEST_TIME))) * (steps - STANDARD_TIME*15))
reward += finish_reward
```

That bonus reads `steps`. It does not read position, speed, direction, or off-track state. **The function gives the agent almost no signal during the lap, and its one large signal is blind to how the lap was driven.**

opt9 also has no standalone fail-closed guard. Off-track appears only inside a compound bonus condition. The next version added the missing gate:

```python
# opt10
if not all_wheels_on_track or is_offtrack:
    reward = 1e-3
```

**The scorer ranked the oldest function last, then located the exact term responsible, without being told the version order.** That is the validation. A scorer that cannot rank a known history correctly is not worth building an agent on top of.

*Honest caveat:* in the real simulator an off-track car is reset and never reaches `progress == 100`, so the terminal bonus would not fire. The harness lets every trajectory finish, which the simulator would not. The defect that survives that caveat is the real one: **opt9 provides no useful per-step gradient, and the simulator, not the reward function, is doing all the safety work.**

### Failure 2: the newest function rewards driving backwards

`reward_function_type3_opt10_rossprocw_v4d.py` scores a reversed lap at **208 percent of a perfect lap**.

The cause is a single unbounded term:

```python
difference = steps - LEAST_STEPS        # negative when the car is fast
cubic_component = normalized_diff ** 3  # stays negative
reward += 50 - (50 * cubic_component)   # so this term reaches +100
```

Every other signal in the file is an order of magnitude smaller: distance about 1 to 15, speed about 1 to 5, steering bonuses of +1.2 and +1.5. The function does check direction, but a penalty cannot compete with +100.

**This is the largest and most recent of the twelve files, at 35 KB.** Added sophistication introduced a reward the agent can exploit. That is the pattern this whole method exists to catch.

*Honest caveat:* in the real simulator, `is_reversed` ends the episode, so a trained model would not literally drive backwards. The unbounded step bonus is still a defect, because it dominates every safety term for any behaviour that reaches a progress milestone quickly.

### The systemic observation

**None of the twelve functions read `params['is_reversed']`.** All of them depend on the simulator to catch reverse driving. That is a hidden dependency, and it is invisible until you test the function outside the simulator.

---

## The gates

Default FAIL. A function passes only when it clears every gate.

| Gate | Requirement |
| --- | --- |
| 1. Discrimination | The optimal trajectory must score highest. |
| 2. Fail closed | An off-track car must score at most 25 percent of optimal. |
| 2b. No inverted reward | A reversed lap must not outscore a perfect lap. |
| 3. Monotonic | More degradation must score lower than less degradation. |
| 4. Runs | The function must not raise on any step. |

Gate 2b is separate from gate 2 on purpose. Reverse detection is normally the simulator's job, so a weak score there is reported as an observation. Scoring *above* a perfect lap is still a defect.

## The trajectories

Each one is generated from the racing line embedded in the reward function itself, so every function is tested against the track it was written for.

| Trajectory | Expected | What it is |
| --- | --- | --- |
| `optimal` | best | Follows the racing line at the prescribed speed |
| `slow_75`, `slow_40` | degraded | Correct line, reduced speed |
| `offset_15cm`, `offset_40cm` | degraded | Parallel to the line, off it |
| `oscillating` | degraded | Weaves across the line, wastes steps |
| `offtrack` | invalid | All wheels off the track |
| `reversed` | invalid | Drives the track backwards |

---

## Run it

Python 3.11 or newer. No dependencies.

```bash
python scorer/run.py data/reward_functions/reward_function.py
python scorer/run.py --all data/reward_functions --json report.json
```

Output for a single function:

```
FAIL  reward_function_type3_opt9.py   (258 racing points)
  trajectory     expect      mean reward   vs optimal
  optimal        best           43.4980
  offtrack       invalid        43.4980        100 %
  FAILURES
    - does not fail closed: an off-track car still scores 100 percent of optimal
```

---

## Limitations, stated plainly

1. **The harness is not the simulator.** Trajectories are synthesised from the racing line. A function that passes here can still fail on a real track.
2. **The racing line stands in for the centreline.** `distance_from_center` is measured from the racing line, not the true track centre. Most of these functions compute their own distance from their embedded line, so the effect is small, but it is not zero.
3. **Track width is assumed at 1.07 m** unless you pass `--track-width`.
4. **Passing every gate does not mean the function is good.** It means the function can tell the difference between the cases tested here. That is a floor, not a ceiling.

Point 4 is the honest one. This test proves a reward function is not obviously broken. It does not prove it is well designed.

---

## Why this exists

Most demonstrations of self-improving agents let the agent grade its own work. A lap time cannot be argued with, and neither can an off-track flag.

The next step is the loop: an agent proposes one change to a reward function, writes its predicted score **before** the run, and a fresh-context judge with no write access decides whether the evidence supports the claim. The prediction error is the second metric. An agent that improves the score but predicts badly got lucky.

This scorer is the foundation for that loop. It had to be built and validated against a known history first.
