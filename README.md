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

## Round 1: an agent fixed a defect the human never fixed

The loop has been run once. The full record is in `rounds/round_01_ledger.jsonl` and the diff is `rounds/round_01_v4d_is_reversed.patch`.

**Target:** `v4d`, which scored a reversed lap at 208 percent of a perfect one. This defect was never fixed by the human. `v4d` is the newest of the twelve files.

**The prediction, recorded before the change:**

> `v4d` rewards a reversed lap because its dominant term is derived from step count alone and is blind to direction. It never reads `params['is_reversed']`. Adding `is_reversed` to the existing fail-closed guard makes the function reject a reversed lap on its own.
> **Predicted mean on the optimal trajectory: 4.1125** (unchanged, because the guard fires only when reversed is true).

**The change**, two lines:

```diff
+        is_reversed = params.get('is_reversed', False)
...
-        ## Zero reward if off track ##
-        if not all_wheels_on_track or is_offtrack:
+        ## Zero reward if off track or driving the wrong way ##
+        if not all_wheels_on_track or is_offtrack or is_reversed:
```

**The result:**

| | Value |
| --- | --- |
| Predicted mean on optimal | 4.1125 |
| Measured mean on optimal | 4.1125 |
| **Prediction error** | **0.0 percent** |
| Reversed lap, before | 208 percent of optimal |
| Reversed lap, after | 0 percent of optimal |
| Regression on other trajectories | none, all identical |

**The round was still rejected.** The verdict on the claim was SUPPORTED, but the gate set still fails, so the function is not accepted as fixed. That is the design working: one correct fix does not make a broken function correct.

### The third defect, found by running the loop

The failing gate is not the one that was targeted. `v4d` raises `UnboundLocalError: cannot access local variable 'prev_waypoint_heading'` on **106 of its 214 steps**, about half the lap. Verified present in the unmodified original, so the change did not cause it.

The cause is structural. Three loops read `prev_waypoint_heading`, but it is assigned only in the first and third:

```
line 517:  heading_prev = prev_waypoint_heading      # loop 1, reads
line 529:  prev_waypoint_heading = waypoint_heading  # loop 1, assigns
line 564:  heading_prev = prev_waypoint_heading      # loop 2, reads
                                                     # loop 2 never assigns
line 615:  heading_prev = prev_waypoint_heading      # loop 3, reads
line 627:  prev_waypoint_heading = waypoint_heading  # loop 3, assigns
```

The second loop was duplicated from the first and the trailing assignment was dropped. **Half of every training episode's reward signal was throwing an exception, and the simulator absorbed it.**

Nobody knew. It is a 35KB file that produced a competitive lap time anyway.

### What this round does and does not show

**It shows:** an agent found and fixed a real defect in code the human wrote and never fixed, predicted the numeric outcome exactly before running, and introduced no regression. The gate then correctly refused to call the function fixed.

**It does not show** that an agent beats 18.121 s. That still needs training and a simulator. This is one round, against a defect the scorer had already localised.

**One honest failure of protocol:** the same party proposed the change and judged it. The method says the judge must be a fresh context that never saw the build. That was not satisfied here, and the contamination is recorded in the ledger rather than hidden.

**Running the loop also found a bug in the loop.** The ledger flagged the round as "tampered" because the target file changed between the prediction and the measurement. But that is the workflow: predict, edit, measure. The check was inverted. The real error condition is a file that is byte-identical at measurement time, which means a change was claimed and nothing was applied. Fixed in `loop/ledger.py`.

## RETRACTION: Round 2's conclusion was wrong

**An earlier version of this README claimed the agent's fixed `v4d` picks a lap 0.60 s faster than the human's. That claim is withdrawn. A stronger test refuted it.**

The measurement was wrong in a specific and instructive way. The correlation test scored each lap by **mean reward per step**. Reinforcement learning maximises **cumulative reward over an episode**, which is the sum.

Dividing by the step count normalises away the step count. That hides the most common defect in a hand-written reward function: **if the per-step reward is positive, a slower lap has more steps and therefore earns more.** The mean looks healthy while the sum is telling the car to crawl.

Rerunning the same 22 laps under both metrics:

| Reward function | rho, **sum** | rho, mean |
| --- | --- | --- |
| `reward_function.py` | 0.997 | 0.999 |
| `opt9` | 0.980 | 0.995 |
| `v1` | 0.891 | 0.978 |
| `opt10` | 0.812 | 0.916 |
| `v4d` | 0.438 | 0.793 |
| `v4b` | **−0.002** | 0.948 |
| `v2` | **−0.016** | 0.967 |
| `v4a` | **−0.093** | 0.947 |
| `v4c` | **−0.094** | 0.953 |
| `v4` | **−0.108** | 0.946 |
| `v4a2` | **−0.137** | 0.940 |
| `v3` | **−0.137** | 0.944 |

**Seven of twelve are uncorrelated or anti-correlated with speed under the metric that governs training.** The mean column says all twelve are fine.

`sim/track_sim.py` now defaults to `mode="total"`. The mean is still available and documented as misleading.

### And the fix made it worse, not better

Training a policy against each version, three seeds:

| seed | human `v4d` | agent `v4d`, crash fixed |
| --- | --- | --- |
| 0 | 24.33 s | **30.67 s** |
| 1 | 24.87 s | **30.53 s** |
| 2 | 17.07 s | **30.53 s** |

Sweeping 42 completing laps explains it:

| | rho(reward, speed) | rho(reward, step count) | cost of its favourite lap |
| --- | --- | --- | --- |
| Human `v4d` (crashes) | −0.237 | +0.237 | +9.8 % |
| Agent `v4d` (fixed) | **−0.907** | **+0.907** | **+50.5 %** |

`v4d` sums a positive per-step reward, so total reward tracks step count almost perfectly. **The crash was suppressing roughly half of those per-step contributions and accidentally masking the pathology.** Fixing the crash made the reward fire on every step and strengthened the incentive to drive slowly, from −0.237 to −0.907.

The agent's change was a correct bug fix and a worse reward function. Both are true.

### What this episode is actually evidence for

The doctrine says a proof must be able to fail. This one did, on its author.

A published claim, backed by a real measurement, was overturned by a stronger measurement of the same system. The failure mode was not a coding error. It was **measuring a convenient quantity instead of the governing one**, which is the same class of mistake the whole method is written to catch.

The original claim is left standing below, struck through, because deleting it would remove the evidence.

---

## ~~Round 2: the agent's reward function picks a faster lap than the human's~~ (RETRACTED, see above)

Round 1 proved a fix worked. It did not prove the fix made the car faster. **This section claimed to, and was wrong. It is kept for the record.**

### The test that settles it

A reward function has one job: rank fast trajectories above slow ones. If maximising the reward does not minimise lap time, the function is wrong.

`sim/track_sim.py` implements a kinematic bicycle model at the DeepRacer control rate, using the real 21-action space and the real racing line. `sim/correlate.py` simulates a grid of 60 driving policies, keeps the ones that complete a lap, and measures the rank correlation between each reward function's score and actual lap time.

**Every constant is taken from your own files.** The 15 Hz timestep comes from the reward functions themselves, which compute `current_actual_time = (step_count - 1) / 15`.

### The head-to-head

22 of 60 policies completed a lap. Lap times ran 16.67 s to 27.73 s. Both functions scored the identical set.

| | rho | picks | cost vs fastest |
| --- | --- | --- | --- |
| Human `v4d`, as shipped | 0.7930 | 17.47 s | 4.8 percent |
| **Agent `v4d`, crash fixed** | **0.9325** | **16.87 s** | **1.2 percent** |
| **Delta** | **+0.1395** | **−0.60 s** | **−3.6 points** |

**The agent's version selects a lap 0.60 seconds faster.**

The prediction was recorded first: rho 0.950 predicted, 0.9325 measured, **1.8 percent error**.

The change was one line, restoring an assignment the second loop was missing:

```diff
+                prev_waypoint_heading = waypoint_heading
```

### Where all twelve land

| Reward function | rho | picks | cost |
| --- | --- | --- | --- |
| `reward_function.py` | 0.999 | 16.67 s | 0.0 % |
| `opt9` | 0.995 | 16.67 s | 0.0 % |
| `v1` | 0.978 | 16.87 s | 1.2 % |
| `v2` | 0.967 | 16.87 s | 1.2 % |
| `v4c` | 0.953 | 16.87 s | 1.2 % |
| `v4b`, `v4a`, `v4`, `v3`, `v4a2` | 0.940 – 0.948 | 16.87 s | 1.2 % |
| `opt10` | 0.916 | 17.47 s | 4.8 % |
| **`v4d`** | **0.793** | **17.47 s** | **4.8 %** |

**The newest and largest function was the worst at the job.**

### A correction to what this README said earlier

`opt9` scores 0.995 here, but the discrimination test called it unable to distinguish anything. Both results are correct, and together they say something neither says alone.

In the discrimination suite every trajectory has the same step count, so `opt9`'s step-count bonus is identical across all of them and it looks constant. In the simulator, laps run 250 to 416 steps, so the same bonus tracks lap time almost perfectly.

`opt9` is a **sparse terminal reward**. It points at the right answer and gives no per-step learning signal. That is a more accurate description than "broken", and the earlier wording is corrected above.

### Limits of this proof, stated plainly

1. **This is not the AWS simulator.** The 0.60 s is on this model's lap times. It does not translate to the 18.121 s AWS figure and no claim is made that it does. What transfers is the ranking.
2. **The bar was low.** The agent fixed a crash. The human's function was broken, not merely suboptimal. Beating broken code is easier than beating good code.
3. **One target, one metric, one round.**
4. **The judge was not independent.** The same party proposed and judged, which the method forbids. It is recorded in the ledger rather than hidden.

### What is now proven, and what is not

**Proven:** an agent found a defect in shipped human code, fixed it in one line, predicted the numeric outcome to within 1.8 percent before running, and the corrected function selects a measurably faster lap on a common test set.

**Not proven:** that an agent beats 18.121 s on AWS. That needs the retired managed service or self-hosted compute, and the number is specific to a platform that no longer exists in that form.

## The portable core: ProofLoop

The rules in this repository are not about racing. They are about deciding what to believe when an agent says it improved something.

`proofloop/` holds that decision layer, separated from the task. It has **no dependency outside the Python standard library**, so a reader can copy it into another project without bringing anything with it.

**An environment supplies five methods.** Nothing else.

| Method | Meaning |
| --- | --- |
| `name()` | the identity of the environment |
| `baseline()` | the score to beat: a human result, or a known method |
| `bound()` | the best score physics or mathematics allows, or none |
| `propose_context()` | what the agent must know to write a candidate |
| `score(artifact, seed)` | the verifiable reward, plus the gate results |

`score` must be deterministic for a given seed. Otherwise the loop cannot tell an improvement from noise.

### Seven rules that cannot be turned off

Each rule exists because this repository made the matching mistake and recorded it in `CHANGELOG.md`.

| Rule | Came from |
| --- | --- |
| R1 Default FAIL | a passing check that proved nothing |
| R2 Limit the tools before the agent acts | a prompt that asked a model to behave |
| R3 Isolate the run | model code executing in the main process |
| R4 Score the governing quantity | scoring a mean when training maximised a sum |
| R5 Check the evaluator against a known bound | a 15.27 s lap against an 18.147 s floor |
| R6 Report the mean across seeds, and the spread | an 18.40 s result that was one lucky seed |
| R7 Keep a retraction, do not delete it | a published claim that was wrong |

### Two environments, so portability is tested and not asserted

An interface with one implementation is a single program with extra steps.

| Environment | Artifact | Verifiable reward | Bound |
| --- | --- | --- | --- |
| `envs/deepracer.py` | a reward function | the lap time of a policy trained on it | 18.147 s, the racing line's own times |
| `envs/tsp.py` | a tour heuristic | the tour length | 29.840, brute force over nine cities |

The travelling-salesman environment shares nothing with DeepRacer. No simulator, no policy, no training. **The loop ran on it with no change to `proofloop/loop.py`.**

```
python proofloop/demo.py --env tsp --rounds 5 --proposer scripted
```

```
round  1: REJECTED  [disallowed_code]     the candidate imported os
round  2: REJECTED  [invalid_output]      the tour visited a city twice
round  3:  no gain  mean 41.264           equals the baseline
round  4: ACCEPTED  mean 29.840           reached the brute-force optimum
round  5: ACCEPTED  mean 32.105           worse, so not kept as best
PROVEN: True
  - no independent party has reviewed this result
```

The last line is the point. The loop states the condition it has not met instead of hiding it.

## Where this sits in the current literature

The lap time and the tour length are **verifiable rewards**: a machine computes them from the outcome and no model judges them. That is the RLVR setting, which DeepSeek-R1 established as a general post-training paradigm.

[SEAGym](https://arxiv.org/abs/2606.17546) describes a self-evolving agent that supplies both the task policy and the harness-update rule, connected through a rollout and update interface. [The Landscape of Agentic Reinforcement Learning for LLMs](https://arxiv.org/abs/2509.02547) frames the shift from single-step decisions to temporally extended, partially observable ones. [Agent2 RL-Bench](https://arxiv.org/html/2604.10547v1) asks whether agents can engineer agentic RL post-training at all.

**This repository does not compete with those.** They run the rollouts. AgentGym-RL, SEAGym, and Harbor already do that well.

This is the layer that most self-improvement work leaves implicit: **the part that says no.** The field's practical problem is that reported self-improvements often do not replicate. This repository has a written record of killing three of its own false positives, including one it had already published.

## What each part demonstrates

This repository copies the pattern from the Node ecosystem. **It imports none of that code**, which is what makes it portable.

| Part of this repository | Node component | What it shows |
| --- | --- | --- |
| Gates G0 to G4 | **NodeProof** | A claim of success needs evidence. The default is FAIL. |
| `receipts/` | **NodeTrace** | Every claim links to the run that produced it. |
| `memory/`, grouped by mechanism | **NodeMem** | The search remembers the class of a failure, not the round number. |
| `search/sandbox.py` | **NodeAgent** | Limit the tools before the model acts. |
| The whole loop | **NodeRL** | Environment, reward, memory, and dataset export in one cycle. |

## Why this exists

Most demonstrations of self-improving agents let the agent grade its own work. A lap time cannot be argued with, and neither can an off-track flag.

The next step is the loop: an agent proposes one change to a reward function, writes its predicted score **before** the run, and a fresh-context judge with no write access decides whether the evidence supports the claim. The prediction error is the second metric. An agent that improves the score but predicts badly got lucky.

This scorer is the foundation for that loop. It had to be built and validated against a known history first.
