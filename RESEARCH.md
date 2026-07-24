# What it would take for an agent to beat the human, and the research that would settle it

Written 2026-07-24, after the Round 2 retraction and the grip-limit fix. **Nothing here is published until the question below is answered.**

---

## 1. The finding that reframes the question

With the grip limit in place, every lap time is finally physical:

| | Lap |
| --- | --- |
| Theoretical perfect lap (sum of the racing line's own per-point times) | **18.147 s** |
| **Best human reward function (`opt9`), trained** | **18.47 s** |
| `opt10` | 18.73 s |
| `v4d` | 19.87 s |
| `v1` | 21.20 s |
| The elaborate `v` family | 28.8 – 29.3 s |
| `reward_function.py` | **DNF**, its optimum crashes the car |

**The human's best result is 1.8 percent off the physical optimum.** There is 0.32 seconds available on this track. Not 5 seconds. Not 2.

So "can an agent beat the human's best" was the wrong question. The ceiling is nearly reached, and any claim of a large improvement would mean the simulator is broken again, not that the agent is clever.

**The real gap is not the ceiling. It is the search.**

The human wrote twelve reward functions over months. **Ten of them are worse than the first two.** The best result came early and the subsequent effort made things worse, culminating in a 35 KB function that crashes on half its steps and trains to a 19.87 s lap.

That is the winnable claim, and it is honest:

> **An agent does not find a better ceiling. It reaches the same ceiling far faster, and does not produce ten dead ends on the way.**

---

## 2. What it would have taken. Four things, and I got three of them wrong first.

### (a) A valid evaluator

Until this afternoon the simulator enforced no grip limit and produced 15.27 s laps against an 18.147 s physical floor. **Any search run against it would have optimised the flaw**, not the driving. That is precisely what happened: the policies "won" by cornering at impossible speeds.

An agent searching against a broken evaluator does not fail loudly. It succeeds, convincingly, at the wrong thing.

**Prerequisite: derive the evaluator's constraints from the data, then verify the results against a known physical bound before trusting a single number.**

### (b) The governing metric, not a convenient one

The correlation test scored laps by **mean reward per step**. Reinforcement learning maximises the **sum**. That one substitution hid the most common reward defect: a positive per-step reward makes a slower lap earn more.

Under the mean, all twelve functions looked healthy at 0.94 to 0.99. Under the sum, **seven of twelve are uncorrelated or anti-correlated with speed.**

**Prerequisite: the metric in the loop must be the metric that governs the outcome. Anything else is a proxy that the search will exploit.**

### (c) The outer objective must be the trained lap time

Rounds 1 and 2 optimised reward *shape*: discrimination gates, then rank correlation. Both are proxies. The agent fixed a real crash and made the trained car **six seconds slower**, and both proxies said it had improved.

**Prerequisite: close the loop on the thing you actually want. Propose a reward, train a policy against it, time the lap, feed that back.** Everything else is a cheaper signal that eventually lies.

### (d) Enough iterations that search beats intuition

One change per round with a human in the middle is not an agent advantage. It is a slower human.

The evaluation now costs **25 seconds for all twelve functions** on CPU. A search over hundreds of reward designs is affordable in an afternoon. That is the actual asymmetry: not that the agent is smarter, but that the loop is now cheap enough to run a thousand times.

---

## 3. Where NodeKit and NodeRL fit, honestly

This is not a forced fit. NodeRL's own description is:

> "Turn failed agent runs into the next better attempt, and into training data. Records what your agent did (NodeTrace), scores the outcome (NodeEval), remembers what worked and failed (NodeMem), and feeds the loop that retries until the task is proven. **NodeRL is the environment + reward + memory + dataset-exporter layer that most agentic-RL efforts are missing.**"

DeepRacer ProofLoop is exactly a NodeRL environment, and the mapping is one to one:

| NodeRL layer | What it is here |
| --- | --- |
| **Environment** | `sim/track_sim.py`, the grip-limited simulator |
| **Reward** (outer) | trained lap time from `sim/train.py`, not the reward function's own output |
| **Trace** (NodeTrace) | `loop/ledger.jsonl`, hypothesis and prediction recorded before the run |
| **Memory** (NodeMem) | which reward designs failed, and the mechanism, so the search does not re-derive "pays per step, drives slowly" a hundred times |
| **Proof gates** (NodeProof) | default FAIL; a design is accepted only if it finishes, is not anti-correlated under the sum, and beats the incumbent |
| **Dataset export** | every (reward design, trained lap time) pair is labelled training data. Twelve human designs plus N agent designs is a real corpus |

**NodeKit supplies the conformance contract**: what a candidate reward function must satisfy before it is allowed to count. Its own README says it "generates applications and then proves what they did." Here the artefact being generated is a reward function and the proof is a trained lap time.

The gates a candidate must clear, all of which came from a failure already observed:

1. **Finishes.** `reward_function.py` trains to a DNF. Its optimum crashes the car.
2. **Not anti-correlated under the sum.** Seven of twelve human designs fail this.
3. **Does not pay per step.** rho(reward, step count) must not exceed rho(reward, speed).
4. **Beats the incumbent trained lap time**, not a proxy.
5. **Prediction recorded before the run.** Prediction error is scored separately from improvement.

---

## 4. The research question

> **Given a fixed simulator and a fixed policy learner, can automated search over reward-function designs reach a human expert's best result faster and more reliably than the human's own iteration history?**

It is well posed because every term is measured:

- **Human baseline:** 12 designs, best trained lap 18.47 s, developed over months, with 10 of 12 worse than the first two. That history is documented in the repo and in the Notion reports.
- **Physical bound:** 18.147 s. Known, so any claim beyond it is a bug report.
- **Agent:** N designs, best trained lap, wall-clock to first reach 18.47 s.
- **Primary metric:** designs evaluated before matching the human's best.
- **Secondary metric:** fraction of proposals that are worse than the incumbent. The human's was 10 of 12.

**A negative result is publishable and likely.** If search cannot beat a competent human's early guess on a problem this small, that is worth knowing and says something real about where agent throughput helps.

---

## 5. Free compute plan

The whole evaluation is CPU-bound pure Python with no dependencies. **All twelve functions train in 25 seconds on 32 cores.**

| Resource | Use |
| --- | --- |
| **Local, 32 cores** | the main search. Nothing here needs a GPU. |
| **Kaggle notebooks** | ~30 hours/week free, 12-hour sessions. Use for the long sweep, and because a public notebook is reproducible in one click. This is also the distribution channel. |
| **Local RTX 4070** | only if the policy learner is upgraded from CEM to a neural policy. Not needed yet. |
| **deepracer-for-cloud** | last, and only to check whether the ranking transfers to the real AWS simulator. Needs about 45 GB; 38 GB free at 97 percent used, so clear space first. |

**Order matters.** Do not touch DRfC until the free loop produces a result worth validating.

---

## 6. What has to be true before anything is published

- [x] The simulator respects a physical bound derived from the data
- [x] The scoring metric is the one that governs training
- [x] The retraction of Round 2 is recorded, not deleted
- [ ] The loop is closed on trained lap time end to end
- [ ] A search of at least a few hundred reward designs has run
- [ ] The result is stated against the human baseline **and** the physical bound
- [ ] An independent fresh-context judge has reviewed it, which has not happened once yet

The last box is the one that has failed in every round so far. The same party has proposed and judged every time, which the method explicitly forbids.
