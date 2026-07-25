# ProofLoop: a portable interface for proof-gated self-improvement

This document uses ASD-STE100 Simplified Technical English.

---

## 1. What this is

ProofLoop is an interface. Any task that has a **verifiable reward** can plug into it. The loop then improves an artifact against that reward, and refuses to accept an improvement that it cannot prove.

The interface is small on purpose. A person must be able to copy it into another project in one afternoon.

## 2. Why this exists

The field has a replication problem. A paper reports that an agent improved itself. Another team runs the same code and the improvement is not there. The cause is usually one of three things:

1. The measured quantity is not the governing quantity.
2. The evaluator has a flaw, and the search finds the flaw instead of the solution.
3. The result came from one random seed.

This repository made all three mistakes and recorded each one. `CHANGELOG.md` holds the evidence. The interface below exists to make those mistakes visible instead of publishable.

## 3. Definitions

**Verifiable reward.** A number that a machine computes from the outcome. No model judges it. A lap time is verifiable. "The answer looks correct" is not.

**Artifact.** The thing the loop improves. A reward function, a prompt, a policy, or a configuration.

**Gate.** A check with a reason. The default result is FAIL.

**Receipt.** A record that links a claim to the run that produced it.

**Mechanism.** The reason a candidate failed, named as a class and not as a round number.

## 4. The interface

An environment must supply five methods. Nothing else.

| Method | Returns | Meaning |
| --- | --- | --- |
| `name()` | text | the identity of the environment |
| `baseline()` | number | the score to beat. A human result, or a known method |
| `bound()` | number or none | the best score physics or mathematics allows |
| `propose_context()` | text | what the agent must know to write a candidate |
| `score(artifact, seed)` | Result | the verifiable reward, plus the gate results |

`score` must be **deterministic for a given seed**. If it is not, the loop cannot tell an improvement from noise.

A `Result` holds:

- `value`: the verifiable reward. Lower is better by convention.
- `gates`: a list of gate results, each with a name, a pass or fail value, and a reason.
- `mechanism`: the failure class, or none when the candidate passed.

## 5. The loop

```
1. read the memory of past failures
2. ask the proposer for one candidate
3. admit or reject the candidate before it runs
4. score the candidate on N seeds
5. accept only if it beats the baseline on the MEAN of those seeds
6. write a receipt
7. write the failure mechanism to memory
8. repeat
```

Step 5 is the rule the field most often breaks. **A single seed is not a result.** The loop must not report a best-of-N score as the outcome.

## 6. The rules that cannot be turned off

These rules come from failures this repository already made.

**R1. Default FAIL.** A candidate passes only when every gate passes.

**R2. Limit the tools before the agent acts.** Check the artifact before it runs. Do not ask the agent to behave.

**R3. Isolate the run.** A bad candidate must not stop the loop.

**R4. Score the governing quantity.** Not a convenient proxy. This repository scored a mean when training maximised a sum, and published a wrong result.

**R5. Check the evaluator against a known bound.** This repository produced a 15.27 s lap when 18.147 s was the physical floor. The search was exploiting the simulator.

**R6. Report the mean across seeds, and the spread.** A margin inside the spread is not a win.

**R7. Keep a retraction, do not delete it.** A corrected claim stays visible with its correction.

## 7. What "proven" means here

A claim is proven when all of these are true:

- [ ] It beats a stated baseline on the mean of at least five seeds.
- [ ] The seed ranges do not overlap.
- [ ] The evaluator is checked against a known bound.
- [ ] The measured quantity is the governing quantity.
- [ ] A receipt exists for every number in the claim.
- [ ] An independent party reviewed it.

The last box has never been ticked in this repository. Every judgement so far was made by the same party that proposed the change. That is recorded, not hidden.

## 8. Portability

An interface with one implementation is not portable. It is a single program with extra steps.

This repository ships **two** environments so the claim can be tested:

1. `envs/deepracer.py` — reward-function design, verified by a trained lap time.
2. `envs/sorting.py` — a small algorithmic task with a verifiable reward and a known optimum.

A third environment written by a reader is the real test. The interface is complete when a reader adds one without changing the loop.

## 9. What this is not

This is not a training framework. It does not replace AgentGym-RL, SEAGym, or Harbor. Those run the rollouts. This decides what to believe about the result.

This is the layer that most self-improvement work leaves implicit: the part that says **no**.
