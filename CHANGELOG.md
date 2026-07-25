# Changelog

This file uses ASD-STE100 Simplified Technical English.

Each entry records what changed and why. An entry that corrects an earlier claim says so. No entry is deleted.

---

## 2026-07-24

### Added: an implementation plan

`PLAN.md` records the six steps that make this harness enforce its own rules.

**Reason:** the repository argues that a proof must be able to fail. Two of its own safety rules were not enforced. A reader must be able to see the gap and the fix.

---

### Changed: RESEARCH.md marks the NodeRL table as PLANNED (Step 1)

The table that maps this loop onto the NodeRL layers now has the word PLANNED in its title. A sentence below the table says that the table is a design intent, and that the code does not exist.

**Reason:** the table described work that is not written. It had no label. A reader could believe the code exists. An unlabelled plan cannot fail a test, because nobody can test it. This is the same class of error as the mean-against-sum defect recorded below.

---

### Added: an allowlist check that uses the abstract syntax tree (Step 2)

`search/sandbox.py` parses each candidate before it runs. The check rejects a candidate for any of these reasons:

- It imports a module other than `math`.
- It calls `open`, `exec`, `eval`, `compile`, `__import__`, or `globals`.
- It reads an attribute whose name starts with two underscore characters.
- It does not define a function with the name `reward_function`.

**Reason:** the prompt asked the model to import only `math`. Nothing enforced the rule. A candidate could read files or open a network connection. NodeAgent enforces the allowlist before the model sees the tools. This repository now does the same.

---

### Added: each candidate runs in a separate process (Step 3)

`search/sandbox.py` runs each candidate in a subprocess with a time limit. The subprocess returns JSON. A crash or a timeout becomes a failed gate.

**Reason:** the loader imported model output into the main process. One bad candidate could stop the whole experiment. A time limit in the same process cannot stop an endless loop.

---

### Added: gate G0 in the evaluator

`search/evaluator.py` now calls `admit()` before it writes or loads a candidate. A candidate that fails the allowlist never reaches the disk. The receipt records the source hash for every candidate, including a rejected one.

**Reason:** a check that a caller can skip is not a gate. The check must sit in the one path that every candidate takes.

**Result:** a candidate that imports the `os` module is rejected with the reason `it imports 'os'. Only ['math'] is allowed`. An honest candidate trains and returns a lap time.

---

### Added: an adversarial test for the gates

`search/test_sandbox.py` attacks each gate on purpose. It tries seven hostile candidates and one endless loop.

**Reason:** PLAN.md says a proof must be able to fail. A gate that nobody attacks is a decoration. The test shows the gate can reject.

**Result:** all gates hold.

| Attack | Result |
| --- | --- |
| imports `os` | rejected |
| imports from `subprocess` | rejected |
| calls `open` | rejected |
| calls `eval` | rejected |
| reads a private attribute | rejected |
| no `reward_function` | rejected |
| does not parse | rejected |
| never stops | timed out after 5 seconds, the harness continued |

---

### Added: ProofLoop, a portable interface (proofloop/)

`proofloop/SPEC.md` defines five methods an environment must supply. `proofloop/core.py`
holds the loop, the failure memory, and the receipts. Neither file imports anything
outside the Python standard library.

**Reason:** the harness was welded to DeepRacer. An idea welded to one problem cannot
travel. The interface separates the rule set from the task.

**Seven rules cannot be turned off.** Each one comes from a mistake this repository
already made and recorded: default FAIL, limit the tools before the agent acts, isolate
the run, score the governing quantity, check the evaluator against a known bound, report
the mean across seeds, and keep a retraction.

---

### Added: a second environment (proofloop/envs/tsp.py)

A nine-city travelling-salesman instance. It has no simulator, no policy, and no
training. The artifact is a Python heuristic. The reward is the tour length. The optimum
is computed by brute force, so rule R5 has a real bound to test.

**Reason:** an interface with one implementation is a single program with extra steps.
Portability is a claim, so it needs evidence.

**Result:** the loop ran on the new environment with no change to `core.py`.

| Round | Outcome |
| --- | --- |
| 1 | REJECTED, `disallowed_code`. The candidate imported `os`. |
| 2 | REJECTED, `invalid_output`. The tour visited a city twice. |
| 3 | No gain. Nearest neighbour equals the baseline at 41.264. |
| 4 | ACCEPTED. Nearest neighbour with 2-opt reached 29.840. |
| 5 | ACCEPTED but worse at 32.105. Not recorded as the best. |

The best result, 29.840, equals the brute-force optimum. Five seeds, standard deviation
0.000, because the task is deterministic.

The loop reports PROVEN, and names the one condition still open: no independent party
has reviewed the result.

---

## Earlier work, for the record

### Retracted: Round 2 claimed the agent picks a faster lap

The claim is withdrawn. The correlation test scored each lap by the mean reward for each step. Training maximises the sum. Division by the step count hides a reward that pays for a longer lap.

Seven of twelve human reward functions are uncorrelated or anti-correlated with speed under the sum. The mean says all twelve are good.

**Kept, not deleted.** The original claim stays in the README with a line through it. Deletion would remove the evidence.

---

### Corrected: the simulator had no grip limit

Every lap time before this correction was impossible. The simulator produced a 15.27 s lap. The theoretical fastest lap is 18.147 s.

The racing line data gives the limit. Lateral acceleration reaches 2.01 m/s squared, which is about 0.21 g. The simulator now models understeer at that limit.

**Reason:** a search against an invalid simulator does not fail. It succeeds at the wrong thing.

---

### Corrected: the description of `opt9`

An earlier note said that `opt9` cannot tell trajectories apart. This is true only when every trajectory has the same step count. In the simulator, laps have different step counts, and `opt9` tracks lap time almost perfectly.

`opt9` is a sparse terminal reward. It points at the right answer. It gives no signal during the lap. This is a better description than "broken".

---

### Result: no agent design beats the human

Seven seeds, inner search of 8 iterations and 32 members:

| Design | Mean | Standard deviation | Did not finish |
| --- | --- | --- | --- |
| Human `opt9` | 18.50 s | 0.651 | 1 |
| Human `opt10` | 18.60 s | **0.123** | 0 |
| Agent, weight search | 18.80 s | 0.589 | 0 |
| Agent, language model | 18.63 s | 0.795 | 0 |

The language model reached 18.40 s on one seed. Across seven seeds the mean is slower than the human. The ranges overlap. The single result was luck.

**A separate finding:** `opt10` has a standard deviation of 0.123. Every other design is between 0.589 and 0.795. `opt10` is the most reliable reward function in the set. A single run never shows this.
