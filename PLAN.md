# Implementation Plan

This document uses ASD-STE100 Simplified Technical English. Sentences are short. The voice is active. Each instruction does one thing.

---

## 1. Purpose

This repository must do two things at the same time.

1. It must run without any dependency. A person clones it and runs it. Nothing else is necessary.
2. It must show the Node ecosystem pattern. A reader must see the pattern clearly, and must be able to copy it into another project.

The repository must not depend on NodeRoom, NodeKit, or Convex. A dependency on a large system stops other people from using the idea. The doctrine must stay separate from the platform.

---

## 2. What the repository does today

The repository trains a driving policy against a reward function. It then measures the lap time. It uses the lap time to decide if a reward function is good.

Two searches are complete:

- **Search A** changes the weights inside a fixed template. It reached 19.13 s.
- **Search B** asks a language model to write a new reward function. It reached 18.40 s on one seed.

Seven seeds show that neither search beats the human. The human reward function `opt9` has a mean of 18.50 s. The language model has a mean of 18.63 s.

---

## 3. What is wrong now

The repository copies the Node ecosystem ideas. It copies them badly. Two of the copies are defects, not style problems.

### Defect 1. The tool allowlist is only a request

The prompt tells the model to import only the `math` module. Nothing enforces this rule. The model can import any module. NodeAgent enforces the allowlist before the model sees the tools. This repository does not.

**Risk:** a candidate reward function can read files, open network connections, or delete data.

### Defect 2. Generated code runs in the main process

The loader imports the model output directly. The code runs in the same Python process as the harness. A bad candidate can stop the harness or change its data.

**Risk:** one candidate can corrupt the whole experiment.

### Problem 3. The receipts have no structure

The ledger writes one JSON line for each round. The line has no standard shape. NodeTrace uses a receipt with a status, a trace, a delta, and a verifier result. This repository does not.

### Problem 4. The memory does not survive

Search B puts the last six failures into the prompt. The memory disappears when the process stops. NodeMem keeps memory on disk and applies quality gates.

### Problem 5. RESEARCH.md contains an unlabelled design table

The file has a table that maps this loop onto the NodeRL layers. The table describes a plan. It does not describe code that exists. The table has no label. A reader can believe the code exists.

---

## 4. The plan

Do the steps in this order. Each step lists what to do, why to do it, and how to check the result.

### Step 1. Label the design table in RESEARCH.md

**Do:** add the word PLANNED to the table title. Add one sentence below the table. The sentence must say that the table is a design intent, and that the code does not exist yet.

**Why:** an unlabelled claim reads as a fact. This repository argues that a proof must be able to fail. An unlabelled plan cannot fail, because nobody can test it.

**Check:** read the table. A new reader must understand in one sentence that the code is not written.

### Step 2. Add an allowlist check that uses the abstract syntax tree

**Do:** parse each candidate with the `ast` module before it runs. Reject the candidate if any of these is true:

- It imports any module except `math`.
- It uses `open`, `exec`, `eval`, `compile`, `__import__`, or `globals`.
- It reads any attribute that starts with two underscore characters.
- It does not define a function with the name `reward_function`.

**Why:** this makes the allowlist a rule instead of a request. The check happens before the code runs. This is the NodeAgent pattern.

**Check:** write a candidate that imports the `os` module. The gate must reject it. The gate must give the reason.

### Step 3. Run each candidate in a separate process

**Do:** run each candidate in a subprocess. Give the subprocess a time limit. Send the result back as JSON. Treat a crash as a failed gate.

**Why:** a bad candidate must not stop the harness. This also makes the time limit real.

**Check:** write a candidate that never stops. The harness must continue, and must record a timeout.

### Step 4. Write a structured receipt for each candidate

**Do:** write one JSON file for each candidate. The file must contain:

| Field | Meaning |
| --- | --- |
| `source_sha256` | the hash of the candidate source |
| `gates` | each gate, with a pass or fail result and a reason |
| `trained_lap_s` | the lap time, or null |
| `seed` | the seed used |
| `verifier` | the result of the independent check |
| `created_at` | the time |

**Why:** NodeProof blocks a completion claim that has no evidence. A receipt is that evidence. A result without a receipt is an opinion.

**Check:** run one candidate. Open the receipt. Every gate must have a reason, and not only a pass or fail value.

### Step 5. Keep the failure memory on disk

**Do:** write each failure to a file. Group the failures by mechanism, and not by round number. Use these mechanism names:

- `step_farming`
- `early_termination`
- `syntax_error`
- `disallowed_import`
- `timeout`
- `does_not_finish`

**Why:** a search that forgets repeats the same mistake. NodeMem keeps the memory and removes duplicates.

**Check:** stop the search. Start it again. The new run must know the earlier failures.

### Step 6. Write the ecosystem map

**Do:** add a section to the README. The section must show which Node component each part of this repository demonstrates. It must state that this repository copies the pattern, and that it does not import the Node code.

**Why:** a reader must be able to find the full system after they read the small one.

**Check:** a reader who does not know the Node ecosystem must be able to name each component after they read the section.

---

## 5. What each part demonstrates

| Part of this repository | Node component | What it shows |
| --- | --- | --- |
| Gates G1 to G4 in `search/evaluator.py` | **NodeProof** | A claim of success needs evidence. The default result is FAIL. |
| Receipts in `receipts/` | **NodeTrace** | Every claim links to the run that produced it. |
| Failure memory in `memory/` | **NodeMem** | The search remembers the mechanism of each failure. |
| Allowlist and subprocess in `search/sandbox.py` | **NodeAgent** | The harness limits the tools before the model runs. |
| The whole loop | **NodeRL** | Environment, reward, memory, and dataset export in one cycle. |

This repository copies these patterns. It does not import the Node code. A reader can use the pattern without the platform.

---

## 6. How to know the work is complete

The work is complete when all these statements are true.

- [ ] A candidate that imports the `os` module is rejected before it runs.
- [ ] A candidate that never stops causes a timeout, and the harness continues.
- [ ] Each candidate has a receipt. Each gate in the receipt has a reason.
- [ ] The failure memory survives a restart.
- [ ] `RESEARCH.md` marks the NodeRL table as PLANNED.
- [ ] The README names each Node component and what it demonstrates.
- [ ] `CHANGELOG.md` records each change, with the reason.

---

## 7. What this plan does not do

This plan does not make the agent beat the human. Seven seeds show that no agent design is faster. That result stays.

This plan makes the harness honest. A harness that argues for proof must enforce its own rules first.
