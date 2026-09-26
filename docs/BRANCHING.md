# How we work: branches, merges and releases

Seven people, two teams, one repository. This document is the whole agreement on how
work moves from someone's laptop to `main`. Read it once before your first branch.

---

## 1. The branch map

```
main ─────────────── protected · receives only release/* and hotfix/*
  ▲                  tags: proposal-v1, checkin, final
  │
release/proposal · release/checkin · release/final
  ▲                  the testing stage: frozen, fully evaluated, fixes only
  │
develop ──────────── protected · where both teams' work meets · CI on every PR
  ▲          ▲          ▲            ▲              ▲              ▲ (later)
infra/<t>  docs/<t>   core/<t>     cardio/dev     lung/dev       app/dev
                                     ▲              ▲              ▲
                     cardio/model/<name>-<topic>    │              app/feat/<name>-<topic>
                     cardio/fusion/<name>-<topic>   │
                     cardio/tta/<name>-<topic>      │
                     cardio/exp/<topic>             │   ← experiments, never merged
                                                    │
                                   lung/model/<name>-<topic>
                                   lung/fusion/<name>-<topic>
                                   lung/tta/<name>-<topic>
                                   lung/exp/<topic>

hotfix/<topic> ← cut from main during demo week, merged back into main AND develop
```

Work always flows **upwards**: your branch → your team's lane → `develop` → a release
→ `main`. Nothing skips a level.

---

## 2. Every branch type

| Branch | Cut from | Merges into | Lives for | Who |
|---|---|---|---|---|
| `main` | — | — | whole project | nobody pushes; releases only |
| `develop` | `main` | `release/*` | whole project | nobody pushes; PRs only |
| `release/<milestone>` | `develop` | `main` (and back into `develop`) | a few days | both team leads |
| `hotfix/<topic>` | `main` | `main` and `develop` | hours | whoever fixes it |
| `cardio/dev` | `develop` | `develop` | whole project | cardio team (3) |
| `lung/dev` | `develop` | `develop` | whole project | lung team (4) |
| `app/dev` | `develop`, **late** | `develop` | end of semester | everyone |
| `cardio/<workstream>/<name>-<topic>` | `cardio/dev` | `cardio/dev` | days to a week | one person |
| `lung/<workstream>/<name>-<topic>` | `lung/dev` | `lung/dev` | days to a week | one person |
| `app/feat/<name>-<topic>` | `app/dev` | `app/dev` | days | one person |
| `cardio/exp/<topic>`, `lung/exp/<topic>` | team lane | **never merged** | as long as useful | anyone |
| `core/<topic>` | `develop` | `develop` | days | anyone; both leads review |
| `docs/<topic>`, `infra/<topic>` | `develop` | `develop` | days | anyone |

**Workstreams** are `model`, `fusion` and `tta`. Each person owns one, and it is the
part they will be asked about in the viva.

---

## 3. Naming

```
<lane>/<workstream>/<name>-<topic>
```

- `<name>` is your first name, lowercase.
- `<topic>` is 2–4 words, hyphenated, and says *what*, not *how long*.

Good:

```
cardio/fusion/meera-late-fusion-logreg
lung/tta/rohan-batchnorm-stats-adapt
lung/model/aisha-mobilenet-baseline
core/metadata-schema
cardio/exp/early-fusion-concat
```

Not good: `fix`, `aviral-branch`, `new-model-final-v2`, `cardio`.

### The one naming rule git enforces for you

**Never create a branch whose name is the prefix of another branch.** Git stores
branches as paths, so if a branch called `cardio` exists, `cardio/dev` cannot — and
vice versa. The same goes for `cardio/fusion` against `cardio/fusion/meera-…`. Every
branch name ends in a leaf; the prefixes (`cardio/`, `cardio/fusion/`) are only ever
folders.

---

## 4. Day-to-day workflow

**Start a piece of work**

```bash
git checkout cardio/dev
git pull
git checkout -b cardio/fusion/meera-late-fusion-logreg
```

**Commit little and often.** A week of small commits is worth more than one large one,
to you and to the course's "Process & Accountability" criterion.

```bash
git add <files>
git commit -m "cardio/fusion: add late-fusion logistic head over model probabilities"
```

Message format: `<lane>/<workstream>: <what changed, imperative>`. If an AI assistant
contributed to the commit, end the message with a trailer naming it, for example
`Co-Authored-By: Claude <noreply@anthropic.com>`.

**Keep your branch current with your lane**

```bash
git fetch
git merge origin/cardio/dev
```

**Open a pull request** into your lane (`cardio/dev`, `lung/dev` or `app/dev`), fill in
the template, and ask a teammate for review. CI must be green before merging.

**After it merges,** delete the branch. GitHub can do this automatically (§8).

---

## 5. Merging

### Merge commits only — never squash

A squash merge collapses all of your commits into one. That erases the week-by-week
record of who did what, which is exactly what the course reads to grade process and to
prepare viva questions. So:

- Pull requests merge with **Create a merge commit**.
- Squash and rebase merging are disabled in the repository settings (§8).

With history intact, anyone can see a person's contribution to an area:

```bash
git shortlog -sn --no-merges -- cardio/fusion/
```

### Lanes keep up with `develop` every week

Once a week, each team merges `develop` into its lane:

```bash
git checkout cardio/dev && git pull
git merge origin/develop
git push
```

Changes to `core/` then reach both teams within days rather than colliding with them at
the end of the semester.

### Lanes merge into `develop` when something works end to end

A lane merges into `develop` through a pull request, reviewed by **both** team leads,
when it has something that runs and has been evaluated — a trained model, a fusion
experiment with results, an adaptation method with before/after numbers. Not on a
timer, and not with half-finished work.

---

## 6. `core/`: the shared pipeline

`core/` holds everything both teams depend on:

- the audio and image front-ends;
- the spectrogram and the out-of-distribution gate;
- the anti-leakage rules and the training and evaluation harness;
- quantisation;
- the derived measurements;
- the **contracts** — the interface both models implement and the metadata schema both
  fusion workstreams consume.

Rules:

- **One definition of the features.** Lanes import from `core`; they never copy a
  front-end into their own folder, where the two copies would drift apart.
- **Every `core/` change needs review from both team leads**, because it changes both
  teams' results. CODEOWNERS enforces this.
- **The contracts land before either lane opens.** Two teams can then build fusion and
  adaptation independently, and the dashboard later consumes a single interface.

### The order in which the baseline arrives

The baseline pipeline is prior work, and it is imported one topic branch at a time, in
dependency order. Each step stands on the ones before it and can be tested on its own:

| # | Branch | Brings in | Why at this point |
|---|---|---|---|
| 1 | `infra/repo-foundation` | this document, the README, the folder skeleton, the PR template, CODEOWNERS | everyone needs to know where work goes before any code lands |
| 2 | `core/data-and-leakage` | dataset manifests, the leakage assertions, their tests, and CI | nothing is trained until the split is provably clean |
| 3 | `core/front-ends` | audio and image front-ends, spectrogram, out-of-distribution gate | features depend only on the data |
| 4 | `core/training-and-baselines` | cross-validated training, hurdle baselines, quantisation, results records | training needs data and features |
| 5 | `core/descriptors` | S1/S2 segmentation, murmur timing, occlusion maps | derived measurements sit on top of trained models |
| 6 | `core/contracts` | model interface, metadata schema, shared evaluation | formalises the baseline's outputs; the team lanes open on top of it |

Only the steps above are imported. Code that served the earlier project's hardware is
left behind.

---

## 7. Experiments, including the ones that fail

Some work exists to find out whether an idea is any good. It goes on
`cardio/exp/<topic>` or `lung/exp/<topic>`, and those branches are **never merged**.

What *is* merged is the finding. Write a short entry in `cardio/experiments/` or
`lung/experiments/` — what was tried, the numbers, the conclusion — and bring it in
through an ordinary pull request. A negative result that is written down is evidence;
one that is deleted is lost work nobody can defend in a viva.

---

## 8. What goes into git, and what never does

| Tracked | Never tracked |
|---|---|
| Source code, tests, configs | Raw datasets |
| **Data manifests** (every fold assignment — the leakage audit trail) | Feature caches |
| Small results files (`reports/*.json`, experiment logs) | Trained weights (`.keras`, `.h5`, `.tflite`, `.onnx`) |
| Figures used in the report | Virtual environments, notebook checkpoints |

Trained weights for a milestone are attached to that milestone's GitHub Release, next
to the tag that produced them.

---

## 9. Releases and milestones

Each milestone is a short-lived `release/*` branch. This is the project's testing stage:
the branch is frozen, fully evaluated and fixed up before it reaches `main`. Only fixes
are allowed on it.

| Milestone | Branch | Tag on `main` |
|---|---|---|
| Proposal (2 Oct 2026) | `release/proposal` | `proposal-v1` |
| Mid-project check-in | `release/checkin` | `checkin` |
| Final submission | `release/final` | `final` |

```bash
git checkout develop && git pull
git checkout -b release/checkin
#   run the full evaluation; commit fixes only
git checkout main && git merge --no-ff release/checkin
git tag -a checkin -m "Mid-project check-in"
git checkout develop && git merge --no-ff release/checkin
git push origin main develop --tags
```

During demo week, urgent fixes go on `hotfix/<topic>`, cut from `main` and merged back
into both `main` and `develop`.

---

## 10. Repository settings checklist

These are set once, in the GitHub web interface, when the team lanes open. The
repository is public, so branch protection is available on the free plan.

**Settings → General → Pull Requests**
- [ ] Allow merge commits — **on**
- [ ] Allow squash merging — **off**
- [ ] Allow rebase merging — **off**
- [ ] Automatically delete head branches — **on**

**Settings → Branches → Add branch protection rule** (or Settings → Rules → Rulesets)

| Pattern | Pull request required | Approvals | Code owner review | Status check `tests` must pass |
|---|---|---|---|---|
| `main` | yes | 2 (both leads) | yes | yes |
| `develop` | yes | 2 (both leads) | yes | yes |
| `release/*` | yes | 1 | — | yes |
| `cardio/dev` | yes | 1 (a cardio teammate) | — | yes |
| `lung/dev` | yes | 1 (a lung teammate) | — | yes |

For every pattern, also enable **Require branches to be up to date before merging** and
**Do not allow force pushes**. The `tests` status check can only be selected after the
CI workflow has run at least once.

**Settings → Collaborators** — add all seven members with *Write* access, then replace
the placeholder handles in `.github/CODEOWNERS`.

---

## 11. Quick reference

| I want to… | Do this |
|---|---|
| start work | branch from **your lane**, named `<lane>/<workstream>/<name>-<topic>` |
| try something risky | `<lane>/exp/<topic>`; merge only the write-up |
| change shared code | `core/<topic>` from `develop`; both leads review |
| get the latest shared code | merge `origin/<your lane>` into your branch |
| ship to the rest of the team | PR into your lane, then the lane PRs into `develop` |
| mark a milestone | `release/<milestone>` → `main`, tag it |
