# How we work with Git

A short guide for the seven of us. If you only read one section, read
**Everyday steps**.

---

## The branches

```
main        ← finished, tested versions only (proposal, check-in, final)
  ▲
develop     ← everyone's work comes together here
  ▲       ▲
cardio    lung        ← one branch per team
  ▲       ▲
your own branches     ← e.g. cardio-meera-late-fusion, lung-rohan-tta
```

Work moves **up**: your branch → your team's branch → `develop` → `main`.

Code that both teams share (`core/`, `tests/`, `docs/`) goes on a `shared-<topic>`
branch made from `develop`.

---

## Naming your branch

```
<team>-<your name>-<what you are doing>
```

Examples: `cardio-meera-late-fusion`, `lung-rohan-tta`, `lung-aisha-mobilenet`.
For shared code: `shared-<topic>`, e.g. `shared-metadata-schema`.

Lowercase, hyphens, no spaces.

---

## Everyday steps

**1. Start from your team's branch**

```bash
git checkout cardio
git pull
git checkout -b cardio-meera-late-fusion
```

**2. Save your work often**, small commits, in plain words:

```bash
git add <files>
git commit -m "add late-fusion model over heart-sound probabilities"
git push -u origin cardio-meera-late-fusion
```

**3. Open a pull request on GitHub** into your team's branch (`cardio` or `lung`).
Fill in the template, and ask one teammate to approve it.

**4. Merge it with "Create a merge commit"**, then press **Delete branch**.

---

## Three rules

1. **Never push straight to `main` or `develop`.** Always go through a pull request.
2. **Always merge with "Create a merge commit", never "Squash".** Squashing hides who
   did what, and the course grades each person's commit history.
3. **Never commit datasets or trained model weights.** `.gitignore` already blocks them.
   Data manifests and small results files *are* committed.

---

## Moving work up

**Team branch → `develop`:** when your team has something that works and has results,
the team lead opens a pull request into `develop`. The other team's lead approves it.

**Every Monday:** each team brings the latest `develop` into its branch, so shared code
reaches everyone early:

```bash
git checkout cardio
git pull
git merge origin/develop
git push
```

**`develop` → `main`:** only at the three milestones. Run the tests, merge, and tag:

```bash
git checkout develop && git pull && python -m pytest tests/
git checkout main && git pull
git merge --no-ff develop
git tag -a proposal-v1 -m "Proposal"     # later: checkin, final
git push origin main --tags
```

| Milestone | Tag |
|---|---|
| Proposal, 2 Oct 2026 | `proposal-v1` |
| Mid-project check-in | `checkin` |
| Final submission | `final` |

---

## When an experiment does not work

Do not merge the branch. Write a short note in `cardio/experiments/` or
`lung/experiments/` saying what you tried, the numbers, and what you concluded, and
merge that note instead. A failed experiment that is written down still counts as a
result.

---

## Later in the semester: the dashboard

When the models, fusion and test-time adaptation are done, create a `dashboard` branch
from `develop`. Everyone then works on `dashboard-<name>-<topic>` branches, exactly as
above.

---

## One-time GitHub settings (repository owner)

**Settings → General → Pull Requests**
- [ ] Allow merge commits — **on**
- [ ] Allow squash merging — **off**
- [ ] Allow rebase merging — **off**
- [ ] Automatically delete head branches — **on**

**Settings → Branches → Add branch protection rule**, once for `main` and once for
`develop`:
- [ ] Require a pull request before merging, with **1** approval
- [ ] Require status checks to pass — select `tests` (it appears after the first
  automated test run)

**Settings → Collaborators** — add all seven members with *Write* access.

---

## How the starting code arrives

The baseline pipeline comes from our earlier project. It is added one step at a time,
in the order it depends on itself:

| Step | Branch | Adds |
|---|---|---|
| 1 | `infra/repo-foundation` (done) | this guide, the README, folders, the pull request template |
| 2 | `shared-data-and-leakage` (done) | dataset downloads, manifests, leakage tests, automatic testing |
| 3 | `shared-front-ends` (done) | audio and image processing, feature cache, spectrogram, out-of-distribution check |
| 4 | `shared-training` (done) | training, simple baselines to beat, 8-bit quantisation, edge-case tests |
| 5 | `shared-descriptors-and-contracts` | derived measurements, and what each model takes in and gives back |

After step 5 the `cardio` and `lung` team branches are created, and team work begins.
