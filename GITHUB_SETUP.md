# Pushing ParkSentinel AI to GitHub

The dataset is now stored gzip-compressed (`data/jan_to_may_police_violation_anonymized791b166.csv.gz`,
~14.8MB instead of 105MB), so it's comfortably under GitHub's 100MB push
limit. **Git LFS is not needed.** Just a normal push.

## 1. Initialize the repo and commit

From inside the `project/` folder (the one with `core_analysis.py`, `README.md`, etc. directly inside it):

```bash
git init
git add .
git commit -m "Initial commit: ParkSentinel AI - Theme 1 submission"
```

## 2. Create the GitHub repo

Go to https://github.com/new, log in as **urmila27-mark**, and create a new repository:
- Name suggestion: `parksentinel-ai`
- Keep it **Public** (judges need to access it without logging in)
- Do NOT initialize with a README, .gitignore, or license — we already have our own, and that would conflict with the push below.

## 3. Connect and push

GitHub will show you a remote URL after creating the repo — it looks like:
`https://github.com/urmila27-mark/parksentinel-ai.git`

```bash
git branch -M main
git remote add origin https://github.com/urmila27-mark/parksentinel-ai.git
git push -u origin main
```

This should be fast — total repo size is now ~16MB, no large-file handling needed.

## 4. Verify

Visit `https://github.com/urmila27-mark/parksentinel-ai` in a browser. You should see all the folders (`streamlit_app/`, `notebooks/`, `cv_pipeline/`, `data/`), and `data/jan_to_may_police_violation_anonymized791b166.csv.gz` should show its real file size (~14.8MB), not an LFS pointer.

## 5. This is your Repository URL for the submission form

```
https://github.com/urmila27-mark/parksentinel-ai
```

---

## Notes on the dataset file

- The app and notebook both auto-detect gzip compression by file extension —
  `pandas.read_csv()` handles `.csv.gz` natively, no special code needed.
- Three columns that were 100% empty in the original CSV (`description`,
  `closed_datetime`, `action_taken_timestamp`) were dropped before
  compressing — this loses zero actual data, since those columns were
  empty for all 298,450 rows in every version of this dataset we've used.
  This was already confirmed in the very first EDA step (see the notebook,
  Section 1).
- If you ever need the original uncompressed 24-column CSV for any reason,
  it's preserved outside this folder structure — ask Claude or check your
  earlier session files.

## Troubleshooting

- **"Updates were rejected" on push** -> You initialized the GitHub repo with a README/license after all. Go to the repo settings and delete it, recreate empty, then retry step 3.
- **Push says file too large** -> Double check you're pushing the `.gz` file, not an accidental uncompressed copy. Run `git ls-files data/` to check exactly which dataset file got staged.
