# ParkSentinel AI — Theme 1: Parking-Induced Congestion

Bangalore Traffic Police Hackathon · Team CYPH3R / Void Walkers · Sofia (Urmila), IIIT Kota

A working prototype for AI-driven parking intelligence: detecting illegal-parking
hotspots and quantifying their impact on traffic flow, using the official BTP
violation dataset.

## What's actually in this project

Everything here is real, runnable code tested against the actual 298,450-row
dataset — nothing is a mockup or a description of what code "would" do.

| Folder | What it is |
|---|---|
| `core_analysis.py` | Single source of truth for all EDA, junction recovery, and CPS logic. Imported by both the notebook and the dashboard so they can never drift out of sync. |
| `notebooks/eda_and_cps.ipynb` | Full exploratory analysis, narrated step by step, including the timezone bug we caught and corrected. Already executed — outputs are saved in the notebook. |
| `streamlit_app/` | The live dashboard. Run with `streamlit run app.py`. |
| `cv_pipeline/enforcement_logic.py` | Real geofencing (point-in-polygon) + dwell-time tracking + evidence generation logic — the "after detection" half of the proposed CV pipeline. |
| `data/` | The dataset, plus exported intermediate CSVs (`junctions_scored.csv`, etc.) produced by the notebook. |

## Quickstart

```bash
cd streamlit_app
pip install -r requirements.txt
streamlit run app.py
```

Opens a 5-tab dashboard:
1. **Overview** — violation breakdown, rejection rates, repeat offenders
2. **Time Analysis** — the corrected (and, for comparison, the original incorrect) time-of-day pattern
3. **CPS Rankings + Heatmap** — live, interactive Coverage Priority Score ranking and geospatial heatmap
4. **Junction Recovery** — recovers the ~49.5% of rows with no junction tag via lat/long nearest-match
5. **Enforcement Demo** — a live, adjustable geofence + dwell-time simulation

You can also upload a *different* BTP-format CSV from the sidebar and every
number on the dashboard recomputes live against it.

## What this is honestly NOT

We'd rather you walk into a judging round knowing these limits than get
caught off guard by a question:

- **CPS is a formula, not a trained model.** The weights (0.40 / 0.30 / 0.20 / 0.10)
  are a reasoned prioritization tied to the brief's stated concerns, not values
  fitted to a real congestion outcome — this dataset has no traffic-speed or
  delay field to fit against. Say this proactively if asked.
- **There is no vision model that classifies congestion from a camera frame.**
  We deliberately did not build this. The dataset contains zero images and zero
  congestion-severity labels, so there is nothing to train such a model on.
  Claiming otherwise would not survive a technical follow-up question.
- **Junction recovery is nearest-centroid matching, not a trained model.**
  It's deterministic and auditable (every match's distance is inspectable),
  which is a feature for an enforcement tool, not a shortcoming.
- **The CV pipeline's vehicle-detection stage (YOLOv8 + ANPR) is specified,
  not demonstrated, in this submission**, because it requires either a live
  camera feed or labeled video we don't have access to in this environment.
  `enforcement_logic.py` is written so that a real detector's output (a stream
  of `Detection` objects) plugs in directly with no changes to the geofence/
  dwell-time/evidence logic — see the module docstring.

## Reproducing the EDA

```bash
cd notebooks
jupyter nbconvert --to notebook --execute --inplace eda_and_cps.ipynb
```

This regenerates `data/junctions_scored.csv`, `data/hourly_distribution.csv`,
and `data/rejection_by_type.csv` from scratch.

## Key findings (all reproducible from `core_analysis.py`)

- Only **1.44%** of all violations are captured 3pm–midnight IST; **81.68%**
  fall between 3am and 1pm. The raw timestamps are in UTC — reading them
  without converting to IST (UTC+5:30) produces the opposite, misleading
  picture. See the Time Analysis tab's toggle to reproduce that mistake live.
- Manual-review rejection rate is **24–36%** across every violation category —
  a measurable, pre-existing accuracy ceiling in the current pipeline.
- **~49.5%** of rows have no junction tag despite having usable lat/long;
  of those, only ~11% are within 500m of a known junction — most genuinely
  aren't near any of the 169 curated junctions, which is itself a finding
  about the limits of that junction list.
- Junction ranking by CPS meaningfully disagrees with ranking by raw violation
  count — see the "rank improvement" table in the CPS tab for concrete examples.
