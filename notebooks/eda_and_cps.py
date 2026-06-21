# %% [markdown]
# # ParkSentinel AI — EDA & Coverage Priority Score (CPS)
#
# **Theme 1: Parking-Induced Congestion — Bangalore Traffic Police**
#
# This notebook documents the full exploratory analysis performed on the
# provided dataset, including a timezone-conversion bug we caught and fixed
# partway through (Section 3), and the computation of the Coverage Priority
# Score used to rank junctions (Section 5).
#
# All logic lives in `core_analysis.py` so the Streamlit dashboard and this
# notebook never drift out of sync — this notebook calls those functions and
# narrates what each one is doing and why.

# %%
import sys
sys.path.append("..")  # so core_analysis.py is importable when run from /notebooks

import pandas as pd
import matplotlib.pyplot as plt
import core_analysis as ca

pd.set_option("display.max_columns", None)
plt.rcParams["figure.figsize"] = (10, 4)

DATA_PATH = "../data/jan_to_may_police_violation_anonymized791b166.csv.gz"

# %% [markdown]
# ## 1. Load and inspect

# %%
raw = ca.load_raw(DATA_PATH)
print("Shape:", raw.shape)
raw.head(3)

# %%
print("Null counts:")
print(raw.isnull().sum())

# %% [markdown]
# Three columns (`description`, `closed_datetime`, `action_taken_timestamp`)
# are entirely null. Four others are null on the same ~125,000 rows,
# consistent with those rows not having completed a review workflow yet.
# `core_analysis.clean()` drops the dead columns and derives the fields we
# need (parsed violation list, IST timestamps, etc).

# %%
df = ca.clean(raw)
df[["created_datetime", "created_ist", "hour", "violation_list", "primary_violation", "is_main_road"]].head()

# %% [markdown]
# ## 2. Dataset overview

# %%
import json
print(json.dumps(ca.headline_stats(df), indent=2))

# %% [markdown]
# ## 3. Time-of-day analysis — and a timezone bug we caught
#
# `created_datetime` is stored in **UTC**. Bengaluru is **UTC+5:30**. The
# first time we ran this analysis, we read the hour directly off the raw UTC
# timestamp without converting — here's what that looked like:

# %%
raw_hour_utc = pd.to_datetime(raw["created_datetime"], errors="coerce", utc=True).dt.hour
total = len(raw)
biz_wrong = ((raw_hour_utc >= 10) & (raw_hour_utc < 18)).sum()
night_wrong = ((raw_hour_utc >= 21) | (raw_hour_utc < 7)).sum()
print(f"[INCORRECT - raw UTC] 10am-6pm: {100*biz_wrong/total:.1f}%   9pm-7am: {100*night_wrong/total:.1f}%")

# %% [markdown]
# This made it look like the system was blind during the day and only
# active overnight — which would have pointed an entire enforcement
# strategy at the wrong half of the day. After converting to IST:

# %%
bucketed = ca.bucketed_distribution(df)
bucketed

# %%
ax = bucketed.plot(x="bucket", y="share_pct", kind="bar", legend=False, color="#0E6E5C")
ax.set_ylabel("% of all violations")
ax.set_title("Violations by time of day (IST) — corrected")
plt.xticks(rotation=30)
plt.tight_layout()
plt.show()

# %% [markdown]
# **Finding #1:** violations are dense from ~3am to 1pm IST and collapse to
# near-zero from 3pm to midnight. Only **1.44%** of all 298,450 violations
# are captured in that 3pm–midnight window — confirmed below, and also
# confirmed to hold specifically within the "PARKING IN A MAIN ROAD" subset.

# %%
gap_pct = ca.coverage_gap_share(df) * 100
print(f"Share of violations in the 3pm-midnight IST gap: {gap_pct:.2f}%")

main_road = df[df["is_main_road"]]
gap_pct_main_road = ca.coverage_gap_share(main_road) * 100
print(f"Same gap, restricted to main-road violations only: {gap_pct_main_road:.2f}%")

# %% [markdown]
# ## 4. Data-quality check — rejection rate by violation type
#
# The `validation_status` field records manual-review outcomes for
# camera-flagged incidents. This is a useful, already-measured baseline
# accuracy figure for the *existing* system, before any new AI is added.

# %%
status_breakdown = ca.validation_status_breakdown(df)
status_breakdown

# %%
rejection = ca.rejection_rate_by_type(df, min_n=200)
rejection

# %%
ax = rejection.plot(x="violation_type", y="rejection_rate_pct", kind="bar", legend=False, color="#C0524F")
ax.set_ylabel("Rejection rate (%)")
ax.set_title("Manual-review rejection rate by violation type")
plt.xticks(rotation=40, ha="right")
plt.tight_layout()
plt.show()

# %% [markdown]
# **Finding #2:** rejection rate is consistently 24–36% across every major
# violation category — not concentrated in one type. This is the accuracy
# ceiling of the *current* pipeline; any proposed CV system should be
# designed to beat it (see the dwell-time-confirmation stage in
# `cv_pipeline/`), not just match it.

# %% [markdown]
# ## 5. Repeat offenders

# %%
import json
print(json.dumps(ca.repeat_offender_summary(df), indent=2))

# %% [markdown]
# ## 6. Junction recovery — recovering the ~49.5% of rows with no junction tag
#
# Roughly half of all rows have `junction_name == "No Junction"`, despite
# every one of them having usable latitude/longitude. Most analyses would
# silently drop these. We recover them by matching each unmapped row to its
# nearest *named* junction centroid (deterministic haversine distance, not a
# trained model), within a 500m cap so we never force a false match.

# %%
recovered = ca.recover_unmapped_junctions(df, max_distance_km=0.5)
import json
print(json.dumps(ca.recovery_summary(recovered), indent=2))

# %% [markdown]
# Only ~11% of unmapped rows are within 500m of a known junction — most are
# genuinely far from any of the 169 curated junctions. That's itself a
# finding about how incomplete the junction list is, not a failure of the
# matching method. CPS below is computed on this recovered junction set, so
# it matches the dashboard's default behavior.

# %% [markdown]
# ## 7. Coverage Priority Score (CPS)
#
# CPS ranks junctions by a blend of four factors instead of raw violation
# count. See the project README for the plain-language explanation of why
# raw counts are misleading and what each weight represents. In short:
#
# ```
# CPS = 0.40 × coverage_gap + 0.30 × main_road_share
#     + 0.20 × repeat_offender_share + 0.10 × violation_density
# ```
#
# Junctions with fewer than 200 total violations are excluded before scoring,
# because percentage-based components get noisy at low sample sizes.

# %%
cps = ca.compute_cps(recovered, min_volume=200, junction_col="junction_name_recovered")
cps[["rank", "junction_name", "n", "CPS", "coverage_gap_pct", "main_road_share_pct", "repeat_share_pct"]].head(15)

# %% [markdown]
# Note junction rank #5, **Nayandahalli Junction** — only 206 total
# violations, which would not place in any top-20 ranking by raw count, but
# its 35.9% main-road share surfaces it as a genuine priority. This is the
# core value of CPS: it disagrees with a simple leaderboard on purpose.

# %%
ax = cps.head(10).plot(x="junction_name", y="CPS", kind="barh", legend=False, color="#0E6E5C")
ax.invert_yaxis()
ax.set_xlabel("CPS")
ax.set_title("Top 10 junctions by Coverage Priority Score")
plt.tight_layout()
plt.show()

# %% [markdown]
# ## 8. Clustering cross-check — an independent method, not just a formula
#
# CPS uses fixed, hand-chosen weights. To check whether its priorities are
# actually supported by the data (rather than an artifact of our weight
# choices), we run **k-means clustering** on the same four underlying
# features, with no weights imposed. This is genuine unsupervised ML — it
# needs no labels, which this dataset doesn't have for a supervised model
# anyway.

# %%
clustered = ca.cluster_junctions(cps, n_clusters=3)
clustered["tier"].value_counts()

# %%
centers = ca.cluster_centers_summary(cps, n_clusters=3)
centers[["tier", "n_junctions", "coverage_gap_pct", "main_road_share_pct", "repeat_share_pct", "density_norm_pct"]]

# %% [markdown]
# The "High priority" tier is defined mainly by **main-road share** (~21% on
# average, vs ~3% in other tiers) — i.e. clustering independently discovered
# that main-road blockage is the dominant distinguishing feature, without
# being told CPS weights it second-highest at 0.30.

# %%
agreement = ca.cps_vs_cluster_agreement(clustered, top_n=10)
print(f"Agreement with CPS top 10: {agreement['pct_agreement']}% ({agreement['n_overlap']}/10)")
print()
print("Junctions CPS ranks highly that clustering does NOT place in 'High priority':")
import pandas as pd
pd.DataFrame(agreement["disagreement_junctions"])

# %% [markdown]
# **70% agreement is a meaningful, honest result** — not perfect (which would
# be suspicious, since the methods would then not really be independent), and
# not low either. The 3 disagreements are explainable: those junctions score
# high on CPS mostly via raw violation density (CPS weight 0.10), while
# clustering's high-priority tier is defined by main-road share. This is a
# real methodological difference between the two approaches, not noise.

# %% [markdown]
# ## 9. Export scored junctions for the dashboard

# %%
cps.to_csv("../data/junctions_scored.csv", index=False)
bucketed.to_csv("../data/hourly_distribution.csv", index=False)
rejection.to_csv("../data/rejection_by_type.csv", index=False)
print("Exported junctions_scored.csv, hourly_distribution.csv, rejection_by_type.csv to ../data/")

# %% [markdown]
# ## 10. Limitations (carried over from the technical proposal)
#
# - The 3pm–midnight gap is a robust pattern in this data, but this dataset
#   alone cannot confirm *why* (camera/staffing schedule vs. genuine
#   behavioral difference). Recommended next step: audit camera uptime logs.
# - The CPS weights are a reasoned prioritization, not values fitted to a
#   real congestion outcome — this dataset has no traffic-speed or delay
#   field to fit against. Phase 3 of the rollout plan (live traffic-speed
#   API) is designed specifically to make that validation possible.
# - ~49.5% of rows have no junction tag and are excluded from CPS scoring
#   entirely; a complete system would recover them via lat/long geofencing.
