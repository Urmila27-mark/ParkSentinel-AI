"""
core_analysis.py
Shared EDA + Coverage Priority Score (CPS) logic for ParkSentinel AI.

This module is the single source of truth for all numbers used in the
dashboard, notebook, and proposal document. Import from here rather than
recomputing logic elsewhere, so the notebook and Streamlit app never drift
out of sync with each other.
"""

import re
import pandas as pd
import numpy as np

IST_OFFSET = pd.Timedelta(hours=5, minutes=30)

# ---------------------------------------------------------------------------
# Loading & cleaning
# ---------------------------------------------------------------------------

DEAD_COLUMNS = ["description", "closed_datetime", "action_taken_timestamp"]


def load_raw(csv_path: str) -> pd.DataFrame:
    """Load the raw BTP violation CSV."""
    df = pd.read_csv(csv_path, low_memory=False)
    return df


def parse_tag_list(cell):
    """Parse a stringified-list cell like '["WRONG PARKING","PARKING IN A MAIN ROAD"]'
    into a Python list of strings. Returns [] for null/empty cells."""
    if pd.isna(cell):
        return []
    return re.findall(r'"([^"]+)"', cell)


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Apply the full cleaning + feature-derivation pipeline used throughout
    this project. Returns a new dataframe; does not mutate the input."""
    df = df.copy()

    # Drop fully-null / unused columns if present
    for col in DEAD_COLUMNS:
        if col in df.columns:
            df = df.drop(columns=col)

    # Timestamps: stored in UTC, dataset is Bengaluru -> must convert to IST.
    # This conversion is the single most important step in this pipeline —
    # see README / proposal Section 3.2 Step 3 for why.
    df["created_datetime"] = pd.to_datetime(df["created_datetime"], errors="coerce", utc=True)
    df["created_ist"] = df["created_datetime"] + IST_OFFSET
    df["hour"] = df["created_ist"].dt.hour
    df["dow"] = df["created_ist"].dt.day_name()
    df["date"] = df["created_ist"].dt.date

    # Parse multi-valued violation_type into a list + a primary tag
    df["violation_list"] = df["violation_type"].apply(parse_tag_list)
    df["primary_violation"] = df["violation_list"].apply(lambda l: l[0] if l else None)
    df["n_violations_tagged"] = df["violation_list"].apply(len)
    df["is_main_road"] = df["violation_list"].apply(lambda l: "PARKING IN A MAIN ROAD" in l)

    return df


# ---------------------------------------------------------------------------
# Time-of-day analysis (Finding #1)
# ---------------------------------------------------------------------------

TIME_BUCKETS = [
    (0, 3, "12am-3am"), (3, 6, "3am-6am"), (6, 9, "6am-9am"), (9, 12, "9am-12pm"),
    (12, 15, "12pm-3pm"), (15, 18, "3pm-6pm"), (18, 21, "6pm-9pm"), (21, 24, "9pm-12am"),
]

GAP_START_HOUR = 15  # 3pm IST
GAP_END_HOUR = 24    # midnight IST


def hourly_distribution(df: pd.DataFrame) -> pd.DataFrame:
    """Return violation share by hour, 0-23, IST."""
    counts = df["hour"].value_counts().reindex(range(24), fill_value=0)
    share = (counts / len(df) * 100).round(2)
    return pd.DataFrame({"hour": range(24), "count": counts.values, "share_pct": share.values})


def bucketed_distribution(df: pd.DataFrame) -> pd.DataFrame:
    """Return violation share in 3-hour buckets, for charting."""
    total = len(df)
    rows = []
    for lo, hi, label in TIME_BUCKETS:
        c = ((df["hour"] >= lo) & (df["hour"] < hi)).sum()
        rows.append({"bucket": label, "count": int(c), "share_pct": round(100 * c / total, 2)})
    return pd.DataFrame(rows)


def coverage_gap_share(df: pd.DataFrame) -> float:
    """Fraction (0-1) of rows falling in the 3pm-midnight gap window."""
    mask = (df["hour"] >= GAP_START_HOUR) & (df["hour"] < GAP_END_HOUR)
    return mask.mean()


# ---------------------------------------------------------------------------
# Rejection-rate analysis (Finding #2)
# ---------------------------------------------------------------------------

def rejection_rate_by_type(df: pd.DataFrame, min_n: int = 200) -> pd.DataFrame:
    """Rejection rate (validation_status == 'rejected') by primary violation
    type, restricted to types with at least min_n reviewed records."""
    reviewed = df[df["validation_status"].notna()]
    rows = []
    for vtype, sub in reviewed.groupby("primary_violation"):
        if len(sub) < min_n:
            continue
        rejected = (sub["validation_status"] == "rejected").sum()
        rows.append({
            "violation_type": vtype,
            "n_reviewed": len(sub),
            "n_rejected": int(rejected),
            "rejection_rate_pct": round(100 * rejected / len(sub), 1),
        })
    out = pd.DataFrame(rows).sort_values("rejection_rate_pct", ascending=False).reset_index(drop=True)
    return out


def validation_status_breakdown(df: pd.DataFrame) -> pd.DataFrame:
    reviewed = df[df["validation_status"].notna()]
    vc = reviewed["validation_status"].value_counts()
    out = pd.DataFrame({
        "status": vc.index,
        "count": vc.values,
        "share_pct": (vc.values / len(reviewed) * 100).round(1),
    })
    return out


# ---------------------------------------------------------------------------
# Repeat offenders
# ---------------------------------------------------------------------------

REPEAT_THRESHOLD = 10


def repeat_offender_vehicles(df: pd.DataFrame, threshold: int = REPEAT_THRESHOLD) -> set:
    vc = df["vehicle_number"].value_counts()
    return set(vc[vc >= threshold].index)


def repeat_offender_summary(df: pd.DataFrame) -> dict:
    vc = df["vehicle_number"].value_counts()
    total = len(df)
    return {
        "n_vehicles_total": int(df["vehicle_number"].nunique()),
        "n_repeat_10plus": int((vc >= 10).sum()),
        "rows_from_repeat_10plus": int(vc[vc >= 10].sum()),
        "rows_from_repeat_10plus_pct": round(100 * vc[vc >= 10].sum() / total, 2),
        "n_repeat_2plus": int((vc >= 2).sum()),
        "rows_from_repeat_2plus_pct": round(100 * vc[vc >= 2].sum() / total, 2),
    }


# ---------------------------------------------------------------------------
# Junction recovery for unmapped ("No Junction") rows
# ---------------------------------------------------------------------------
# ~49.5% of rows have no junction tag at all, despite having usable lat/long.
# Most teams will silently drop these from any junction-level analysis. We
# recover them by assigning each unmapped row to its nearest named junction
# (by centroid), capped at a maximum distance so we never force a false
# match for a row that's genuinely far from any known junction.

EARTH_RADIUS_KM = 6371.0
MAX_RECOVERY_DISTANCE_KM = 0.5  # ~500m cap: don't force-match distant rows


def _haversine_km(lat1, lon1, lat2, lon2):
    """Vectorized haversine. Inputs should already be in radians and
    pre-shaped for broadcasting (converting to radians AFTER broadcasting
    is what makes this slow on large arrays, so callers should convert
    each 1-D array to radians first, then reshape for broadcasting)."""
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return 2 * EARTH_RADIUS_KM * np.arcsin(np.sqrt(a))


def junction_centroids(df: pd.DataFrame) -> pd.DataFrame:
    """Mean lat/long per named junction, used as the matching target for
    recovery. Excludes 'No Junction' itself."""
    named = df[df["junction_name"] != "No Junction"]
    cent = named.groupby("junction_name")[["latitude", "longitude"]].mean().reset_index()
    return cent


def recover_unmapped_junctions(
    df: pd.DataFrame,
    max_distance_km: float = MAX_RECOVERY_DISTANCE_KM,
) -> pd.DataFrame:
    """Return a copy of df with a new column `junction_name_recovered`:
    the original junction_name where present, else the nearest named
    junction's centroid if within max_distance_km, else 'Unrecovered'.

    This is nearest-centroid assignment, not a trained model — it is fast,
    fully deterministic, and auditable (every recovered row's distance is
    inspectable), which matters more than sophistication for an enforcement
    tool where a human will eventually act on the assignment.
    """
    df = df.copy()
    cent = junction_centroids(df)

    mask_unmapped = df["junction_name"] == "No Junction"
    unmapped = df[mask_unmapped]

    if len(unmapped) == 0 or len(cent) == 0:
        df["junction_name_recovered"] = df["junction_name"]
        df["recovery_distance_km"] = np.nan
        return df

    # Vectorized nearest-centroid search (168 centroids -> cheap even for 150k rows).
    # Convert to radians BEFORE broadcasting to (N x M) -- doing it after is
    # what made an earlier version of this function slow.
    lat_c = np.radians(cent["latitude"].values)[None, :]
    lon_c = np.radians(cent["longitude"].values)[None, :]
    lat_u = np.radians(unmapped["latitude"].values)[:, None]
    lon_u = np.radians(unmapped["longitude"].values)[:, None]

    dists = _haversine_km(lat_u, lon_u, lat_c, lon_c)
    nearest_idx = np.argmin(dists, axis=1)
    nearest_dist = dists[np.arange(len(unmapped)), nearest_idx]
    nearest_name = cent["junction_name"].values[nearest_idx]

    recovered_name = np.where(nearest_dist <= max_distance_km, nearest_name, "Unrecovered")

    df["junction_name_recovered"] = df["junction_name"]
    df.loc[mask_unmapped, "junction_name_recovered"] = recovered_name
    df["recovery_distance_km"] = np.nan
    df.loc[mask_unmapped, "recovery_distance_km"] = nearest_dist

    return df


def recovery_summary(df_recovered: pd.DataFrame) -> dict:
    """Stats on how much of the unmapped data was successfully recovered."""
    total = len(df_recovered)
    was_unmapped = (df_recovered["junction_name"] == "No Junction")
    n_unmapped = int(was_unmapped.sum())
    recovered_ok = was_unmapped & (df_recovered["junction_name_recovered"] != "Unrecovered")
    n_recovered = int(recovered_ok.sum())
    return {
        "total_rows": total,
        "n_originally_unmapped": n_unmapped,
        "pct_originally_unmapped": round(100 * n_unmapped / total, 2),
        "n_recovered": n_recovered,
        "pct_of_unmapped_recovered": round(100 * n_recovered / n_unmapped, 2) if n_unmapped else 0.0,
        "median_recovery_distance_km": round(
            float(df_recovered.loc[recovered_ok, "recovery_distance_km"].median()), 4
        ) if n_recovered else None,
    }




CPS_WEIGHTS = {
    "coverage_gap": 0.40,
    "main_road_share": 0.30,
    "repeat_share": 0.20,
    "density_norm": 0.10,
}

DEFAULT_MIN_VOLUME = 200


def compute_cps(
    df: pd.DataFrame,
    min_volume: int = DEFAULT_MIN_VOLUME,
    weights: dict = None,
    junction_col: str = "junction_name",
) -> pd.DataFrame:
    """Compute the Coverage Priority Score per junction.

    Parameters
    ----------
    df : cleaned dataframe (output of `clean()`)
    min_volume : junctions with fewer than this many total violations are
        excluded, because percentage-based components become noisy at low
        sample sizes (see proposal Section 5.1, "Methodological note").
    weights : optional override of CPS_WEIGHTS, must contain the same keys.
    junction_col : which column to group by. Pass "junction_name_recovered"
        (after calling recover_unmapped_junctions) to include rows that were
        originally unmapped but matched to a nearby named junction.

    Returns
    -------
    DataFrame sorted by CPS descending, one row per junction, with the raw
    component values alongside the final score so the ranking is auditable.
    """
    w = weights or CPS_WEIGHTS
    repeat_vehicles = repeat_offender_vehicles(df)

    exclude_values = {"No Junction", "Unrecovered"}
    named = df[~df[junction_col].isin(exclude_values)].copy()
    named["is_repeat"] = named["vehicle_number"].isin(repeat_vehicles)

    def gap_share(g):
        return ((g["hour"] >= GAP_START_HOUR) & (g["hour"] < GAP_END_HOUR)).mean()

    grouped = named.groupby(junction_col)
    agg = grouped.agg(
        n=("is_main_road", "count"),
        main_road_share=("is_main_road", "mean"),
        repeat_share=("is_repeat", "mean"),
    ).reset_index().rename(columns={junction_col: "junction_name"})
    agg["coverage_gap"] = grouped.apply(gap_share).values

    agg = agg[agg["n"] >= min_volume].copy()
    if agg.empty:
        return agg

    rng = agg["n"].max() - agg["n"].min()
    agg["density_norm"] = (agg["n"] - agg["n"].min()) / rng if rng > 0 else 0.0

    agg["CPS"] = (
        w["coverage_gap"] * agg["coverage_gap"]
        + w["main_road_share"] * agg["main_road_share"]
        + w["repeat_share"] * agg["repeat_share"]
        + w["density_norm"] * agg["density_norm"]
    ) * 100

    agg = agg.sort_values("CPS", ascending=False).reset_index(drop=True)
    agg.insert(0, "rank", range(1, len(agg) + 1))

    # round for display
    for c in ["main_road_share", "repeat_share", "coverage_gap", "density_norm"]:
        agg[c + "_pct"] = (agg[c] * 100).round(1)
    agg["CPS"] = agg["CPS"].round(2)

    # attach centroid lat/long for mapping
    cent = named.groupby(junction_col)[["latitude", "longitude"]].mean()
    cent.index.name = "junction_name"
    agg = agg.merge(cent, on="junction_name", how="left")

    return agg


# ---------------------------------------------------------------------------
# Unsupervised clustering -- a second, independent method
# ---------------------------------------------------------------------------
# CPS uses fixed, hand-chosen weights (0.40/0.30/0.20/0.10) on four junction-
# level features. That is a defensible but judgment-based formula, not a
# learned result. K-means clustering on the SAME four features, with no
# weights imposed, lets natural groupings emerge from the data on their own.
# This is real, if simple, unsupervised ML -- it needs no labels (which this
# dataset doesn't have for a supervised model anyway), and it gives us an
# independent check on whether CPS's priority ranking is something the data
# actually supports, or just an artifact of our chosen weights.

CLUSTER_FEATURES = ["coverage_gap", "main_road_share", "repeat_share", "density_norm"]


def cluster_junctions(cps_df: pd.DataFrame, n_clusters: int = 3, random_state: int = 42) -> pd.DataFrame:
    """Run k-means on the same four CPS component features (standardized),
    label clusters as High/Medium/Low priority by their mean CPS, and
    return cps_df with two new columns: `cluster` (raw label) and `tier`.

    This does NOT use the CPS score itself as a clustering input -- only
    the four underlying components -- so it's a genuinely independent
    cross-check, not circular reasoning dressed up as one.
    """
    from sklearn.cluster import KMeans
    from sklearn.preprocessing import StandardScaler

    if len(cps_df) < n_clusters:
        out = cps_df.copy()
        out["cluster"] = 0
        out["tier"] = "Insufficient data"
        return out

    out = cps_df.copy()
    X = out[CLUSTER_FEATURES].values
    X_scaled = StandardScaler().fit_transform(X)

    km = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
    labels = km.fit_predict(X_scaled)
    out["cluster"] = labels

    # Order clusters by mean CPS so the tier labels are meaningful, even
    # though CPS was not a clustering input -- this is just for labeling,
    # not for the clustering itself.
    cluster_order = out.groupby("cluster")["CPS"].mean().sort_values(ascending=False).index
    tier_labels = ["High priority", "Medium priority", "Low priority"] + \
                  [f"Tier {i+1}" for i in range(3, n_clusters)]
    rank_map = {cl: tier_labels[i] for i, cl in enumerate(cluster_order)}
    out["tier"] = out["cluster"].map(rank_map)

    return out


def cluster_centers_summary(cps_df: pd.DataFrame, n_clusters: int = 3, random_state: int = 42) -> pd.DataFrame:
    """Return cluster centers in original (interpretable) feature units,
    with the same tier labels used in cluster_junctions, so the dashboard
    can show *why* each tier is high/medium/low -- which feature dominates."""
    from sklearn.cluster import KMeans
    from sklearn.preprocessing import StandardScaler

    X = cps_df[CLUSTER_FEATURES].values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    km = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
    labels = km.fit_predict(X_scaled)

    tmp = cps_df.copy()
    tmp["cluster"] = labels
    cluster_order = tmp.groupby("cluster")["CPS"].mean().sort_values(ascending=False).index
    tier_labels = ["High priority", "Medium priority", "Low priority"] + \
                  [f"Tier {i+1}" for i in range(3, n_clusters)]
    rank_map = {cl: tier_labels[i] for i, cl in enumerate(cluster_order)}

    centers = scaler.inverse_transform(km.cluster_centers_)
    centers_df = pd.DataFrame(centers, columns=CLUSTER_FEATURES)
    centers_df["tier"] = [rank_map[i] for i in range(n_clusters)]
    centers_df["n_junctions"] = [int((labels == i).sum()) for i in range(n_clusters)]
    for c in CLUSTER_FEATURES:
        centers_df[c + "_pct"] = (centers_df[c] * 100).round(1)
    return centers_df.sort_values("tier")


def cps_vs_cluster_agreement(cps_df_with_tier: pd.DataFrame, top_n: int = 10) -> dict:
    """How much does the CPS top-N ranking agree with the clustering's
    'High priority' tier? Returns counts and the disagreeing junctions with
    a brief reason, so the dashboard can explain divergence rather than
    just report a number."""
    top_by_cps = set(cps_df_with_tier.nsmallest(top_n, "rank")["junction_name"])
    high_tier = set(cps_df_with_tier[cps_df_with_tier["tier"] == "High priority"]["junction_name"])
    overlap = top_by_cps & high_tier
    cps_only = top_by_cps - high_tier  # high CPS rank, but clustering disagrees

    disagreements = []
    for jn in cps_only:
        row = cps_df_with_tier[cps_df_with_tier["junction_name"] == jn].iloc[0]
        # diagnose: which feature is comparatively low vs the High-priority cluster mean
        disagreements.append({
            "junction_name": jn,
            "CPS": row["CPS"],
            "tier": row["tier"],
            "coverage_gap_pct": row["coverage_gap_pct"],
            "main_road_share_pct": row["main_road_share_pct"],
            "repeat_share_pct": row["repeat_share_pct"],
            "density_norm_pct": row["density_norm_pct"],
        })

    return {
        "top_n": top_n,
        "n_overlap": len(overlap),
        "pct_agreement": round(100 * len(overlap) / top_n, 1),
        "overlap_junctions": sorted(overlap),
        "disagreement_junctions": disagreements,
    }




def headline_stats(df: pd.DataFrame) -> dict:
    return {
        "total_records": len(df),
        "date_min": str(df["created_ist"].min().date()),
        "date_max": str(df["created_ist"].max().date()),
        "n_devices": int(df["device_id"].nunique()),
        "n_junctions": int(df["junction_name"].nunique()),
        "n_police_stations": int(df["police_station"].nunique()),
        "n_vehicles": int(df["vehicle_number"].nunique()),
        "n_violation_types": int(df["primary_violation"].nunique()),
        "gap_share_pct": round(coverage_gap_share(df) * 100, 2),
    }
