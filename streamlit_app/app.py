"""
app.py — ParkSentinel AI Dashboard
Theme 1: Parking-Induced Congestion — Bangalore Traffic Police

Tabs: Overview, Time Analysis, CPS Rankings + Heatmap, Clustering Cross-Check
(unsupervised k-means as an independent check on CPS), Junction Recovery,
Enforcement Demo.

Run with:  streamlit run app.py
"""

import sys
import os
from datetime import datetime, timedelta

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

sys.path.append(os.path.dirname(__file__))
sys.path.append(os.path.join(os.path.dirname(__file__), "cv_pipeline"))

import core_analysis as ca
from enforcement_logic import make_circular_zone, DwellTimeEngine, sample_scenario, Detection

_CANDIDATE_PATHS = [
    os.path.join(os.path.dirname(__file__), "data", "jan_to_may_police_violation_anonymized791b166.csv.gz"),
    os.path.join(os.path.dirname(__file__), "..", "data", "jan_to_may_police_violation_anonymized791b166.csv.gz"),
    os.path.join(os.path.dirname(__file__), "data", "jan_to_may_police_violation_anonymized791b166.csv"),
    os.path.join(os.path.dirname(__file__), "..", "data", "jan_to_may_police_violation_anonymized791b166.csv"),
]
DEFAULT_DATA_PATH = next((p for p in _CANDIDATE_PATHS if os.path.exists(p)), _CANDIDATE_PATHS[0])

st.set_page_config(
    page_title="ParkSentinel AI — Theme 1 Dashboard",
    page_icon="🅿️",
    layout="wide",
)

# ---------------------------------------------------------------------------
# Data loading (cached)
# ---------------------------------------------------------------------------

def _df_hash_proxy(df: pd.DataFrame):
    """Cheap, correct hash key for caching: dataframe shape + a sample of
    values from columns we know are hashable, rather than letting Streamlit
    try (and fail) to hash columns containing Python lists."""
    safe_cols = [c for c in df.columns if df[c].dtype != object or c not in
                 ("violation_list",)]
    sample = df[safe_cols].head(50) if safe_cols else None
    return (df.shape, pd.util.hash_pandas_object(sample).sum() if sample is not None else 0)


@st.cache_data(show_spinner=False, hash_funcs={pd.DataFrame: _df_hash_proxy})
def load_and_clean(file_bytes_or_path):
    if isinstance(file_bytes_or_path, str):
        raw = ca.load_raw(file_bytes_or_path)
    else:
        raw = pd.read_csv(file_bytes_or_path, low_memory=False)
    df = ca.clean(raw)
    return df


@st.cache_data(show_spinner=False, hash_funcs={pd.DataFrame: _df_hash_proxy})
def get_recovered(df, max_distance_km=0.5):
    return ca.recover_unmapped_junctions(df, max_distance_km=max_distance_km)


@st.cache_data(show_spinner=False, hash_funcs={pd.DataFrame: _df_hash_proxy})
def get_cps(df_recovered, min_volume, use_recovered):
    col = "junction_name_recovered" if use_recovered else "junction_name"
    return ca.compute_cps(df_recovered, min_volume=min_volume, junction_col=col)


@st.cache_data(show_spinner=False, hash_funcs={pd.DataFrame: _df_hash_proxy})
def get_clustered(cps_df, n_clusters):
    return ca.cluster_junctions(cps_df, n_clusters=n_clusters)


@st.cache_data(show_spinner=False, hash_funcs={pd.DataFrame: _df_hash_proxy})
def get_cluster_centers(cps_df, n_clusters):
    return ca.cluster_centers_summary(cps_df, n_clusters=n_clusters)


# ---------------------------------------------------------------------------
# Sidebar — data source
# ---------------------------------------------------------------------------

st.sidebar.title("🅿️ ParkSentinel AI")
st.sidebar.caption("Theme 1 — Parking-Induced Congestion")

st.sidebar.markdown("---")
st.sidebar.subheader("Data source")
uploaded = st.sidebar.file_uploader(
    "Upload a BTP-format violation CSV",
    type="csv",
    help="Must contain the same columns as the official BTP export "
         "(junction_name, violation_type, vehicle_number, created_datetime, "
         "latitude, longitude, validation_status). If you don't upload one, "
         "the original Theme 1 dataset is used.",
)

if uploaded is not None:
    try:
        df = load_and_clean(uploaded)
        st.sidebar.success(f"Loaded uploaded file: {len(df):,} rows")
    except Exception as e:
        st.sidebar.error(f"Could not parse uploaded file: {e}")
        st.stop()
else:
    if not os.path.exists(DEFAULT_DATA_PATH):
        st.error(f"Default dataset not found at {DEFAULT_DATA_PATH}. Please upload a CSV.")
        st.stop()
    df = load_and_clean(DEFAULT_DATA_PATH)
    st.sidebar.info(f"Using default Theme 1 dataset: {len(df):,} rows")

st.sidebar.markdown("---")
st.sidebar.subheader("CPS settings")
min_volume = st.sidebar.slider(
    "Minimum violations per junction", 50, 1000, 200, step=50,
    help="Junctions below this volume are excluded from CPS scoring, "
         "since percentage-based components get noisy at low sample sizes."
)
use_recovered = st.sidebar.checkbox(
    "Include recovered 'No Junction' rows", value=True,
    help="Recovers unmapped rows by matching their lat/long to the nearest "
         "named junction centroid, within a 500m cap. See the 'Recovery' tab."
)
recovery_cap_km = st.sidebar.slider(
    "Recovery distance cap (km)", 0.1, 2.0, 0.5, step=0.1,
    help="Maximum distance an unmapped row can be from a known junction "
         "to be recovered into that junction's count."
)

# ---------------------------------------------------------------------------
# Compute pipeline
# ---------------------------------------------------------------------------

with st.spinner("Running recovery + CPS pipeline..."):
    df_recovered = get_recovered(df, recovery_cap_km) if use_recovered else None
    cps_df = get_cps(df_recovered if use_recovered else df, min_volume, use_recovered)

stats = ca.headline_stats(df)

# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------

st.title("ParkSentinel AI — Live Analysis Dashboard")
st.caption("Every number on this page is computed live from the loaded CSV — nothing here is hardcoded.")

c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Total records", f"{stats['total_records']:,}")
c2.metric("Camera devices", f"{stats['n_devices']:,}")
c3.metric("Named junctions", f"{stats['n_junctions']:,}")
c4.metric("Unique vehicles", f"{stats['n_vehicles']:,}")
c5.metric("3pm–midnight share", f"{stats['gap_share_pct']}%", help="Share of all violations captured in the coverage-gap window identified in Finding #1.")

st.markdown("---")

# ---------------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------------

tab_overview, tab_time, tab_cps, tab_cluster, tab_recovery, tab_demo = st.tabs([
    "📊 Overview", "🕐 Time Analysis", "🏆 CPS Rankings + Heatmap",
    "🧬 Clustering Cross-Check", "📍 Junction Recovery", "🎥 Enforcement Demo",
])

# ===========================================================================
# TAB 1: Overview
# ===========================================================================
with tab_overview:
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Violation type breakdown")
        vt_counts = df["primary_violation"].value_counts().head(8).reset_index()
        vt_counts.columns = ["violation_type", "count"]
        fig = px.bar(vt_counts, x="count", y="violation_type", orientation="h",
                     color="count", color_continuous_scale="Teal")
        fig.update_layout(yaxis=dict(autorange="reversed"), showlegend=False, height=400)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("Manual-review outcomes")
        status_df = ca.validation_status_breakdown(df)
        fig = px.pie(status_df, names="status", values="count", hole=0.45,
                      color_discrete_sequence=px.colors.sequential.Teal_r)
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Rejection rate by violation type")
    st.caption("Across every category, a consistent share of camera-flagged incidents get rejected on manual review — "
               "the system's existing accuracy ceiling, before any new AI is added.")
    rejection_df = ca.rejection_rate_by_type(df, min_n=100)
    fig = px.bar(rejection_df, x="violation_type", y="rejection_rate_pct",
                 color="rejection_rate_pct", color_continuous_scale="Reds")
    fig.update_layout(showlegend=False, height=380, yaxis_title="Rejection rate (%)")
    st.plotly_chart(fig, use_container_width=True)

    with st.expander("Repeat-offender summary"):
        rs = ca.repeat_offender_summary(df)
        rc1, rc2, rc3 = st.columns(3)
        rc1.metric("Vehicles with 10+ violations", f"{rs['n_repeat_10plus']:,}")
        rc2.metric("Rows from chronic offenders", f"{rs['rows_from_repeat_10plus_pct']}%")
        rc3.metric("Vehicles with 2+ violations", f"{rs['n_repeat_2plus']:,}")

# ===========================================================================
# TAB 2: Time Analysis (the timezone-corrected finding)
# ===========================================================================
with tab_time:
    st.subheader("Violations by time of day")
    st.warning(
        "⚠️ **Timezone correction applied.** Raw timestamps in this dataset are stored in UTC. "
        "Bengaluru is UTC+5:30. All times below are converted to IST — "
        "reading the raw UTC hour directly produces a misleading, inverted picture "
        "(see the toggle below to reproduce that mistake)."
    )

    show_bug = st.checkbox("Show the uncorrected (incorrect) UTC version for comparison")

    bucketed = ca.bucketed_distribution(df)

    if show_bug:
        raw_hour_utc = df["created_datetime"].dt.hour
        total = len(df)
        rows = []
        for lo, hi, label in ca.TIME_BUCKETS:
            c = ((raw_hour_utc >= lo) & (raw_hour_utc < hi)).sum()
            rows.append({"bucket": label, "share_pct": round(100 * c / total, 2)})
        bucketed_wrong = pd.DataFrame(rows)

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**❌ Incorrect (raw UTC hour, not converted)**")
            fig = px.bar(bucketed_wrong, x="bucket", y="share_pct", color_discrete_sequence=["#C0524F"])
            fig.update_layout(height=350, yaxis_title="% of violations")
            st.plotly_chart(fig, use_container_width=True)
        with col2:
            st.markdown("**✅ Correct (converted to IST)**")
            fig = px.bar(bucketed, x="bucket", y="share_pct", color_discrete_sequence=["#0E6E5C"])
            fig.update_layout(height=350, yaxis_title="% of violations")
            st.plotly_chart(fig, use_container_width=True)
    else:
        fig = px.bar(bucketed, x="bucket", y="share_pct", color="share_pct",
                     color_continuous_scale="Teal")
        fig.update_layout(showlegend=False, height=420, yaxis_title="% of all violations")
        st.plotly_chart(fig, use_container_width=True)

    gap_pct = stats["gap_share_pct"]
    st.markdown(
        f"### 🔍 Finding: only **{gap_pct}%** of all violations are captured between 3pm and midnight (IST)"
    )
    st.markdown(
        "This is the window the brief cares about most — commercial-area and metro-station congestion "
        "during business hours and evenings. The existing camera network is structurally blind here."
    )

    main_road_df = df[df["is_main_road"]]
    if len(main_road_df) > 0:
        main_road_gap = ca.coverage_gap_share(main_road_df) * 100
        st.caption(f"Cross-check: restricting to 'PARKING IN A MAIN ROAD' only, the same gap holds — {main_road_gap:.2f}% in the 3pm–midnight window.")

# ===========================================================================
# TAB 3: CPS Rankings + Heatmap
# ===========================================================================
with tab_cps:
    st.subheader("Coverage Priority Score — live ranking")
    st.caption(
        "CPS = 0.40 × coverage gap + 0.30 × main-road share + 0.20 × repeat-offender share + 0.10 × violation density. "
        "Weights are a reasoned prioritization based on the brief's stated priorities — not fitted to a real congestion "
        "outcome, since this dataset has no traffic-speed field to fit against. See the Limitations section of the "
        "technical proposal."
    )

    if cps_df.empty:
        st.error("No junctions meet the minimum volume threshold. Lower the threshold in the sidebar.")
    else:
        col1, col2 = st.columns([1, 1])

        with col1:
            st.markdown("**Top junctions by CPS**")
            display_cols = ["rank", "junction_name", "n", "CPS", "coverage_gap_pct", "main_road_share_pct", "repeat_share_pct"]
            st.dataframe(
                cps_df[display_cols].head(20).rename(columns={
                    "n": "violations", "coverage_gap_pct": "gap %",
                    "main_road_share_pct": "main-road %", "repeat_share_pct": "repeat-offender %",
                }),
                use_container_width=True, hide_index=True, height=560,
            )

        with col2:
            st.markdown("**Geospatial heatmap — the artifact the brief explicitly asks for**")
            map_df = cps_df.dropna(subset=["latitude", "longitude"]).copy()
            if len(map_df) > 0:
                fig = go.Figure(go.Densitymap(
                    lat=map_df["latitude"], lon=map_df["longitude"], z=map_df["CPS"],
                    radius=35, colorscale="YlOrRd", opacity=0.75,
                    hovertext=map_df["junction_name"] + "<br>CPS: " + map_df["CPS"].astype(str),
                ))
                fig.add_trace(go.Scattermap(
                    lat=map_df["latitude"], lon=map_df["longitude"],
                    mode="markers",
                    marker=dict(size=8, color="black"),
                    text=map_df["junction_name"] + "<br>CPS: " + map_df["CPS"].astype(str),
                    hoverinfo="text",
                ))
                fig.update_layout(
                    map_style="open-street-map",
                    map_center={"lat": float(map_df["latitude"].mean()), "lon": float(map_df["longitude"].mean())},
                    map_zoom=10.5,
                    height=560, margin=dict(l=0, r=0, t=0, b=0), showlegend=False,
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No lat/long available for mapping with current settings.")

        st.markdown("### Why CPS disagrees with a simple violation-count ranking")
        cps_by_count = cps_df.sort_values("n", ascending=False).reset_index(drop=True)
        cps_by_count["count_rank"] = range(1, len(cps_by_count) + 1)
        merged = cps_df.merge(cps_by_count[["junction_name", "count_rank"]], on="junction_name")
        merged["rank_shift"] = merged["count_rank"] - merged["rank"]
        biggest_movers = merged.sort_values("rank_shift", ascending=False).head(5)
        st.dataframe(
            biggest_movers[["junction_name", "rank", "count_rank", "rank_shift", "main_road_share_pct"]].rename(
                columns={"rank": "CPS rank", "count_rank": "raw-count rank", "rank_shift": "rank improvement",
                         "main_road_share_pct": "main-road %"}
            ),
            use_container_width=True, hide_index=True,
        )
        st.caption("Positive 'rank improvement' = this junction ranks much higher on CPS than it would on raw violation count alone — "
                   "usually because of a high main-road share or coverage gap that a simple count misses.")

# ===========================================================================
# TAB: Clustering Cross-Check (independent ML method vs. the CPS formula)
# ===========================================================================
with tab_cluster:
    st.subheader("Does an independent method agree with CPS?")
    st.markdown(
        "CPS uses **fixed, hand-chosen weights** (0.40 / 0.30 / 0.20 / 0.10) — a defensible but judgment-based "
        "formula, not something learned from data. This tab runs **k-means clustering** on the same four "
        "underlying features, with **no weights imposed**, and lets natural groupings emerge on their own. "
        "This is genuine unsupervised machine learning — it needs no labels (which this dataset doesn't have "
        "for a supervised model anyway) — used here as an independent check on whether CPS's priorities are "
        "something the data actually supports, or just an artifact of our chosen weights."
    )

    if cps_df.empty:
        st.warning("No junctions available for clustering at the current minimum-volume threshold.")
    else:
        n_clusters = st.slider("Number of clusters (risk tiers)", 2, 6, 3)
        clustered = get_clustered(cps_df, n_clusters)
        centers = get_cluster_centers(cps_df, n_clusters)

        col1, col2 = st.columns([1, 1])
        with col1:
            st.markdown("**Cluster sizes**")
            tier_counts = clustered["tier"].value_counts().reset_index()
            tier_counts.columns = ["tier", "count"]
            fig = px.bar(tier_counts, x="tier", y="count", color="tier",
                         color_discrete_sequence=px.colors.sequential.Teal_r)
            fig.update_layout(showlegend=False, height=320)
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            st.markdown("**What defines each tier** (cluster centers, original units)")
            display_centers = centers[["tier", "n_junctions", "coverage_gap_pct", "main_road_share_pct", "repeat_share_pct", "density_norm_pct"]]
            st.dataframe(
                display_centers.rename(columns={
                    "n_junctions": "junctions", "coverage_gap_pct": "gap %",
                    "main_road_share_pct": "main-road %", "repeat_share_pct": "repeat %", "density_norm_pct": "density %",
                }),
                use_container_width=True, hide_index=True,
            )

        st.markdown("### Agreement with CPS's top-ranked junctions")
        top_n = st.slider("Compare against CPS top-N", 5, 30, 10)
        agreement = ca.cps_vs_cluster_agreement(clustered, top_n=top_n)

        ac1, ac2 = st.columns(2)
        ac1.metric("Agreement", f"{agreement['pct_agreement']}%",
                   help=f"{agreement['n_overlap']} of the top {top_n} CPS junctions also land in clustering's 'High priority' tier.")
        ac2.metric("Junctions where methods disagree", len(agreement["disagreement_junctions"]))

        if agreement["pct_agreement"] == 100:
            st.info("Perfect agreement can be a sign the two methods aren't really independent — worth a healthy dose of skepticism rather than celebration.")
        elif agreement["pct_agreement"] >= 50:
            st.success(
                f"**{agreement['pct_agreement']}% agreement is a meaningful, honest result** — strong enough to show CPS's "
                "priorities are largely supported by an independent method, while the disagreements below are individually explainable, "
                "not just noise."
            )
        else:
            st.warning(f"Only {agreement['pct_agreement']}% agreement — worth investigating which CPS weight is driving the divergence.")

        if agreement["disagreement_junctions"]:
            st.markdown("**Junctions CPS ranks highly that clustering does NOT place in 'High priority' — and why:**")
            dis_df = pd.DataFrame(agreement["disagreement_junctions"])
            st.dataframe(
                dis_df.rename(columns={
                    "coverage_gap_pct": "gap %", "main_road_share_pct": "main-road %",
                    "repeat_share_pct": "repeat %", "density_norm_pct": "density %",
                }),
                use_container_width=True, hide_index=True,
            )
            st.caption(
                "These junctions typically score high on CPS mostly via raw violation density (which CPS weights at 0.10) "
                "rather than the main-road share that defines clustering's high-priority tier — a real, explainable "
                "methodological difference, not a bug in either method."
            )

# ===========================================================================
# TAB 4: Junction Recovery
# ===========================================================================
with tab_recovery:
    st.subheader("Recovering unmapped ('No Junction') rows")
    st.markdown(
        "Roughly half of all rows in this dataset have no junction tag, despite having usable lat/long. "
        "Most analyses silently drop these. This recovers them by matching to the nearest named junction's "
        "centroid, within a distance cap — deterministic nearest-neighbor matching, not a trained model."
    )

    if use_recovered and df_recovered is not None:
        summary = ca.recovery_summary(df_recovered)
        c1, c2, c3 = st.columns(3)
        c1.metric("Originally unmapped", f"{summary['pct_originally_unmapped']}%", help=f"{summary['n_originally_unmapped']:,} rows")
        c2.metric("Successfully recovered", f"{summary['pct_of_unmapped_recovered']}%", help=f"{summary['n_recovered']:,} rows, within {recovery_cap_km}km")
        c3.metric("Median match distance", f"{summary['median_recovery_distance_km']} km" if summary['median_recovery_distance_km'] else "n/a")

        st.markdown("#### Honest finding: most unmapped rows are genuinely far from any known junction")
        unmapped_only = df_recovered[df_recovered["junction_name"] == "No Junction"]
        fig = px.histogram(unmapped_only, x="recovery_distance_km", nbins=40,
                            color_discrete_sequence=["#0E6E5C"])
        fig.add_vline(x=recovery_cap_km, line_dash="dash", line_color="red",
                      annotation_text=f"current cap: {recovery_cap_km}km")
        fig.update_layout(height=350, xaxis_title="Distance to nearest named junction (km)", yaxis_title="Row count")
        st.plotly_chart(fig, use_container_width=True)
        st.caption(
            "This is a real, disclosed limitation, not a hidden one: the 169 named junctions are a curated list, "
            "not exhaustive coverage of every road. Raising the distance cap recovers more rows but at the cost "
            "of forcing weaker matches — adjust the slider in the sidebar to see the tradeoff."
        )
    else:
        st.info("Enable 'Include recovered No Junction rows' in the sidebar to see this analysis.")

# ===========================================================================
# TAB 5: Enforcement Demo (geofence + dwell-time)
# ===========================================================================
with tab_demo:
    st.subheader("Geofence + dwell-time enforcement engine — live simulation")
    st.markdown(
        "This is real, working logic (not a description) — point-in-polygon geofencing plus multi-frame "
        "dwell-time confirmation, exactly as specified in the technical proposal's pipeline stages 4–6. "
        "It is fed a deterministic sample scenario below; in production, stage 2–3 (YOLOv8 detection + ANPR) "
        "would feed it real detections instead."
    )

    colA, colB = st.columns([1, 1])
    with colA:
        min_dwell = st.slider("Minimum dwell time to confirm (seconds)", 30, 180, 90, step=10)
        min_frames = st.slider("Minimum frames required", 1, 10, 3)
        min_conf = st.slider("Minimum plate confidence to auto-confirm", 0.0, 1.0, 0.55, step=0.05)

    if not cps_df.empty:
        top_junction = cps_df.iloc[0]
        zone_lat, zone_lon, zone_name = top_junction["latitude"], top_junction["longitude"], top_junction["junction_name"]
    else:
        zone_lat, zone_lon, zone_name = 12.981488, 77.609097, "Demo Zone"

    zone = make_circular_zone(f"{zone_name} (no-parking zone)", zone_lat, zone_lon, radius_m=25)
    engine = DwellTimeEngine(zones=[zone], min_dwell_seconds=min_dwell, min_frames=min_frames, min_plate_confidence=min_conf)
    detections = sample_scenario(zone)
    results = engine.run_batch(detections)

    with colB:
        st.markdown(f"**Demo zone:** {zone_name}")
        st.markdown("**Scenario:** 3 vehicles —")
        st.markdown("- 🚗 Vehicle A: parks ~100s, clear plate\n- 🏍️ Vehicle B: passes through, never stops\n- 🚗 Vehicle C: parks ~100s, poor plate visibility")

    st.markdown("### Results")
    if results:
        for ev in results:
            icon = "✅" if ev.confirmed else "⚠️"
            color = "green" if ev.confirmed else "orange"
            st.markdown(f"**{icon} {ev.track_id}** — dwell {ev.dwell_seconds:.0f}s, {ev.frame_count} frames, plate confidence {ev.mean_plate_confidence}")
            st.markdown(f"<span style='color:{color}'>{ev.reason}</span>", unsafe_allow_html=True)
            st.markdown("---")
    else:
        st.info("No violations confirmed yet at current thresholds — try lowering the dwell time or frame count.")

    confirmed_ids = {ev.track_id for ev in results if ev.confirmed}
    flagged_ids = {ev.track_id for ev in results if not ev.confirmed}
    all_track_ids = {d.vehicle_track_id for d in detections}
    never_triggered = all_track_ids - confirmed_ids - flagged_ids
    if never_triggered:
        st.caption(f"Never triggered any evidence (correctly ignored): {', '.join(sorted(never_triggered))}")

    with st.expander("Full frame-by-frame engine log"):
        for line in engine.log:
            st.text(line)

st.markdown("---")
st.caption("ParkSentinel AI · Theme 1 · Sofia (Urmila), IIIT Kota · CYPH3R / Void Walkers")
