import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import io
import traceback

# Import your core engine functions here
from consolidated_tcx_to_motion_map_30 import (
    parse_tcx_to_rows, prepare_run_df, add_deltas, add_smoothed_speed,
    summarize_motion_segments, prepare_for_csv, infer_activity_timezone_name,
    compute_performance_stats, collapse_run_streams_for_map, build_lookup,
    enrich_segments, build_segments_payload, compute_metric_stats,
    compute_run_stats, write_html
)

# --- CONFIGURATION ---
st.set_page_config(page_title="Motion Map", layout="wide")

# --- UI HEADER ---
st.title("👟📍📈 Motion Map Analyzer")

st.markdown("""
An interactive, web-based running data analytics application that parses standard **.tcx** fitness files, 
automatically smooths raw GPS/sensor noise, applies movement state classifications (running, walking, stopped), 
and renders a rich performance overlay directly onto a web map.
""")

with st.expander("✨ Application Features", expanded=False):
    st.markdown("""
    * **Advanced Motion Segmentation:** Vectorized classification algorithms that divide your activity into highly precise Running, Walking, and Stopped segments.
    * **Interactive Metric Overlays:** High-fidelity overlays for Pace, Heart Rate, and Cadence distributed seamlessly across map geometry.
    * **Press-and-Hold Highlighting:** Smooth JavaScript-driven interactions enabling users to hold performance blocks (like Per-Km Splits or HR zones) to isolate exact route sections on the map.
    * **Privacy Controls:** Optional on-the-fly privacy zones that trim the first and last 500 meters of your activity to mask sensitive start/end addresses.
    * **Privacy-First Processing:** Secure architecture executing completely in RAM; user data is parsed in-memory and immediately destroyed post-session.
    """)

st.divider()

# --- SIDEBAR CONTROLS ---
with st.sidebar:
    st.header("Settings")
    # MVP Feature: Privacy Zones
    apply_privacy = st.checkbox(
        "Enable Privacy Zone", 
        value=True, 
        help="Hides the first and last 500 meters of your route to protect home/start locations."
    )
    
    st.divider()
    
    # MVP Feature: Demo Mode
    st.subheader("Don't have a file?")
    load_demo = st.button("Load Demo Run")

# --- FILE UPLOADER ---
# Streamlit holds this file entirely in RAM
uploaded_file = st.file_uploader("Upload a .tcx file", type=["tcx"])

# --- PROCESSING PIPELINE ---
file_to_process = None

if load_demo:
    # Ensure you have a file named 'demo.tcx' in your folder for this to work
    try:
        file_to_process = open("demo.tcx", "rb")
    except FileNotFoundError:
        st.sidebar.error("Demo file not found on server.")

elif uploaded_file is not None:
    file_to_process = uploaded_file

if file_to_process:
    # MVP Feature: Loading State
    with st.spinner("Analyzing your activity... This usually takes 5-10 seconds."):
        try:
            # 1. Parse in-memory TCX directly to a Pandas DataFrame
            rows = list(parse_tcx_to_rows(file_to_process))
            raw_run_df = pd.DataFrame(rows)
            
            if raw_run_df.empty:
                raise ValueError("The uploaded file contains no valid tracking data.")

            # 2. Prepare Data
            run_df = prepare_run_df(raw_run_df)
            
            # --- ERROR HANDLING: Graceful Failures ---
            if run_df["latitude"].isna().all() or run_df["longitude"].isna().all():
                raise ValueError("No GPS data found. Is this an indoor treadmill run?")
            
            if "cadence" not in run_df.columns or run_df["cadence"].isna().all():
                st.warning("No cadence data detected. Motion segmentation might be less accurate (Is this a cycling file?)")

            run_df = add_deltas(run_df)
            run_df = add_smoothed_speed(run_df)

            # --- PRIVACY ZONES (In-Memory Filtering) ---
            if apply_privacy and "distance_m" in run_df.columns:
                max_dist = run_df["distance_m"].max()
                if max_dist > 1500: # Only apply if the run is longer than 1.5km
                    privacy_mask = (run_df["distance_m"] >= 500) & (run_df["distance_m"] <= (max_dist - 500))
                    run_df = run_df.loc[privacy_mask].copy()
                    # Re-calculate deltas so the map doesn't draw a weird line connecting the gap
                    run_df = add_deltas(run_df)

            # 3. Analytics Pipeline
            temp_plot_df = run_df.dropna(subset=["latitude", "longitude"])
            tz_name = infer_activity_timezone_name(temp_plot_df)

            motion_segments_df = summarize_motion_segments(run_df)
            motion_segments_csv = prepare_for_csv(
                motion_segments_df, time_cols=["start_time", "end_time"], 
                round_decimals=2, tz_name=tz_name
            )

            perfstats = compute_performance_stats(run_df, tz_name=tz_name)
            run_df_collapsed = collapse_run_streams_for_map(run_df, tz_name=tz_name)
            lookup = build_lookup(run_df_collapsed)

            seg_df_enriched = enrich_segments(motion_segments_csv, lookup)
            segments, plot_df = build_segments_payload(run_df_collapsed, seg_df_enriched)
            
            if not segments:
                raise ValueError("No segments could be generated. Route may be too short after privacy zones are applied.")

            metricstats = compute_metric_stats(seg_df_enriched)
            runstats = compute_run_stats(run_df, seg_df_enriched, tz_name)

            # 4. Generate HTML String (In-Memory)
            # Notice we pass None for outpath because we altered write_html to return a string
            html_content = write_html(
                None, segments, plot_df, metricstats, runstats, perfstats
            )

            # --- MVP UI: Render the HTML ---
            # We use Streamlit Components to render the raw HTML directly on the page
            st.success("Analysis Complete!")
            components.html(html_content, height=850, scrolling=True)

        except ValueError as ve:
            # Catch our custom Graceful Failures
            st.error(f"⚠️ **Could not process activity:** {str(ve)}")
        except Exception as e:
            # Catch unexpected Python tracebacks cleanly
            st.error("🚨 **An unexpected error occurred while processing the file.**")
            with st.expander("Show technical details"):
                st.code(traceback.format_exc())