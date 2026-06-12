#!/usr/bin/env python3
import argparse
import csv
import json
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
import pandas as pd

from functools import lru_cache
from geopy.geocoders import Nominatim

from zoneinfo import ZoneInfo
from timezonefinder import TimezoneFinder

import time  # <-- ADD THIS
from contextlib import contextmanager  # <-- ADD THIS

NS = {
    "tcx": "http://www.garmin.com/xmlschemas/TrainingCenterDatabase/v2",
    "ax": "http://www.garmin.com/xmlschemas/ActivityExtension/v2",
}

TIME_FMT = "%Y-%m-%d %H:%M:%S"
MOVING_SPEED_THRESH = 0.8
STOP_SPEED_THRESH = 0.8
WALK_CADENCE_MAX = 140
WALK_SPEED_MAX = 2
DISPLAY_DECIMALS = 2
SMOOTHWINDOW = 5.0
MIN_SEGMENT_TIME_S = 5.0     # seconds: below this is considered tiny
MIN_SEGMENT_DIST_M = 5.0     # meters: below this is considered tiny
CADENCE_MULTIPLE = 2
ENRICH_SEGMENTS_TOLERANCE= "30s"#"15s"
TF = TimezoneFinder()
DEFAULT_TIMEZONE = "UTC"

#python consolidated_tcx_to_motion_map.6.1.py Morning_Run.20260518.0623.tcx Morning_Run.20260518.0623.csv
def parse_args():
    p = argparse.ArgumentParser(
        description="Convert Strava TCX to CSV, derive run/walk/stop segments using getrunstats.15.py logic, and generate motion map HTML."
    )
    p.add_argument("tcx_file", help="Input Strava/Garmin TCX file")
    p.add_argument(
        "-o",
        "--map-out",
        help="Output HTML motion map path (default: <prefix>_motion_map.html)",
    )
    p.add_argument(
        "--prefix",
        help="Optional prefix for intermediate files (default: tcx stem in same folder)",
    )
    p.add_argument(
        "--benchmark",
        action="store_true",
        help="Print execution time for pipeline stages and element counts.",
    )
    return p.parse_args()


def get_child_text(elem, path, default=None):
    if elem is None:
        return default
    child = elem.find(path, NS)
    return child.text if child is not None and child.text is not None else default


def parse_tcx_to_rows(tcx_path: Path):
    tree = ET.parse(tcx_path)
    root = tree.getroot()
    activities = root.find("tcx:Activities", NS)
    if activities is None:
        return
    for activity in activities.findall("tcx:Activity", NS):
        sport = activity.get("Sport")
        activity_id_elem = activity.find("tcx:Id", NS)
        activity_id = activity_id_elem.text if activity_id_elem is not None else None
        for lap in activity.findall("tcx:Lap", NS):
            lap_start_time = lap.get("StartTime")
            lap_total_time = get_child_text(lap, "tcx:TotalTimeSeconds")
            lap_distance = get_child_text(lap, "tcx:DistanceMeters")
            track = lap.find("tcx:Track", NS)
            if track is None:
                continue
            for tp in track.findall("tcx:Trackpoint", NS):
                time = get_child_text(tp, "tcx:Time")
                lat = get_child_text(tp, "tcx:Position/tcx:LatitudeDegrees")
                lon = get_child_text(tp, "tcx:Position/tcx:LongitudeDegrees")
                altitude = get_child_text(tp, "tcx:AltitudeMeters")
                distance = get_child_text(tp, "tcx:DistanceMeters")
                hr = get_child_text(tp, "tcx:HeartRateBpm/tcx:Value")
                cadence = get_child_text(tp, "tcx:Cadence")
                tpx = tp.find("tcx:Extensions/ax:TPX", NS)
                if tpx is None:
                    tpx = tp.find(".//{http://www.garmin.com/xmlschemas/ActivityExtension/v2}TPX")
                speed = None
                run_cadence = None
                watts = None
                if tpx is not None:
                    speed = get_child_text(tpx, "ax:Speed")
                    run_cadence = get_child_text(tpx, "ax:RunCadence")
                    watts = get_child_text(tpx, "ax:Watts")
                    if cadence is None and run_cadence is not None:
                        cadence = run_cadence
                yield {
                    "activity_id": activity_id,
                    "sport": sport,
                    "lap_start_time": lap_start_time,
                    "lap_total_time_s": lap_total_time,
                    "lap_distance_m": lap_distance,
                    "time": time,
                    "latitude": lat,
                    "longitude": lon,
                    "altitude_m": altitude,
                    "distance_m": distance,
                    "heart_rate_bpm": hr,
                    "cadence": cadence,
                    "speed_m_s": speed,
                    "run_cadence": run_cadence,
                    "watts": watts,
                }


def write_csv(rows, out_path: Path):
    rows = list(rows)
    if not rows:
        raise ValueError("No trackpoints found in TCX")
    fieldnames = [
        "activity_id", "sport", "lap_start_time", "lap_total_time_s", "lap_distance_m",
        "time", "latitude", "longitude", "altitude_m", "distance_m", "heart_rate_bpm",
        "cadence", "speed_m_s", "run_cadence", "watts"
    ]
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

def format_local_time(ts, tz_name=DEFAULT_TIMEZONE):
    if ts is None or pd.isna(ts):
        return "na"
    ts = pd.to_datetime(ts, utc=True, errors="coerce")
    if pd.isna(ts):
        return "na"
    try:
        return ts.tz_convert(ZoneInfo(tz_name)).strftime(TIME_FMT)
    except Exception:
        return ts.tz_convert(ZoneInfo(DEFAULT_TIMEZONE)).strftime(TIME_FMT)

def prepare_for_csv(df: pd.DataFrame, time_cols=None, round_decimals=DISPLAY_DECIMALS, tz_name=DEFAULT_TIMEZONE):
    if df.empty:
        return df
    out = df.copy()
    if time_cols:
        for col in time_cols:
            if col in out.columns:
                out[col] = out[col].apply(lambda x: format_local_time(x, tz_name=tz_name))
    for col in out.select_dtypes(include=["float", "float32", "float64"]).columns:
        out[col] = out[col].round(round_decimals)
    return out

def collapse_streams(df: pd.DataFrame) -> pd.DataFrame:
    """
    Collapse multi-stream TCX CSV (GPS, HR, speed rows) into one row per timestamp.
    Uses vectorized groupby aggregation for massive performance gains over manual loops.
    """
    if df.empty:
        return df

    df = df.copy()
    df["time"] = pd.to_datetime(df["time"], errors="coerce")
    df = df.dropna(subset=["time"]).sort_values("time")

    agg_rules = {}
    
    # Coordinates: take the first valid (non-null) entry per second
    for col in ["latitude", "longitude", "altitude_m"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
            agg_rules[col] = "first"
            
    # Metrics: take the mean of all entries in that second
    for col in ["heart_rate_bpm", "cadence", "speed_m_s", "run_cadence"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
            agg_rules[col] = "mean"

    if not agg_rules:
        return df.drop_duplicates(subset=["time"]).reset_index(drop=True)

    # Perform the aggregation in C-space
    collapsed = df.groupby("time", as_index=False).agg(agg_rules)
    
    return collapsed.sort_values("time").reset_index(drop=True)


def rebuild_distance_from_coords(df: pd.DataFrame) -> pd.DataFrame:
    """
    Always rebuild distance_m from latitude/longitude when they are present.
    Falls back to existing distance_m if coords are missing or unusable.
    """
    df = df.copy()
    if {"latitude", "longitude"} <= set(df.columns):
        lat_deg = pd.to_numeric(df["latitude"], errors="coerce")
        lon_deg = pd.to_numeric(df["longitude"], errors="coerce")
        if lat_deg.notna().sum() > 1 and lon_deg.notna().sum() > 1:
            lat = np.radians(lat_deg)
            lon = np.radians(lon_deg)
            dlat = lat.diff()
            dlon = lon.diff()
            R = 6371000.0
            a = np.sin(dlat / 2) ** 2 + np.cos(lat).shift(1) * np.cos(lat) * np.sin(dlon / 2) ** 2
            c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
            dist_delta = R * c
            dist_delta.iloc[0] = 0.0
            df["distance_m"] = dist_delta.cumsum().ffill()
            return df

    # Fallback: keep any existing distance_m (or later speed-based fallback)
    if "distance_m" not in df.columns:
        df["distance_m"] = np.nan
    return df


def prepare_run_df(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "time" not in df.columns:
        raise ValueError("'time' column missing")

    df = collapse_streams(df)

    if "cadence" not in df.columns and "run_cadence" in df.columns:
        df["cadence"] = df["run_cadence"]
    elif "cadence" in df.columns and "run_cadence" in df.columns:
        df["cadence"] = df["cadence"].fillna(df["run_cadence"])

    if "cadence" in df.columns:
        df["cadence"] = pd.to_numeric(df["cadence"], errors="coerce")

        run_cadence_missing = (
            "run_cadence" in df.columns and df["run_cadence"].notna().sum() == 0
        )
        cadence_median = df["cadence"].median(skipna=True)

        if run_cadence_missing and pd.notna(cadence_median) and 60 <= cadence_median < 110:
            df["cadence"] = df["cadence"] * CADENCE_MULTIPLE

    df = rebuild_distance_from_coords(df)

    if df["distance_m"].isna().all() and "speed_m_s" in df.columns:
        time_delta = df["time"].diff().dt.total_seconds().fillna(0.0)
        dist_delta = pd.to_numeric(df["speed_m_s"], errors="coerce").fillna(0.0) * time_delta
        df["distance_m"] = dist_delta.cumsum()

    return df

def add_deltas(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Time delta
    df["time_delta_s"] = df["time"].diff().dt.total_seconds().fillna(0.0)

    # Distance delta
    if "distance_m" in df.columns:
        df["distance_delta_m"] = df["distance_m"].diff().fillna(0.0)
    else:
        df["distance_delta_m"] = np.nan

    # Fill missing speed from distance if needed
    if "speed_m_s" not in df.columns or df["speed_m_s"].isna().all():
        with np.errstate(divide="ignore", invalid="ignore"):
            df["speed_m_s"] = df["distance_delta_m"] / df["time_delta_s"].replace(0, np.nan)
    return df


def add_smoothed_speed(df: pd.DataFrame, window_s: float = 5.0) -> pd.DataFrame:
    df = df.copy()
    if "distance_m" not in df.columns:
        return df

    df_ts = df[["time", "distance_m"]].dropna(subset=["time", "distance_m"]).copy()
    df_ts["time"] = pd.to_datetime(df_ts["time"], errors="coerce")
    df_ts["distance_m"] = pd.to_numeric(df_ts["distance_m"], errors="coerce")
    df_ts = df_ts.dropna(subset=["time", "distance_m"]).set_index("time").sort_index()

    if df_ts.empty:
        if "speed_m_s" in df.columns:
            df["speed_smooth_m_s"] = pd.to_numeric(df["speed_m_s"], errors="coerce").fillna(0.0)
        else:
            df["speed_smooth_m_s"] = 0.0
        return df

    dist_roll = df_ts["distance_m"].rolling(f"{int(window_s)}s", min_periods=2).apply(
        lambda x: x.iloc[-1] - x.iloc[0],
        raw=False,
    )

    df["time"] = pd.to_datetime(df["time"], errors="coerce")
    df = df.merge(dist_roll.rename("dist_rolling"), left_on="time", right_index=True, how="left")

    if "dist_rolling" not in df.columns:
        df["dist_rolling"] = np.nan

    df["speed_smooth_m_s"] = df["dist_rolling"] / window_s

    if "speed_m_s" in df.columns:
        df["speed_smooth_m_s"] = df["speed_smooth_m_s"].fillna(
            pd.to_numeric(df["speed_m_s"], errors="coerce")
        )
    else:
        df["speed_smooth_m_s"] = df["speed_smooth_m_s"].fillna(0.0)

    return df

class SegmentStatsCalculator:
    """Pre-computes numpy arrays and cumulative sums to make segment slice math O(1)."""
    def __init__(self, work: pd.DataFrame):
        self.time = work["time"]
        
        # Cumulative sums for O(1) interval math
        td = pd.to_numeric(work.get("time_delta_s", pd.Series(np.zeros(len(work)))), errors="coerce").fillna(0.0).values
        dd = pd.to_numeric(work.get("distance_delta_m", pd.Series(np.zeros(len(work)))), errors="coerce").fillna(0.0).values
        self.cum_time = np.cumsum(td)
        self.cum_dist = np.cumsum(dd)
        
        def get_vals(col):
            if col in work.columns:
                return pd.to_numeric(work[col], errors="coerce").values
            return np.full(len(work), np.nan)
        
        self.hr = get_vals("heart_rate_bpm")
        self.cad = get_vals("cadence")
        self.speed_raw = get_vals("speed_m_s")
        self.speed_smooth = get_vals("speed_smooth_m_s")
        
        def get_str_labels(col):
            if col in work.columns:
                return work[col].values
            return np.full(len(work), None, dtype=object)

        self.raw_labels = get_str_labels("raw_motion_label")
        self.smoothed_labels = get_str_labels("motion_label")

    def get_stats(self, start_idx: int, end_idx: int) -> dict:
        start_idx = int(start_idx)
        end_idx = int(end_idx)
        n_points = end_idx - start_idx + 1
        
        if n_points > 1:
            duration_s = float(self.cum_time[end_idx] - self.cum_time[start_idx])
            distance_m = float(self.cum_dist[end_idx] - self.cum_dist[start_idx])
        else:
            duration_s = 0.0
            distance_m = 0.0

        sl = slice(start_idx, end_idx + 1)

        def safe_nanmean(arr):
            chunk = arr[sl]
            valid = chunk[~np.isnan(chunk)]
            return float(np.mean(valid)) if len(valid) > 0 else np.nan

        avg_hr = safe_nanmean(self.hr)
        avg_cad = safe_nanmean(self.cad)
        avg_speed_raw = safe_nanmean(self.speed_raw)
        avg_speed_smooth = safe_nanmean(self.speed_smooth)

        avg_speed = distance_m / duration_s if duration_s > 0 else np.nan
        avg_pace = ((duration_s / 60.0) / (distance_m / 1000.0)) if (duration_s > 0 and distance_m > 0) else np.nan

        # Mode and First calculations avoiding expensive Pandas logic
        valid_raw = [x for x in self.raw_labels[sl] if isinstance(x, str)]
        valid_smooth = [x for x in self.smoothed_labels[sl] if isinstance(x, str)]
        
        raw_label_first = valid_raw[0] if valid_raw else None
        smoothed_label_first = valid_smooth[0] if valid_smooth else None
        
        def mode_str(arr):
            if not arr: return None
            vals, counts = np.unique(arr, return_counts=True)
            return vals[np.argmax(counts)]

        raw_label_mode = mode_str(valid_raw)
        smoothed_label_mode = mode_str(valid_smooth)

        if duration_s <= 0 or np.isnan(avg_speed):
            final_label = "stopped"
        elif avg_speed < STOP_SPEED_THRESH:
            final_label = "stopped"
        elif avg_speed <= WALK_SPEED_MAX and (np.isnan(avg_cad) or avg_cad < WALK_CADENCE_MAX):
            final_label = "walking"
        else:
            final_label = "running"

        return {
            "label": final_label,
            "final_label": final_label,
            "raw_label_first": raw_label_first,
            "raw_label_mode": raw_label_mode,
            "smoothed_label_first": smoothed_label_first,
            "smoothed_label_mode": smoothed_label_mode,
            "start_time": self.time.iloc[start_idx],
            "end_time": self.time.iloc[end_idx],
            "duration_s": duration_s,
            "distance_m": distance_m,
            "avg_speed_m_s": avg_speed,
            "avg_speed_raw_m_s": avg_speed_raw,
            "avg_speed_smooth_m_s": avg_speed_smooth,
            "avg_pace_min_per_km": avg_pace,
            "avg_hr_bpm": avg_hr,
            "avg_cadence_spm": avg_cad,
            "n_points": n_points,
            "start_idx": start_idx,
            "end_idx": end_idx,
        }


def _merge_tiny_segments(calc: SegmentStatsCalculator, initial_segs: pd.DataFrame) -> list:
    if initial_segs.empty:
        return []

    stats_list = [calc.get_stats(int(r.start_idx), int(r.end_idx)) for r in initial_segs.itertuples(index=False)]

    def is_tiny(st):
        return (st["duration_s"] < MIN_SEGMENT_TIME_S and st["distance_m"] < MIN_SEGMENT_DIST_M) or (st["distance_m"] == 0.0)

    while True:
        tiny_idx = -1
        for i, st in enumerate(stats_list):
            if is_tiny(st):
                tiny_idx = i
                break
        
        if tiny_idx == -1:
            break
        
        if len(stats_list) == 1:
            break
            
        i = tiny_idx
        
        # Merge forward
        if i == 0:
            merged_start = stats_list[0]["start_idx"]
            merged_end = stats_list[1]["end_idx"]
            stats_list[0:2] = [calc.get_stats(merged_start, merged_end)]
            continue
            
        # Merge backward
        if i == len(stats_list) - 1:
            merged_start = stats_list[i-1]["start_idx"]
            merged_end = stats_list[i]["end_idx"]
            stats_list[i-1:i+1] = [calc.get_stats(merged_start, merged_end)]
            continue
        
        # Middle segment: check neighbors
        prev_stats = stats_list[i-1]
        this_stats = stats_list[i]
        next_stats = stats_list[i+1]
        
        prev_same = prev_stats["smoothed_label_mode"] == next_stats["smoothed_label_mode"]
        
        if prev_same:
            merged_start = prev_stats["start_idx"]
            merged_end = next_stats["end_idx"]
            stats_list[i-1:i+2] = [calc.get_stats(merged_start, merged_end)]
        else:
            prev_str = float(prev_stats["duration_s"]) + float(prev_stats["distance_m"]) / 10.0
            next_str = float(next_stats["duration_s"]) + float(next_stats["distance_m"]) / 10.0
            
            if prev_str >= next_str:
                merged_start = prev_stats["start_idx"]
                merged_end = this_stats["end_idx"]
                stats_list[i-1:i+1] = [calc.get_stats(merged_start, merged_end)]
            else:
                merged_start = this_stats["start_idx"]
                merged_end = next_stats["end_idx"]
                stats_list[i:i+2] = [calc.get_stats(merged_start, merged_end)]

    return stats_list


def classify_motion_row(speed_m_s, cadence_spm):
    if pd.isna(speed_m_s):
        return "stopped"
    if speed_m_s < STOP_SPEED_THRESH:
        return "stopped"
    if speed_m_s <= WALK_SPEED_MAX and (pd.isna(cadence_spm) or cadence_spm < WALK_CADENCE_MAX):
        return "walking"
    return "running"


def _majority_label_series(labels: pd.Series, window: int = 5) -> pd.Series:
    labels = pd.Series(labels).astype("object")
    if labels.empty or window <= 1:
        return labels

    order = ["stopped", "walking", "running"]
    rank = {k: i for i, k in enumerate(order)}

    out = []
    half = window // 2
    vals = labels.tolist()
    n = len(vals)

    for i in range(n):
        lo = max(0, i - half)
        hi = min(n, i + half + 1)
        chunk = [v for v in vals[lo:hi] if pd.notna(v)]
        if not chunk:
            out.append(np.nan)
            continue
        counts = pd.Series(chunk).value_counts()
        max_count = counts.max()
        tied = [lab for lab, cnt in counts.items() if cnt == max_count]
        tied.sort(key=lambda x: rank.get(x, 999))
        out.append(tied[-1])

    return pd.Series(out, index=labels.index)


def _initial_motion_segments(work: pd.DataFrame) -> pd.DataFrame:
    change = work["motion_label"].ne(work["motion_label"].shift()).cumsum()
    records = []
    for _, seg in work.groupby(change):
        if seg.empty:
            continue
        records.append(
            {
                "label": seg["motion_label"].iloc[0],
                "start_idx": int(seg.index[0]),
                "end_idx": int(seg.index[-1]),
            }
        )
    return pd.DataFrame(records)

def summarize_motion_segments(df: pd.DataFrame, smoothing_window: int = 5) -> pd.DataFrame:
    work = df.copy().sort_values("time").reset_index(drop=True)

    speed_smooth = pd.to_numeric(work.get("speed_smooth_m_s"), errors="coerce")
    speed_raw = pd.to_numeric(work.get("speed_m_s"), errors="coerce")
    cadence = pd.to_numeric(
        work.get("cadence", pd.Series(index=work.index, dtype=float)),
        errors="coerce"
    )

    speed_for_label = speed_smooth.fillna(speed_raw)

    work["raw_motion_label"] = [
        classify_motion_row(s, c) for s, c in zip(speed_for_label, cadence)
    ]

    if smoothing_window and smoothing_window > 1:
        work["motion_label"] = _majority_label_series(work["raw_motion_label"], window=smoothing_window)
    else:
        work["motion_label"] = work["raw_motion_label"]

    initial = _initial_motion_segments(work)
    
    # Initialize the fast numpy calculator
    calc = SegmentStatsCalculator(work)
    # The merge function now returns a fully computed list of dictionaries
    final_records = _merge_tiny_segments(calc, initial)

    out = pd.DataFrame(final_records)

    preferred_order = [
        "label", "final_label", "raw_label_first", "raw_label_mode",
        "smoothed_label_first", "smoothed_label_mode", "start_time",
        "end_time", "duration_s", "distance_m", "avg_speed_m_s",
        "avg_speed_raw_m_s", "avg_speed_smooth_m_s", "avg_pace_min_per_km",
        "avg_hr_bpm", "avg_cadence_spm", "n_points", "start_idx", "end_idx",
    ]

    cols = [c for c in preferred_order if c in out.columns] + [c for c in out.columns if c not in preferred_order]
    return out[cols]


def utc_to_local_string(series: pd.Series, tz_name=DEFAULT_TIMEZONE) -> pd.Series:
    ts = pd.to_datetime(series, utc=True, errors="coerce")
    try:
        return ts.dt.tz_convert(ZoneInfo(tz_name)).dt.strftime(TIME_FMT)
    except Exception:
        return ts.dt.tz_convert(ZoneInfo(DEFAULT_TIMEZONE)).dt.strftime(TIME_FMT)

def normalize_local_string(series: pd.Series) -> pd.Series:
    ts = pd.to_datetime(series, errors="coerce")
    return ts.dt.strftime(TIME_FMT)


def first_valid(series: pd.Series):
    s = series.dropna()
    return s.iloc[0] if not s.empty else None


def collapse_run_streams_for_map(df: pd.DataFrame, tz_name=DEFAULT_TIMEZONE) -> pd.DataFrame:
    """
    Prepare the final collapsed run data for map output using vectorized aggregation.
    """
    df = df.copy()
    df["time"] = utc_to_local_string(df["time"], tz_name=tz_name)
    
    numeric_cols = [
        "latitude", "longitude", "altitude_m", "distance_m",
        "heart_rate_bpm", "cadence", "speed_m_s", "run_cadence", "watts"
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    agg_rules = {}
    
    # Metadata and coordinates: take first valid
    first_cols = [
        "activity_id", "sport", "lap_start_time", "lap_total_time_s", 
        "lap_distance_m", "latitude", "longitude", "altitude_m", "distance_m"
    ]
    for col in first_cols:
        if col in df.columns:
            agg_rules[col] = "first"
            
    # Metrics: take the mean
    mean_cols = ["heart_rate_bpm", "cadence", "speed_m_s", "run_cadence", "watts"]
    for col in mean_cols:
        if col in df.columns:
            agg_rules[col] = "mean"

    if not agg_rules:
        out = df.drop_duplicates(subset=["time"]).reset_index(drop=True)
    else:
        out = df.groupby("time", as_index=False).agg(agg_rules)
        out = out.sort_values("time").reset_index(drop=True)

    # Reconcile cadence
    if "cadence" in out.columns and "run_cadence" in out.columns:
        out["cadence"] = out["cadence"].fillna(out["run_cadence"])
    elif "cadence" not in out.columns and "run_cadence" in out.columns:
        out["cadence"] = out["run_cadence"]
        
    return out

def build_lookup(run_df_collapsed: pd.DataFrame) -> pd.DataFrame:
    needed = {"time", "latitude", "longitude"}
    missing = needed - set(run_df_collapsed.columns)
    if missing:
        raise ValueError(f"Missing columns for coordinate lookup: {sorted(missing)}")

    lookup = run_df_collapsed.copy()
    lookup["time_dt"] = pd.to_datetime(lookup["time"], errors="coerce")
    lookup["latitude"] = pd.to_numeric(lookup["latitude"], errors="coerce")
    lookup["longitude"] = pd.to_numeric(lookup["longitude"], errors="coerce")

    lookup = lookup.dropna(subset=["time_dt", "latitude", "longitude"])
    lookup = lookup.sort_values("time_dt").reset_index(drop=True)

    return lookup[["time_dt", "latitude", "longitude"]]

def enrich_segments(seg_df: pd.DataFrame, lookup: pd.DataFrame, tolerance=ENRICH_SEGMENTS_TOLERANCE) -> pd.DataFrame:
    seg_df = seg_df.copy()

    seg_df["start_time"] = normalize_local_string(seg_df["start_time"])
    seg_df["end_time"] = normalize_local_string(seg_df["end_time"])

    for col in ["avg_pace_min_per_km", "avg_hr_bpm", "avg_cadence_spm", "distance_m", "duration_s"]:
        if col in seg_df.columns:
            seg_df[col] = pd.to_numeric(seg_df[col], errors="coerce")

    seg_df["start_time_dt"] = pd.to_datetime(seg_df["start_time"], errors="coerce")
    seg_df["end_time_dt"] = pd.to_datetime(seg_df["end_time"], errors="coerce")

    start_match = pd.merge_asof(
        seg_df[["start_time_dt"]].sort_values("start_time_dt"),
        lookup.sort_values("time_dt"),
        left_on="start_time_dt",
        right_on="time_dt",
        direction="nearest",
        tolerance=pd.Timedelta(tolerance),
    )

    end_match = pd.merge_asof(
        seg_df[["end_time_dt"]].sort_values("end_time_dt"),
        lookup.sort_values("time_dt"),
        left_on="end_time_dt",
        right_on="time_dt",
        direction="nearest",
        tolerance=pd.Timedelta(tolerance),
    )

    start_match.index = seg_df.sort_values("start_time_dt").index
    end_match.index = seg_df.sort_values("end_time_dt").index

    seg_df.loc[start_match.index, "start_latitude"] = start_match["latitude"].values
    seg_df.loc[start_match.index, "start_longitude"] = start_match["longitude"].values
    seg_df.loc[end_match.index, "end_latitude"] = end_match["latitude"].values
    seg_df.loc[end_match.index, "end_longitude"] = end_match["longitude"].values

    return seg_df


def build_segments_payload(run_df_collapsed: pd.DataFrame, seg_df: pd.DataFrame):
    style_map = {
        "running": {"dashArray": None},
        "walking": {"dashArray": "10 8"},
        "stopped": {"dashArray": "2 10"},
    }

    plot_df = run_df_collapsed.dropna(subset=["latitude", "longitude"]).copy()
    plot_df["time_dt"] = pd.to_datetime(plot_df["time"], errors="coerce")
    plot_df["latitude"] = pd.to_numeric(plot_df["latitude"], errors="coerce")
    plot_df["longitude"] = pd.to_numeric(plot_df["longitude"], errors="coerce")
    plot_df = plot_df.dropna(subset=["time_dt", "latitude", "longitude"]).sort_values("time_dt").reset_index(drop=True)

    payload = []

    for _, row in seg_df.iterrows():
        label = str(row.get("label", "unknown")).strip().lower()
        st = row["start_time"]
        et = row["end_time"]

        st_dt = pd.to_datetime(st, errors="coerce")
        et_dt = pd.to_datetime(et, errors="coerce")

        seg_pts = plot_df[
            (plot_df["time_dt"] >= st_dt) & (plot_df["time_dt"] <= et_dt)
        ][["latitude", "longitude", "time", "time_dt"]].copy()

        # Fallback: if no points in range, take nearest start/end points
        if seg_pts.empty and pd.notna(st_dt) and pd.notna(et_dt) and not plot_df.empty:
            start_idx = (plot_df["time_dt"] - st_dt).abs().idxmin()
            end_idx = (plot_df["time_dt"] - et_dt).abs().idxmin()

            lo = min(start_idx, end_idx)
            hi = max(start_idx, end_idx)

            seg_pts = plot_df.loc[lo:hi, ["latitude", "longitude", "time", "time_dt"]].copy()

            # If still only one point, duplicate it so Leaflet can draw something
            if seg_pts.empty:
                nearest_idx = (plot_df["time_dt"] - st_dt).abs().idxmin()
                seg_pts = plot_df.loc[[nearest_idx], ["latitude", "longitude", "time", "time_dt"]].copy()

        if seg_pts.empty:
            continue

        coords = seg_pts[["latitude", "longitude"]].values.tolist()
        if len(coords) == 1:
            coords = coords + coords

        payload.append({
            "label": label,
            "start_time": st,
            "end_time": et,
            "distance_m": None if pd.isna(row.get("distance_m")) else float(row.get("distance_m")),
            "duration_s": None if pd.isna(row.get("duration_s")) else float(row.get("duration_s")),
            "avg_pace_min_per_km": None if pd.isna(row.get("avg_pace_min_per_km")) else float(row.get("avg_pace_min_per_km")),
            "avg_hr_bpm": None if pd.isna(row.get("avg_hr_bpm")) else float(row.get("avg_hr_bpm")),
            "avg_cadence_spm": None if pd.isna(row.get("avg_cadence_spm")) else float(row.get("avg_cadence_spm")),
            "coords": coords,
            "dashArray": style_map.get(label, {"dashArray": "4 6"})["dashArray"],
        })

    return payload, plot_df


def weighted_percentile(values, weights, q):
    if len(values) == 0:
        return None
    order = np.argsort(values)
    v = np.asarray(values)[order]
    w = np.asarray(weights)[order]
    cum_w = np.cumsum(w)
    total_w = cum_w[-1]
    if total_w <= 0:
        return None
    target = q * total_w
    idx = np.searchsorted(cum_w, target, side="left")
    idx = min(idx, len(v) - 1)
    return float(v[idx])


def compute_weighted_histogram(values, weights, vmin, vmax, bins=8):
    if len(values) == 0:
        return [1.0], [0.0, 1.0]
    hist_weights, hist_edges = np.histogram(values, bins=bins, range=(vmin, vmax), weights=weights)
    return hist_weights.tolist(), hist_edges.tolist()


def compute_metric_stats(seg_df: pd.DataFrame):
    required_base_cols = {"label", "distance_m"}
    missing_base = required_base_cols - set(seg_df.columns)
    if missing_base:
        raise ValueError(
            f"compute_metric_stats: missing required columns: {sorted(missing_base)}"
        )

    stats = {}
    metric_specs = {
        "pace": {
            "col": "avg_pace_min_per_km",
            "exclude_labels": {"stopped"},
            "q_low": 0.05,
            "q_high": 0.95,
            "bins": 8,
        },
        "hr": {
            "col": "avg_hr_bpm",
            "exclude_labels": set(),
            "q_low": 0.05,
            "q_high": 0.95,
            "bins": 8,
        },
        "cadence": {
            "col": "avg_cadence_spm",
            "exclude_labels": {"stopped"},
            "q_low": 0.05,
            "q_high": 0.95,
            "bins": 8,
        },
    }

    metric_cols = {spec["col"] for spec in metric_specs.values()}
    missing_metric_cols = metric_cols - set(seg_df.columns)
    if missing_metric_cols:
        raise ValueError(
            f"compute_metric_stats: missing metric columns: {sorted(missing_metric_cols)}"
        )

    for metric, spec in metric_specs.items():
        col = spec["col"]
        work = seg_df.copy()

        if spec["exclude_labels"]:
            work = work[~work["label"].isin(spec["exclude_labels"])]

        work[col] = pd.to_numeric(work[col], errors="coerce")
        work["distance_m"] = pd.to_numeric(work["distance_m"], errors="coerce")
        work = work.dropna(subset=[col, "distance_m"])
        work = work[work["distance_m"] > 0]

        if work.empty:
            stats[metric] = {
                "min": 0.0,
                "mid": 0.5,
                "max": 1.0,
                "hist_edges": [0.0, 1.0],
                "hist_weights": [1.0],
                "underflow_weight": 0.0,
                "overflow_weight": 0.0,
                "total_weight": 1.0,
                "weight_unit": "m",
            }
            continue

        values = work[col].to_numpy(dtype=float)
        weights = work["distance_m"].to_numpy(dtype=float)

        q_low = weighted_percentile(values, weights, spec["q_low"])
        q_high = weighted_percentile(values, weights, spec["q_high"])

        trimmed_mask = (values >= q_low) & (values <= q_high)
        trimmed_values = values[trimmed_mask]
        trimmed_weights = weights[trimmed_mask]

        if len(trimmed_values) == 0:
            trimmed_values = values
            trimmed_weights = weights

        vmin = float(np.min(trimmed_values))
        vmax = float(np.max(trimmed_values))
        vmid = weighted_percentile(trimmed_values, trimmed_weights, 0.5)

        if vmax <= vmin:
            vmax = vmin + 1e-9

        hist_weights, hist_edges = compute_weighted_histogram(
            trimmed_values,
            trimmed_weights,
            vmin,
            vmax,
            bins=spec["bins"],
        )

        underflow_weight = float(np.sum(weights[values < vmin]))
        overflow_weight = float(np.sum(weights[values > vmax]))

        stats[metric] = {
            "min": vmin,
            "mid": float(vmid),
            "max": vmax,
            "hist_edges": [float(x) for x in hist_edges],
            "hist_weights": [float(x) for x in hist_weights],
            "underflow_weight": underflow_weight,
            "overflow_weight": overflow_weight,
            "total_weight": float(np.sum(weights)),
            "trimmed_weight": float(np.sum(trimmed_weights)),
            "weight_unit": "m",
        }

    return stats

def format_hms(seconds):
    if seconds is None or pd.isna(seconds):
        return "n/a"
    seconds = int(round(float(seconds)))
    h = seconds // 3600
    m = (seconds % 3600) // 60
    s = seconds % 60
    return f"{h:02d}:{m:02d}:{s:02d}"

def format_pace(min_per_km):
    if min_per_km is None or pd.isna(min_per_km) or np.isinf(min_per_km):
        return "n/a"
    total_sec = int(round(float(min_per_km) * 60.0))
    mm = total_sec // 60
    ss = total_sec % 60
    return f"{mm}:{ss:02d} min/km"

def format_speed(speed_m_s):
    if speed_m_s is None or pd.isna(speed_m_s):
        return "n/a"
    return f"{float(speed_m_s):.2f} m/s"

def format_km(distance_m):
    if distance_m is None or pd.isna(distance_m):
        return "n/a"
    return f"{float(distance_m) / 1000.0:.2f} km"

def format_meters(m):
    if m is None or pd.isna(m):
        return "n/a"
    return f"{float(m):.1f} m"

def basic_time_distance(df: pd.DataFrame):
    if df.empty:
        return {
            "total_distance_m": np.nan,
            "elapsed_time_s": np.nan,
            "moving_time_s": np.nan,
            "moving_distance_m": np.nan,
        }

    total_distance_m = (
        float(df["distance_m"].max() - df["distance_m"].min())
        if "distance_m" in df.columns and df["distance_m"].notna().any()
        else np.nan
    )
    elapsed_time_s = float((df["time"].iloc[-1] - df["time"].iloc[0]).total_seconds())

    speed_for_motion = df.get("speed_smooth_m_s", df.get("speed_m_s"))
    if speed_for_motion is None:
        moving_time_s = np.nan
        moving_distance_m = np.nan
    else:
        moving_mask = pd.to_numeric(speed_for_motion, errors="coerce") >= MOVING_SPEED_THRESH
        moving_time_s = float(df.loc[moving_mask, "time_delta_s"].sum()) if "time_delta_s" in df.columns else np.nan
        moving_distance_m = float(df.loc[moving_mask, "distance_delta_m"].sum()) if "distance_delta_m" in df.columns else np.nan

    return {
        "total_distance_m": total_distance_m,
        "elapsed_time_s": elapsed_time_s,
        "moving_time_s": moving_time_s,
        "moving_distance_m": moving_distance_m,
    }

def compute_ascent_descent(df: pd.DataFrame):
    if "altitude_m" not in df.columns:
        return np.nan, np.nan
    alt = pd.to_numeric(df["altitude_m"], errors="coerce")
    delta = alt.diff().fillna(0.0)
    ascent = float(delta.clip(lower=0).sum())
    descent = float((-delta.clip(upper=0)).sum())
    return ascent, descent

def summarize_motion_totals(seg_df: pd.DataFrame):
    out = {}
    if seg_df.empty:
        return out

    work = seg_df.copy()
    for col in ["duration_s", "distance_m"]:
        if col in work.columns:
            work[col] = pd.to_numeric(work[col], errors="coerce")

    grouped = work.groupby("label", dropna=False)
    for label, g in grouped:
        key = str(label).strip().lower()
        out[key] = {
            "segments": int(len(g)),
            "duration_s": float(g["duration_s"].sum()) if "duration_s" in g.columns else np.nan,
            "distance_m": float(g["distance_m"].sum()) if "distance_m" in g.columns else np.nan,
        }
    return out

@lru_cache(maxsize=128)
def lookup_timezone_name(lat: float, lon: float) -> str:
    try:
        tz_name = TF.timezone_at(lat=float(lat), lng=float(lon))
        if tz_name:
            return tz_name
    except Exception:
        pass
    return DEFAULT_TIMEZONE

def infer_activity_timezone_name(plot_df: pd.DataFrame) -> str:
    try:
        if plot_df.empty or not {"latitude", "longitude"}.issubset(plot_df.columns):
            return DEFAULT_TIMEZONE
        lat = pd.to_numeric(plot_df["latitude"], errors="coerce")
        lon = pd.to_numeric(plot_df["longitude"], errors="coerce")
        valid = pd.DataFrame({"lat": lat, "lon": lon}).dropna()
        if valid.empty:
            return DEFAULT_TIMEZONE
        center_lat = float((valid["lat"].min() + valid["lat"].max()) / 2.0)
        center_lon = float((valid["lon"].min() + valid["lon"].max()) / 2.0)
        return lookup_timezone_name(center_lat, center_lon)
    except Exception:
        return DEFAULT_TIMEZONE

@lru_cache(maxsize=128)
def reverse_geocode_city(lat: float, lon: float) -> str:
    """
    Reverse geocodes lat/lon to a city/town name.
    Includes a highly specific user-agent and extended timeout for cloud deployments.
    """
    if pd.isna(lat) or pd.isna(lon):
        return ""
        
    try:
        # FIX 1: Unique User-Agent prevents OpenStreetMap from blocking the cloud IP
        geolocator = Nominatim(user_agent="motion_map_analyzer_streamlit_v1.0")
        
        # FIX 2: Increased timeout to 10 seconds for slower cloud network routing
        location = geolocator.reverse((lat, lon), exactly_one=True, timeout=10)
        
        if not location:
            return ""
            
        address = location.raw.get("address", {})
        
        # OpenStreetMap stores city names inconsistently depending on the region's size
        city = (
            address.get("city") or 
            address.get("town") or 
            address.get("municipality") or 
            address.get("village") or 
            address.get("suburb")
        )
        
        if city:
            return city
            
        # Fallback to state/county if city is somehow completely missing
        return address.get("state") or address.get("county") or ""
        
    except Exception as e:
        # Fails gracefully without crashing the app
        print(f"Geocoding Error: {e}")
        return ""


def build_run_summary_title(runstats: dict, plot_df: pd.DataFrame) -> str:
    start_text = runstats.get("start_time")
    title_date = "Unknown date"
    try:
        dt = pd.to_datetime(start_text, errors="coerce")
        if pd.notna(dt):
            title_date = dt.strftime("%d-%b-%Y")
    except Exception:
        pass

    location_text = "Unknown location"
    try:
        if not plot_df.empty and {"latitude", "longitude"}.issubset(plot_df.columns):
            lat = pd.to_numeric(plot_df["latitude"], errors="coerce").dropna()
            lon = pd.to_numeric(plot_df["longitude"], errors="coerce").dropna()
            if not lat.empty and not lon.empty:
                center_lat = float((lat.min() + lat.max()) / 2.0)
                center_lon = float((lon.min() + lon.max()) / 2.0)
                location_text = reverse_geocode_city(center_lat, center_lon)
    except Exception:
        pass

    return f"Run Summary — {title_date} — {location_text}"

def compute_run_stats(df: pd.DataFrame, segdf: pd.DataFrame, tz_name=DEFAULT_TIMEZONE):
    ti = basic_time_distance(df)

    moving_speed = np.nan
    moving_pace = np.nan
    if (
        pd.notna(ti["moving_time_s"])
        and pd.notna(ti["moving_distance_m"])
        and ti["moving_time_s"] > 0
        and ti["moving_distance_m"] > 0
    ):
        moving_speed = ti["moving_distance_m"] / ti["moving_time_s"]
        moving_pace = ti["moving_time_s"] / 60.0 / (ti["moving_distance_m"] / 1000.0)

    max_speed = np.nan
    max_pace = np.nan
    if "speed_m_s" in df.columns:
        speed = pd.to_numeric(df["speed_m_s"], errors="coerce")
        speed = speed[speed > 0]
        if not speed.empty:
            max_speed = float(speed.max())
            max_pace = 1000.0 / max_speed / 60.0

    avg_hr = np.nan
    max_hr = np.nan
    if "heart_rate_bpm" in df.columns:
        hr = pd.to_numeric(df["heart_rate_bpm"], errors="coerce")
        if hr.notna().any():
            avg_hr = float(hr.mean())
            max_hr = float(hr.max())

    avg_cad = np.nan
    max_cad = np.nan
    if "cadence" in df.columns:
        cad = pd.to_numeric(df["cadence"], errors="coerce")
        if cad.notna().any():
            avg_cad = float(cad.mean())
            max_cad = float(cad.max())

    ascent, descent = compute_ascent_descent(df)
    motion_totals = summarize_motion_totals(segdf)

    start_time = df["time"].iloc[0] if not df.empty else None
    end_time = df["time"].iloc[-1] if not df.empty else None

    return {
        "start_time": format_local_time(start_time, tz_name=tz_name) if start_time is not None else "na",
        "end_time": format_local_time(end_time, tz_name=tz_name) if end_time is not None else "na",
        "total_distance_m": ti["total_distance_m"],
        "elapsed_time_s": ti["elapsed_time_s"],
        "moving_time_s": ti["moving_time_s"],
        "moving_distance_m": ti["moving_distance_m"],
        "avg_speed_m_s": moving_speed,
        "avg_pace_min_per_km": moving_pace,
        "max_speed_m_s": max_speed,
        "max_pace_min_per_km": max_pace,
        "avg_hr_bpm": avg_hr,
        "max_hr_bpm": max_hr,
        "avg_cadence_spm": avg_cad,
        "max_cadence_spm": max_cad,
        "ascent_m": ascent,
        "descent_m": descent,
        "motion_totals": motion_totals,
        "segment_count": int(len(segdf)),
        "trackpoint_count": int(len(df)),
        "timezone_name": tz_name,
    }

def pace_from_speed(speed_m_s: pd.Series) -> pd.Series:
    """
    Pace (min/km) from speed (m/s), matching getrunstats.15.py logic.
    """
    speed = pd.to_numeric(speed_m_s, errors="coerce")
    with np.errstate(divide="ignore", invalid="ignore"):
        pace_min_per_km = 1000.0 / (60.0 * speed)
    pace_min_per_km.replace([np.inf, -np.inf], np.nan, inplace=True)
    return pace_min_per_km


def best_rolling_pace(df: pd.DataFrame, window_m: float) -> dict | None:
    if "distance_m" not in df.columns:
        return None

    work = df.copy()
    work["distance_m"] = pd.to_numeric(work["distance_m"], errors="coerce")
    work = work.dropna(subset=["distance_m"]).sort_values("distance_m").reset_index(drop=True)
    if work.empty:
        return None

    distances = work["distance_m"].to_numpy()
    n = len(distances)
    if n < 2:
        return None

    times = pd.to_datetime(work["time"], errors="coerce").to_numpy()
    # Use Local Time string if available to match the frontend Map data
    time_strs = work["time_str"].to_numpy() if "time_str" in work.columns else times
    
    best_pace = None
    best_start_time = None
    best_end_time = None
    end_idx = 0

    for start_idx in range(n):
        start_dist = distances[start_idx]
        target = start_dist + window_m

        while end_idx < n and distances[end_idx] < target:
            end_idx += 1
        if end_idx >= n:
            break

        time_window = (times[end_idx] - times[start_idx]) / np.timedelta64(1, "s")
        if time_window <= 0:
            continue

        pace_min_per_km = (time_window / 60.0) / (window_m / 1000.0)
        if best_pace is None or pace_min_per_km < best_pace:
            best_pace = pace_min_per_km
            best_start_time = str(time_strs[start_idx])
            best_end_time = str(time_strs[end_idx])

    if best_pace is None:
        return None

    return {
        "window_m": float(window_m),
        "pace_min_per_km": float(best_pace),
        "start_time": best_start_time,
        "end_time": best_end_time,
    }


def distance_splits(df: pd.DataFrame, split_m: float) -> pd.DataFrame:
    if "distance_m" not in df.columns:
        return pd.DataFrame()

    work = df.copy()
    work["distance_m"] = pd.to_numeric(work["distance_m"], errors="coerce")
    work["time"] = pd.to_datetime(work["time"], errors="coerce")
    work = work.dropna(subset=["distance_m", "time"]).sort_values("distance_m").reset_index(drop=True)
    if work.empty:
        return pd.DataFrame()

    max_dist = work["distance_m"].max()
    if not np.isfinite(max_dist) or max_dist <= 0:
        return pd.DataFrame()

    split_edges = np.arange(0.0, max_dist + split_m, split_m)
    records: list[dict] = []

    for i in range(len(split_edges) - 1):
        start = split_edges[i]
        end = split_edges[i + 1]
        mask = (work["distance_m"] >= start) & (work["distance_m"] < end)
        seg = work.loc[mask]
        if seg.empty:
            continue

        seg_time_s = (seg["time"].iloc[-1] - seg["time"].iloc[0]).total_seconds()
        seg_dist_m = seg["distance_m"].iloc[-1] - seg["distance_m"].iloc[0]
        if seg_dist_m <= 0 or seg_time_s <= 0:
            continue

        avg_speed = seg_dist_m / seg_time_s
        avg_pace = (seg_time_s / 60.0) / (seg_dist_m / 1000.0)

        st_str = seg["time_str"].iloc[0] if "time_str" in seg.columns else seg["time"].iloc[0].strftime(TIME_FMT)
        et_str = seg["time_str"].iloc[-1] if "time_str" in seg.columns else seg["time"].iloc[-1].strftime(TIME_FMT)

        records.append(
            {
                "split_index": i + 1,
                "start_distance_m": float(start),
                "end_distance_m": float(end),
                "distance_m": float(seg_dist_m),
                "duration_s": float(seg_time_s),
                "avg_speed_m_s": float(avg_speed),
                "avg_pace_min_per_km": float(avg_pace),
                "start_time": st_str,
                "end_time": et_str,
            }
        )

    return pd.DataFrame.from_records(records)


def compute_performance_stats(df: pd.DataFrame, tz_name=DEFAULT_TIMEZONE) -> dict:
    if df.empty:
        return {
            "best_rolling": [],
            "km_splits": [],
            "hr_bands": [],
            "cadence_bands": [],
            "ef_run": None,
        }

    work = df.copy()
    work["time"] = pd.to_datetime(work["time"], errors="coerce")
    work = work.dropna(subset=["time"]).sort_values("time").reset_index(drop=True)

    # Convert timestamps to Local Time strings so they perfectly match the HTML Map data
    work["time_str"] = utc_to_local_string(work["time"], tz_name=tz_name)

    for col in [
        "distance_m", "time_delta_s", "distance_delta_m", 
        "speed_m_s", "speed_smooth_m_s", "heart_rate_bpm", "cadence",
    ]:
        if col in work.columns:
            work[col] = pd.to_numeric(work[col], errors="coerce")

    if "speed_m_s" in work.columns:
        work["pace_min_per_km"] = pace_from_speed(work["speed_m_s"])
    else:
        work["pace_min_per_km"] = np.nan

    best_list: list[dict] = []
    for window in (400.0, 1000.0, 5000.0):
        br = best_rolling_pace(work, window)
        if br is not None:
            best_list.append(br)

    splits_df = distance_splits(work, 1000.0)
    if splits_df.empty:
        km_splits: list[dict] = []
    else:
        km_splits = [
            {
                "index": int(row["split_index"]),
                "distance_m": float(row["distance_m"]),
                "duration_s": float(row["duration_s"]),
                "avg_pace_min_per_km": float(row["avg_pace_min_per_km"]),
                "start_time": str(row["start_time"]),
                "end_time": str(row["end_time"]),
            }
            for _, row in splits_df.iterrows()
        ]

    hr_bands_stats = compute_hr_band_stats(work)
    cad_bands_stats = compute_cadence_band_stats(work)

    ef_run = efficiency_index(work, moving=True)
    ef_run_val = None if np.isnan(ef_run) else float(ef_run)

    return {
        "best_rolling": best_list,
        "km_splits": km_splits,
        "hr_bands": hr_bands_stats,
        "cadence_bands": cad_bands_stats,
        "ef_run": ef_run_val,
    }


def _compute_band_stats(
    df: pd.DataFrame,
    value_col: str,
    bands: list[tuple[str, float, float]],
) -> list[dict]:
    required = {"time_delta_s", "distance_delta_m", value_col}
    if not required.issubset(df.columns):
        return []

    speed_for_motion = df.get("speed_smooth_m_s", df.get("speed_m_s"))
    if speed_for_motion is None:
        return []

    work = df.copy()
    work["time_delta_s"] = pd.to_numeric(work["time_delta_s"], errors="coerce")
    work["distance_delta_m"] = pd.to_numeric(work["distance_delta_m"], errors="coerce")
    work[value_col] = pd.to_numeric(work[value_col], errors="coerce")
    speed_for_motion = pd.to_numeric(speed_for_motion, errors="coerce")

    moving_mask = speed_for_motion >= MOVING_SPEED_THRESH
    work = work.loc[moving_mask].copy()
    if work.empty:
        return []

    out: list[dict] = []
    for label, low, high in bands:
        mask = work[value_col].between(low, high, inclusive="left")
        seg = work.loc[mask]
        if seg.empty:
            continue

        time_s = float(seg["time_delta_s"].sum())
        dist_m = float(seg["distance_delta_m"].sum())

        if dist_m > 0 and time_s > 0:
            avg_pace = (time_s / 60.0) / (dist_m / 1000.0)
        else:
            avg_pace = np.nan

        ef_val = efficiency_index(seg, moving=False)

        out.append(
            {
                "band": label,
                "min_val": float(low) if low != -np.inf else -9999.0,
                "max_val": float(high) if high != np.inf else 9999.0,
                "time_s": time_s,
                "distance_m": dist_m,
                "avg_pace_min_per_km": None if np.isnan(avg_pace) else float(avg_pace),
                "ef": None if np.isnan(ef_val) else float(ef_val),
            }
        )

    return out


def efficiency_index(df: pd.DataFrame, moving: bool = True) -> float:
    """
    Simple EF‑style index (speed / HR) from getrunstats.15.py efficiency_index().[file:295]
    Expressed as (avg_speed * 100) / avg_hr for readability.
    """
    if "heart_rate_bpm" not in df.columns or "speed_m_s" not in df.columns:
        return np.nan

    speed_for_motion = df.get("speed_smooth_m_s", df["speed_m_s"])
    speed_for_motion = pd.to_numeric(speed_for_motion, errors="coerce")

    if moving:
        mask = speed_for_motion >= MOVING_SPEED_THRESH
    else:
        mask = speed_for_motion.notna()

    subset = df.loc[mask].copy()
    if subset.empty:
        return np.nan

    avg_speed = pd.to_numeric(subset["speed_m_s"], errors="coerce").mean()
    avg_hr = pd.to_numeric(subset["heart_rate_bpm"], errors="coerce").mean()

    if avg_hr <= 0 or np.isnan(avg_hr):
        return np.nan

    return float((avg_speed * 100.0) / avg_hr)




def compute_hr_band_stats(df: pd.DataFrame) -> list[dict]:
    """
    Time in HR bands with dist, avg pace, EF; based on hr_bands() +
    band segmentation logic from getrunstats.15.py.[file:295]
    """
    if "heart_rate_bpm" not in df.columns:
        return []

    bands = [
        ("<130", -np.inf, 130.0),
        ("130–150", 130.0, 150.0),
        ("150–170", 150.0, 170.0),
        (">170", 170.0, np.inf),
    ]
    return _compute_band_stats(df, "heart_rate_bpm", bands)


def compute_cadence_band_stats(df: pd.DataFrame) -> list[dict]:
    """
    Time in cadence bands with dist, avg pace, EF; based on cadence_bands()
    logic from getrunstats.15.py.[file:295]
    """
    if "cadence" not in df.columns:
        return []

    bands = [
        ("<160", -np.inf, 160.0),
        ("160–175", 160.0, 175.0),
        (">175", 175.0, np.inf),
    ]
    return _compute_band_stats(df, "cadence", bands)

def write_html(
    outpath: Path,
    segments,
    plotdf: pd.DataFrame,
    metricstats: dict,
    runstats: dict,
    perfstats: dict,
):
    if plotdf.empty:
        center = [12.9716, 77.5946]
        track_points: list[dict] = []
    else:
        latmin = float(plotdf["latitude"].min())
        latmax = float(plotdf["latitude"].max())
        lonmin = float(plotdf["longitude"].min())
        lonmax = float(plotdf["longitude"].max())
        center = [(latmin + latmax) / 2.0, (lonmin + lonmax) / 2.0]
        track_points = (
            plotdf.loc[
                :,
                [c for c in ["time", "latitude", "longitude", "heart_rate_bpm", "cadence"] if c in plotdf.columns],
            ]
            .copy()
            .dropna(subset=["latitude", "longitude"])
            .to_dict(orient="records")
        )

    summary_title = build_run_summary_title(runstats, plotdf)

    segments_json = json.dumps(segments, ensure_ascii=False)
    metricstats_json = json.dumps(metricstats, ensure_ascii=False)
    runstats_json = json.dumps(runstats, ensure_ascii=False)
    perfstats_json = json.dumps(perfstats, ensure_ascii=False)
    center_json = json.dumps(center, ensure_ascii=False)
    summary_title_json = json.dumps(summary_title, ensure_ascii=False)
    track_points_json = json.dumps(track_points, ensure_ascii=False)

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Motion Map</title>
  <link rel="preconnect" href="https://unpkg.com">
  <link
    rel="stylesheet"
    href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css"
    integrity="sha256-p4NxAoJBhIIN+hmNHrzRCf9tD/miZyoHS5obTRR9BMY="
    crossorigin=""
  >
<style>
:root {{
  --sidebar-width: 500px;
  --sidebar-min: 300px;
  --sidebar-max: 700px;
  --bg: #f5f7fb;
  --panel: #ffffff;
  --text: #18212f;
  --muted: #627083;
  --line: #d7deea;
  --soft: #eef2f8;
  --accent: #1e5eff;
  --shadow: 0 8px 24px rgba(16, 24, 40, 0.08);
  --radius: 14px;
  --running: #1058d1;
  --walking: #d97706;
  --stopped: #7c8595;
  --summary-bg: #eef4ff;
  --summary-border: #c8d9ff;
  --map-bg: #f8fafc;
  --map-border: #dbe4f0;
  --metric-bg: #f2fbf7;
  --metric-border: #cfe9dc;
}}

*, *::before, *::after {{
  box-sizing: border-box;
}}

html, body {{
  margin: 0;
  height: 100%;
  font-family: Inter, system-ui, -apple-system, "Segoe UI", Roboto, sans-serif;
  color: var(--text);
  background: var(--bg);
}}

body {{
  overflow: hidden;
}}

.app {{
  height: 100vh;
  display: grid;
  grid-template-columns: minmax(var(--sidebar-min), var(--sidebar-width)) 8px minmax(0, 1fr);
}}

.sidebar {{
  background: var(--panel);
  border-right: 1px solid var(--line);
  overflow-y: auto;
  overflow-x: hidden;
  padding: 14px 14px 16px;
}}

.resizer {{
  cursor: col-resize;
  background: linear-gradient(
    to right,
    transparent 0%,
    #d7deea 40%,
    #c2cada 50%,
    #d7deea 60%,
    transparent 100%
  );
}}

.map-wrap {{
  position: relative;
  min-width: 0;
}}

#map {{
  width: 100%;
  height: 100%;
  background: #f8fafc;
}}

.panel-block {{
  padding: 10px 12px 12px;
  margin-bottom: 10px;
  border: 1px solid var(--line);
  border-radius: 12px;
  background: #ffffff;
  box-shadow: 0 1px 0 rgba(15, 23, 42, 0.02);
}}

.panel-block.summary-section {{
  background: var(--summary-bg);
  border-color: var(--summary-border);
}}

.panel-block.map-section {{
  background: var(--map-bg);
  border-color: var(--map-border);
}}

.panel-block.metric-section {{
  background: var(--metric-bg);
  border-color: var(--metric-border);
}}

.panel-block.footer {{
  background: transparent;
  border: 1px dashed var(--line);
  box-shadow: none;
}}

/* Trays / drawers */

.tray {{
  border-radius: 12px;
  border: 1px solid var(--line);
  background: rgba(255, 255, 255, 0.4);
  box-shadow: 0 1px 0 rgba(15, 23, 42, 0.02);
  margin-bottom: 10px;
  overflow: hidden;
}}

.tray-header {{
  width: 100%;
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 8px 10px;
  border: none;
  border-bottom: 1px solid var(--line);
  background: #f3f4ff;
  cursor: pointer;
  font-size: 1.1rem;
  font-weight: 600;
  color: var(--text);
}}

.tray-title {{
  display: inline-flex;
  align-items: center;
  gap: 6px;
}}

.tray-chevron {{
  width: 16px;
  height: 16px;
  display: inline-flex;
  align-items: center;
  justify-content: center;
  font-size: 0.82rem;
  color: var(--muted);
  transform: rotate(0deg);
  transition: transform 0.16s ease;
}}

.tray-chevron.is-open {{
  transform: rotate(180deg);
}}

.tray-body {{
  padding: 8px 6px 10px;
  max-height: 0;
  overflow: hidden;
  transition: max-height 0.18s ease;
}}

.tray-body.is-open {{
  max-height: 2000px;
}}

h1 {{
  margin: 0 0 4px;
  font-size: 1.22rem;
  line-height: 1.15;
}}

h2 {{
  margin: 0 0 6px;
  font-size: 0.92rem;
  line-height: 1.15;
}}

.subtle-copy {{
  margin: 0;
  color: var(--muted);
  font-size: 0.82rem;
  line-height: 1.4;
}}

.summary-grid {{
  display: grid;
  grid-template-columns: 1fr auto 1fr auto;
  gap: 4px 10px;
  font-size: 0.8rem;
  align-items: center;
}}

.summary-wide {{
  grid-column: 1 / -1;
}}

.metric {{
  padding: 1px 0;
}}

.metric-label {{
  color: var(--muted);
  line-height: 1.15;
}}

.metric-value {{
  font-weight: 600;
  text-align: left;
  line-height: 1.15;
  font-size: 0.9rem;
  font-variant-numeric: tabular-nums;
}}

.totals-grid {{
  display: grid;
  grid-template-columns: repeat(3, minmax(0, 1fr));
  gap: 6px;
  margin-top: 4px;
}}

.totals-card {{
  border: 1px solid var(--line);
  border-radius: 10px;
  padding: 6px 7px;
  background: rgba(255, 255, 255, 0.7);
  min-width: 0;
}}

.totals-card .totals-head {{
  font-size: 0.72rem;
  color: var(--muted);
  margin-bottom: 2px;
  text-align: center;
}}

.totals-card .totals-value {{
  font-size: 1.0rem;
  font-weight: 700;
  text-align: center;
  font-variant-numeric: tabular-nums;
  white-space: nowrap;
}}

.legend-inline {{
  display: flex;
  flex-wrap: wrap;
  justify-content: center;
  gap: 8px 10px;
  padding-top: 2px;
}}

.motion-type-row {{
  display: flex;
  flex-wrap: wrap;
  gap: 8px 10px;
  align-items: center;
  justify-content: flex-start;
}}

.motion-type-item {{
  display: inline-flex;
  align-items: center;
  gap: 7px;
}}

.legend-line-inline {{
  width: 32px;
  height: 0;
  border-top: 4px solid;
  border-radius: 999px;
  flex: 0 0 auto;
  opacity: 1;
}}

.legend-line-inline.walking {{
  border-top-style: dashed;
}}

.legend-line-inline.stopped {{
  border-top-style: dotted;
}}

.button-row {{
  display: flex;
  flex-wrap: wrap;
  gap: 6px;
}}

.btn {{
  border: 1px solid var(--line);
  background: #ffffff;
  color: var(--text);
  border-radius: 999px;
  padding: 6px 10px;
  font-size: 0.8rem;
  line-height: 1;
  cursor: pointer;
  transition:
    background 0.12s ease,
    color 0.12s ease,
    border-color 0.12s ease,
    opacity 0.12s ease,
    transform 0.12s ease;
}}

.btn:hover {{
  background: #f7faff;
  transform: translateY(-1px);
}}

.btn.is-active {{
  background: var(--accent);
  color: #ffffff;
  border-color: var(--accent);
}}

.btn.is-inactive {{
  background: #ffffff;
  color: var(--text);
  border-color: var(--line);
  opacity: 1;
}}

.btn.motion-btn {{
  min-width: 92px;
  justify-content: center;
}}

.histogram {{
  display: grid;
  grid-template-columns: repeat(8, minmax(0, 1fr));
  align-items: flex-end;
  gap: 6px;
  height: 132px;
  margin-top: 8px;
}}

.hist-bar {{
  position: relative;
  min-height: 12px;
  border-radius: 8px 8px 4px 4px;
  cursor: pointer;
  opacity: 0.92;
  transition:
    transform 0.12s ease,
    opacity 0.12s ease,
    box-shadow 0.12s ease;
  border: none;
}}

.hist-bar:hover {{
  transform: translateY(-2px);
  opacity: 1;
}}

.hist-bar.active {{
  outline: 2px solid #111827;
  outline-offset: 2px;
  opacity: 1;
  box-shadow: 0 0 0 2px rgba(255, 255, 255, 0.9) inset;
}}

.hist-axis {{
  display: flex;
  justify-content: space-between;
  gap: 8px;
  font-size: 0.74rem;
  color: var(--muted);
  margin-top: 8px;
  font-variant-numeric: tabular-nums;
}}

.segment-list {{
  display: grid;
  gap: 6px;
  margin-top: 10px;
  max-height: 260px;
  overflow: auto;
  padding-right: 2px;
}}

.segment-row {{
  display: grid;
  grid-template-columns: 12px 1fr auto auto auto;
  gap: 8px;
  align-items: center;
  font-size: 0.8rem;
  padding: 4px 0;
  border-bottom: 1px solid rgba(0, 0, 0, 0.04);
}}

.segment-row.is-highlighted .segment-main,
.segment-row.is-highlighted .segment-meta {{
  font-weight: 700;
  color: var(--text);
}}

.segment-swatch {{
  width: 12px;
  height: 12px;
  border-radius: 3px;
}}

.segment-main {{
  min-width: 0;
}}

.segment-meta {{
  color: var(--muted);
  font-variant-numeric: tabular-nums;
  white-space: nowrap;
}}

.footer {{
  padding-top: 8px;
  font-size: 0.74rem;
  color: var(--muted);
}}

/* Performance Stats tables */

.performance-section {{
  background: #ffffff;
  border-color: var(--line);
}}

.perf-table {{
  display: grid;
  gap: 3px;
  font-size: 0.78rem;
}}

.perf-empty {{
  font-size: 0.76rem;
  color: var(--muted);
}}

.perf-row {{
  display: grid;
  align-items: baseline;
  padding: 2px 0;
  border-bottom: 1px solid rgba(0, 0, 0, 0.04);
}}

/* Interactive highlighting styling for Press & Hold */
.perf-row.is-clickable {{
  cursor: pointer;
  transition: background-color 0.12s ease;
  padding: 4px 6px;
  margin-left: -6px;
  border-radius: 6px;
  border-bottom: none;
  user-select: none;
  -webkit-user-select: none;
  -webkit-touch-callout: none;
}}

.perf-row.is-clickable:hover {{
  background-color: rgba(30, 94, 255, 0.08);
}}

.perf-row.is-clickable.is-active {{
  background-color: rgba(30, 94, 255, 0.12);
  border-left: 3px solid var(--accent);
  padding-left: 3px; 
  font-weight: 500;
}}

.perf-row-header {{
  font-weight: 600;
  color: var(--muted);
  border-bottom: 1px solid rgba(0, 0, 0, 0.08);
  padding-left: 0;
}}

.perf-table--rolling .perf-row {{
  grid-template-columns: 1.1fr 1fr;
}}

.perf-table--splits .perf-row {{
  grid-template-columns: 0.8fr 1fr 1fr;
}}

.perf-table--bands .perf-row {{
  grid-template-columns: 1.0fr 1fr 1fr 1fr 0.8fr;
}}

.perf-label {{
  white-space: nowrap;
}}

.perf-value {{
  text-align: right;
  font-variant-numeric: tabular-nums;
}}

.marker-dot {{
  width: 12px;
  height: 12px;
  border-radius: 999px;
  border: 2px solid #ffffff;
  box-shadow: 0 0 0 1px rgba(15, 23, 42, 0.18);
}}

.marker-dot.start {{
  background: #16a34a;
}}

.marker-dot.finish {{
  background: #dc2626;
}}

.route-badge {{
  min-width: 22px;
  height: 22px;
  padding: 0 6px;
  border-radius: 999px;
  display: inline-flex;
  align-items: center;
  justify-content: center;
  background: rgba(255, 255, 255, 0.95);
  color: #111827;
  border: 1px solid rgba(17, 24, 39, 0.18);
  box-shadow: 0 2px 6px rgba(15, 23, 42, 0.10);
  font-size: 11px;
  font-weight: 700;
  line-height: 1;
  font-variant-numeric: tabular-nums;
  white-space: nowrap;
}}

.route-badge.time {{
  padding: 0 7px;
  min-width: 44px;
}}

.direction-arrow {{
  width: 26px;
  height: 26px;
  display: flex;
  align-items: center;
  justify-content: center;
  transform-origin: 50% 50%;
  pointer-events: none;
  color: rgba(17, 24, 39, 0.82);
  font-size: 24px;
  font-weight: 800;
  line-height: 1;
  text-shadow:
    0 0 2px rgba(255, 255, 255, 0.95),
    0 1px 2px rgba(255, 255, 255, 0.90),
    0 0 6px rgba(255, 255, 255, 0.65);
}}

.direction-arrow::before {{
  content: "\\219F";
  display: block;
}}

.leaflet-control-attribution {{
  font-size: 10px;
}}

@media (max-width: 900px) {{
  body {{
    overflow: auto;
  }}
  .app {{
    grid-template-columns: 1fr;
    grid-template-rows: auto 1fr;
    height: auto;
    min-height: 100vh;
  }}
  .sidebar {{
    max-height: 50vh;
    border-right: none;
    border-bottom: 1px solid var(--line);
  }}
  .resizer {{
    display: none;
  }}
  .map-wrap,
  #map {{
    min-height: 52vh;
  }}
}}
</style>
</head>
<body>
<div class="app" id="app">
  <aside class="sidebar" id="sidebar">
    <section class="panel-block summary-section">
      <h1 id="summaryTitle"></h1>
      <div id="runSummary" class="summary-grid"></div>
    </section>

    <section class="tray">
      <button type="button" class="tray-header" data-tray-toggle="motion">
        <span class="tray-title">Motion Map</span>
        <span class="tray-chevron is-open" aria-hidden="true">&#9660;</span>
      </button>
      <div class="tray-body is-open" id="tray-motion">
        <section class="panel-block map-section">
          <p class="subtle-copy">
            Explore motion-type segments, switch overlay metrics, compare the segment colours
            with the distribution below, and toggle map annotations and display layers.
          </p>
        </section>

        <section class="panel-block map-section">
          <h2>Motion Types</h2>
          <div class="motion-type-row" id="motionTypeButtons"></div>
        </section>

        <section class="panel-block map-section">
          <h2>Markers</h2>
          <div class="button-row" id="markerButtons"></div>
        </section>

        <section class="panel-block map-section">
          <h2>Direction</h2>
          <div class="button-row" id="directionButtons"></div>
        </section>

        <section class="panel-block map-section">
          <h2>Base Map</h2>
          <div class="button-row" id="basemapButtons"></div>
        </section>

        <section class="panel-block map-section">
          <h2>Overlay Metric</h2>
          <div class="button-row" id="metricButtons"></div>
        </section>

        <section class="panel-block metric-section">
          <h2>Colour Scale</h2>
          <div class="button-row" id="themeButtons"></div>
        </section>

        <section class="panel-block metric-section">
          <h2>Metric Distribution</h2>
          <div id="metricHistogram" class="histogram"></div>
          <div id="metricAxis" class="hist-axis"></div>
          <div id="metricSegmentList" class="segment-list"></div>
        </section>
      </div>
    </section>

    <section class="tray">
      <button type="button" class="tray-header" data-tray-toggle="performance">
        <span class="tray-title">Performance Stats</span>
        <span class="tray-chevron" aria-hidden="true">&#9660;</span>
      </button>
      <div class="tray-body" id="tray-performance">
        <section class="panel-block performance-section">
          <h2>Best rolling pace (400m, 1km, 5k)</h2>
          <div id="perfBestRolling" class="perf-table perf-table--rolling"></div>
        </section>

        <section class="panel-block performance-section">
          <h2>Per‑km splits</h2>
          <div id="perfKmSplits" class="perf-table perf-table--splits"></div>
        </section>

        <section class="panel-block performance-section">
          <h2>Time in HR bands</h2>
          <div id="perfHrBands" class="perf-table perf-table--bands"></div>
        </section>

        <section class="panel-block performance-section">
          <h2>Time in cadence bands</h2>
          <div id="perfCadBands" class="perf-table perf-table--bands"></div>
        </section>

        <section class="panel-block performance-section">
          <h2>Efficiency Factor (EF)</h2>
          <div id="perfEf" class="perf-table perf-table--rolling"></div>
        </section>
      </div>
    </section>

    <section class="panel-block footer">
      Generated from TCX-derived motion segments.
    </section>
  </aside>

  <div class="resizer" id="resizer" aria-hidden="true"></div>

  <main class="map-wrap">
    <div id="map"></div>
  </main>
</div>

<script 
src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js" 
integrity="sha256-20nQCchB9co0qIjJZRGuk2/Z9VM+kNiyxNV1lvTlZBo=" 
crossorigin="">
</script>
<script>
const segments = {segments_json};
const metricStats = {metricstats_json};
const runStats = {runstats_json};
const perfStats = {perfstats_json};
const center = {center_json};
const summaryTitle = {summary_title_json};
const trackPoints = {track_points_json};

const THEME_DEFS = {{
  viridis: ["#440154", "#414487", "#2a788e", "#22a884", "#7ad151", "#fde725"],
  cividis: ["#00204c", "#2e4a7d", "#575d6d", "#7d7c78", "#a59c74", "#fee838"],
  turbo: ["#30123b", "#4666d6", "#2fb47c", "#e1d925", "#f89441", "#b40f20"],
  warmcool: ["#6e40aa", "#417de0", "#1ac7c2", "#7bd23c", "#f9c74f", "#f94144"],
  eclectic: ["#C52233", "#A73B4A", "#6467CC", "#FFAE03", "#5A8646", "#327985"],
}};

const MOTION_TYPES = [
  {{ key: "running", label: "Running", className: "", cssVar: "--running", dashArray: null }},
  {{ key: "walking", label: "Walking", className: "walking", cssVar: "--walking", dashArray: "10 8" }},
  {{ key: "stopped", label: "Stopped", className: "stopped", cssVar: "--stopped", dashArray: "2 10" }},
];

const MARKER_DEFS = [
  {{ key: "km", label: "Kilometre markers" }},
  {{ key: "time", label: "Time markers" }},
];

const BASEMAP_DEFS = {{
  standard: {{
    label: "Standard",
    url: "https://tile.openstreetmap.org/{{z}}/{{x}}/{{y}}.png",
    options: {{
      maxZoom: 19,
      attribution: "&copy; OpenStreetMap contributors",
    }},
  }},
  topo: {{
    label: "Topo",
    url: "https://{{s}}.tile.opentopomap.org/{{z}}/{{x}}/{{y}}.png",
    options: {{
      maxZoom: 17,
      subdomains: "abc",
      attribution: "Map data &copy; OpenStreetMap contributors, SRTM | Map style &copy; OpenTopoMap",
    }},
  }},
  dark: {{
    label: "Dark",
    url: "https://s.basemaps.cartocdn.com/dark_all/{{z}}/{{x}}/{{y}}.png",
    options: {{
      maxZoom: 20,
      subdomains: "abcd",
      attribution: "&copy; OpenStreetMap contributors &copy; CARTO",
    }},
  }},
  nomap: {{
    label: "No-Map",
    url: null,
    options: {{}},
  }},
}};

const ARROW_DEF = {{ key: "directionArrows", label: "Direction arrows" }};

let activeMetric = "none";
let activeTheme = "viridis";
let activeBinIndex = null;
let map = null;
let activeMotionTypes = new Set(MOTION_TYPES.map((x) => x.key));
let markerVisibility = {{ km: false, time: false }};
let arrowVisibility = false;
let activeBasemap = "standard";
let currentTileLayer = null;
let routeData = null;
let routeBounds = null;
let currentZoom = 15;
let highlightLayerGroup = L.layerGroup();

const layerGroups = {{
  motion: {{
    running: L.layerGroup(),
    walking: L.layerGroup(),
    stopped: L.layerGroup(),
  }},
  markers: {{
    startFinish: L.layerGroup(),
    km: L.layerGroup(),
    time: L.layerGroup(),
  }},
  arrows: L.layerGroup(),
}};

/* Utility helpers */

function clamp(x, lo, hi) {{
  return Math.max(lo, Math.min(hi, x));
}}

function lerp(a, b, t) {{
  return a + (b - a) * t;
}}

function hexToRgb(hex) {{
  const m = (hex || "#888888").replace("#", "");
  return [
    parseInt(m.slice(0, 2), 16),
    parseInt(m.slice(2, 4), 16),
    parseInt(m.slice(4, 6), 16),
  ];
}}

function rgbToHex(r, g, b) {{
  return "#" + [r, g, b].map((v) => Math.round(v).toString(16).padStart(2, "0")).join("");
}}

function interpColors(stops, t) {{
  const safeStops = Array.isArray(stops) && stops.length ? stops : ["#888888", "#aaaaaa"];
  const n = safeStops.length - 1;
  if (n <= 0) return safeStops[0] || "#888888";
  const x = clamp(t * n, 0, n);
  const i = Math.min(n - 1, Math.floor(x));
  const f = x - i;
  const a = hexToRgb(safeStops[i]);
  const b = hexToRgb(safeStops[i + 1]);
  return rgbToHex(lerp(a[0], b[0], f), lerp(a[1], b[1], f), lerp(a[2], b[2], f));
}}

function safeNumber(v) {{
  const n = Number(v);
  return Number.isFinite(n) ? n : NaN;
}}

function haversineMeters(a, b) {{
  if (!a || !b) return 0;
  const lat1 = safeNumber(a.lat);
  const lon1 = safeNumber(a.lng);
  const lat2 = safeNumber(b.lat);
  const lon2 = safeNumber(b.lng);
  if (![lat1, lon1, lat2, lon2].every(Number.isFinite)) return 0;
  const toRad = Math.PI / 180;
  const dLat = (lat2 - lat1) * toRad;
  const dLon = (lon2 - lon1) * toRad;
  const p1 = lat1 * toRad;
  const p2 = lat2 * toRad;
  const h =
    Math.sin(dLat / 2) ** 2 +
    Math.cos(p1) * Math.cos(p2) * Math.sin(dLon / 2) ** 2;
  return 6371000 * 2 * Math.atan2(Math.sqrt(h), Math.sqrt(Math.max(0, 1 - h)));
}}

function format3DigitInt(value) {{
  const n = safeNumber(value);
  if (!Number.isFinite(n)) return "na";
  return Math.round(n).toString().padStart(3, "0");
}}

function formatPace(v) {{
  const n = safeNumber(v);
  if (!Number.isFinite(n) || n <= 0) return "na";
  const s = Math.round(n * 60);
  const m = Math.floor(s / 60);
  const ss = s % 60;
  return `${{m.toString()}}:${{ss.toString().padStart(2, "0")}} min/km`;
}}

function formatKm(v) {{
  const n = safeNumber(v);
  return Number.isFinite(n) ? (n / 1000).toFixed(2) + " km" : "na";
}}

function formatHms(v) {{
  const n = safeNumber(v);
  if (!Number.isFinite(n)) return "na";
  const s = Math.round(n);
  const h = Math.floor(s / 3600);
  const m = Math.floor((s % 3600) / 60);
  const ss = s % 60;
  return `${{h.toString().padStart(2, "0")}}:${{m.toString().padStart(2, "0")}}:${{ss
    .toString()
    .padStart(2, "0")}}`;
}}

function formatElapsedLabel(seconds) {{
  const s = safeNumber(seconds);
  if (!Number.isFinite(s) || s < 0) return "na";
  const whole = Math.round(s);
  const h = Math.floor(whole / 3600);
  const m = Math.floor((whole % 3600) / 60);
  return `${{h.toString().padStart(2, "0")}}:${{m.toString().padStart(2, "0")}}`;
}}

function formatTimeOnly(isoText) {{
  if (!isoText) return "na";
  const dt = new Date(String(isoText).replace(" ", "T"));
  if (Number.isNaN(dt.getTime())) return String(isoText);
  return dt.toLocaleTimeString("en-GB", {{ hour12: false }});
}}

function formatEf(v) {{
  const n = safeNumber(v);
  return Number.isFinite(n) ? n.toFixed(2) : "na";
}}

const METRIC_DEFS = {{
  none: {{ label: "None", key: "none" }},
  pace: {{
    label: "Pace",
    key: "pace",
    field: "avg_pace_min_per_km",
    fmt: (v) => formatPace(safeNumber(v)),
  }},
  hr: {{
    label: "HR",
    key: "hr",
    field: "avg_hr_bpm",
    fmt: (v) => (Number.isFinite(safeNumber(v)) ? `${{format3DigitInt(v)}} bpm` : "na"),
  }},
  cadence: {{
    label: "Cadence",
    key: "cadence",
    field: "avg_cadence_spm",
    fmt: (v) => (Number.isFinite(safeNumber(v)) ? `${{format3DigitInt(v)}} spm` : "na"),
  }},
}};

/* Colour + metric helpers */

function baseMotionColor(label) {{
  const styles = getComputedStyle(document.documentElement);
  if (label === "walking") return styles.getPropertyValue("--walking").trim() || "#d97706";
  if (label === "stopped") return styles.getPropertyValue("--stopped").trim() || "#7c8595";
  return styles.getPropertyValue("--running").trim() || "#1058d1";
}}

function metricValue(seg) {{
  const def = METRIC_DEFS[activeMetric];
  if (!def || !def.field) return NaN;
  return safeNumber(seg ? seg[def.field] : NaN);
}}

function getMetricColor(value) {{
  if (activeMetric === "none" || !Number.isFinite(value)) return null;
  const stats = metricStats ? metricStats[activeMetric] : null;
  if (!stats) return null;
  const minV = safeNumber(stats.min);
  const maxV = safeNumber(stats.max);
  if (!Number.isFinite(minV) || !Number.isFinite(maxV)) return null;
  const span = Math.max(maxV - minV, 1e-9);
  const t = clamp((value - minV) / span, 0, 1);
  return interpColors(THEME_DEFS[activeTheme], t);
}}

function buildTooltip(seg) {{
  return (
    `<strong>${{seg.label}} segment</strong><br>` +
    `Time: ${{formatTimeOnly(seg.start_time)}} – ${{formatTimeOnly(seg.end_time)}}<br>` +
    `Distance: ${{formatKm(safeNumber(seg.distance_m))}}<br>` +
    `Duration: ${{formatHms(safeNumber(seg.duration_s))}}<br>` +
    `Pace: ${{formatPace(safeNumber(seg.avg_pace_min_per_km))}}<br>` +
    `HR: ${{Number.isFinite(safeNumber(seg.avg_hr_bpm)) ? safeNumber(seg.avg_hr_bpm).toFixed(0) + " bpm" : "na"}}<br>` +
    `Cadence: ${{Number.isFinite(safeNumber(seg.avg_cadence_spm)) ? safeNumber(seg.avg_cadence_spm).toFixed(0) + " spm" : "na"}}`
  );
}}

function motionTotalCell(label, value) {{
  return `
    <div class="totals-card">
      <div class="totals-head">${{label}}</div>
      <div class="totals-value">${{value}}</div>
    </div>
  `;
}}

/* Press & Hold Highlighting Logic */

function clearHighlight() {{
  highlightLayerGroup.clearLayers();
}}

function drawHighlight(pointsArrays) {{
  clearHighlight();
  if (!pointsArrays || pointsArrays.length === 0) return;

  pointsArrays.forEach(pts => {{
    if (!pts || pts.length < 2) return;
    const coords = pts.map(p => [p.lat, p.lng]);
    
    L.polyline(coords, {{
      color: '#facc15',
      weight: getBaseLineWeight() + 6,
      opacity: 0.6,
      lineCap: 'round',
      lineJoin: 'round'
    }}).addTo(highlightLayerGroup);

    L.polyline(coords, {{
      color: '#1e3a8a',
      weight: getBaseLineWeight() - 1,
      opacity: 0.9,
      lineCap: 'round',
      lineJoin: 'round'
    }}).addTo(highlightLayerGroup);
  }});
  // Explicitly NOT calling map.fitBounds here so the view stays steady when pressing
}}

function highlightTimeRange(startTimeStr, endTimeStr) {{
  if (!startTimeStr || !endTimeStr) return;
  const t0 = new Date(startTimeStr.replace(" ", "T")).getTime();
  const t1 = new Date(endTimeStr.replace(" ", "T")).getTime();
  
  const chunk = routeData.filter(p => {{
    if (!p.dt) return false;
    const t = p.dt.getTime();
    return t >= t0 && t <= t1;
  }});

  drawHighlight([chunk]);
}}

function highlightMetricBand(metric, minVal, maxVal) {{
  // 1. Calculate a 5-point moving average to smooth out second-by-second flickering
  const windowSize = 5;
  const smoothedVals = routeData.map((p, i, arr) => {{
    let sum = 0;
    let count = 0;
    const start = Math.max(0, i - 2);
    const end = Math.min(arr.length - 1, i + 2);
    for (let j = start; j <= end; j++) {{
       const v = arr[j][metric];
       if (Number.isFinite(v) && v > 0) {{ // Ignore 0s (stopped)
         sum += v;
         count++;
       }}
    }}
    return count > 0 ? sum / count : p[metric];
  }});

  let chunks = [];
  let currentChunk = [];

  routeData.forEach((p, i) => {{
    const val = smoothedVals[i];
    
    if (val >= minVal && val < maxVal) {{
      // 2. Boundary Extension: Grab the previous point to connect the line seamlessly
      if (currentChunk.length === 0 && i > 0) {{
        currentChunk.push(routeData[i - 1]);
      }}
      currentChunk.push(p);
    }} else {{
      if (currentChunk.length > 0) {{
        // 2. Boundary Extension: Grab this immediate out-of-band point to close seamlessly
        currentChunk.push(p);
        chunks.push(currentChunk);
        currentChunk = [];
      }}
    }}
  }});
  
  if (currentChunk.length > 0) {{
    chunks.push(currentChunk);
  }}

  drawHighlight(chunks);
}}


function bindPressAndHold(el, onStart, onEnd) {{
  let active = false;
  const start = (e) => {{
    if (e && e.type === 'touchstart' && e.cancelable) e.preventDefault();
    if (active) return;
    active = true;
    el.classList.add('is-active');
    onStart();
  }};
  const end = (e) => {{
    if (!active) return;
    active = false;
    el.classList.remove('is-active');
    onEnd();
  }};
  el.addEventListener('mousedown', start);
  el.addEventListener('touchstart', start, {{passive: false}});
  
  el.addEventListener('mouseup', end);
  el.addEventListener('mouseleave', end);
  el.addEventListener('touchend', end);
  el.addEventListener('touchcancel', end);
}}

/* Run summary + performance stats rendering */

function renderRunSummary() {{
  const titleEl = document.getElementById("summaryTitle");
  if (titleEl) titleEl.textContent = summaryTitle || "Run Summary";

  const el = document.getElementById("runSummary");
  if (!el || !runStats) return;

  const mt = runStats.motion_totals || {{}};
  const running = mt.running || {{}};
  const walking = mt.walking || {{}};
  const stopped = mt.stopped || {{}};

  const rows = [
    ["Start", formatTimeOnly(runStats.start_time || null), "End", formatTimeOnly(runStats.end_time || null)],
    ["Moving Distance", formatKm(safeNumber(runStats.moving_distance_m)), "Moving Time", formatHms(safeNumber(runStats.moving_time_s))],
    ["Avg Pace", formatPace(safeNumber(runStats.avg_pace_min_per_km)), "Max Pace", formatPace(safeNumber(runStats.max_pace_min_per_km))],
    [
      "Avg HR",
      Number.isFinite(safeNumber(runStats.avg_hr_bpm)) ? `${{safeNumber(runStats.avg_hr_bpm).toFixed(0)}} bpm` : "na",
      "Max HR",
      Number.isFinite(safeNumber(runStats.max_hr_bpm)) ? `${{safeNumber(runStats.max_hr_bpm).toFixed(0)}} bpm` : "na",
    ],
    [
      "Avg Cadence",
      Number.isFinite(safeNumber(runStats.avg_cadence_spm)) ? `${{safeNumber(runStats.avg_cadence_spm).toFixed(0)}} spm` : "na",
      "Max Cadence",
      Number.isFinite(safeNumber(runStats.max_cadence_spm)) ? `${{safeNumber(runStats.max_cadence_spm).toFixed(0)}} spm` : "na",
    ],
    [
      "Ascent",
      Number.isFinite(safeNumber(runStats.ascent_m)) ? `${{safeNumber(runStats.ascent_m).toFixed(0)}} m` : "na",
      "Descent",
      Number.isFinite(safeNumber(runStats.descent_m)) ? `${{safeNumber(runStats.descent_m).toFixed(0)}} m` : "na",
    ],
  ];

  const baseRows = rows
    .map(
      ([l1, v1, l2, v2]) => `
      <div class="metric">
        <span class="metric-label">${{l1}}</span>
        <div class="metric metric-value">${{v1}}</div>
      </div>
      <div class="metric">
        <span class="metric-label">${{l2}}</span>
        <div class="metric metric-value">${{v2}}</div>
      </div>
    `
    )
    .join("");

  const distanceTable = `
    <div class="metric summary-wide">
      <span class="metric-label">Distance</span>
    </div>
    <div class="metric summary-wide">
      <div class="totals-grid">
        ${{motionTotalCell("Running", formatKm(safeNumber(running.distance_m)))}}
        ${{motionTotalCell("Walking", formatKm(safeNumber(walking.distance_m)))}}
        ${{motionTotalCell("Stopped", formatKm(safeNumber(stopped.distance_m)))}}
      </div>
    </div>
  `;

  const timeTable = `
    <div class="metric summary-wide">
      <span class="metric-label">Time</span>
    </div>
    <div class="metric summary-wide">
      <div class="totals-grid">
        ${{motionTotalCell("Running", formatHms(safeNumber(running.duration_s)))}}
        ${{motionTotalCell("Walking", formatHms(safeNumber(walking.duration_s)))}}
        ${{motionTotalCell("Stopped", formatHms(safeNumber(stopped.duration_s)))}}
      </div>
    </div>
  `;

  el.innerHTML = baseRows + distanceTable + timeTable;
}}

function renderPerformanceStats() {{
  if (!perfStats) return;

  const bestEl = document.getElementById("perfBestRolling");
  const splitsEl = document.getElementById("perfKmSplits");
  const hrEl = document.getElementById("perfHrBands");
  const cadEl = document.getElementById("perfCadBands");
  const efEl = document.getElementById("perfEf");

  /* Best rolling pace */
  if (bestEl) {{
    const rows = (perfStats.best_rolling || []).map((br) => {{
      const w = safeNumber(br.window_m);
      let label = "window";
      if (w < 1000) label = `${{w.toFixed(0)}} m`;
      else if (w >= 5000) label = `${{(w / 1000).toFixed(0)}} km`;
      else label = `${{(w / 1000).toFixed(1)}} km`;
      return `
        <div class="perf-row is-clickable perf-time-range" data-t0="${{br.start_time}}" data-t1="${{br.end_time}}">
          <span class="perf-label">${{label}}</span>
          <span class="perf-value">${{formatPace(safeNumber(br.pace_min_per_km))}}</span>
        </div>
      `;
    }});
    if (!rows.length) {{
      bestEl.innerHTML = `<div class="perf-empty">No rolling pace data.</div>`;
    }} else {{
      bestEl.innerHTML =
        `<div class="perf-row perf-row-header">
          <span class="perf-label">Window</span>
          <span class="perf-value">Best pace</span>
        </div>` + rows.join("");
    }}
  }}

  /* Per-km splits */
  if (splitsEl) {{
    const rows = (perfStats.km_splits || []).map((row) => `
      <div class="perf-row is-clickable perf-time-range" data-t0="${{row.start_time}}" data-t1="${{row.end_time}}">
        <span class="perf-label">Km ${{row.index}}</span>
        <span class="perf-value">${{formatHms(safeNumber(row.duration_s))}}</span>
        <span class="perf-value">${{formatPace(safeNumber(row.avg_pace_min_per_km))}}</span>
      </div>
    `);
    if (!rows.length) {{
      splitsEl.innerHTML = `<div class="perf-empty">No km splits (distance data missing).</div>`;
    }} else {{
      splitsEl.innerHTML =
        `<div class="perf-row perf-row-header">
          <span class="perf-label">Split</span>
          <span class="perf-value">Time</span>
          <span class="perf-value">Pace</span>
        </div>` + rows.join("");
    }}
  }}

  /* HR bands */
  if (hrEl) {{
    const rows = (perfStats.hr_bands || []).map((row) => `
      <div class="perf-row is-clickable perf-metric-band" data-metric="heart_rate_bpm" data-min="${{row.min_val}}" data-max="${{row.max_val}}">
        <span class="perf-label">${{row.band}}</span>
        <span class="perf-value">${{formatHms(safeNumber(row.time_s))}}</span>
        <span class="perf-value">${{formatKm(safeNumber(row.distance_m))}}</span>
        <span class="perf-value">${{formatPace(safeNumber(row.avg_pace_min_per_km))}}</span>
        <span class="perf-value">${{formatEf(row.ef)}}</span>
      </div>
    `);
    if (!rows.length) {{
      hrEl.innerHTML = `<div class="perf-empty">No HR-band data.</div>`;
    }} else {{
      hrEl.innerHTML =
        `<div class="perf-row perf-row-header">
          <span class="perf-label">Band</span>
          <span class="perf-value">Time</span>
          <span class="perf-value">Distance</span>
          <span class="perf-value">Avg pace</span>
          <span class="perf-value">EF</span>
        </div>` + rows.join("");
    }}
  }}

  /* Cadence bands */
  if (cadEl) {{
    const rows = (perfStats.cadence_bands || []).map((row) => `
      <div class="perf-row is-clickable perf-metric-band" data-metric="cadence" data-min="${{row.min_val}}" data-max="${{row.max_val}}">
        <span class="perf-label">${{row.band}}</span>
        <span class="perf-value">${{formatHms(safeNumber(row.time_s))}}</span>
        <span class="perf-value">${{formatKm(safeNumber(row.distance_m))}}</span>
        <span class="perf-value">${{formatPace(safeNumber(row.avg_pace_min_per_km))}}</span>
        <span class="perf-value">${{formatEf(row.ef)}}</span>
      </div>
    `);
    if (!rows.length) {{
      cadEl.innerHTML = `<div class="perf-empty">No cadence-band data.</div>`;
    }} else {{
      cadEl.innerHTML =
        `<div class="perf-row perf-row-header">
          <span class="perf-label">Band</span>
          <span class="perf-value">Time</span>
          <span class="perf-value">Distance</span>
          <span class="perf-value">Avg pace</span>
          <span class="perf-value">EF</span>
        </div>` + rows.join("");
    }}
  }}

  /* Global EF */
  if (efEl) {{
    const val = perfStats.ef_run;
    efEl.innerHTML = `
      <div class="perf-row perf-row-header" style="cursor:default">
        <span class="perf-label">Run EF (speed/HR, moving)</span>
        <span class="perf-value">Value</span>
      </div>
      <div class="perf-row">
        <span class="perf-label">Entire run</span>
        <span class="perf-value">${{formatEf(val)}}</span>
      </div>
    `;
  }}

  // Bind the Press & Hold event listeners dynamically
  document.querySelectorAll('.perf-time-range').forEach(el => {{
    bindPressAndHold(el, 
      () => highlightTimeRange(el.dataset.t0, el.dataset.t1),
      () => clearHighlight()
    );
  }});

  document.querySelectorAll('.perf-metric-band').forEach(el => {{
    bindPressAndHold(el, 
      () => highlightMetricBand(el.dataset.metric, Number(el.dataset.min), Number(el.dataset.max)),
      () => clearHighlight()
    );
  }});
}}

/* Trays */

function initTrays() {{
  document.querySelectorAll("[data-tray-toggle]").forEach((btn) => {{
    const key = btn.getAttribute("data-tray-toggle");
    const body = document.getElementById(`tray-${{key}}`);
    if (!body) return;
    const chevron = btn.querySelector(".tray-chevron");
    const isOpen = body.classList.contains("is-open");
    btn.setAttribute("aria-expanded", isOpen ? "true" : "false");
    if (chevron) chevron.classList.toggle("is-open", isOpen);

    btn.addEventListener("click", () => {{
      const nowOpen = body.classList.toggle("is-open");
      btn.setAttribute("aria-expanded", nowOpen ? "true" : "false");
      if (chevron) chevron.classList.toggle("is-open", nowOpen);
      if (map) {{
        setTimeout(() => map.invalidateSize(), 160);
      }}
    }});
  }});
}}

    function buildRouteData() {{
      const pts = (Array.isArray(trackPoints) ? trackPoints : [])
        .map((p, idx) => {{
          const lat = safeNumber(p.latitude);
          const lng = safeNumber(p.longitude);
          const timeText = p.time || null;
          const hr = safeNumber(p.heart_rate_bpm);
          const cad = safeNumber(p.cadence);
          const dt = timeText ? new Date(String(timeText).replace(" ", "T")) : null;
          return {{
            index: idx,
            lat,
            lng,
            time: timeText,
            heart_rate_bpm: hr,
            cadence: cad,
            dt: dt instanceof Date && !Number.isNaN(dt.getTime()) ? dt : null
          }};
        }})
        .filter(p => Number.isFinite(p.lat) && Number.isFinite(p.lng));

      let cumulative = 0;
      const firstTimed = pts.find(p => p.dt instanceof Date && !Number.isNaN(p.dt.getTime()));
      const startMs = firstTimed ? firstTimed.dt.getTime() : NaN;

      for (let i = 0; i < pts.length; i += 1) {{
        if (i > 0) cumulative += haversineMeters(pts[i - 1], pts[i]);
        pts[i].cumDist = cumulative;
        pts[i].elapsedS = pts[i].dt && Number.isFinite(startMs) ? Math.max(0, (pts[i].dt.getTime() - startMs) / 1000) : NaN;
      }}

      routeData = pts;
      routeBounds = pts.length ? L.latLngBounds(pts.map(p => [p.lat, p.lng])) : null;
    }}

    function findPointAlongDistance(targetMeters) {{
      if (!Array.isArray(routeData) || routeData.length === 0) return null;
      const target = clamp(targetMeters, 0, routeData[routeData.length - 1].cumDist || 0);
      if (target <= 0) return {{ ...routeData[0] }};
      for (let i = 1; i < routeData.length; i += 1) {{
        const a = routeData[i - 1];
        const b = routeData[i];
        if (target <= b.cumDist) {{
          const span = Math.max(b.cumDist - a.cumDist, 1e-9);
          const t = clamp((target - a.cumDist) / span, 0, 1);
          const timeA = a.dt ? a.dt.getTime() : NaN;
          const timeB = b.dt ? b.dt.getTime() : NaN;
          const interpTime = Number.isFinite(timeA) && Number.isFinite(timeB) ? new Date(lerp(timeA, timeB, t)) : null;
          const elapsedA = safeNumber(a.elapsedS);
          const elapsedB = safeNumber(b.elapsedS);
          const interpElapsed = Number.isFinite(elapsedA) && Number.isFinite(elapsedB) ? lerp(elapsedA, elapsedB, t) : NaN;
          return {{
            lat: lerp(a.lat, b.lat, t),
            lng: lerp(a.lng, b.lng, t),
            cumDist: target,
            dt: interpTime,
            elapsedS: interpElapsed,
            time: interpTime ? interpTime.toISOString() : (a.time || b.time || null),
            segmentIndex: i - 1
          }};
        }}
      }}
      const last = routeData[routeData.length - 1];
      return {{ ...last }};
    }}

    function findPointAlongElapsed(targetSeconds) {{
      if (!Array.isArray(routeData) || routeData.length === 0) return null;
      const timed = routeData.filter(p => Number.isFinite(safeNumber(p.elapsedS)));
      if (!timed.length) return null;
      const totalElapsed = safeNumber(timed[timed.length - 1].elapsedS);
      const target = clamp(targetSeconds, 0, totalElapsed);
      if (target <= 0) return {{ ...timed[0] }};
      for (let i = 1; i < timed.length; i += 1) {{
        const a = timed[i - 1];
        const b = timed[i];
        const ea = safeNumber(a.elapsedS);
        const eb = safeNumber(b.elapsedS);
        if (target <= eb) {{
          const span = Math.max(eb - ea, 1e-9);
          const t = clamp((target - ea) / span, 0, 1);
          const timeA = a.dt ? a.dt.getTime() : NaN;
          const timeB = b.dt ? b.dt.getTime() : NaN;
          const interpTime = Number.isFinite(timeA) && Number.isFinite(timeB) ? new Date(lerp(timeA, timeB, t)) : null;
          return {{
            lat: lerp(a.lat, b.lat, t),
            lng: lerp(a.lng, b.lng, t),
            cumDist: lerp(a.cumDist, b.cumDist, t),
            dt: interpTime,
            elapsedS: target,
            time: interpTime ? interpTime.toISOString() : (a.time || b.time || null),
            segmentIndex: i - 1
          }};
        }}
      }}
      return {{ ...timed[timed.length - 1] }};
    }}

    function getMarkerDistanceStepMeters() {{
      return currentZoom >= 17 ? 500 : 1000;
    }}

    function getTimeMarkerStepSeconds() {{
      return currentZoom >= 17 ? 15 * 60 : 30 * 60;
    }}

    function getArrowSpacingMeters() {{
      if (!routeData.length) return 600;
      const total = routeData[routeData.length - 1].cumDist || 0;
      const base = clamp(total / 10, 350, 1500);
      if (currentZoom >= 17) return Math.max(260, base * 0.75);
      if (currentZoom >= 15) return base;
      return Math.min(2000, base * 1.2);
    }}

    function getBaseLineWeight() {{
      const z = Number.isFinite(safeNumber(currentZoom)) ? currentZoom : (map ? safeNumber(map.getZoom()) : 15);
      if (z <= 11) return 1.4;
      if (z <= 12) return 1.8;
      if (z <= 13) return 2.3;
      if (z <= 14) return 2.9;
      if (z <= 15) return 3.7;
      if (z <= 16) return 4.7;
      if (z <= 17) return 5.8;
      return 6.8;
    }}

    function routeBearingDegrees(a, b) {{
      if (!a || !b) return 0;
      const lat1 = safeNumber(a.lat) * Math.PI / 180;
      const lat2 = safeNumber(b.lat) * Math.PI / 180;
      const dLon = (safeNumber(b.lng) - safeNumber(a.lng)) * Math.PI / 180;
      const y = Math.sin(dLon) * Math.cos(lat2);
      const x = Math.cos(lat1) * Math.sin(lat2) - Math.sin(lat1) * Math.cos(lat2) * Math.cos(dLon);
      const brng = Math.atan2(y, x) * 180 / Math.PI;
      return (brng + 360) % 360;
    }}

    function areLayerPointsFarEnough(candidateLatLng, acceptedLatLngs, minPixels) {{
      if (!map || !acceptedLatLngs.length) return true;
      const candidatePt = map.latLngToLayerPoint(candidateLatLng);
      return acceptedLatLngs.every(existing => candidatePt.distanceTo(map.latLngToLayerPoint(existing)) >= minPixels);
    }}

    function buildStartFinishLayers() {{
      layerGroups.markers.startFinish.clearLayers();
      if (!routeData.length) return;

      const start = routeData[0];
      const finish = routeData[routeData.length - 1];

      const startMarker = L.marker([start.lat, start.lng], {{
        icon: L.divIcon({{
          className: "",
          html: '<div class="marker-dot start"></div>',
          iconSize: [12, 12],
          iconAnchor: [6, 6]
        }})
      }}).bindTooltip("Start", {{ direction: "top", offset: [0, -6], sticky: true }});

      const finishMarker = L.marker([finish.lat, finish.lng], {{
        icon: L.divIcon({{
          className: "",
          html: '<div class="marker-dot finish"></div>',
          iconSize: [12, 12],
          iconAnchor: [6, 6]
        }})
      }}).bindTooltip("Finish", {{ direction: "top", offset: [0, -6], sticky: true }});

      layerGroups.markers.startFinish.addLayer(startMarker);
      layerGroups.markers.startFinish.addLayer(finishMarker);
    }}

    function buildKmMarkerLayers() {{
      layerGroups.markers.km.clearLayers();
      if (!routeData.length) return;

      const total = routeData[routeData.length - 1].cumDist || 0;
      const step = getMarkerDistanceStepMeters();
      const acceptedLatLngs = [];
      const maxCount = 200;

      for (let dist = step, count = 0; dist <= total + 1e-6 && count < maxCount; dist += step) {{
        const point = findPointAlongDistance(dist);
        if (!point) continue;
        const latLng = L.latLng(point.lat, point.lng);
        if (!areLayerPointsFarEnough(latLng, acceptedLatLngs, currentZoom >= 17 ? 18 : 22)) continue;

        const isHalf = step === 500 && Math.round(dist) % 1000 !== 0;
        const label = isHalf ? `${{(dist / 1000).toFixed(1)}}` : `${{Math.round(dist / 1000)}}`;
        const tooltipLabel = isHalf ? `${{(dist / 1000).toFixed(1)}} km` : `${{Math.round(dist / 1000)}} km`;

        const marker = L.marker([point.lat, point.lng], {{
          icon: L.divIcon({{
            className: "",
            html: `<div class="route-badge">${{label}}</div>`,
            iconSize: [isHalf ? 32 : 24, 24],
            iconAnchor: [isHalf ? 16 : 12, 12]
          }})
        }}).bindTooltip(tooltipLabel, {{ direction: "top", offset: [0, -8], sticky: true }});

        acceptedLatLngs.push(latLng);
        layerGroups.markers.km.addLayer(marker);
      }}
    }}

    function buildTimeMarkerTargets() {{
      const timed = routeData.filter(p => Number.isFinite(safeNumber(p.elapsedS)));
      if (!timed.length) return [];
      const totalElapsed = safeNumber(timed[timed.length - 1].elapsedS);
      const step = getTimeMarkerStepSeconds();
      const targets = [0];

      for (let t = step; t < totalElapsed; t += step) {{
        targets.push(t);
      }}

      if (totalElapsed > 0) targets.push(totalElapsed);
      return targets;
    }}

    function buildTimeMarkerLayers() {{
      layerGroups.markers.time.clearLayers();
      if (!routeData.length) return;

      const targets = buildTimeMarkerTargets();
      const acceptedLatLngs = [];
      const minPixels = currentZoom >= 17 ? 34 : 42;

      targets.forEach((elapsedS, idx) => {{
        const point = findPointAlongElapsed(elapsedS);
        if (!point) return;
        const latLng = L.latLng(point.lat, point.lng);
        const isBoundary = idx === 0 || idx === targets.length - 1;
        if (!isBoundary && !areLayerPointsFarEnough(latLng, acceptedLatLngs, minPixels)) return;
        if (isBoundary && !areLayerPointsFarEnough(latLng, acceptedLatLngs, 24)) return;

        const label = formatElapsedLabel(elapsedS);
        const marker = L.marker([point.lat, point.lng], {{
          icon: L.divIcon({{
            className: "",
            html: `<div class="route-badge time">${{label}}</div>`,
            iconSize: [48, 22],
            iconAnchor: [24, 11]
          }})
        }}).bindTooltip(`Elapsed ${{label}}`, {{ direction: "top", offset: [0, -8], sticky: true }});

        acceptedLatLngs.push(latLng);
        layerGroups.markers.time.addLayer(marker);
      }});
    }}

    function buildDirectionArrowLayers() {{
      layerGroups.arrows.clearLayers();
      if (!routeData.length || !arrowVisibility || !map) return;

      const total = routeData[routeData.length - 1].cumDist || 0;
      if (total < 250) return;

      const spacing = getArrowSpacingMeters();
      const margin = Math.min(spacing * 0.45, 250);
      const targets = [];
      for (let d = margin; d < total - margin; d += spacing) targets.push(d);

      const acceptedLatLngs = [];
      targets.forEach(target => {{
        const point = findPointAlongDistance(target);
        const behind = findPointAlongDistance(Math.max(0, target - Math.max(20, spacing * 0.12)));
        const ahead = findPointAlongDistance(Math.min(total, target + Math.max(20, spacing * 0.12)));
        if (!point || !ahead || !behind) return;

        const latLng = L.latLng(point.lat, point.lng);
        if (!areLayerPointsFarEnough(latLng, acceptedLatLngs, currentZoom >= 17 ? 24 : 32)) return;

        const angle = routeBearingDegrees(behind, ahead);
        const cssAngle = angle;
        const marker = L.marker([point.lat, point.lng], {{
          interactive: false,
          keyboard: false,
          icon: L.divIcon({{
            className: "",
            html: `<div class="direction-arrow" style="transform: rotate(${{cssAngle}}deg)"></div>`,
            iconSize: [26, 26],
            iconAnchor: [13, 13]
          }})
        }});

        acceptedLatLngs.push(latLng);
        layerGroups.arrows.addLayer(marker);
      }});
    }}

    function buildMapAnnotations() {{
      buildStartFinishLayers();
      buildKmMarkerLayers();
      buildTimeMarkerLayers();
      buildDirectionArrowLayers();
    }}

    function syncAnnotationLayers() {{
      if (!map) return;

      buildKmMarkerLayers();
      buildTimeMarkerLayers();

      if (!map.hasLayer(layerGroups.markers.startFinish)) map.addLayer(layerGroups.markers.startFinish);

      if (markerVisibility.km) {{
        if (!map.hasLayer(layerGroups.markers.km)) map.addLayer(layerGroups.markers.km);
      }} else if (map.hasLayer(layerGroups.markers.km)) {{
        map.removeLayer(layerGroups.markers.km);
      }}

      if (markerVisibility.time) {{
        if (!map.hasLayer(layerGroups.markers.time)) map.addLayer(layerGroups.markers.time);
      }} else if (map.hasLayer(layerGroups.markers.time)) {{
        map.removeLayer(layerGroups.markers.time);
      }}

      if (arrowVisibility) {{
        buildDirectionArrowLayers();
        if (!map.hasLayer(layerGroups.arrows)) map.addLayer(layerGroups.arrows);
      }} else if (map.hasLayer(layerGroups.arrows)) {{
        map.removeLayer(layerGroups.arrows);
      }}
    }}

    function ensureBasemap() {{
      if (!map) return;
      if (currentTileLayer && map.hasLayer(currentTileLayer)) {{
        map.removeLayer(currentTileLayer);
      }}
      currentTileLayer = null;
      const def = BASEMAP_DEFS[activeBasemap];
      if (!def || !def.url) return;
      currentTileLayer = L.tileLayer(def.url, def.options || {{}});
      currentTileLayer.addTo(map);
    }}

    function renderButtons() {{
      const motionEl = document.getElementById("motionTypeButtons");
      motionEl.innerHTML = MOTION_TYPES.map(def => {{
        const isActive = activeMotionTypes.has(def.key);
        const styleColor = baseMotionColor(def.key);
        return `<div class="motion-type-item">
          <button type="button" class="btn motion-btn ${{isActive ? "is-active" : "is-inactive"}}" data-motion="${{def.key}}">${{def.label}}</button>
          <span class="legend-line-inline ${{def.className}}" style="border-top-color:${{styleColor}}"></span>
        </div>`;
      }}).join("");
      motionEl.querySelectorAll("[data-motion]").forEach(btn =>
        btn.addEventListener("click", () => {{
          const key = btn.dataset.motion;
          if (activeMotionTypes.has(key)) activeMotionTypes.delete(key);
          else activeMotionTypes.add(key);
          renderButtons();
          applyStyles();
        }})
      );

      const markerEl = document.getElementById("markerButtons");
      markerEl.innerHTML = MARKER_DEFS.map(def =>
        `<button type="button" class="btn ${{markerVisibility[def.key] ? "is-active" : ""}}" data-marker="${{def.key}}">${{def.label}}</button>`
      ).join("");
      markerEl.querySelectorAll("[data-marker]").forEach(btn =>
        btn.addEventListener("click", () => {{
          const key = btn.dataset.marker;
          markerVisibility[key] = !markerVisibility[key];
          renderButtons();
          syncAnnotationLayers();
        }})
      );

      const directionEl = document.getElementById("directionButtons");
      directionEl.innerHTML = `<button type="button" class="btn ${{arrowVisibility ? "is-active" : ""}}" data-arrow="directionArrows">${{ARROW_DEF.label}}</button>`;
      directionEl.querySelectorAll("[data-arrow]").forEach(btn =>
        btn.addEventListener("click", () => {{
          arrowVisibility = !arrowVisibility;
          renderButtons();
          syncAnnotationLayers();
        }})
      );

      const basemapEl = document.getElementById("basemapButtons");
      basemapEl.innerHTML = Object.entries(BASEMAP_DEFS).map(([key, def]) =>
        `<button type="button" class="btn ${{key === activeBasemap ? "is-active" : ""}}" data-basemap="${{key}}">${{def.label}}</button>`
      ).join("");
      basemapEl.querySelectorAll("[data-basemap]").forEach(btn =>
        btn.addEventListener("click", () => {{
          activeBasemap = btn.dataset.basemap;
          renderButtons();
          ensureBasemap();
        }})
      );

      const mb = document.getElementById("metricButtons");
      mb.innerHTML = Object.values(METRIC_DEFS).map(def =>
        `<button type="button" class="btn ${{def.key === activeMetric ? "is-active" : ""}}" data-metric="${{def.key}}">${{def.label}}</button>`
      ).join("");
      mb.querySelectorAll("[data-metric]").forEach(btn =>
        btn.addEventListener("click", () => {{
          activeMetric = btn.dataset.metric;
          activeBinIndex = null;
          renderButtons();
          applyStyles();
          renderMetricDistribution();
        }})
      );

      const tb = document.getElementById("themeButtons");
      tb.innerHTML = Object.keys(THEME_DEFS).map(k =>
        `<button type="button" class="btn ${{k === activeTheme ? "is-active" : ""}}" data-theme="${{k}}">${{k}}</button>`
      ).join("");
      tb.querySelectorAll("[data-theme]").forEach(btn =>
        btn.addEventListener("click", () => {{
          activeTheme = btn.dataset.theme;
          renderButtons();
          applyStyles();
          renderMetricDistribution();
        }})
      );
    }}

    function segmentMatchesActiveBin(seg) {{
      if (activeMetric === "none" || activeBinIndex === null) return true;
      const stats = metricStats ? metricStats[activeMetric] : null;
      if (!stats) return false;
      const edges = Array.isArray(stats.hist_edges) ? stats.hist_edges : [];
      const v = metricValue(seg);
      if (!Number.isFinite(v) || edges.length < 2) return false;
      const lo = safeNumber(edges[activeBinIndex]);
      const hi = safeNumber(edges[activeBinIndex + 1]);
      if (!Number.isFinite(lo) || !Number.isFinite(hi)) return false;
      if (activeBinIndex === edges.length - 2) return v >= lo && v <= hi;
      return v >= lo && v < hi;
    }}

    function initMap() {{
      if (map) return;
      const mapEl = document.getElementById("map");
      if (!mapEl) {{
        console.error("Map container #map not found");
        return;
      }}

      buildRouteData();

      map = L.map("map", {{ zoomControl: true, preferCanvas: true, worldCopyJump: false }}).setView(center, 15);
      currentZoom = map.getZoom();

      Object.values(layerGroups.motion).forEach(group => group.addTo(map));
      layerGroups.markers.startFinish.addTo(map);
      
      highlightLayerGroup.addTo(map);

      ensureBasemap();

      const allCoords = [];
      segments.forEach((seg, idx) => {{
        seg.index = idx;
        const coords = Array.isArray(seg.coords) ? seg.coords : [];
        if (!coords.length) return;
        allCoords.push(...coords);

        const motionKey = String(seg.label || "").toLowerCase();
        const parentGroup = layerGroups.motion[motionKey] || layerGroups.motion.running;

        const layer = L.polyline(coords, {{
          color: baseMotionColor(motionKey),
          weight: getBaseLineWeight(),
          opacity: 0.92,
          dashArray: seg.dashArray || null,
          lineCap: "round",
          lineJoin: "round"
        }});

        layer.bindTooltip(buildTooltip(seg), {{ sticky: true }});
        layer.addTo(parentGroup);
        seg.layer = layer;
      }});

      buildMapAnnotations();
      syncAnnotationLayers();

      if (routeBounds && routeBounds.isValid()) map.fitBounds(routeBounds, {{ padding: [24, 24] }});
      else if (allCoords.length) map.fitBounds(allCoords, {{ padding: [24, 24] }});
      else map.setView(center, 15);

      currentZoom = map.getZoom();
      applyStyles();
      syncAnnotationLayers();

      map.on("zoomend", () => {{
        currentZoom = map.getZoom();
        applyStyles();
        syncAnnotationLayers();
      }});

      requestAnimationFrame(() => {{ if (map) map.invalidateSize(); }});
      setTimeout(() => {{ if (map) map.invalidateSize(); }}, 150);
    }}

    function applyStyles() {{
      const baseWeight = getBaseLineWeight();

      segments.forEach(seg => {{
        if (!seg.layer) return;
        const motionKey = String(seg.label || "").toLowerCase();
        const motionVisible = activeMotionTypes.has(motionKey);
        const v = metricValue(seg);
        const metricColor = getMetricColor(v);
        const color = metricColor || baseMotionColor(motionKey);
        const histVisible = segmentMatchesActiveBin(seg);
        const visible = motionVisible && histVisible;

        seg.layer.setStyle({{
          color,
          opacity: visible ? 0.95 : 0,
          weight: visible ? baseWeight : Math.max(1.1, baseWeight * 0.65),
          dashArray: seg.dashArray || null
        }});
      }});
    }}

    function renderMetricDistribution() {{
      const histogramEl = document.getElementById("metricHistogram");
      const axisEl = document.getElementById("metricAxis");
      const listEl = document.getElementById("metricSegmentList");

      if (activeMetric === "none") {{
        histogramEl.innerHTML = "";
        axisEl.innerHTML = "<span>Select an overlay metric to view the histogram and segment list.</span><span></span>";
        listEl.innerHTML = segments.map(seg =>
          `<div class="segment-row">
            <div class="segment-swatch" style="background:${{baseMotionColor(seg.label)}}"></div>
            <div class="segment-main">${{formatTimeOnly(seg.start_time)}} - ${{formatTimeOnly(seg.end_time)}}</div>
            <div class="segment-meta">${{seg.label}}</div>
            <div class="segment-meta">${{format3DigitInt(seg.avg_hr_bpm)}} bpm</div>
            <div class="segment-meta">${{format3DigitInt(seg.avg_cadence_spm)}} spm</div>
          </div>`
        ).join("");
        return;
      }}

      const stats = metricStats ? metricStats[activeMetric] : null;
      const def = METRIC_DEFS[activeMetric];
      if (!stats || !def) {{
        histogramEl.innerHTML = "";
        axisEl.innerHTML = "<span>No metric data</span><span></span>";
        listEl.innerHTML = "";
        return;
      }}

      const edges = Array.isArray(stats.hist_edges) ? stats.hist_edges : [];
      const weights = Array.isArray(stats.hist_weights) ? stats.hist_weights : [];
      const maxW = Math.max(1, ...weights.map(w => safeNumber(w)).filter(Number.isFinite));

      histogramEl.innerHTML = weights.map((w, i) => {{
        const lo = safeNumber(edges[i]);
        const hi = safeNumber(edges[i + 1]);
        const mid = Number.isFinite(lo) && Number.isFinite(hi) ? (lo + hi) / 2 : NaN;
        const color = getMetricColor(mid) || "#94a3b8";
        const h = Math.max(12, Math.round((safeNumber(w) / maxW) * 132));
        const activeCls = activeBinIndex === i ? "active" : "";
        const title = `${{def.fmt(lo)}} - ${{def.fmt(hi)}} · ${{(safeNumber(w) / 1000).toFixed(2)}} km`;
        return `<button type="button" class="hist-bar ${{activeCls}}" data-bin="${{i}}" title="${{title}}" style="height:${{h}}px;background:${{color}}"></button>`;
      }}).join("");

      axisEl.innerHTML = `<span>${{def.fmt(safeNumber(stats.min))}}</span><span>${{def.fmt(safeNumber(stats.max))}}</span>`;

      histogramEl.querySelectorAll("[data-bin]").forEach(btn =>
        btn.addEventListener("click", () => {{
          const idx = Number(btn.dataset.bin);
          activeBinIndex = activeBinIndex === idx ? null : idx;
          applyStyles();
          renderMetricDistribution();
        }})
      );

      const rows = segments
        .filter(seg => Number.isFinite(metricValue(seg)))
        .map(seg => {{
          const v = metricValue(seg);
          const color = getMetricColor(v) || baseMotionColor(seg.label);
          const highlighted = segmentMatchesActiveBin(seg);
          return {{ seg, v, color, highlighted }};
        }});

      listEl.innerHTML = rows.map(({{ seg, v, color, highlighted }}) =>
       `<div class="segment-row ${{highlighted ? "is-highlighted" : ""}}">
         <div class="segment-swatch" style="background:${{color}}"></div>
         <div class="segment-main">${{formatTimeOnly(seg.start_time)}} - ${{formatTimeOnly(seg.end_time)}}</div>
         <div class="segment-meta">${{seg.label}}</div>
         <div class="segment-meta">${{def.fmt(v)}}</div>
         <div class="segment-meta">${{formatKm(safeNumber(seg.distance_m))}}</div>
       </div>`
     ).join("");
    }}

    function enableResizer() {{
      const app = document.getElementById("app");
      const sidebar = document.getElementById("sidebar");
      const resizer = document.getElementById("resizer");
      if (!app || !sidebar || !resizer) return;

      let active = false;
      resizer.addEventListener("mousedown", () => {{
        active = true;
        document.body.style.userSelect = "none";
      }});

      window.addEventListener("mousemove", (ev) => {{
        if (!active) return;
        const minW = 300;
        const maxW = Math.min(700, Math.floor(window.innerWidth * 0.6));
        const nextW = clamp(ev.clientX, minW, maxW);
        app.style.setProperty("--sidebar-width", nextW + "px");
        if (map) map.invalidateSize();
      }});

      window.addEventListener("mouseup", () => {{
        active = false;
        document.body.style.userSelect = "";
        setTimeout(() => {{ if (map) map.invalidateSize(); }}, 50);
      }});
    }}

function boot() {{
  try {{ renderRunSummary(); }} catch (e) {{ console.error("renderRunSummary failed", e); }}
  try {{ renderButtons(); }} catch (e) {{ console.error("renderButtons failed", e); }}
  try {{ initTrays(); }} catch (e) {{ console.error("initTrays failed", e); }}
  try {{ renderPerformanceStats(); }} catch (e) {{ console.error("renderPerformanceStats failed", e); }}
  try {{ initMap(); }} catch (e) {{ console.error("initMap failed", e); }}
  try {{ applyStyles(); }} catch (e) {{ console.error("applyStyles failed", e); }}
  try {{ renderMetricDistribution(); }} catch (e) {{ console.error("renderMetricDistribution failed", e); }}
  try {{ enableResizer(); }} catch (e) {{ console.error("enableResizer failed", e); }}
}}

if (document.readyState === "loading") {{
  document.addEventListener("DOMContentLoaded", boot, {{ once: true }});
}} else {{
  boot();
}}

window.addEventListener("load", () => {{ if (map) map.invalidateSize(); }});
window.addEventListener("resize", () => {{
  if (map) {{
    currentZoom = map.getZoom();
    map.invalidateSize();
  }}
}});
</script>
</body>
</html>
"""
    if outpath is not None:
            Path(outpath).write_text(html, encoding="utf-8")
            
    return html



@contextmanager
def measure_time(step_name: str, timings: dict, enabled: bool):
    """Context manager to optionally record execution time of code blocks."""
    if not enabled:
        yield
        return
    t0 = time.perf_counter()
    yield
    t1 = time.perf_counter()
    timings[step_name] = t1 - t0

def main():
    args = parse_args()
    tcx_path = Path(args.tcx_file)
    prefix = Path(args.prefix) if args.prefix else tcx_path.with_suffix("")
    csv_out = Path(f"{prefix}.csv")
    seg_out = Path(f"{prefix}.segments.runwalkstop.csv")
    map_out = Path(args.map_out) if args.map_out else Path(f"{prefix}.motionmap.html")

    is_bench = args.benchmark
    timings = {}
    total_start = time.perf_counter()

    with measure_time("1. Parse TCX to Dictionary", timings, is_bench):
        rows = list(parse_tcx_to_rows(tcx_path))
    
    with measure_time("2. Write Initial CSV to Disk", timings, is_bench):
        write_csv(rows, csv_out)

    with measure_time("3. Read CSV to DataFrame", timings, is_bench):
        raw_run_df = pd.read_csv(csv_out)
        
    if raw_run_df.empty:
        raise ValueError("Run dataframe is empty after parsing")

    with measure_time("4. Prepare & Smooth Data", timings, is_bench):
        run_df = prepare_run_df(raw_run_df)
        run_df = add_deltas(run_df)
        run_df = add_smoothed_speed(run_df)
        
        # EXTRACT TIMEZONE EARLY
        temp_plot_df = raw_run_df.copy()
        if "latitude" in temp_plot_df.columns:
            temp_plot_df["latitude"] = pd.to_numeric(temp_plot_df["latitude"], errors="coerce")
        if "longitude" in temp_plot_df.columns:
            temp_plot_df["longitude"] = pd.to_numeric(temp_plot_df["longitude"], errors="coerce")
        temp_plot_df = temp_plot_df.dropna(subset=["latitude", "longitude"]) if {"latitude", "longitude"}.issubset(temp_plot_df.columns) else pd.DataFrame()
        tz_name = infer_activity_timezone_name(temp_plot_df)

    with measure_time("5. Identify Motion Segments", timings, is_bench):
        motion_segments_df = summarize_motion_segments(run_df)

    with measure_time("6. Write Segments CSV to Disk", timings, is_bench):
        motion_segments_csv = prepare_for_csv(
            motion_segments_df,
            time_cols=["start_time", "end_time"],
            round_decimals=DISPLAY_DECIMALS,
            tz_name=tz_name,
        )
        motion_segments_csv.to_csv(seg_out, index=False)

    with measure_time("7. Compute Stats & Build HTML Payload", timings, is_bench):
        perfstats = compute_performance_stats(run_df, tz_name=tz_name)
        
        # FIX: Pass the fully prepared run_df with corrected cadence instead of raw_run_df
        run_df_collapsed = collapse_run_streams_for_map(run_df, tz_name=tz_name)
        lookup = build_lookup(run_df_collapsed)

        seg_df = motion_segments_csv.copy()
        if seg_df.empty:
            raise ValueError("No motion segments were produced")

        seg_df_enriched = enrich_segments(seg_df, lookup)
        segments, plot_df = build_segments_payload(run_df_collapsed, seg_df_enriched)
        if not segments:
            raise ValueError("No segment geometry available for map output")

        metricstats = compute_metric_stats(seg_df_enriched)
        runstats = compute_run_stats(run_df, seg_df_enriched, tz_name)

    with measure_time("8. Generate & Write HTML to Disk", timings, is_bench):
        write_html(map_out, segments, plot_df, metricstats, runstats, perfstats)

    total_time = time.perf_counter() - total_start

    print(csv_out)
    print(seg_out)
    print(map_out)

    if is_bench:
        print("\n" + "="*50)
        print("⏱️  PERFORMANCE BENCHMARK")
        print("="*50)
        print(f"Total Trackpoints: {len(raw_run_df):,}")
        print(f"Motion Segments:   {len(motion_segments_df):,}")
        print("-" * 50)
        for step, elapsed in timings.items():
            print(f"{step:<40} {elapsed:.4f}s")
        print("-" * 50)
        print(f"{'TOTAL EXECUTION TIME':<40} {total_time:.4f}s")
        print("="*50 + "\n")

if __name__ == "__main__":
    main()
