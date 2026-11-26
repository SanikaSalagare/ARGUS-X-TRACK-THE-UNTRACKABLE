"""Data preparation for the Analytics page (pandas only, so it can be tested without Streamlit/Plotly)."""
import re

import pandas as pd

SEG_COLS = ["argus_id", "start", "length", "recovered", "end"]


def parse_recovery(detail):
    """'tracker 7, similarity 0.91, gone 43 frames' -> (0.91, 43)."""
    sim = re.search(r"similarity ([0-9.]+)", detail or "")
    gone = re.search(r"gone (\d+) frames", detail or "")
    return (float(sim.group(1)) if sim else None, int(gone.group(1)) if gone else None)


def recovery_table(events):
    rows = []
    for e in events:
        if e["kind"] == "RECOVERED":
            sim, gone = parse_recovery(e["detail"])
            rows.append(dict(frame=e["frame"], argus_id=e["argus_id"], similarity=sim, gone_frames=gone))
    return pd.DataFrame(rows, columns=["frame", "argus_id", "similarity", "gone_frames"])


def timeline_segments(tr, ids, max_ids=40):
    """One row per continuous visible stretch of an identity (a gap between rows = not visible).
    Only the `max_ids` longest-seen identities are kept so the chart stays readable."""
    if tr.empty or ids.empty:
        return pd.DataFrame(columns=SEG_COLS)
    top = ids.sort_values("frames_seen", ascending=False).head(max_ids)
    recovered = set(top.loc[top["reid_count"] > 0, "argus_id"])
    rows = []
    for aid, g in tr[tr["argus_id"].isin(set(top["argus_id"]))].groupby("argus_id"):
        frames = sorted(set(int(f) for f in g["frame"]))
        start = prev = frames[0]
        for f in frames[1:]:
            if f != prev + 1:
                rows.append((aid, start, prev - start + 1, aid in recovered, prev))
                start = f
            prev = f
        rows.append((aid, start, prev - start + 1, aid in recovered, prev))
    return pd.DataFrame(rows, columns=SEG_COLS)


def activity(tr, frames, window=15):
    """Objects visible per frame plus a rolling mean."""
    if tr.empty:
        return pd.DataFrame(columns=["frame", "objects", "smooth"])
    n = max(int(frames or 0), int(tr["frame"].max()))
    s = tr.groupby("frame").size().reindex(range(1, n + 1), fill_value=0)
    return pd.DataFrame({"frame": s.index, "objects": s.values,
                         "smooth": s.rolling(window, min_periods=1, center=True).mean().round(2).values})


def footpoints(tr):
    """Where objects stood (bottom-centre of each box), for the movement heatmap."""
    if tr.empty:
        return pd.DataFrame(columns=["x", "y"])
    return pd.DataFrame({"x": (tr["x1"] + tr["x2"]) / 2, "y": tr["y2"]})


def summary(ids, events, tr):
    """Headline numbers for the KPI cards."""
    rec = sum(e["kind"] == "RECOVERED" for e in events)
    out = dict(identities=len(ids), recoveries=rec, detections=len(tr), peak=0, avg_visible=0.0,
               longest_id="-", longest_frames=0, recovered_people=0)
    if not ids.empty:
        out["avg_visible"] = round(float(ids["frames_seen"].mean()), 1)
        top = ids.sort_values("frames_seen", ascending=False).iloc[0]
        out["longest_id"], out["longest_frames"] = top["argus_id"], int(top["frames_seen"])
        out["recovered_people"] = int((ids["reid_count"] > 0).sum())
    if not tr.empty:
        out["peak"] = int(tr.groupby("frame").size().max())
    return out
