"""ARGUS-X dashboard v0.3 (Streamlit). Run from the project root: streamlit run frontend/app.py"""
import json
import platform
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from backend.app.core.argus_engine import ArgusEngine
from backend.app.core.video_job import VideoJob, probe
from backend.app.database import db
try:
    from frontend import analytics_data as ad
except ImportError:  # script folder is on sys.path when run by Streamlit
    import analytics_data as ad

VERSION = "0.3"
st.set_page_config(page_title="ARGUS-X", page_icon="👁️", layout="wide")
db.init()
st.markdown("""<style>
[data-testid="stMetric"]{background:#151E29;border:1px solid #263241;border-radius:10px;padding:12px 16px}
[data-testid="stSidebar"]{background:#111821;border-right:1px solid #263241}
.stButton>button{border:1px solid #263241}
.kpi{background:linear-gradient(160deg,#18222F,#111821);border:1px solid #263241;border-radius:12px;padding:14px 16px;min-height:112px}
.kpi .kl{font-size:.78rem;color:#8FA3BA;text-transform:uppercase;letter-spacing:.06em}
.kpi .kv{font-size:2rem;font-weight:700;color:#F3F6FA;line-height:1.25}
.kpi .ks{font-size:.78rem;color:#8FA3BA}
[data-baseweb="tab-list"]{gap:6px}
</style>""", unsafe_allow_html=True)

S = st.session_state
SET_PATH = ROOT / "outputs" / "settings.json"
DEFAULTS = dict(missing_after=5, archive_after=300, model="yolo26n.pt", device="auto", tracker="bytetrack.yaml",
                track_buffer=60, confirm_frames=4)
CLASSES = {"person": 0, "bicycle": 1, "car": 2, "motorcycle": 3, "bus": 5, "truck": 7}
PAGES = ["Dashboard", "Video & Camera", "Live Tracking", "Object Manager", "Identity History", "Analytics", "Settings"]


def load_settings():
    try:
        return {**DEFAULTS, **json.loads(SET_PATH.read_text())}
    except Exception:
        return dict(DEFAULTS)


def pretty(fig, height=360):
    """Shared dark chart style for the Analytics page."""
    fig.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                      height=height, margin=dict(l=10, r=10, t=55, b=10), font=dict(color="#C9D4E3"),
                      title_font_size=16, legend=dict(orientation="h", y=-0.18))
    fig.update_xaxes(gridcolor="rgba(255,255,255,0.06)", zeroline=False)
    fig.update_yaxes(gridcolor="rgba(255,255,255,0.06)", zeroline=False)
    return fig


S.setdefault("job", None)
S.setdefault("cfg", load_settings())


# ---------------------------------------------------------------- helpers
def settings_panel(k):
    st.subheader("Processing settings")
    conf = st.slider("Confidence threshold", 0.1, 0.9, 0.3, 0.05, key=k + "conf")
    imgsz = st.select_slider("Image size", [320, 480, 640, 800, 960], 640, key=k + "sz")
    maxf = st.number_input("Maximum frames (0 = all)", 0, 200000, 0, key=k + "mf")
    cls = st.multiselect("Object classes", list(CLASSES), ["person"], key=k + "cls")
    reid = st.toggle("Appearance re-identification", True, key=k + "reid",
                     help="Gives a person who disappears and comes back the same ARGUS ID.")
    appearance = st.selectbox("Appearance model", ["colour", "neural"], key=k + "app", disabled=not reid,
                              help="colour = HSV histograms. neural = MobileNetV3 ImageNet features (downloads ~10 MB once). "
                                   "Neither is a person-ReID-trained network: tune the threshold on your own clip.")
    thr = st.slider("Similarity threshold", 0.5, 0.99, 0.80, 0.01, key=k + "thr", disabled=not reid,
                    help="Lower = recognises people more often but can merge look-alikes. Higher = safer, misses more.")
    window = st.slider("Re-link window (frames a person may be gone)", 20, 300, 90, 10, key=k + "win",
                       disabled=not reid)
    gate = st.toggle("Location gating", True, key=k + "gate", disabled=not reid,
                     help="Reject matches that are too far from where the object was last seen.")
    return dict(conf=conf, imgsz=imgsz, max_frames=maxf, classes=[CLASSES[c] for c in cls] or None, reid=reid,
                thr=thr, appearance=appearance, gate=gate, window=window)


def start_job(source, run, kind, label):
    if S.job is not None and S.job.is_alive():
        st.rerun()  # duplicate click while a session is running
    c = S.cfg
    mp = ROOT / "models" / c["model"]
    with st.spinner("Loading model..."):
        eng = ArgusEngine(str(mp) if mp.exists() else c["model"], run["conf"], run["imgsz"], run["classes"],
                          run["reid"], run["thr"], c["missing_after"], c["archive_after"], c["tracker"],
                          run["appearance"], run["gate"], c["device"], db.max_argus_no() + 1,
                          confirm_frames=c["confirm_frames"], reid_window=run["window"],
                          track_buffer=c["track_buffer"])
    S.job = VideoJob(source, eng, ROOT / "outputs", run["max_frames"], kind, label)
    S.job.start()
    st.rerun()


def pick_session(key):
    ss = db.sessions()
    if not ss:
        return None
    labels = [f"#{s['id']} - {s['source']} ({s['started']})" for s in ss]
    return ss[labels.index(st.selectbox("Session", labels, key=key))]


def current_rows():
    j = S.job
    if j is not None and j.is_alive():
        return j.engine.snapshot(), j.engine.events
    ss = db.sessions()
    if ss:
        return db.identities(ss[0]["id"]), db.events(ss[0]["id"])
    return [], []


def thumbs(sid, aid):
    d = ROOT / "outputs" / "thumbs" / f"s{sid}"
    return sorted(d.glob(f"{aid}_*.jpg")) if d.exists() else []


@st.fragment(run_every=0.5)
def live_view():
    j = S.job
    if j is None or not j.is_alive():
        st.rerun()
        return
    if j.latest is not None:
        st.image(j.latest, channels="RGB", output_format="JPEG", width="stretch")
    if j.total:
        st.progress(min(j.frame_no / j.total, 1.0), f"Frame {j.frame_no} / {j.total}  |  {j.fps_proc} FPS")
    else:
        st.caption(f"Frames processed: {j.frame_no}  |  {j.fps_proc} FPS")
    c = j.engine.counts()
    a, b, d, e = st.columns(4)
    a.metric("Active", c["ACTIVE"])
    b.metric("Missing", c["MISSING"])
    d.metric("Archived", c["ARCHIVED"])
    e.metric("Recovered", j.engine.recovered)


def job_result(kind):
    j = S.job
    if j is None or j.kind != kind:
        return
    if j.error:
        st.error(j.error)
    if not j.frame_no:
        return
    ev = j.engine.events
    st.success(f"Session {j.state}.")
    m = st.columns(5)
    m[0].metric("Frames", j.frame_no)
    m[1].metric("Identities", len(j.engine.ids))
    m[2].metric("Recovered", sum(e["kind"] == "RECOVERED" for e in ev))
    m[3].metric("Duration (s)", j.seconds)
    m[4].metric("Avg FPS", j.fps_proc)
    if j.output and Path(j.output).exists():
        st.video(j.output)
        d1, d2 = st.columns(2)
        with open(j.output, "rb") as f:
            d1.download_button("Download annotated video", f, file_name=Path(j.output).name, mime="video/mp4")
        if j.session_id:
            csv = pd.DataFrame(db.tracks(j.session_id)).to_csv(index=False).encode()
            d2.download_button("Download tracking CSV", csv, file_name=f"argus_tracks_s{j.session_id}.csv",
                               mime="text/csv")
        st.caption(f"Saved to: {j.output}")


# ---------------------------------------------------------------- sidebar
running = S.job is not None and S.job.is_alive()
with st.sidebar:
    st.markdown("## 👁️ ARGUS-X")
    st.caption("Track the Untrackable")
    page = st.radio("Navigation", PAGES, label_visibility="collapsed")
    st.divider()
    try:
        import torch
        dev = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
    except Exception:
        dev = "CPU (torch not found)"
    st.caption(f"Status: {'🟢 processing' if running else '⚪ idle'}")
    st.caption(f"Device: {dev}")
    st.caption(f"Version {VERSION}")

# ---------------------------------------------------------------- pages
if page == "Dashboard":
    st.title("Dashboard")
    st.caption("Detect an object. Track it. Remember it. Recognize it again.")
    rows, events = current_rows()
    ss = db.sessions()
    cnt = {k: sum(r["status"] == k for r in rows) for k in ("ACTIVE", "MISSING", "ARCHIVED")}
    frames = S.job.frame_no if S.job is not None else (ss[0]["frames"] if ss else 0)
    for col, (lab, val) in zip(st.columns(6), [("Total identities", len(rows)), ("Active", cnt["ACTIVE"]),
                                               ("Missing", cnt["MISSING"]), ("Archived", cnt["ARCHIVED"]),
                                               ("Frames processed", frames), ("Total sessions", len(ss))]):
        col.metric(lab, val)
    a, b = st.columns(2)
    with a:
        st.subheader("Active session")
        if running:
            st.info(f"Processing {S.job.label or S.job.source}: frame {S.job.frame_no}, {S.job.fps_proc} FPS")
        else:
            st.caption("No session is running.")
        st.subheader("System performance")
        last = ss[0] if ss else None
        if last and last.get("seconds"):
            st.write(f"Last session: {last['frames']} frames in {last['seconds']} s "
                     f"({round(last['frames'] / last['seconds'], 1)} FPS) on {dev}")
        else:
            st.caption("Run a session to see speed figures.")
        st.subheader("Object categories")
        if rows:
            cdf = pd.DataFrame(rows).groupby("cls").size().reset_index(name="count")
            st.plotly_chart(px.bar(cdf, x="cls", y="count"), width="stretch")
        else:
            st.caption("No identities yet.")
    with b:
        st.subheader("Recent sessions")
        if ss:
            st.dataframe(pd.DataFrame(ss).drop(columns=["output"], errors="ignore").head(8), hide_index=True,
                         width="stretch")
        else:
            st.info("No sessions yet. Process a video first.")
        st.subheader("Recent identity events")
        if events:
            st.dataframe(pd.DataFrame(events).tail(10).iloc[::-1], hide_index=True, width="stretch")
        else:
            st.info("No identity events yet.")

elif page == "Video & Camera":
    st.title("Video & Camera")
    left, right = st.columns([2.2, 1])
    with right:
        cfg = settings_panel("v_")
    with left:
        up = st.file_uploader("Upload a video", type=["mp4", "avi", "mov", "mkv"])
        path = None
        if up is not None:
            path = ROOT / "uploads" / f"{up.file_id}_{up.name}"
            path.parent.mkdir(exist_ok=True)
            if not path.exists():
                path.write_bytes(up.getbuffer())
            meta = probe(str(path))
            if meta["frames"] <= 0:
                st.error("This file could not be read as a video.")
                path = None
            else:
                m = st.columns(4)
                m[0].metric("Resolution", f"{meta['width']}x{meta['height']}")
                m[1].metric("FPS", meta["fps"])
                m[2].metric("Frames", meta["frames"])
                m[3].metric("Length (s)", meta["seconds"])
                with st.expander("Original video"):
                    st.video(str(path))
        b1, b2, b3 = st.columns(3)
        if b1.button("Start processing", type="primary", disabled=path is None or running):
            start_job(str(path), cfg, "video", up.name)
        if b2.button("Stop", disabled=not running):
            S.job.stop_flag.set()
        if b3.button("Reset", disabled=running):
            S.job = None
            st.rerun()
        if running:
            live_view()
        else:
            job_result("video")

elif page == "Live Tracking":
    st.title("Live Tracking")
    st.caption("Uses a camera on the machine running Streamlit, or an RTSP/HTTP stream URL (RTSP is untested).")
    left, right = st.columns([2.2, 1])
    with right:
        src = st.text_input("Camera index or stream URL", "0")
        cfg = settings_panel("l_")
    with left:
        b1, b2 = st.columns(2)
        if b1.button("Start camera", type="primary", disabled=running):
            start_job(int(src) if src.strip().isdigit() else src.strip(), cfg, "camera",
                      f"camera {src.strip()}")
        if b2.button("Stop camera", disabled=not running):
            S.job.stop_flag.set()
        if running:
            live_view()
            st.subheader("Current objects")
            act = [r for r in S.job.engine.snapshot() if r["status"] == "ACTIVE"]
            if act:
                st.dataframe(pd.DataFrame(act), hide_index=True, width="stretch")
            else:
                st.caption("No objects visible right now.")
        else:
            job_result("camera")

elif page == "Object Manager":
    st.title("Object Manager")
    if running:
        st.info("A session is running. Live identities are shown; details and rename are available when it ends.")
        st.dataframe(pd.DataFrame(S.job.engine.snapshot()), hide_index=True, width="stretch")
    else:
        sess = pick_session("om_s")
        rows = db.identities(sess["id"]) if sess else []
        if not rows:
            st.info("No identities yet.")
        else:
            df = pd.DataFrame(rows)
            c1, c2, c3 = st.columns(3)
            q = c1.text_input("Search (ID or name)")
            fc = c2.multiselect("Class", sorted(df["cls"].unique()))
            fs = c3.multiselect("Status", ["ACTIVE", "MISSING", "ARCHIVED"])
            if q:
                df = df[df["argus_id"].str.contains(q, case=False) | df["name"].str.contains(q, case=False)]
            if fc:
                df = df[df["cls"].isin(fc)]
            if fs:
                df = df[df["status"].isin(fs)]
            st.dataframe(df, hide_index=True, width="stretch")
            st.subheader("Identity details")
            sel = st.selectbox("Identity", list(df["argus_id"]) or [r["argus_id"] for r in rows])
            rec = next(r for r in rows if r["argus_id"] == sel)
            m = st.columns(5)
            m[0].metric("Name", rec["name"])
            m[1].metric("Class", rec["cls"])
            m[2].metric("Frames seen", rec["frames_seen"])
            m[3].metric("First / last frame", f"{rec['first_frame']} / {rec['last_frame']}")
            m[4].metric("Recoveries", rec["reid_count"])
            th = thumbs(sess["id"], sel)
            if th:
                cap = []
                for p in th:
                    tag = p.stem.split("_", 1)[1]
                    cap.append("First seen" if tag == "first" else tag.replace("_", " frame ").capitalize())
                st.image([str(p) for p in th], caption=cap, width=130)
            else:
                st.caption("No thumbnails saved for this identity.")
            tr = pd.DataFrame(db.tracks(sess["id"]))
            if not tr.empty and (tr["argus_id"] == sel).any():
                g = tr[tr["argus_id"] == sel].assign(cx=lambda d: (d.x1 + d.x2) / 2, cy=lambda d: (d.y1 + d.y2) / 2)
                fig = px.scatter(g, x="cx", y="cy", color="frame", title="Trajectory (image coordinates)")
                fig.update_yaxes(autorange="reversed")
                st.plotly_chart(fig, width="stretch")
            ev = [e for e in db.events(sess["id"]) if e["argus_id"] == sel]
            st.write("Match and lifecycle history")
            st.dataframe(pd.DataFrame(ev), hide_index=True, width="stretch")
            st.subheader("Rename identity")
            new = st.text_input("New display name")
            if st.button("Rename") and new.strip():
                db.rename(sess["id"], sel, new.strip())
                st.rerun()

elif page == "Identity History":
    st.title("Identity History")
    allsess = st.checkbox("Show identities from all sessions")
    if allsess:
        rows, ev, sess = db.identities(), [], None
    else:
        sess = pick_session("ih_s")
        rows = db.identities(sess["id"]) if sess else []
        ev = db.events(sess["id"]) if sess else []
    if not rows:
        st.info("No saved sessions yet.")
    else:
        q = st.text_input("Search (ID or name)")
        df = pd.DataFrame(rows)
        if q:
            df = df[df["argus_id"].str.contains(q, case=False) | df["name"].str.contains(q, case=False)]
        st.subheader("Identities")
        st.dataframe(df, hide_index=True, width="stretch")
        if sess:
            st.subheader("Identity events")
            if ev:
                kinds = st.multiselect("Event types", sorted({e["kind"] for e in ev}))
                edf = pd.DataFrame(ev)
                if kinds:
                    edf = edf[edf["kind"].isin(kinds)]
                st.dataframe(edf, hide_index=True, width="stretch")
            else:
                st.caption("No events in this session.")

elif page == "Analytics":
    st.title("Analytics")
    st.caption("What ARGUS-X saw, who it recognised again, and where people spent their time.")
    sess = pick_session("an_s")
    if not sess:
        st.info("No data yet. Process a video first.")
    else:
        ids = pd.DataFrame(db.identities(sess["id"]))
        ev = db.events(sess["id"])
        tr = pd.DataFrame(db.tracks(sess["id"]))
        rt = ad.recovery_table(ev)
        if ids.empty:
            st.info("This session produced no identities (nothing was detected, or every detection was too brief).")
        else:
            sm = ad.summary(ids, ev, tr)
            secs = pd.to_numeric(sess.get("seconds"), errors="coerce")
            fps_txt = f"{sess['frames'] / secs:.1f}" if secs and secs > 0 else "-"
            cards = [
                ("Unique identities", sm["identities"], f"{sm['peak']} on screen at once (peak)", "#00D4FF"),
                ("Re-identified", sm["recoveries"], f"{sm['recovered_people']} people got their old ID back", "#F4B942"),
                ("Detections", f"{sm['detections']:,}", f"across {sess['frames']} frames", "#7C5CFF"),
                ("Avg time on screen", sm["avg_visible"], f"frames (longest: {sm['longest_id']}, {sm['longest_frames']})", "#2ECC71"),
                ("Processing speed", fps_txt, f"FPS, {secs if pd.notna(secs) else '-'} s total", "#FF6B6B"),
            ]
            for col, (lab, val, sub, colr) in zip(st.columns(5), cards):
                col.markdown(f'<div class="kpi" style="border-top:3px solid {colr}"><div class="kl">{lab}</div>'
                             f'<div class="kv">{val}</div><div class="ks">{sub}</div></div>', unsafe_allow_html=True)
            st.write("")
            t1, t2, t3, t4, t5, t6 = st.tabs(["Overview", "Identities", "Re-ID recoveries", "Timeline", "Movement",
                                              "Sessions"])

            with t1:
                try:
                    a, b = st.columns(2)
                    fig = px.pie(ids, names="status", hole=0.62, title="Identity status", color="status",
                                 color_discrete_map={"ACTIVE": "#2ECC71", "MISSING": "#F4B942", "ARCHIVED": "#7B8794"})
                    fig.update_traces(textinfo="label+value")
                    a.plotly_chart(pretty(fig, 340), width="stretch")
                    cdf = ids.groupby("cls").size().reset_index(name="count")
                    fig = px.bar(cdf, x="count", y="cls", orientation="h", text="count", title="Object categories")
                    fig.update_traces(marker_color="#00D4FF")
                    b.plotly_chart(pretty(fig, 340), width="stretch")
                    act = ad.activity(tr, sess["frames"])
                    if not act.empty:
                        fig = go.Figure()
                        fig.add_scatter(x=act["frame"], y=act["objects"], mode="lines", name="per frame",
                                        line=dict(width=1, color="rgba(0,212,255,0.35)"))
                        fig.add_scatter(x=act["frame"], y=act["smooth"], mode="lines", name="smoothed", fill="tozeroy",
                                        line=dict(width=3, color="#00D4FF"), fillcolor="rgba(0,212,255,0.15)")
                        fig.update_layout(title="People on screen over time", xaxis_title="frame", yaxis_title="objects")
                        st.plotly_chart(pretty(fig, 340), width="stretch")
                except Exception as exc:
                    st.warning(f"This section could not be drawn: {exc}")

            with t2:
                try:
                    d = ids.copy()
                    d["Re-identified"] = d["reid_count"].apply(lambda n: "Yes" if n > 0 else "No")
                    a, b = st.columns([2, 1])
                    top = d.sort_values("frames_seen", ascending=False).head(25)
                    fig = px.bar(top, x="argus_id", y="frames_seen", color="Re-identified",
                                 title="Longest-tracked identities (top 25)",
                                 color_discrete_map={"Yes": "#F4B942", "No": "#00D4FF"})
                    a.plotly_chart(pretty(fig, 340), width="stretch")
                    fig = px.histogram(d, x="frames_seen", nbins=20, title="How long identities stay visible")
                    fig.update_traces(marker_color="#7C5CFF")
                    b.plotly_chart(pretty(fig, 340), width="stretch")
                    show = d[["argus_id", "name", "cls", "status", "first_frame", "last_frame", "frames_seen",
                              "reid_count", "confidence"]]
                    st.dataframe(show, hide_index=True, width="stretch", column_config={
                        "frames_seen": st.column_config.ProgressColumn(
                            "Frames seen", min_value=0, max_value=max(int(d["frames_seen"].max()), 1), format="%d"),
                        "reid_count": st.column_config.NumberColumn("Recoveries"),
                        "confidence": st.column_config.NumberColumn("Confidence", format="%.2f")})
                except Exception as exc:
                    st.warning(f"This section could not be drawn: {exc}")

            with t3:
                try:
                    if rt.empty:
                        st.info("No re-identifications in this session. They happen when someone disappears and "
                                "comes back looking the same. Turn on Appearance re-identification and process a clip "
                                "where people are hidden for a moment.")
                    else:
                        m = st.columns(3)
                        m[0].metric("Times someone was recognised again", len(rt))
                        m[1].metric("Average similarity", f"{rt['similarity'].mean():.2f}")
                        m[2].metric("Average time gone (frames)", f"{rt['gone_frames'].mean():.0f}")
                        st.write("Same ID, before and after the gap:")
                        base = ROOT / "outputs" / "thumbs" / f"s{sess['id']}"
                        for _, r in rt.head(12).iterrows():
                            c = st.columns([1, 1, 4])
                            fr = int(r["frame"])
                            for col, tag, cap in ((c[0], "before", "Before"), (c[1], "after", "After")):
                                p = base / f"{r['argus_id']}_{tag}_{fr}.jpg"
                                if p.exists():
                                    col.image(str(p), caption=cap, width=110)
                            gone = f"{int(r['gone_frames'])} frames" if pd.notna(r["gone_frames"]) else "a while"
                            sim = f"{r['similarity']:.2f}" if pd.notna(r["similarity"]) else "-"
                            c[2].markdown(f"### {r['argus_id']}\nBack at frame **{fr}** after being gone for "
                                          f"**{gone}**. Appearance similarity **{sim}**.")
                        if len(rt) > 12:
                            with st.expander(f"All {len(rt)} recoveries"):
                                st.dataframe(rt, hide_index=True, width="stretch")
                except Exception as exc:
                    st.warning(f"This section could not be drawn: {exc}")

            with t4:
                try:
                    seg = ad.timeline_segments(tr, ids, 40)
                    if seg.empty:
                        st.info("No tracking data for a timeline.")
                    else:
                        order = [i for i in ids.sort_values("first_frame")["argus_id"] if i in set(seg["argus_id"])]
                        seg["Re-identified"] = seg["recovered"].map({True: "Yes", False: "No"})
                        fig = px.bar(seg, base="start", x="length", y="argus_id", orientation="h",
                                     color="Re-identified", color_discrete_map={"Yes": "#F4B942", "No": "#00D4FF"},
                                     category_orders={"argus_id": order[::-1]},
                                     title="Who was visible when (gaps = out of sight, stars = recognised again)")
                        r2 = rt[rt["argus_id"].isin(order)]
                        if not r2.empty:
                            fig.add_scatter(x=r2["frame"], y=r2["argus_id"], mode="markers", name="re-identified here",
                                            marker=dict(symbol="star", size=14, color="#FFFFFF",
                                                        line=dict(width=1, color="#F4B942")))
                        fig.update_layout(xaxis_title="frame", yaxis_title="")
                        st.plotly_chart(pretty(fig, max(380, 20 * len(order) + 140)), width="stretch")
                        if ids.shape[0] > 40:
                            st.caption("Showing the 40 identities seen the longest.")
                except Exception as exc:
                    st.warning(f"This section could not be drawn: {exc}")

            with t5:
                try:
                    fp = ad.footpoints(tr)
                    if fp.empty:
                        st.info("No tracking data for a heatmap.")
                    else:
                        fig = px.density_heatmap(fp, x="x", y="y", nbinsx=48, nbinsy=27,
                                                 color_continuous_scale="Turbo",
                                                 title="Where people stood (bottom centre of each box)")
                        fig.update_yaxes(autorange="reversed", title="y (pixels)")
                        fig.update_xaxes(title="x (pixels)")
                        st.plotly_chart(pretty(fig, 460), width="stretch")
                except Exception as exc:
                    st.warning(f"This section could not be drawn: {exc}")

            with t6:
                try:
                    sdf = pd.DataFrame(db.sessions())
                    sdf["session"] = "#" + sdf["id"].astype(str)
                    sdf["fps"] = (sdf["frames"] / pd.to_numeric(sdf["seconds"], errors="coerce")).round(1)
                    a, b = st.columns(2)
                    fig = px.bar(sdf, x="session", y="frames", color="identities", title="Frames per session",
                                 color_continuous_scale="Tealgrn")
                    a.plotly_chart(pretty(fig, 320), width="stretch")
                    fig = px.line(sdf, x="session", y="fps", markers=True, title="Processing speed (FPS)")
                    fig.update_traces(line_color="#FF6B6B", marker_size=10)
                    b.plotly_chart(pretty(fig, 320), width="stretch")
                    st.dataframe(sdf.drop(columns=["output", "session"], errors="ignore"), hide_index=True,
                                 width="stretch")
                except Exception as exc:
                    st.warning(f"This section could not be drawn: {exc}")

elif page == "Settings":
    st.title("Settings")
    c = dict(S.cfg)
    models = sorted(p.name for p in (ROOT / "models").glob("*.pt")) or ["yolo26n.pt"]
    c["model"] = st.selectbox("Detection model", models, index=models.index(c["model"]) if c["model"] in models else 0)
    c["device"] = st.selectbox("Processing device", ["auto", "cpu", "cuda"], index=["auto", "cpu", "cuda"].index(c["device"]))
    c["tracker"] = st.selectbox("Tracker", ["bytetrack.yaml", "botsort.yaml"],
                                index=["bytetrack.yaml", "botsort.yaml"].index(c["tracker"]),
                                help="botsort.yaml adds camera-motion compensation (untested here).")
    c["track_buffer"] = st.number_input("Tracker memory (frames a lost track is remembered)", 10, 300,
                                        int(c["track_buffer"]),
                                        help="Higher keeps the same ID through longer occlusions. Applies to the next session.")
    c["confirm_frames"] = st.number_input("Frames needed to confirm a new identity", 1, 15, int(c["confirm_frames"]),
                                          help="A new track is judged on this many frames before it gets an ID, "
                                               "which filters flickers and gives re-identification a cleaner look.")
    c["missing_after"] = st.number_input("Missing-frame threshold (frames before MISSING)", 1, 300, int(c["missing_after"]))
    c["archive_after"] = st.number_input("Archive threshold (frames before ARCHIVED)", 10, 100000, int(c["archive_after"]))
    if st.button("Save settings"):
        if c["archive_after"] <= c["missing_after"]:
            st.error("The archive threshold must be larger than the missing threshold.")
        else:
            S.cfg = c
            SET_PATH.parent.mkdir(exist_ok=True)
            SET_PATH.write_text(json.dumps(c))
            st.success("Saved. Applies to the next session.")
    st.caption(f"Output folder: {ROOT / 'outputs'}")
    st.subheader("System information")
    info = {"Python": platform.python_version(), "Application": f"ARGUS-X {VERSION}"}
    try:
        import torch
        info.update({"PyTorch": torch.__version__, "CUDA available": str(torch.cuda.is_available()),
                     "GPU": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none"})
    except Exception:
        info["PyTorch"] = "not installed"
    try:
        import ultralytics
        info["Ultralytics"] = ultralytics.__version__
    except Exception:
        info["Ultralytics"] = "not installed"
    mp = ROOT / "models" / c["model"]
    info["Model file"] = f"{c['model']} ({mp.stat().st_size / 1e6:.1f} MB)" if mp.exists() else f"{c['model']} (will be downloaded)"
    st.table(pd.DataFrame(info.items(), columns=["Item", "Value"]))
