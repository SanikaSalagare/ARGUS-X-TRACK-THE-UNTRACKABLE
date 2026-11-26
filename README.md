# ARGUS-X - Track the Untrackable (v0.3.1)

## What changed in 0.3.1
- Re-identification is ON by default (colour model, threshold 0.80). Recoveries are drawn on the video: orange box,
  "RECOVERED" label, and a line from where the person was last seen. Missing people leave a grey ghost box.
- A new tracker must be seen for N frames (default 4) before it gets an ARGUS ID, and is matched on the average of those
  frames. One-or-two-frame flickers no longer create identities.
- Colour fingerprint now uses the torso only and Bhattacharyya similarity; the location gate allows ~15 px/frame at
  1080p and checks box size (the settings that did best in the MOT17 sweeps).
- Tracker memory setting (default 60 frames) so ByteTrack itself keeps an ID through longer occlusions.
- Live preview is small and throttled; the final ffmpeg re-encode uses the veryfast preset.
- Analytics page rebuilt with tabs: Overview, Identities, Re-ID recoveries (before/after photos), Timeline, Movement
  heatmap, Sessions.

**Tested here (fake detector, no GPU):** all engine/database logic including recovery with confirmation, flicker
rejection, look-alike and distance rejection; analytics data preparation; every dashboard page executed against a
stand-in Streamlit with seeded data.
**Not yet run on real video/hardware:** the speed-ups (no timings measured), the tracker-memory setting, the new
Analytics charts' appearance in a browser, and the recovery rate on real footage. Check the "Avg FPS" figure after a run.

## Status
**Tested by automated tests here (fake detector, no GPU):** identity recovery logic, lifecycle events, location gating,
ID numbering across sessions, database round trip, rename, and rendering of all 7 dashboard pages with seeded data.
Run: `python backend/test_identity_memory.py`

**Seen working by the developer on real video:** upload, background processing, live preview, stop, history, analytics, settings.

**Written but NOT yet run on real video/hardware:** neural appearance model, thumbnails, trajectory plot, tracking CSV export,
webcam, RTSP/stream URL, BoT-SORT tracker option, playback of the result video after the ffmpeg fix, ghost boxes.

**Experimental:** appearance ReID (colour histograms, or ImageNet MobileNet features). Neither is a person-ReID-trained
network. Tune the similarity threshold on your own occlusion clip.

**Not built:** multi-camera identity sharing, open-vocabulary detection, vehicle-specific ReID, SQLAlchemy, streamlit-webrtc, PyAV.

## Run (PowerShell, from this project folder, using the existing venv)
    & "C:\ARGUS-X\.venv\Scripts\python.exe" -m pip install -r requirements.txt
    & "C:\ARGUS-X\.venv\Scripts\python.exe" -m streamlit run frontend/app.py

Put `yolo26n.pt` in `models/`. Keep this project in its own folder.
