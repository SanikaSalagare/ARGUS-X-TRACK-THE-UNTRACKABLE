"""Background video worker: runs the engine on a file or camera without blocking the dashboard."""
import logging
import shutil
import subprocess
import threading
import time
from pathlib import Path

import cv2

from backend.app.database import db

ROOT = Path(__file__).resolve().parents[3]
(ROOT / "outputs").mkdir(exist_ok=True)
logging.basicConfig(filename=str(ROOT / "outputs" / "argus.log"), level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("argus")


def probe(path):
    """Video metadata for the upload panel."""
    c = cv2.VideoCapture(path)
    try:
        fps, n = c.get(cv2.CAP_PROP_FPS) or 0, int(c.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        return dict(fps=round(fps, 1), width=int(c.get(3)), height=int(c.get(4)), frames=n,
                    seconds=round(n / fps, 1) if fps else 0)
    finally:
        c.release()


class VideoJob(threading.Thread):
    def __init__(self, source, engine, out_dir, max_frames=0, kind="video", label=None):
        super().__init__(daemon=True)
        self.source, self.engine, self.max_frames = source, engine, int(max_frames)
        self.kind, self.label, self.out_dir = kind, label, Path(out_dir)
        self.stop_flag = threading.Event()
        self.latest, self.frame_no, self.total = None, 0, 0
        self.preview_every, self.preview_width = 3, 800  # the browser preview is small and throttled: it is not the saved video
        self.state, self.error, self.output, self.session_id = "running", None, None, None
        self.t0, self.t1 = time.time(), None

    @property
    def seconds(self):
        return round((self.t1 or time.time()) - self.t0, 1)

    @property
    def fps_proc(self):
        return round(self.frame_no / self.seconds, 1) if self.seconds > 0 else 0.0

    def _ffmpeg(self):
        if shutil.which("ffmpeg"):
            return "ffmpeg"
        try:
            import imageio_ffmpeg
            return imageio_ffmpeg.get_ffmpeg_exe()
        except Exception:
            return None

    def run(self):
        cap, writer, raw = cv2.VideoCapture(self.source), None, None
        label = self.label or (Path(self.source).name if isinstance(self.source, str) else f"camera {self.source}")
        try:
            self.session_id = db.create_session(label)
            self.engine.thumb_dir = str(self.out_dir / "thumbs" / f"s{self.session_id}")
            log.info("session %s started: %s", self.session_id, label)
            if not cap.isOpened():
                raise RuntimeError("Cannot open the video source (unsupported or corrupted file, bad URL, or camera busy).")
            fps = cap.get(cv2.CAP_PROP_FPS)
            fps = fps if 1 <= fps <= 120 else 25
            if self.kind == "video":
                self.total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            if self.max_frames:
                self.total = min(self.total or self.max_frames, self.max_frames)
            self.out_dir.mkdir(exist_ok=True)
            raw = self.out_dir / f"argus_{int(time.time())}_raw.mp4"
            while not self.stop_flag.is_set():
                ok, frame = cap.read()
                if not ok:
                    if self.kind == "camera":
                        self.error = "Camera disconnected or stopped sending frames."
                    break
                out = self.engine.process(frame)
                if writer is None:
                    h, w = out.shape[:2]
                    writer = cv2.VideoWriter(str(raw), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
                writer.write(out)
                self.frame_no = self.engine.frame
                if self.frame_no % self.preview_every == 1 or self.preview_every == 1:
                    h, w = out.shape[:2]
                    k = min(1.0, self.preview_width / w)
                    small = cv2.resize(out, (int(w * k), int(h * k)), interpolation=cv2.INTER_AREA) if k < 1 else out
                    self.latest = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
                if self.max_frames and self.frame_no >= self.max_frames:
                    break
            if self.frame_no == 0 and not self.error:
                self.error = "No frames could be read from this source."
            self.state = "stopped" if self.stop_flag.is_set() else "finished"
        except Exception as e:  # report to the UI instead of dying silently
            self.error, self.state = str(e), "error"
            log.exception("job failed")
        finally:
            self.t1 = time.time()
            cap.release()
            if writer is not None:
                writer.release()
            if raw is not None and raw.exists() and self.frame_no:
                final = raw.with_name(raw.name.replace("_raw", ""))
                exe, ok = self._ffmpeg(), False
                if exe:  # re-encode to H.264 so browsers can play it
                    r = subprocess.run([exe, "-y", "-i", str(raw), "-vcodec", "libx264", "-preset", "veryfast",
                                        "-crf", "23", "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(final)], capture_output=True)
                    ok = r.returncode == 0 and final.exists()
                if ok:
                    raw.unlink()
                self.output = str(final if ok else raw)
            try:
                if self.session_id:
                    db.finish_session(self.session_id, self.frame_no, self.output or "", self.engine.snapshot(),
                                      self.engine.events, self.engine.tracks, self.seconds)
                    log.info("session %s saved: %s frames, %s identities", self.session_id, self.frame_no,
                             len(self.engine.ids))
            except Exception as e:
                self.error = self.error or f"Database save failed: {e}"
                log.exception("db save failed")
