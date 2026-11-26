"""ARGUS-X core engine v0.3.1.
Detection: YOLO | Tracking: ByteTrack or BoT-SORT | Identity: ARGUS IDs separate from tracker IDs.
ReID (optional): colour or neural appearance embeddings, location gating, an uncertainty guard, and a short
confirmation period so a new tracker is judged on several clean frames instead of its first (often partial) one."""
import os
from collections import deque
from pathlib import Path

import cv2
import numpy as np

_MODELS = {}
FONT = cv2.FONT_HERSHEY_SIMPLEX
OUT_DIR = Path(__file__).resolve().parents[3] / "outputs"


def get_model(path):
    if path not in _MODELS:
        from ultralytics import YOLO
        _MODELS[path] = YOLO(path)
    return _MODELS[path]


def pick_device(choice="auto"):
    if choice == "cpu":
        return "cpu"
    try:
        import torch
        return 0 if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


def make_tracker_yaml(name, buffer):
    """Copy Ultralytics' own tracker config with a longer track_buffer (how long a lost track is remembered).
    Falls back to the stock config if anything is missing."""
    try:
        import yaml
        from ultralytics.utils import ROOT as U_ROOT
        cfg = yaml.safe_load((U_ROOT / "cfg" / "trackers" / name).read_text())
        cfg["track_buffer"] = int(buffer)
        out = OUT_DIR / "trackers"
        out.mkdir(parents=True, exist_ok=True)
        path = out / f"{Path(name).stem}_buf{int(buffer)}.yaml"
        path.write_text(yaml.safe_dump(cfg))
        return str(path)
    except Exception:
        return name


def crop(frame, box):
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = [int(v) for v in box]
    x1, y1, x2, y2 = max(0, x1), max(0, y1), min(w, x2), min(h, y2)
    return frame[y1:y2, x1:x2] if x2 - x1 >= 8 and y2 - y1 >= 16 else None


def iou(a, b):
    x1, y1, x2, y2 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def colour_embed(c):
    """Colour appearance vector: HSV histogram of the torso (15%-60% of the box height, so head and legs are
    skipped). Stored as sqrt of the normalised histogram, so the dot product of two vectors is the Bhattacharyya
    coefficient, which separates clothing colours better than a plain cosine on raw histograms."""
    h = c.shape[0]
    top, bot = int(h * 0.15), max(int(h * 0.60), int(h * 0.15) + 2)
    hsv = cv2.cvtColor(c[top:bot], cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1], None, [16, 8], [0, 180, 0, 256]).flatten()
    s = hist.sum()
    if s <= 0:
        return None
    v = np.sqrt(hist / s).astype(np.float32)
    n = np.linalg.norm(v)
    return v / n if n > 0 else None


def color_for(no):
    rng = np.random.RandomState(no * 7919)
    return tuple(int(c) for c in rng.randint(70, 255, 3))


class ArgusEngine:
    def __init__(self, model_path="yolo26n.pt", conf=0.3, imgsz=640, classes=None, reid=False, reid_thr=0.80,
                 missing_after=5, archive_after=300, tracker="bytetrack.yaml", appearance="colour", gate=True,
                 device="auto", start_no=1, thumb_dir=None, confirm_frames=4, reid_window=90, track_buffer=None):
        self.model = get_model(model_path)
        self.model.predictor = None  # reset tracker state for a new session
        self.device = pick_device(device)
        self.conf, self.imgsz, self.classes = conf, imgsz, classes
        self.tracker = make_tracker_yaml(tracker, track_buffer) if track_buffer else tracker
        self.reid, self.reid_thr, self.gate = reid, reid_thr, gate
        self.missing_after, self.archive_after = missing_after, archive_after
        self.confirm_frames, self.reid_window = max(1, int(confirm_frames)), int(reid_window)
        self.neural = None
        if reid and appearance == "neural":
            from backend.app.reidentification.embedder import NeuralEmbedder
            self.neural = NeuralEmbedder(self.device)
        self.ids, self.t2a, self.events, self.tracks, self.prov = {}, {}, [], [], {}
        self.frame, self.next_no, self.thumb_dir = 0, int(start_no), thumb_dir
        self.recovered, self._shape = 0, (480, 640, 3)

    # ---- helpers
    def status(self, rec):
        gap = self.frame - rec["last"]
        if gap <= self.missing_after:
            return "ACTIVE"
        return "MISSING" if gap <= self.archive_after else "ARCHIVED"

    def _event(self, no, kind, detail, frame=None):
        self.events.append(dict(frame=self.frame if frame is None else frame, argus_id=f"ARGUS-{no:04d}",
                                kind=kind, detail=detail))

    def _thumb(self, no, tag, c):
        if self.thumb_dir and c is not None:
            os.makedirs(self.thumb_dir, exist_ok=True)
            cv2.imwrite(os.path.join(self.thumb_dir, f"ARGUS-{no:04d}_{tag}.jpg"), c)

    def _emb(self, c):
        if c is None:
            return None
        return self.neural(c) if self.neural is not None else colour_embed(c)

    def _plausible(self, rec, box, gap):
        """Location gating: a lost person can only have moved so far in `gap` frames, and keeps a similar size."""
        if not self.gate:
            return True
        diag = (self._shape[0] ** 2 + self._shape[1] ** 2) ** 0.5
        lx, ly = (rec["box"][0] + rec["box"][2]) / 2, (rec["box"][1] + rec["box"][3]) / 2
        nx, ny = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
        if ((nx - lx) ** 2 + (ny - ly) ** 2) ** 0.5 > diag * max(0.03, 0.0068 * gap):
            return False
        old_h = max(rec["box"][3] - rec["box"][1], 1e-6)
        return 0.5 <= (box[3] - box[1]) / old_h <= 2.0

    def _update(self, rec, box, conf, tid):
        rec.update(last=self.frame, box=box, conf=conf, tracker_id=tid)
        rec["seen"] += 1
        rec["trail"].append((int((box[0] + box[2]) / 2), int(box[3])))

    def _match(self, e, cls, box, active):
        """Match identities using the average of the three best gallery similarities."""
        cands = []

        for no, rec in self.ids.items():
            gap = self.frame - rec["last"]

            if no in active or rec["cls"] != cls or gap < 1 or not rec["gallery"]:
                continue

            if gap > min(self.archive_after, self.reid_window):
                continue

            if not self._plausible(rec, box, gap):
                continue

            similarities = sorted(
                (float(np.dot(g, e)) for g in rec["gallery"]),
                reverse=True
            )
            score = float(np.mean(similarities[:3]))
            cands.append((score, no))

        cands.sort(reverse=True)

        if not cands or cands[0][0] < self.reid_thr:
            return None

        if len(cands) > 1 and cands[0][0] - cands[1][0] < 0.03:
            return "uncertain", cands[0][0], cands[0][1]

        return "match", cands[0][0], cands[0][1]
    def _absorb(self, rec, p, tid):
        """Attach the frames seen during confirmation to the identity."""
        for _, b in p["boxes"]:
            self._update(rec, b, p["conf"], tid)
        aid = f"ARGUS-{rec['no']:04d}"
        for f, b in p["boxes"][:-1]:  # the latest frame is recorded by process()
            self.tracks.append((f, aid, int(b[0]), int(b[1]), int(b[2]), int(b[3])))

    def _confirm(self, tid, active):
        p = self.prov.pop(tid)
        box, cls, c = p["boxes"][-1][1], p["cls"], p["crop"]
        e = None
        if self.reid:
            embs = [x for x in (p["embs"] or [self._emb(c)]) if x is not None]
            if embs:
                m = np.mean(embs, axis=0)
                n = np.linalg.norm(m)
                e = m / n if n > 0 else None
        verdict = self._match(e, cls, box, active) if e is not None else None
        if verdict and verdict[0] == "match":
            sim, no = verdict[1], verdict[2]
            rec = self.ids[no]
            ob, gone = rec["box"], self.frame - rec["last"]
            self._thumb(no, f"before_{self.frame}", rec.get("last_crop"))
            self._thumb(no, f"after_{self.frame}", c)
            rec.update(reid=rec["reid"] + 1, flash=45, state="ACTIVE", last_sim=sim,
                       recover_from=(int((ob[0] + ob[2]) / 2), int((ob[1] + ob[3]) / 2)))
            if c is not None:
                rec["last_crop"] = c.copy()
            rec["gallery"].append(e)
            self.t2a[tid] = no
            self._absorb(rec, p, tid)
            self.recovered += 1
            self._event(no, "RECOVERED", f"tracker {tid}, similarity {sim:.2f}, gone {gone} frames")
            return rec
        if verdict:
            self._event(verdict[2], "UNCERTAIN", f"tracker {tid}: ambiguous match, new identity created")
        no = self.next_no
        self.next_no += 1
        rec = dict(no=no, name=f"ARGUS-{no:04d}", cls=cls, tracker_id=tid, first=p["first"], last=self.frame,
                   seen=0, conf=p["conf"], box=box, reid=0, flash=0, state="ACTIVE", gallery=deque(maxlen=30),
                   trail=deque(maxlen=40), last_crop=None if c is None else c.copy(), last_sim=0.0,
                   recover_from=None)
        if e is not None:
            rec["gallery"].append(e)
        self.ids[no] = rec
        self.t2a[tid] = no
        self._absorb(rec, p, tid)
        self._thumb(no, "first", c)
        self._event(no, "REGISTERED", f"{cls}, tracker {tid}", frame=p["first"])
        return rec

    def _lifecycle(self):
        for no, rec in self.ids.items():
            st = self.status(rec)
            if st != rec["state"]:
                kind = {"MISSING": "MISSING", "ARCHIVED": "ARCHIVED", "ACTIVE": "RETURNED"}[st]
                self._event(no, kind, f"last seen frame {rec['last']}")
                rec["state"] = st

    # ---- main entry
    def process(self, frame):
        self.frame += 1
        self._shape = frame.shape
        r = self.model.track(frame, persist=True, tracker=self.tracker, conf=self.conf, imgsz=self.imgsz,
                             device=self.device, classes=self.classes, verbose=False)[0]
        dets = []
        if r.boxes is not None and r.boxes.id is not None:
            for tid, b, c, cf in zip(r.boxes.id.int().tolist(), r.boxes.xyxy.cpu().numpy(),
                                     r.boxes.cls.int().tolist(), r.boxes.conf.tolist()):
                dets.append((tid, b, r.names[c], float(cf)))
        boxes = [d[1] for d in dets]
        shown, active, pending, probing = [], set(), [], []
        for i, (tid, b, cls, cf) in enumerate(dets):
            clean = all(iou(b, boxes[j]) < 0.15 for j in range(len(dets)) if j != i)
            no = self.t2a.get(tid)
            if no is None:  # not an identity yet: collect evidence for a few frames first
                p = self.prov.setdefault(tid, dict(first=self.frame, boxes=[], embs=[], cls=cls, crop=None,
                                                   last=self.frame, conf=cf))
                p["last"], p["conf"] = self.frame, cf
                p["boxes"].append((self.frame, b))
                c = crop(frame, b)
                if c is not None:
                    p["crop"] = c.copy()
                    if self.reid and clean:
                        e = self._emb(c)
                        if e is not None:
                            p["embs"].append(e)
                pending.append(tid)
                continue
            rec = self.ids[no]
            if self.reid and self.frame % 5 == 0 and clean:
                c = crop(frame, b)  # learn appearance only from clean, non-overlapped views
                e = self._emb(c)
                if e is not None:
                    rec["gallery"].append(e)
                    rec["last_crop"] = c.copy()
            self._update(rec, b, cf, tid)
            active.add(no)
            shown.append(rec)
        for tid in pending:
            p = self.prov[tid]
            if len(p["boxes"]) >= self.confirm_frames:
                rec = self._confirm(tid, active)
                active.add(rec["no"])
                shown.append(rec)
            else:
                probing.append(p["boxes"][-1][1])
        for tid in [t for t, p in self.prov.items() if self.frame - p["last"] > 10]:
            del self.prov[tid]  # a flicker that never became a stable track: not an identity
        self._lifecycle()
        for rec in shown:
            x1, y1, x2, y2 = [int(v) for v in rec["box"]]
            self.tracks.append((self.frame, f"ARGUS-{rec['no']:04d}", x1, y1, x2, y2))
        out = self._draw(frame.copy(), shown, active, probing)
        c = self.counts()
        cv2.putText(out, f"ARGUS-X | frame {self.frame} | active {c['ACTIVE']} | missing {c['MISSING']} | "
                         f"identities {len(self.ids)} | recovered {self.recovered}",
                    (10, 24), FONT, 0.6, (255, 255, 255), 2)
        return out

    def _draw(self, frame, shown, active, probing=()):
        for b in probing:  # still being confirmed
            x1, y1, x2, y2 = [int(v) for v in b]
            cv2.rectangle(frame, (x1, y1), (x2, y2), (200, 200, 200), 1)
        for rec in shown:
            x1, y1, x2, y2 = [int(v) for v in rec["box"]]
            hot = rec["flash"] > 0
            col = (0, 165, 255) if hot else color_for(rec["no"])
            aid = f"ARGUS-{rec['no']:04d}"
            label = f"{aid if rec['name'] == aid else aid + ' ' + rec['name']} {rec['cls']} {rec['conf']:.2f}"
            if hot:
                label = f"{aid} RECOVERED {rec['last_sim']:.2f}"
                if rec["recover_from"]:  # line from where this person was last seen to where they are now
                    ox, oy = rec["recover_from"]
                    cv2.line(frame, (ox, oy), (int((x1 + x2) / 2), int((y1 + y2) / 2)), col, 2, cv2.LINE_AA)
                    cv2.circle(frame, (ox, oy), 7, col, -1)
                rec["flash"] -= 1
            cv2.rectangle(frame, (x1, y1), (x2, y2), col, 4 if hot else 2)
            tw = cv2.getTextSize(label, FONT, 0.5, 1)[0][0]
            cv2.rectangle(frame, (x1, max(0, y1 - 20)), (x1 + tw + 8, max(20, y1)), col, -1)
            cv2.putText(frame, label, (x1 + 4, max(14, y1 - 6)), FONT, 0.5, (0, 0, 0), 1)
            pts = np.array(rec["trail"], np.int32).reshape(-1, 1, 2)
            if len(pts) > 1:
                cv2.polylines(frame, [pts], False, col, 2)
        for no, rec in self.ids.items():  # ghost box where a recently missing object was last seen
            if no not in active and rec["state"] == "MISSING" and self.frame - rec["last"] <= self.reid_window:
                x1, y1, x2, y2 = [int(v) for v in rec["box"]]
                cv2.rectangle(frame, (x1, y1), (x2, y2), (150, 150, 150), 1)
                cv2.putText(frame, f"ARGUS-{no:04d} MISSING", (x1, max(14, y1 - 6)), FONT, 0.45, (150, 150, 150), 1)
        return frame

    # ---- data access
    def counts(self):
        c = {"ACTIVE": 0, "MISSING": 0, "ARCHIVED": 0}
        for rec in self.ids.values():
            c[self.status(rec)] += 1
        return c

    def snapshot(self):
        return [dict(argus_id=f"ARGUS-{r['no']:04d}", name=r["name"], cls=r["cls"], tracker_id=r["tracker_id"],
                     status=self.status(r), first_frame=r["first"], last_frame=r["last"], frames_seen=r["seen"],
                     reid_count=r["reid"], confidence=round(float(r["conf"]), 3)) for r in self.ids.values()]
