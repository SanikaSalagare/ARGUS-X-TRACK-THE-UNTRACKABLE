"""eval_mot17.py - plain tracker IDs vs colour-ReID stitching on MOT17 (train split).

Run:  python backend/eval_mot17.py --seqs MOT17-02-DPM
"""
import argparse, csv, os, pickle, sys, time
sys.path.insert(0, r"C:\ARGUS-X")

import cv2
import numpy as np
try:
    from scipy.optimize import linear_sum_assignment
except Exception:
    def linear_sum_assignment(cost):
        """Pure-Python Hungarian algorithm (minimises cost), same output as scipy."""
        cost = np.asarray(cost, dtype=float)
        transposed = cost.shape[0] > cost.shape[1]
        if transposed:
            cost = cost.T
        n, m = cost.shape
        C = cost.tolist()
        INF = float("inf")
        u = [0.0] * (n + 1); v = [0.0] * (m + 1)
        p = [0] * (m + 1); way = [0] * (m + 1)
        for i in range(1, n + 1):
            p[0] = i; j0 = 0
            minv = [INF] * (m + 1); used = [False] * (m + 1)
            while True:
                used[j0] = True
                i0 = p[j0]; delta = INF; j1 = 0
                for j in range(1, m + 1):
                    if not used[j]:
                        cur = C[i0 - 1][j - 1] - u[i0] - v[j]
                        if cur < minv[j]:
                            minv[j] = cur; way[j] = j0
                        if minv[j] < delta:
                            delta = minv[j]; j1 = j
                for j in range(m + 1):
                    if used[j]:
                        u[p[j]] += delta; v[j] -= delta
                    else:
                        minv[j] -= delta
                j0 = j1
                if p[j0] == 0:
                    break
            while True:
                j1 = way[j0]; p[j0] = p[j1]; j0 = j1
                if j0 == 0:
                    break
        rows, cols = [], []
        for j in range(1, m + 1):
            if p[j]:
                rows.append(p[j] - 1); cols.append(j - 1)
        rows = np.array(rows, dtype=int); cols = np.array(cols, dtype=int)
        if transposed:
            rows, cols = cols, rows
        order = np.argsort(rows)
        return rows[order], cols[order]
from backend.app.detectors.yolo_detector import YOLODetector

ROOT = r"C:\ARGUS-X2\datasets\MOT17\MOT17\train"
MODEL = r"C:\ARGUS-X2\models\yolo26n.pt"
OUT = r"C:\ARGUS-X2\outputs"


# ---------------------------------------------------------------- ground truth
def load_gt(seq):
    gt = {}
    with open(os.path.join(ROOT, seq, "gt", "gt.txt")) as fh:
        for line in fh:
            p = line.strip().split(",")
            if len(p) < 8:
                continue
            # keep only "considered" pedestrian rows (flag==1, class==1)
            if int(float(p[6])) != 1 or int(float(p[7])) != 1:
                continue
            x, y, w, h = map(float, p[2:6])
            gt.setdefault(int(p[0]), []).append((int(p[1]), x, y, x + w, y + h))
    return gt


# ------------------------------------------------------------ appearance
def hs_hist(frame, box):
    """HSV histogram of the torso region (15%-60% of box height)."""
    H, W = frame.shape[:2]
    x1, y1, x2, y2 = [int(v) for v in box]
    x1, y1, x2, y2 = max(0, x1), max(0, y1), min(W, x2), min(H, y2)
    if x2 - x1 < 4 or y2 - y1 < 8:
        return None
    top = y1 + int((y2 - y1) * 0.15)
    bot = y1 + int((y2 - y1) * 0.60)
    crop = frame[top:bot, x1:x2]
    if crop.size == 0:
        return None
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    h = cv2.calcHist([hsv], [0, 1], None, [16, 8], [0, 180, 0, 256]).flatten()
    s = h.sum()
    return (h / s).astype(np.float32) if s > 0 else None


# ------------------------------------------------------------ run tracker once
def run_tracker(det, seq, args):
    cache = os.path.join(OUT, f"cache_{seq}_{args.conf}_{args.imgsz}_{args.max_frames}.pkl")
    if os.path.exists(cache) and not args.no_cache:
        print(f"  using cached detections: {cache}")
        with open(cache, "rb") as fh:
            return pickle.load(fh)
    d = os.path.join(ROOT, seq, "img1")
    files = sorted(f for f in os.listdir(d) if f.lower().endswith(".jpg"))
    if args.max_frames:
        files = files[:args.max_frames]
    det.reset_tracker()
    frames, t0 = {}, time.time()
    for k, fn in enumerate(files, 1):
        frame = cv2.imread(os.path.join(d, fn))
        dets = det.track(frame, conf=args.conf, imgsz=args.imgsz, persist=True)
        rows = []
        for dt in dets:
            if dt.tracker_id is None or dt.class_name != "person":
                continue
            box = tuple(float(v) for v in dt.bbox[:4])
            rows.append((int(dt.tracker_id), box, hs_hist(frame, box)))
        frames[int(os.path.splitext(fn)[0])] = rows
        if k % 100 == 0:
            print(f"  {k}/{len(files)} frames ({time.time() - t0:.0f}s)")
    os.makedirs(OUT, exist_ok=True)
    with open(cache, "wb") as fh:
        pickle.dump(frames, fh)
    return frames


# ------------------------------------------------------------ colour ReID
def build_tracklets(frames):
    T = {}
    for f in sorted(frames):
        for tid, box, h in frames[f]:
            t = T.setdefault(tid, dict(start=f, end=f, sbox=box, ebox=box, sh=[], eh=[]))
            t["end"], t["ebox"] = f, box
            if h is not None:
                if len(t["sh"]) < 5:
                    t["sh"].append(h)
                t["eh"] = (t["eh"] + [h])[-5:]
    for t in T.values():
        t["shm"] = np.mean(t["sh"], axis=0) if t["sh"] else None
        t["ehm"] = np.mean(t["eh"], axis=0) if t["eh"] else None
    return T


def _c(b):
    return (b[0] + b[2]) / 2, (b[1] + b[3]) / 2, b[2] - b[0], b[3] - b[1]


def stitch(T, thr, max_gap, px_per_frame):
    """Link a new tracklet to a recently ended one if colour + position agree."""
    order = sorted(T, key=lambda k: T[k]["start"])
    out, claimed = {}, set()
    for tid in order:
        t = T[tid]
        out[tid] = tid
        if thr is None or t["shm"] is None:
            continue
        best, best_s = None, thr
        cx, cy, w, h = _c(t["sbox"])
        for pid in order:
            p = T[pid]
            if pid == tid or pid in claimed or p["ehm"] is None or p["end"] >= t["start"]:
                continue
            gap = t["start"] - p["end"]
            if gap > max_gap:
                continue
            px, py, pw, ph = _c(p["ebox"])
            if np.hypot(cx - px, cy - py) > max(60.0, px_per_frame * gap):
                continue
            if not (0.5 <= (h / max(ph, 1e-6)) <= 2.0):
                continue
            s = float(np.sqrt(t["shm"] * p["ehm"]).sum())  # Bhattacharyya coeff.
            if s >= best_s:
                best, best_s = pid, s
        if best is not None:
            out[tid] = out[best]
            claimed.add(best)
    return out


def apply_ids(frames, mapping):
    return {f: [(mapping[tid], box) for tid, box, _ in rows] for f, rows in frames.items()}


# ------------------------------------------------------------ metrics
def iou_matrix(a, b):
    ix1 = np.maximum(a[:, None, 0], b[None, :, 0]); iy1 = np.maximum(a[:, None, 1], b[None, :, 1])
    ix2 = np.minimum(a[:, None, 2], b[None, :, 2]); iy2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(ix2 - ix1, 0, None) * np.clip(iy2 - iy1, 0, None)
    aa = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1]); ab = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    return inter / (aa[:, None] + ab[None, :] - inter + 1e-9)


def evaluate(gt, pred, frame_ids):
    pair, last, idsw, matches, n_gt, n_pr = {}, {}, 0, 0, 0, 0
    gids, pids = set(), set()
    for f in frame_ids:
        g, p = gt.get(f, []), pred.get(f, [])
        n_gt += len(g); n_pr += len(p)
        gids.update(x[0] for x in g); pids.update(x[0] for x in p)
        if not g or not p:
            continue
        gb = np.array([x[1:] for x in g]); pb = np.array([x[1] for x in p])
        iou = iou_matrix(gb, pb)
        ok = iou >= 0.5
        for a, b in zip(*np.nonzero(ok)):
            key = (g[a][0], p[b][0]); pair[key] = pair.get(key, 0) + 1
        r, c = linear_sum_assignment(np.where(ok, 1 - iou, 1e6))
        for a, b in zip(r, c):
            if not ok[a, b]:
                continue
            gid, pid = g[a][0], p[b][0]
            matches += 1
            if gid in last and last[gid] != pid:
                idsw += 1
            last[gid] = pid
    idtp = 0
    if pair:
        gi = {k: i for i, k in enumerate(sorted({k[0] for k in pair}))}
        pi = {k: i for i, k in enumerate(sorted({k[1] for k in pair}))}
        M = np.zeros((len(gi), len(pi)))
        for (a, b), n in pair.items():
            M[gi[a], pi[b]] = n
        r, c = linear_sum_assignment(-M)
        idtp = M[r, c].sum()
    return dict(
        IDF1=100 * 2 * idtp / max(n_gt + n_pr, 1),
        IDsw=idsw,
        MOTA=100 * (1 - ((n_gt - matches) + (n_pr - matches) + idsw) / max(n_gt, 1)),
        IDs=len(pids), GT_IDs=len(gids),
    )


# ------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seqs", nargs="+", default=["MOT17-02-DPM"])
    ap.add_argument("--max-frames", type=int, default=0, help="0 = whole sequence")
    ap.add_argument("--conf", type=float, default=0.3)
    ap.add_argument("--imgsz", type=int, default=960)
    ap.add_argument("--thresholds", nargs="+", type=float, default=[0.80, 0.85, 0.90, 0.95])
    ap.add_argument("--max-gap", type=int, default=150, help="max frames a track may be lost")
    ap.add_argument("--speed", type=float, default=40.0, help="max px/frame for re-link distance")
    ap.add_argument("--no-cache", action="store_true")
    args = ap.parse_args()

    det = YOLODetector(MODEL)
    rows_out = []
    for seq in args.seqs:
        print(f"\n=== {seq}")
        gt = load_gt(seq)
        frames = run_tracker(det, seq, args)
        fids = sorted(frames)
        T = build_tracklets(frames)
        variants = [("ByteTrack (baseline)", None)] + [(f"Colour ReID @ {t:.2f}", t) for t in args.thresholds]
        print(f"{'Method':<24}{'IDF1':>7}{'IDsw':>7}{'MOTA':>8}{'IDs':>6}{'GT':>5}")
        for name, thr in variants:
            pred = apply_ids(frames, stitch(T, thr, args.max_gap, args.speed))
            m = evaluate(gt, pred, fids)
            print(f"{name:<24}{m['IDF1']:>7.1f}{m['IDsw']:>7d}{m['MOTA']:>8.1f}{m['IDs']:>6d}{m['GT_IDs']:>5d}")
            rows_out.append(dict(seq=seq, method=name, **m))
    os.makedirs(OUT, exist_ok=True)
    path = os.path.join(OUT, "mot17_results.csv")
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows_out[0])); w.writeheader(); w.writerows(rows_out)
    print(f"\nSaved {path}")


if __name__ == "__main__":
    main()