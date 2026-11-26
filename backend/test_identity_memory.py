"""Logic tests for the identity engine and database (no GPU / no YOLO weights needed).
Run from the project root:  python backend/test_identity_memory.py"""
import sys
import tempfile
import types
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from backend.app.core import argus_engine as E
from backend.app.database import db


class FakeBoxes:
    def __init__(s, ids, boxes):
        s.id = types.SimpleNamespace(int=lambda: types.SimpleNamespace(tolist=lambda: ids))
        s.xyxy = types.SimpleNamespace(cpu=lambda: types.SimpleNamespace(numpy=lambda: np.array(boxes, float)))
        s.cls = types.SimpleNamespace(int=lambda: types.SimpleNamespace(tolist=lambda: [0] * len(ids)))
        s.conf = types.SimpleNamespace(tolist=lambda: [0.9] * len(ids))


class FakeModel:
    names, predictor = {0: "person"}, None

    def __init__(s, script):
        s.script = list(script)

    def track(s, frame, **kw):
        ids, boxes = s.script.pop(0)
        return [types.SimpleNamespace(boxes=FakeBoxes(ids, boxes) if ids else types.SimpleNamespace(id=None),
                                      names=s.names)]


def frame(colours=((0, 0, 255), (255, 0, 0))):
    f = np.zeros((200, 200, 3), np.uint8)
    for x in (40, 150):  # same upper/lower colours ("person") at two places
        f[20:70, x:x + 40], f[70:120, x:x + 40] = colours
    return f


def engine(script, **kw):
    kw.setdefault("confirm_frames", 1)  # old tests: judge a new tracker on its first frame
    E._MODELS["fake"] = FakeModel(script)
    return E.ArgusEngine("fake", device="cpu", **kw)


def run(eng, n, f=None):
    f = frame() if f is None else f
    for _ in range(n):
        eng.process(f)


def run_each(eng, frames):
    for f in frames:
        eng.process(f)


NEAR, FAR = [[40, 20, 80, 120]], [[150, 20, 190, 120]]


def test_recovery_and_lifecycle():
    e = engine([([1], NEAR)] * 6 + [([], NEAR)] * 8 + [([7], NEAR)], reid=True, reid_thr=0.8, missing_after=2)
    run(e, 15)
    kinds = [x["kind"] for x in e.events]
    assert len(e.ids) == 1 and "RECOVERED" in kinds, kinds
    assert "MISSING" in kinds, "lifecycle MISSING event not logged"
    print("ok: recovery + MISSING event")


def test_location_gate_blocks_far_match():
    e = engine([([1], NEAR)] * 6 + [([], NEAR)] * 8 + [([7], FAR)], reid=True, reid_thr=0.8, missing_after=2)
    run(e, 15)
    assert len(e.ids) == 2, "far-away look-alike must NOT be merged"
    e2 = engine([([1], NEAR)] * 6 + [([], NEAR)] * 8 + [([7], FAR)], reid=True, reid_thr=0.8, missing_after=2, gate=False)
    run(e2, 15)
    assert len(e2.ids) == 1, "without the gate the look-alike is merged"
    print("ok: location gating")


def test_reid_off_never_merges():
    e = engine([([1], NEAR)] * 3 + [([], NEAR)] * 5 + [([7], NEAR)], reid=False)
    run(e, 9)
    assert len(e.ids) == 2
    print("ok: ReID off creates a new identity")


def test_global_numbering():
    e = engine([([1], NEAR)], start_no=50)
    run(e, 1)
    assert "ARGUS-0050" in [r["argus_id"] for r in e.snapshot()]
    print("ok: start_no")


def test_database():
    db.DB_PATH = Path(tempfile.mkdtemp()) / "t.db"
    db.init()
    e = engine([([1], NEAR)] * 3, start_no=db.max_argus_no() + 1)
    run(e, 3)
    sid = db.create_session("t.mp4")
    db.finish_session(sid, 3, "", e.snapshot(), e.events, e.tracks, 1.5)
    assert db.max_argus_no() == 1 and len(db.tracks(sid)) == 3 and len(db.identities(sid)) == 1
    db.rename(sid, "ARGUS-0001", "Person A")
    assert db.identities(sid)[0]["name"] == "Person A"
    print("ok: database round trip, rename, ID numbering")


GREEN = frame(((0, 255, 0), (0, 255, 0)))


def test_confirmation_ignores_flicker():
    e = engine([([1], NEAR)] * 2 + [([], NEAR)] * 12, reid=True, confirm_frames=4)
    run(e, 14)
    assert len(e.ids) == 0 and not e.tracks, "a 2-frame flicker must not become an identity"
    print("ok: flicker never becomes an identity")


def test_recovery_with_confirmation_keeps_one_id():
    script = [([1], NEAR)] * 8 + [([], NEAR)] * 10 + [([7], NEAR)] * 6
    e = engine(script, reid=True, reid_thr=0.8, missing_after=2, confirm_frames=4)
    run(e, len(script))
    assert len(e.ids) == 1, [r["argus_id"] for r in e.snapshot()]
    assert e.ids[1]["reid"] == 1 and e.recovered == 1
    ids_in_tracks = {t[1] for t in e.tracks}
    assert ids_in_tracks == {"ARGUS-0001"}, ids_in_tracks
    frames_with_id = sorted(t[0] for t in e.tracks)
    assert frames_with_id == list(range(1, 9)) + list(range(19, 25)), frames_with_id  # every visible frame is recorded
    ev = [x for x in e.events if x["kind"] == "RECOVERED"][0]
    assert "similarity" in ev["detail"] and "gone" in ev["detail"]
    print("ok: same person keeps ONE id after disappearing (with confirmation)")


def test_different_person_gets_new_id():
    script = [([1], NEAR)] * 8 + [([], NEAR)] * 10 + [([7], NEAR)] * 6
    e = engine(script, reid=True, reid_thr=0.8, missing_after=2, confirm_frames=4)
    run(e, 8)
    run(e, 10)
    run(e, 6, GREEN)  # a green person appears where the red/blue one was
    assert len(e.ids) == 2, "a different-looking person must not inherit the old ID"
    print("ok: different clothing -> new id")


def test_far_lookalike_with_confirmation():
    script = [([1], NEAR)] * 8 + [([], NEAR)] * 10 + [([7], FAR)] * 6
    e = engine(script, reid=True, reid_thr=0.8, missing_after=2, confirm_frames=4)
    run(e, len(script))
    assert len(e.ids) == 2
    print("ok: far look-alike not merged (with confirmation)")


def test_too_long_gone_not_recovered():
    script = [([1], NEAR)] * 8 + [([], NEAR)] * 150 + [([7], NEAR)] * 6
    e = engine(script, reid=True, reid_thr=0.8, missing_after=2, confirm_frames=4, reid_window=60)
    run(e, len(script))
    assert len(e.ids) == 2
    print("ok: re-link window respected")


def test_embedding_separates_clothes():
    red = E.colour_embed(frame()[20:120, 40:80])
    same = E.colour_embed(frame()[20:120, 150:190])
    green = E.colour_embed(GREEN[20:120, 40:80])
    assert float(red @ same) > 0.95 and float(red @ green) < 0.5
    print("ok: colour embedding: same clothes high, different clothes low")


if __name__ == "__main__":
    for t in (test_recovery_and_lifecycle, test_location_gate_blocks_far_match, test_reid_off_never_merges,
              test_global_numbering, test_database, test_confirmation_ignores_flicker,
              test_recovery_with_confirmation_keeps_one_id, test_different_person_gets_new_id,
              test_far_lookalike_with_confirmation, test_too_long_gone_not_recovered,
              test_embedding_separates_clothes):
        t()
    print("ALL TESTS PASSED")
