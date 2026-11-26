"""SQLite storage: sessions, identities, events and per-frame tracks."""
import sqlite3
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
DB_PATH = ROOT / "outputs" / "argus_x.db"


def _conn():
    DB_PATH.parent.mkdir(exist_ok=True)
    c = sqlite3.connect(DB_PATH)
    c.row_factory = sqlite3.Row
    return c


def run(sql, args=(), fetch=False):
    c = _conn()
    try:
        cur = c.execute(sql, args)
        c.commit()
        return [dict(r) for r in cur.fetchall()] if fetch else cur.lastrowid
    finally:
        c.close()


def init():
    c = _conn()
    try:
        c.executescript("""
        CREATE TABLE IF NOT EXISTS sessions(id INTEGER PRIMARY KEY AUTOINCREMENT, source TEXT, started TEXT,
            frames INTEGER, identities INTEGER, output TEXT);
        CREATE TABLE IF NOT EXISTS identities(session_id INTEGER, argus_id TEXT, name TEXT, cls TEXT, tracker_id INTEGER,
            status TEXT, first_frame INTEGER, last_frame INTEGER, frames_seen INTEGER, reid_count INTEGER, confidence REAL);
        CREATE TABLE IF NOT EXISTS events(session_id INTEGER, frame INTEGER, argus_id TEXT, kind TEXT, detail TEXT);
        CREATE TABLE IF NOT EXISTS tracks(session_id INTEGER, frame INTEGER, argus_id TEXT, x1 INTEGER, y1 INTEGER,
            x2 INTEGER, y2 INTEGER);
        CREATE INDEX IF NOT EXISTS ix_tracks ON tracks(session_id, argus_id);
        """)
        try:
            c.execute("ALTER TABLE sessions ADD COLUMN seconds REAL")
        except sqlite3.OperationalError:
            pass  # column already exists
        c.commit()
    finally:
        c.close()


def create_session(source):
    return run("INSERT INTO sessions(source,started,frames,identities,output) VALUES(?,?,?,?,?)",
               (source, time.strftime("%Y-%m-%d %H:%M:%S"), 0, 0, ""))


def finish_session(sid, frames, output, rows, events, tracks, seconds):
    c = _conn()
    try:
        c.execute("UPDATE sessions SET frames=?, identities=?, output=?, seconds=? WHERE id=?",
                  (frames, len(rows), output, seconds, sid))
        c.executemany("INSERT INTO identities VALUES(?,?,?,?,?,?,?,?,?,?,?)",
                      [(sid, r["argus_id"], r["name"], r["cls"], r["tracker_id"], r["status"], r["first_frame"],
                        r["last_frame"], r["frames_seen"], r["reid_count"], r["confidence"]) for r in rows])
        c.executemany("INSERT INTO events VALUES(?,?,?,?,?)",
                      [(sid, e["frame"], e["argus_id"], e["kind"], e["detail"]) for e in events])
        c.executemany("INSERT INTO tracks VALUES(?,?,?,?,?,?,?)", [(sid, *t) for t in tracks])
        c.commit()
    finally:
        c.close()


def max_argus_no():
    r = run("SELECT MAX(CAST(SUBSTR(argus_id,7) AS INTEGER)) AS m FROM identities", fetch=True)
    return (r[0]["m"] or 0) if r else 0


def sessions():
    return run("SELECT * FROM sessions ORDER BY id DESC", fetch=True)


def identities(sid=None):
    if sid is None:
        return run("SELECT i.*, s.source FROM identities i JOIN sessions s ON s.id=i.session_id "
                   "ORDER BY i.session_id DESC, i.argus_id", fetch=True)
    return run("SELECT argus_id,name,cls,tracker_id,status,first_frame,last_frame,frames_seen,reid_count,confidence "
               "FROM identities WHERE session_id=?", (sid,), fetch=True)


def events(sid):
    return run("SELECT frame,argus_id,kind,detail FROM events WHERE session_id=? ORDER BY frame", (sid,), fetch=True)


def tracks(sid):
    return run("SELECT frame,argus_id,x1,y1,x2,y2 FROM tracks WHERE session_id=? ORDER BY frame", (sid,), fetch=True)


def rename(sid, argus_id, name):
    run("UPDATE identities SET name=? WHERE session_id=? AND argus_id=?", (name, sid, argus_id))
