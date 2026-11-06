import json
import sqlite3
import time
import uuid


class Conflict(ValueError):
    pass


class JobStore:
    def __init__(self, path):
        self.path = str(path)
        with self.db() as db:
            db.execute("""CREATE TABLE IF NOT EXISTS jobs (
              id TEXT PRIMARY KEY, idem TEXT UNIQUE NOT NULL, request_hash TEXT NOT NULL,
              compiled TEXT NOT NULL, state TEXT NOT NULL, proxy_id TEXT,
              created REAL NOT NULL, error TEXT)""")

    def db(self):
        db = sqlite3.connect(self.path, timeout=10)
        db.row_factory = sqlite3.Row
        return db

    def get(self, job_id):
        with self.db() as db:
            row = db.execute("SELECT * FROM jobs WHERE id=?", (job_id,)).fetchone()
        return dict(row) if row else None

    def lookup(self, key, request_hash):
        with self.db() as db:
            row = db.execute("SELECT * FROM jobs WHERE idem=?", (key,)).fetchone()
        if row and row["request_hash"] != request_hash:
            raise Conflict("Idempotency-Key was already used with different parameters")
        return dict(row) if row else None

    def claim(self, key, request_hash, compiled, job_id):
        with self.db() as db:
            db.execute("BEGIN IMMEDIATE")
            row = db.execute("SELECT * FROM jobs WHERE idem=?", (key,)).fetchone()
            if row:
                if row["request_hash"] != request_hash:
                    raise Conflict("Idempotency-Key was already used with different parameters")
                return dict(row), False
            db.execute("INSERT INTO jobs VALUES (?,?,?,?,?,?,?,?)",
                       (job_id, key, request_hash, json.dumps(compiled, ensure_ascii=False), "submitting", None, time.time(), None))
        return self.get(job_id), True

    def update(self, job_id, state, proxy_id=None, error=None):
        with self.db() as db:
            db.execute("UPDATE jobs SET state=?, proxy_id=COALESCE(?,proxy_id), error=? WHERE id=?", (state, proxy_id, error, job_id))

    def list(self):
        with self.db() as db:
            return [dict(row) for row in db.execute("SELECT * FROM jobs ORDER BY created DESC LIMIT 100")]

    def by_proxy(self, proxy_id):
        with self.db() as db:
            row = db.execute("SELECT * FROM jobs WHERE proxy_id=?", (proxy_id,)).fetchone()
        return dict(row) if row else None

