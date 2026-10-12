"""Reviewer decisions that outlive a single run.

One SQLite file per engine home (``<app data>/decisions.sqlite``) holds the
current decision for every reviewed key plus an append-only audit log. Each
run's review queue is matched against it so that a filer name you already
approved or rejected is never shown as "to decide" again.

Why SQLite from the standard library rather than DuckDB: nothing extra to bundle
into the sidecar, single-writer workloads of a few thousand rows, and a file
anyone can open with any SQLite viewer.
"""
from __future__ import annotations

import csv
import sqlite3
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

DECISIONS = ("approved", "rejected")
_COLUMNS = ("key", "debt_state", "secp_state", "secp_name", "decision", "suggested", "mapped_name",
            "firm_id", "match_method", "match_score", "confidence", "run_id", "decided_at", "norm_name", "source")

_SCHEMA = """
CREATE TABLE IF NOT EXISTS decisions (
    key          TEXT PRIMARY KEY,
    debt_state   TEXT,
    secp_state   TEXT,
    secp_name    TEXT NOT NULL,
    decision     TEXT NOT NULL CHECK (decision IN ('approved', 'rejected')),
    suggested    TEXT,
    mapped_name  TEXT,
    firm_id      INTEGER,
    match_method TEXT,
    match_score  REAL,
    confidence   TEXT,
    run_id       TEXT,
    decided_at   TEXT NOT NULL,
    norm_name    TEXT,                      -- normalised secp_name: decisions propagate across state pairs / spellings
    source       TEXT                       -- NULL = a reviewer decided; 'inherited:<key>' = propagated from that key
);
CREATE INDEX IF NOT EXISTS decisions_decision ON decisions (decision);
CREATE TABLE IF NOT EXISTS decision_log (
    id          INTEGER PRIMARY KEY,
    key         TEXT NOT NULL,
    action      TEXT NOT NULL,              -- approved | rejected | cleared
    suggested   TEXT,
    mapped_name TEXT,
    firm_id     INTEGER,
    run_id      TEXT,
    at          TEXT NOT NULL
);
"""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def norm_name(value: Any) -> str:
    """Case/whitespace-insensitive comparison key for a bank name."""
    return "" if value is None else " ".join(str(value).split()).upper()


class DecisionStore:
    """Thread-safe access to the decisions database (one connection, serialised)."""

    _open: Dict[Path, "DecisionStore"] = {}
    _open_lock = threading.Lock()

    @classmethod
    def at(cls, path: Path) -> "DecisionStore":
        """One store object per file, so every run shares the connection and cache version."""
        path = Path(path).expanduser().resolve()
        with cls._open_lock:
            store = cls._open.get(path)
            if store is None:
                store = cls._open[path] = cls(path)
            return store

    def __init__(self, path: Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._db = sqlite3.connect(self.path, check_same_thread=False, isolation_level=None)
        self._db.row_factory = sqlite3.Row
        self._db.execute("PRAGMA journal_mode=WAL")
        self._db.execute("PRAGMA synchronous=NORMAL")
        self._db.executescript(_SCHEMA)
        self._migrate()
        self.version = 0  # bumped on every write; callers use it to invalidate caches

    def _migrate(self) -> None:
        """Add columns introduced after the first release to an existing database."""
        have = {r["name"] for r in self._db.execute("PRAGMA table_info(decisions)")}
        with self._db:
            if "norm_name" not in have:
                self._db.execute("ALTER TABLE decisions ADD COLUMN norm_name TEXT")
            if "source" not in have:
                self._db.execute("ALTER TABLE decisions ADD COLUMN source TEXT")
            self._db.execute("CREATE INDEX IF NOT EXISTS decisions_norm_name ON decisions (norm_name)")

    # ------------------------------------------------------------------ reads
    def get(self, key: str) -> Optional[Dict[str, Any]]:
        with self._lock:
            row = self._db.execute("SELECT * FROM decisions WHERE key = ?", (key,)).fetchone()
        return dict(row) if row else None

    def get_many(self, keys: Iterable[str]) -> Dict[str, Dict[str, Any]]:
        keys = list(dict.fromkeys(keys))
        out: Dict[str, Dict[str, Any]] = {}
        with self._lock:
            for i in range(0, len(keys), 900):  # SQLite's default variable limit is 999
                chunk = keys[i:i + 900]
                q = f"SELECT * FROM decisions WHERE key IN ({','.join('?' * len(chunk))})"
                out.update({r["key"]: dict(r) for r in self._db.execute(q, chunk)})
        return out

    def by_norm_name(self, norm_names: Iterable[str]) -> Dict[str, List[Dict[str, Any]]]:
        """Every decision whose normalised filer name is one of ``norm_names``."""
        names = [n for n in dict.fromkeys(norm_names) if n]
        out: Dict[str, List[Dict[str, Any]]] = {}
        with self._lock:
            for i in range(0, len(names), 900):
                chunk = names[i:i + 900]
                q = f"SELECT * FROM decisions WHERE norm_name IN ({','.join('?' * len(chunk))}) ORDER BY decided_at DESC"
                for r in self._db.execute(q, chunk):
                    out.setdefault(r["norm_name"], []).append(dict(r))
        return out

    def cleared_keys(self) -> set:
        """Keys whose most recent audit action is 'cleared' - the reviewer undid them
        on purpose, so propagation must not decide them again."""
        with self._lock:
            rows = self._db.execute(
                "SELECT key FROM decision_log WHERE action = 'cleared' AND id IN "
                "(SELECT MAX(id) FROM decision_log GROUP BY key)").fetchall()
        return {r["key"] for r in rows}

    def missing_norm_names(self) -> List[Dict[str, Any]]:
        with self._lock:
            return [dict(r) for r in self._db.execute("SELECT key, secp_name FROM decisions WHERE norm_name IS NULL")]

    def set_norm_names(self, pairs: Iterable[tuple]) -> int:
        """Backfill ``norm_name`` for rows written before the column existed."""
        pairs = list(pairs)
        with self._lock, self._db:
            self._db.executemany("UPDATE decisions SET norm_name = ? WHERE key = ?", [(n, k) for k, n in pairs])
        return len(pairs)

    def approved(self) -> List[Dict[str, Any]]:
        with self._lock:
            return [dict(r) for r in self._db.execute("SELECT * FROM decisions WHERE decision = 'approved'")]

    def counts(self) -> Dict[str, int]:
        with self._lock:
            rows = self._db.execute("SELECT decision, COUNT(*) AS n FROM decisions GROUP BY decision").fetchall()
        out = {"approved": 0, "rejected": 0}
        out.update({r["decision"]: r["n"] for r in rows})
        out["total"] = out["approved"] + out["rejected"]
        return out

    def list(
        self, *, decision: str = "", search: str = "", run_id: str = "",
        offset: int = 0, limit: int = 100, sort: str = "decided_at", descending: bool = True,
    ) -> Dict[str, Any]:
        where, args = [], []
        if decision:
            where.append("decision = ?")
            args.append(decision)
        if run_id:
            where.append("run_id = ?")
            args.append(run_id)
        if search.strip():
            like = f"%{search.strip()}%"
            where.append("(secp_name LIKE ? OR suggested LIKE ? OR mapped_name LIKE ?)")
            args += [like, like, like]
        clause = f" WHERE {' AND '.join(where)}" if where else ""
        sort = sort if sort in _COLUMNS else "decided_at"
        order = f" ORDER BY {sort} {'DESC' if descending else 'ASC'}, key"
        with self._lock:
            total = self._db.execute(f"SELECT COUNT(*) FROM decisions{clause}", args).fetchone()[0]
            rows = self._db.execute(f"SELECT * FROM decisions{clause}{order} LIMIT ? OFFSET ?",
                                    [*args, limit, offset]).fetchall()
        return {"total": int(total), "offset": offset, "limit": limit, "columns": list(_COLUMNS),
                "rows": [dict(r) for r in rows], "counts": self.counts()}

    # ----------------------------------------------------------------- writes
    def set(self, key: str, **fields: Any) -> None:
        """Insert or replace the decision for ``key`` (and append to the audit log)."""
        decision = fields.get("decision")
        if decision not in DECISIONS:
            raise ValueError(f"decision must be one of {DECISIONS}, got {decision!r}")
        if "|" not in key:
            raise ValueError(f"malformed key {key!r}")
        debt, secp, name = key.split("|", 2)
        row = {
            "key": key, "debt_state": debt or None, "secp_state": secp or None, "secp_name": name,
            "decision": decision, "suggested": fields.get("suggested"), "mapped_name": fields.get("mapped_name"),
            "firm_id": _int_or_none(fields.get("firm_id")), "match_method": fields.get("match_method"),
            "match_score": fields.get("match_score"), "confidence": fields.get("confidence"),
            "run_id": fields.get("run_id"), "decided_at": fields.get("decided_at") or _now(),
            "norm_name": fields.get("norm_name"), "source": fields.get("source"),
        }
        cols = ", ".join(row)
        sets = ", ".join(f"{c} = excluded.{c}" for c in row if c != "key")
        with self._lock, self._db:
            self._db.execute(
                f"INSERT INTO decisions ({cols}) VALUES ({','.join('?' * len(row))}) "
                f"ON CONFLICT(key) DO UPDATE SET {sets}", list(row.values()))
            action = decision if not row["source"] else f"{decision} ({row['source']})"
            self._db.execute(
                "INSERT INTO decision_log (key, action, suggested, mapped_name, firm_id, run_id, at) VALUES (?,?,?,?,?,?,?)",
                (key, action, row["suggested"], row["mapped_name"], row["firm_id"], row["run_id"], row["decided_at"]))
            self.version += 1

    def clear(self, key: str, run_id: Optional[str] = None) -> bool:
        with self._lock, self._db:
            cur = self._db.execute("DELETE FROM decisions WHERE key = ?", (key,))
            if cur.rowcount:
                self._db.execute("INSERT INTO decision_log (key, action, run_id, at) VALUES (?, 'cleared', ?, ?)",
                                 (key, run_id, _now()))
                self.version += 1
            return bool(cur.rowcount)

    def import_legacy(self, decisions: Dict[str, Dict[str, Any]], run_id: str) -> int:
        """One-off import of a run's old ``decisions.json``; existing keys win."""
        n = 0
        for key, d in decisions.items():
            if d.get("decision") in DECISIONS and self.get(key) is None:
                self.set(key, decision=d["decision"], suggested=d.get("suggested"), mapped_name=d.get("mapped_name"),
                         firm_id=d.get("firm_id"), match_method=d.get("match_method"), match_score=d.get("match_score"),
                         confidence=d.get("confidence"), run_id=run_id, decided_at=d.get("at"))
                n += 1
        return n

    def export_csv(self, path: Path) -> int:
        with self._lock:
            rows = self._db.execute("SELECT * FROM decisions ORDER BY decided_at DESC").fetchall()
        with open(path, "w", newline="", encoding="utf-8") as fh:
            w = csv.writer(fh)
            w.writerow(_COLUMNS)
            w.writerows(tuple(r[c] for c in _COLUMNS) for r in rows)
        return len(rows)

    def history(self, key: str) -> List[Dict[str, Any]]:
        with self._lock:
            return [dict(r) for r in self._db.execute(
                "SELECT * FROM decision_log WHERE key = ? ORDER BY id DESC", (key,))]


def _int_or_none(value: Any) -> Optional[int]:
    if value is None or value == "":
        return None
    try:
        f = float(value)
    except (TypeError, ValueError):
        return None
    return None if f != f else int(f)
