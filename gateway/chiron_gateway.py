"""Chiron gateway: BLE central + event store + local dashboard.

Runs on the Raspberry Pi 4 (or any machine with a BLE radio). One asyncio
process holds everything -- bleak clients, the SQLite writer and the FastAPI
app all share the event loop, so there is no IPC and nothing extra to fail.

    python chiron_gateway.py                 # connect to real boards
    python chiron_gateway.py --simulate      # no hardware needed

Simulation mode replays the ACTUAL exported golden-vector manifests, including
the offline INT8 predictions, so the dashboard shows exactly what the hardware
will show. That makes the whole UI testable and demo-rehearsable before a board
is ever plugged in, and it doubles as the reference the real device is checked
against.

See docs/09_DASHBOARD_SPEC.md.
"""

from __future__ import annotations

import argparse
import base64
import asyncio
import contextlib
import json
import logging
import os
import random
import sqlite3
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any

from fastapi import (FastAPI, File, Form, HTTPException, UploadFile,
                     WebSocket, WebSocketDisconnect)
from fastapi.responses import (FileResponse, PlainTextResponse,
                               Response)
from fastapi.staticfiles import StaticFiles

import chiron_live

# Default microphone capture length for the live cardiac path.
CAPTURE_DEFAULT_SECONDS = 5

# Reference media (PNG/WAV exports of the golden vectors). Ships alongside
# the gateway; absent installations simply return 404 and the UI hides the
# preview rather than breaking.
MEDIA_DIR = os.environ.get(
    "CHIRON_MEDIA_DIR",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "media"))

from chiron_protocol import (
    CHAR_CLASS_NAMES, CHAR_COMMAND, CHAR_NODE_INFO, CHAR_RESULT, CHAR_STATUS,
    CHAR_UPLOAD, CMD_RUN_UPLOADED, UPLOADED_VECTOR_ID,
    CLASS_UNCERTAIN, CMD_RESET_STATS, CMD_RUN_ALL, CMD_RUN_ONE, CMD_SET_INTERVAL,
    CMD_SET_MODE, CMD_SET_THRESHOLD, DEFAULT_THRESHOLD,
    NODE_NAMES, NODE_TYPE_CARDIO, NODE_TYPE_PULMO,
    SERVICE_UUID, InferenceResult, NodeInfo, NodeStatus, pack_command,
    parse_class_names,
)

LOG = logging.getLogger("chiron")

# BlueZ allows a single discovery session per adapter; concurrent scans from
# the two node tasks fail with org.bluez.Error.InProgress. See ble_node_task.
_SCAN_LOCK = asyncio.Lock()

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent
DB_PATH = HERE / "events.db"
STATIC_DIR = HERE / "static"

RETENTION_DAYS_DEFAULT = 30

# Per-case detail (measured indicators + offline model output), produced by
# core/build_case_data.py. Absent file is non-fatal: the dashboard simply hides the
# case-detail view rather than the whole gateway failing to start.
CASE_DATA_PATH = HERE / "case_data.json"
try:
    CASE_DATA = json.loads(CASE_DATA_PATH.read_text(encoding="utf-8"))
except Exception as _exc:  # noqa: BLE001 - startup must never depend on this
    CASE_DATA = {"meta": {}, "cases": {}}
    logging.getLogger("chiron").warning(
        "case_data.json unavailable (%s); case detail disabled", _exc)


# ===========================================================================
# Storage
# ===========================================================================
SCHEMA = """
CREATE TABLE IF NOT EXISTS events (
    id                INTEGER PRIMARY KEY AUTOINCREMENT,
    ts_utc            TEXT    NOT NULL,
    node_type         INTEGER NOT NULL,
    node_address      TEXT    NOT NULL,
    class_id          INTEGER NOT NULL,
    class_name        TEXT    NOT NULL,
    confidence        REAL    NOT NULL,
    top2_class_id     INTEGER,
    top2_class_name   TEXT,
    top2_confidence   REAL,
    vector_id         INTEGER,
    expected_class_id INTEGER,
    expected_name     TEXT,
    is_correct        INTEGER,
    front_ms          INTEGER,
    inference_ms      INTEGER,
    self_test         INTEGER,
    sequence          INTEGER
);
CREATE INDEX IF NOT EXISTS idx_events_ts   ON events(ts_utc);
CREATE INDEX IF NOT EXISTS idx_events_node ON events(node_type, ts_utc);

CREATE TABLE IF NOT EXISTS nodes (
    node_type        INTEGER PRIMARY KEY,
    address          TEXT,
    name             TEXT,
    class_names      TEXT,
    model_version    TEXT,
    build_id         TEXT,
    arena_used_bytes INTEGER,
    model_size_bytes INTEGER,
    vector_count     INTEGER,
    input_description TEXT,
    modality         TEXT,
    last_seen_utc    TEXT,
    connected        INTEGER DEFAULT 0
);
"""


class Store:
    def __init__(self, path: Path):
        self.db = sqlite3.connect(path, check_same_thread=False)
        self.db.row_factory = sqlite3.Row
        self.db.executescript(SCHEMA)
        self.db.commit()

    def record_event(self, row: dict) -> int:
        cols = ", ".join(row)
        marks = ", ".join("?" * len(row))
        cur = self.db.execute(f"INSERT INTO events ({cols}) VALUES ({marks})",
                              list(row.values()))
        self.db.commit()
        return int(cur.lastrowid)

    def upsert_node(self, node_type: int, **fields) -> None:
        fields["node_type"] = node_type
        cols = ", ".join(fields)
        marks = ", ".join("?" * len(fields))
        updates = ", ".join(f"{k}=excluded.{k}" for k in fields if k != "node_type")
        self.db.execute(
            f"INSERT INTO nodes ({cols}) VALUES ({marks}) "
            f"ON CONFLICT(node_type) DO UPDATE SET {updates}",
            list(fields.values()))
        self.db.commit()

    def nodes(self) -> list[dict]:
        return [dict(r) for r in self.db.execute("SELECT * FROM nodes ORDER BY node_type")]

    def known_address(self, node_type: int) -> str:
        """Last address this node was seen at, across restarts.

        Used when a board is connected at the BlueZ level but not advertising --
        a peripheral goes silent while connected, so if the gateway died without
        closing the link the board is invisible to a scan despite being healthy.
        """
        r = self.db.execute("SELECT address FROM nodes WHERE node_type=?",
                            (node_type,)).fetchone()
        return (r["address"] if r and r["address"] else "")

    def events(self, limit: int = 100, node_type: int | None = None) -> list[dict]:
        if node_type is None:
            q = "SELECT * FROM events ORDER BY id DESC LIMIT ?"
            args: tuple = (limit,)
        else:
            q = "SELECT * FROM events WHERE node_type=? ORDER BY id DESC LIMIT ?"
            args = (node_type, limit)
        return [dict(r) for r in self.db.execute(q, args)]

    def stats(self) -> dict:
        out = {}
        for r in self.db.execute(
                "SELECT node_type, COUNT(*) n, "
                "SUM(CASE WHEN is_correct=1 THEN 1 ELSE 0 END) ok, "
                "SUM(CASE WHEN is_correct IS NOT NULL THEN 1 ELSE 0 END) graded, "
                "SUM(CASE WHEN class_id=255 THEN 1 ELSE 0 END) uncertain, "
                "AVG(inference_ms) mean_inf, AVG(front_ms) mean_front "
                "FROM events GROUP BY node_type"):
            d = dict(r)
            graded = d["graded"] or 0
            d["accuracy"] = (d["ok"] / graded) if graded else None
            out[str(d["node_type"])] = d
        return out

    def purge(self) -> int:
        n = self.db.execute("SELECT COUNT(*) c FROM events").fetchone()["c"]
        self.db.execute("DELETE FROM events")
        self.db.commit()
        return int(n)

    def enforce_retention(self, days: int) -> int:
        """Storage limitation: drop anything older than the window."""
        cutoff = (datetime.now(timezone.utc) - timedelta(days=days)).isoformat()
        cur = self.db.execute("DELETE FROM events WHERE ts_utc < ?", (cutoff,))
        self.db.commit()
        return cur.rowcount


# ===========================================================================
# Live state shared with the web layer
# ===========================================================================
@dataclass
class NodeState:
    node_type: int
    name: str
    address: str = ""
    connected: bool = False
    class_names: list[str] = field(default_factory=list)
    info: dict = field(default_factory=dict)
    status: dict = field(default_factory=dict)
    last_result: dict | None = None
    last_seen: str = ""
    threshold: float = 0.60

    def class_name(self, cid: int) -> str:
        if cid == CLASS_UNCERTAIN:
            return "UNCERTAIN"
        if 0 <= cid < len(self.class_names):
            return self.class_names[cid]
        return f"class{cid}"

    def to_dict(self) -> dict:
        return {
            "node_type": self.node_type, "name": self.name,
            "address": self.address, "connected": self.connected,
            "class_names": self.class_names, "info": self.info,
            "status": self.status, "last_result": self.last_result,
            "last_seen": self.last_seen, "threshold": self.threshold,
        }


class Hub:
    """Fan-out to websocket clients plus the shared node registry."""

    def __init__(self, store: Store):
        self.store = store
        self.nodes: dict[int, NodeState] = {
            NODE_TYPE_CARDIO: NodeState(NODE_TYPE_CARDIO, NODE_NAMES[NODE_TYPE_CARDIO]),
            NODE_TYPE_PULMO: NodeState(NODE_TYPE_PULMO, NODE_NAMES[NODE_TYPE_PULMO]),
        }
        self.clients: set[WebSocket] = set()
        self.senders: dict[int, Any] = {}      # node_type -> command sender
        self.uploaders: dict[int, Any] = {}    # node_type -> bulk sample sender
        self.live_meta: dict[int, Any] = {}    # node_type -> provenance of the pending live sample

    async def broadcast(self, msg: dict) -> None:
        dead = []
        for ws in list(self.clients):
            try:
                await ws.send_json(msg)
            except Exception:
                dead.append(ws)
        for ws in dead:
            self.clients.discard(ws)

    async def on_result(self, node_type: int, res: InferenceResult) -> None:
        node = self.nodes[node_type]
        node.last_seen = res.ts_utc
        cname = node.class_name(res.class_id)
        ename = (node.class_name(res.expected_class)
                 if res.ground_truth_known else None)

        row = {
            "ts_utc": res.ts_utc, "node_type": node_type,
            "node_address": node.address, "class_id": res.class_id,
            "class_name": cname, "confidence": res.confidence,
            "top2_class_id": res.top2_class_id,
            "top2_class_name": node.class_name(res.top2_class_id),
            "top2_confidence": res.top2_confidence,
            "vector_id": res.vector_id,
            "expected_class_id": (res.expected_class
                                  if res.ground_truth_known else None),
            "expected_name": ename,
            "is_correct": (None if res.is_correct is None else int(res.is_correct)),
            "front_ms": res.front_ms, "inference_ms": res.inference_ms,
            "self_test": int(bool(res.flags & 0x02)), "sequence": res.sequence,
        }

        # record_event builds its INSERT from the row's keys, so the row handed
        # to it must contain columns and nothing else. Provenance is presentation
        # metadata: it is attached AFTER the insert, for the broadcast and the
        # card, and never reaches the database.
        self.store.record_event(row)

        # A live sample carries no ground truth and, for cardiac audio, comes
        # from a signal chain the model was never validated on. Attach that
        # judgement so the dashboard can qualify the result instead of
        # presenting an unvalidated class as a finding.
        if res.vector_id == UPLOADED_VECTOR_ID:
            row = dict(row, live=True, **(self.live_meta.pop(node_type, {}) or {}))

        node.last_result = row
        await self.broadcast({"type": "inference", **row})

    async def on_status(self, node_type: int, st: NodeStatus) -> None:
        node = self.nodes[node_type]
        node.status = st.to_dict()
        node.last_seen = datetime.now(timezone.utc).isoformat()
        await self.broadcast({"type": "status", "node_type": node_type,
                              **node.status})

    async def set_connected(self, node_type: int, connected: bool,
                            address: str = "") -> None:
        node = self.nodes[node_type]
        node.connected = connected
        if address:
            node.address = address

        fields = dict(connected=int(connected), name=node.name,
                      last_seen_utc=datetime.now(timezone.utc).isoformat())
        # Never overwrite a stored address with an empty one. On a fresh process
        # node.address is "" until the first successful connect, so the
        # disconnect path would erase the very address the not-advertising
        # fallback in ble_node_task depends on -- and a board whose link BlueZ
        # is still holding never advertises, so nothing could re-discover it.
        if node.address:
            fields["address"] = node.address
        self.store.upsert_node(node_type, **fields)
        await self.broadcast({"type": "connection", "node_type": node_type,
                              "connected": connected, "address": node.address})

    async def send_sample(self, node_type: int, payload: bytes,
                          meta: dict | None = None) -> float:
        """Push a live sample to a node and ask it to classify it."""
        self.live_meta[node_type] = meta or {}
        up = self.uploaders.get(node_type)
        if up is None:
            raise HTTPException(503, f"node {node_type} not connected")
        elapsed = await up(payload)
        await self.send_command(node_type, CMD_RUN_UPLOADED, 0)
        return elapsed

    async def send_command(self, node_type: int, opcode: int, param: int) -> None:
        sender = self.senders.get(node_type)
        if sender is None:
            raise HTTPException(503, f"node {node_type} not connected")
        await sender(pack_command(opcode, param))


# ===========================================================================
# Real BLE transport
# ===========================================================================
async def ble_node_task(hub: Hub, node_type: int) -> None:
    """Discover, connect and stay connected to one node, forever.

    Reconnects with exponential backoff. Deliberately independent per node, so
    one board being absent never blocks the other -- the single most common way
    a two-node demo dies.
    """
    from bleak import BleakClient, BleakScanner

    want_name = NODE_NAMES[node_type]
    backoff = 1.0

    while True:
        try:
            LOG.info("[%s] scanning...", want_name)
            device = None
            # BlueZ permits only ONE discovery session per adapter, so two node
            # tasks scanning at once make one of them fail with
            # org.bluez.Error.InProgress -- forever, since both retry together.
            # Windows/WinRT tolerates concurrent scans, so this bug is invisible
            # on a laptop and fatal on the Pi. Serialise the scans.
            # The advertising payload only carries 8 characters of the GAP name,
            # so a node configured as "Chiron-Cardio" goes on air as "Chiron-C".
            # Matching more than 8 chars matches nothing at all. 8 is still
            # enough to separate Chiron-C(ardio) from Chiron-P(ulmo), and the
            # node type is re-verified from NodeInfo once connected.
            want_prefix = want_name[:8].lower()
            async with _SCAN_LOCK:
                for d in await BleakScanner.discover(timeout=8.0):
                    if d.name and d.name.lower().startswith(want_prefix):
                        device = d
                        break

            # A peripheral stops advertising while it is connected. If the
            # gateway was killed without closing the link, BlueZ can still hold
            # the connection, so the board is invisible to a scan even though it
            # is healthy and right there. Fall back to the last known address.
            target = device
            if target is None:
                known = hub.nodes[node_type].address or hub.store.known_address(node_type)
                if known:
                    LOG.info("[%s] not advertising; trying known address %s",
                             want_name, known)
                    target = known
            if target is None:
                raise RuntimeError("not found")

            LOG.info("[%s] connecting to %s", want_name,
                     getattr(target, "address", target))
            async with BleakClient(target) as client:
                node = hub.nodes[node_type]
                resolved_addr = getattr(target, "address", target)
                node.address = resolved_addr

                raw = await client.read_gatt_char(CHAR_CLASS_NAMES)
                node.class_names = parse_class_names(raw)
                info = NodeInfo.unpack(await client.read_gatt_char(CHAR_NODE_INFO))
                node.info = info.to_dict()

                hub.store.upsert_node(
                    node_type, address=resolved_addr, name=want_name,
                    class_names="\n".join(node.class_names),
                    model_version=info.model_version, build_id=info.build_id,
                    arena_used_bytes=info.arena_used_bytes,
                    model_size_bytes=info.model_size_bytes,
                    vector_count=info.vector_count,
                    input_description=node.info["input_description"],
                    modality=node.info["modality"])

                async def _send(payload: bytes, c=client) -> None:
                    await c.write_gatt_char(CHAR_COMMAND, payload, response=False)
                hub.senders[node_type] = _send

                # Bulk path for live samples. Kept separate from _send so a
                # partially written sample can never be mistaken for a command.
                async def _upload(payload: bytes, c=client) -> float:
                    return await chiron_live.push_sample(c, CHAR_UPLOAD, payload)
                hub.uploaders[node_type] = _upload

                # Push the confidence floor on every (re)connect so the UI and the
                # firmware cannot disagree. Without this the slider showed the
                # NodeState default (0.60) while the device applied its own
                # compiled-in default -- a floor the demo was not actually using.
                node.threshold = DEFAULT_THRESHOLD[node_type]
                await _send(pack_command(CMD_SET_THRESHOLD,
                                         int(round(node.threshold * 1000))))

                loop = asyncio.get_running_loop()

                def on_result(_h, data: bytearray) -> None:
                    with contextlib.suppress(Exception):
                        res = InferenceResult.unpack(bytes(data))
                        asyncio.run_coroutine_threadsafe(
                            hub.on_result(node_type, res), loop)

                def on_status(_h, data: bytearray) -> None:
                    with contextlib.suppress(Exception):
                        st = NodeStatus.unpack(bytes(data))
                        asyncio.run_coroutine_threadsafe(
                            hub.on_status(node_type, st), loop)

                await client.start_notify(CHAR_RESULT, on_result)
                await client.start_notify(CHAR_STATUS, on_status)

                await hub.set_connected(node_type, True, resolved_addr)
                LOG.info("[%s] connected: %d classes, %d vectors, arena %d B",
                         want_name, info.class_count, info.vector_count,
                         info.arena_used_bytes)
                backoff = 1.0

                while client.is_connected:
                    await asyncio.sleep(1.0)

        except Exception as exc:                       # noqa: BLE001
            LOG.warning("[%s] %s -- retrying in %.0fs", want_name, exc, backoff)

            # Self-heal a stale BlueZ link. If a board is re-powered while the
            # Pi holds the connection, BlueZ can keep reporting Connected: yes
            # for a session that is actually dead. The node then neither
            # advertises (peripherals go quiet while connected) nor accepts a
            # connect-by-address, so it is invisible until something clears the
            # link. Observed for real after a board reset; without this the node
            # stays offline indefinitely.
            addr = hub.nodes[node_type].address or hub.store.known_address(node_type)
            if addr and backoff >= 4.0:
                with contextlib.suppress(Exception):
                    proc = await asyncio.create_subprocess_exec(
                        "bluetoothctl", "disconnect", addr,
                        stdout=asyncio.subprocess.DEVNULL,
                        stderr=asyncio.subprocess.DEVNULL)
                    await asyncio.wait_for(proc.wait(), timeout=8)
                    LOG.info("[%s] cleared a stale link to %s", want_name, addr)
        finally:
            hub.senders.pop(node_type, None)
            with contextlib.suppress(Exception):
                await hub.set_connected(node_type, False)

        await asyncio.sleep(backoff)
        backoff = min(backoff * 2, 30.0)


# ===========================================================================
# Simulated transport (no hardware)
# ===========================================================================
class SimulatedNode:
    """Replays the real exported golden-vector manifest for one node.

    Everything it emits -- class names, expected labels, the model's own
    predictions and confidences -- comes from vectors/chiron_*_manifest.json,
    i.e. the actual offline INT8 results. The dashboard therefore sees exactly
    what the hardware will send, which is what makes this useful for rehearsal
    rather than just a placeholder.
    """

    def __init__(self, hub: Hub, node_type: int, manifest: Path):
        self.hub = hub
        self.node_type = node_type
        self.m = json.loads(manifest.read_text())
        self.vectors = self.m["vectors"]
        self.classes = self.m["classes"]
        self.seq = 0
        self.runs = 0
        self.correct = 0
        self.mode = 0
        self.interval_ms = 2000
        # Defaults mirror the firmware: 0.50 for the 5-class cardio node, 0.55
        # for the 2-class pulmo node (0.50 could never fire on a binary model).
        self.threshold = 0.50 if node_type == NODE_TYPE_CARDIO else 0.55
        self.next_index = 0
        self.pending: list[tuple[int, int]] = []
        self.t0 = time.time()

    def _info(self) -> NodeInfo:
        """Node identity, matching what the firmware reports.

        Values come from the manifest where it has them and from the measured
        design budget otherwise; the real device overwrites all of this with
        arena_used_bytes() and friends the moment it connects.
        """
        modality = self.m.get("modality", "audio")
        name = "cardio" if self.node_type == NODE_TYPE_CARDIO else "pulmo"
        tfl = REPO_ROOT / "models" / f"chiron_{name}_v1_int8.tflite"
        size = tfl.stat().st_size if tfl.exists() else 0

        if modality == "image":
            win = int(self.m.get("image_size", 96))
            sr = 0
            arena = 200_000
        else:
            samples = int(self.m.get("window_samples", 24000))
            rate = int(self.m.get("sample_rate", 8000))
            win = int(samples * 1000 / rate / 100)     # units of 100 ms
            sr = rate // 1000
            arena = 60_000
        return NodeInfo(
            node_type=self.node_type, class_count=len(self.classes),
            vector_count=len(self.vectors), model_version="1.0.0",
            arena_used_bytes=arena, model_size_bytes=size,
            window_or_size=win, sample_rate_khz=sr, build_id="51mu1a7e")

    async def start(self) -> None:
        node = self.hub.nodes[self.node_type]
        node.class_names = list(self.classes)
        info = self._info()
        node.info = info.to_dict()
        node.address = f"SIM:{self.node_type:02X}"
        node.threshold = self.threshold

        self.hub.store.upsert_node(
            self.node_type, address=node.address, name=node.name,
            class_names="\n".join(self.classes),
            model_version=info.model_version, build_id=info.build_id,
            arena_used_bytes=info.arena_used_bytes,
            model_size_bytes=info.model_size_bytes,
            vector_count=info.vector_count,
            input_description=node.info["input_description"],
            modality=node.info["modality"])

        async def _send(payload: bytes) -> None:
            opcode = payload[0]
            param = int.from_bytes(payload[1:3], "little")
            if opcode == CMD_RUN_ONE:
                self.pending.append((param, 0))
            elif opcode == CMD_RUN_ALL:
                self.pending.extend((i, 1) for i in range(len(self.vectors)))
            elif opcode == CMD_SET_MODE:
                self.mode = param
            elif opcode == CMD_SET_INTERVAL:
                self.interval_ms = param or 2000
            elif opcode == CMD_SET_THRESHOLD:
                self.threshold = max(0.05, min(0.99, param / 1000.0))
                self.hub.nodes[self.node_type].threshold = self.threshold
            elif opcode == CMD_RESET_STATS:
                self.runs = self.correct = self.seq = 0

        self.hub.senders[self.node_type] = _send
        await self.hub.set_connected(self.node_type, True, node.address)
        await self._emit_status(1)

    async def _emit_status(self, state: int) -> None:
        await self.hub.on_status(self.node_type, NodeStatus(
            state=state, error_code=0, runs_completed=self.runs,
            runs_correct=self.correct, uptime_s=int(time.time() - self.t0)))

    async def _run_vector(self, index: int, self_test: bool) -> None:
        v = self.vectors[index % len(self.vectors)]
        probs = v["probs"]
        order = sorted(range(len(probs)), key=lambda i: -probs[i])
        best, second = order[0], (order[1] if len(order) > 1 else order[0])
        conf = probs[best]

        # Timings from the measured design budget, with mild jitter so the
        # dashboard looks alive rather than synthetic.
        front = (25 if self.node_type == NODE_TYPE_CARDIO else 2) + random.randint(-2, 3)
        infer = (19 if self.node_type == NODE_TYPE_CARDIO else 42) + random.randint(-3, 4)

        cls = best if conf >= self.threshold else CLASS_UNCERTAIN
        expected = v["expected_label"]
        flags = 0
        if cls != CLASS_UNCERTAIN:
            flags |= 0x01
        if self_test:
            flags |= 0x02
        if expected is not None:
            flags |= 0x04
            if cls == expected:
                flags |= 0x08

        self.runs += 1
        if cls == expected:
            self.correct += 1

        res = InferenceResult(
            node_type=self.node_type, class_id=cls, confidence=conf,
            flags=flags, vector_id=index, expected_class=expected,
            top2_class_id=second, top2_confidence=probs[second],
            front_ms=max(front, 1), inference_ms=max(infer, 1), sequence=self.seq)
        self.seq += 1
        await self.hub.on_result(self.node_type, res)

    async def run(self) -> None:
        await self.start()
        last_auto = 0.0
        while True:
            if self.pending:
                await self._emit_status(2)
                while self.pending:
                    idx, st = self.pending.pop(0)
                    await self._run_vector(idx, bool(st))
                    await asyncio.sleep(0.25)      # visible pacing for the demo
                await self._emit_status(1)
            elif self.mode == 2 and (time.time() - last_auto) * 1000 >= self.interval_ms:
                last_auto = time.time()
                await self._run_vector(self.next_index, False)
                self.next_index = (self.next_index + 1) % len(self.vectors)
                await self._emit_status(1)
            else:
                await asyncio.sleep(0.1)


# ===========================================================================
# Web app
# ===========================================================================
def build_app(hub: Hub, store: Store, simulate: bool,
              retention_days: int) -> FastAPI:
    app = FastAPI(title="Project Chiron Gateway")

    if STATIC_DIR.is_dir():
        app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

    @app.get("/")
    async def index():
        f = STATIC_DIR / "index.html"
        if not f.exists():
            return PlainTextResponse("dashboard not installed", 404)
        # index.html carries the ?v= stamps that bust the CSS/JS caches, so it
        # must never itself be served from cache. Without this the browser keeps
        # an old document, keeps requesting the old stamped assets, and the
        # dashboard silently runs a previous deployment.
        return FileResponse(f, headers={
            "Cache-Control": "no-store, no-cache, must-revalidate, max-age=0",
            "Pragma": "no-cache",
        })

    @app.get("/api/case/{node_type}/{vector_id}")
    async def api_case(node_type: int, vector_id: int):
        """Per-case detail for the clinical view.

        Returns only facts: what was measured from the signal, and what the model
        produced. The interpretation text (urgency, suggested actions, typical
        associations) lives client-side in static/i18n.js keyed by class, because
        it is generic reference material rather than anything derived from this
        recording -- and the UI labels it that way.
        """
        node = CASE_DATA.get("cases", {}).get(str(node_type), {})
        rec = node.get(str(vector_id))
        if rec is None:
            raise HTTPException(404, f"no case {node_type}/{vector_id}")
        return rec

    @app.get("/api/meta")
    async def api_meta():
        return CASE_DATA.get("meta", {})

    @app.get("/api/nodes")
    async def api_nodes():
        return {"simulate": simulate,
                "nodes": [n.to_dict() for n in hub.nodes.values()]}

    @app.get("/api/events")
    async def api_events(limit: int = 100, node: int | None = None):
        return {"events": store.events(limit=min(limit, 1000), node_type=node)}

    @app.get("/api/stats")
    async def api_stats():
        return {"stats": store.stats(), "retention_days": retention_days}

    @app.post("/api/command/{node_type}")
    async def api_command(node_type: int, body: dict):
        opcode = int(body.get("opcode", 0))
        param = int(body.get("param", 0))
        if opcode not in (CMD_RUN_ONE, CMD_RUN_ALL, CMD_SET_MODE,
                          CMD_SET_INTERVAL, CMD_RESET_STATS, CMD_SET_THRESHOLD):
            raise HTTPException(400, f"bad opcode {opcode}")
        await hub.send_command(node_type, opcode, param)
        if opcode == CMD_SET_THRESHOLD:
            hub.nodes[node_type].threshold = max(0.05, min(0.99, param / 1000.0))
            await hub.broadcast({"type": "threshold", "node_type": node_type,
                                 "threshold": hub.nodes[node_type].threshold})
        return {"ok": True, "opcode": opcode, "param": param}

    _spec_cache: dict = {}

    @app.get("/api/spectrogram/{node_type}/{vector_id}")
    async def api_spectrogram(node_type: int, vector_id: int):
        """Spectrogram of a reference heart sound, rendered from its WAV.

        Cardiac only: the pulmonary reference cases are radiographs and already
        display as images. Rendered on first request and cached, so stepping
        back and forth through cases costs nothing.
        """
        if node_type != NODE_TYPE_CARDIO:
            raise HTTPException(404, "spectrograms apply to heart sound only")
        key = (node_type, vector_id)
        if key not in _spec_cache:
            folder = os.path.join(MEDIA_DIR, "cardio_audio")
            if not os.path.isdir(folder):
                raise HTTPException(404, "media not installed on this gateway")
            prefix = "vec%03d_" % vector_id
            path = next((os.path.join(folder, f) for f in sorted(os.listdir(folder))
                         if f.startswith(prefix) and f.endswith(".wav")), None)
            if path is None:
                raise HTTPException(404, f"no audio for vector {vector_id}")
            with open(path, "rb") as fh:
                wav = fh.read()
            try:
                pcm = chiron_live.audio_to_payload(wav, normalise=False)
            except Exception as exc:
                raise HTTPException(500, f"could not decode reference audio: {exc}")
            uri = chiron_live.spectrogram_png(pcm)
            if not uri:
                raise HTTPException(501, "spectrogram rendering unavailable")
            _spec_cache[key] = base64.b64decode(uri.split(",", 1)[1])
        return Response(content=_spec_cache[key], media_type="image/png",
                        headers={"Cache-Control": "public, max-age=86400"})

    @app.get("/api/media/{node_type}/{vector_id}")
    async def api_media(node_type: int, vector_id: int):
        """Serve the source image or audio behind a reference case.

        The vectors exist as C arrays inside the firmware, which a browser cannot
        render. MEDIA_DIR holds the same bytes exported as PNG and WAV, so the
        operator can look at the radiograph or listen to the heart sound that
        produced a result rather than taking the classification on trust.
        """
        sub = "pulmo_images" if node_type == NODE_TYPE_PULMO else "cardio_audio"
        ext = ".png" if node_type == NODE_TYPE_PULMO else ".wav"
        folder = os.path.join(MEDIA_DIR, sub)
        if not os.path.isdir(folder):
            raise HTTPException(404, "media not installed on this gateway")
        prefix = "vec%03d_" % vector_id
        for fn in sorted(os.listdir(folder)):
            if fn.startswith(prefix) and fn.endswith(ext):
                return FileResponse(os.path.join(folder, fn),
                                    media_type="image/png" if ext == ".png" else "audio/wav",
                                    headers={"Cache-Control": "public, max-age=86400"})
        raise HTTPException(404, f"no media for node {node_type} vector {vector_id}")

    @app.post("/api/live/image")
    async def api_live_image(file: UploadFile = File(...)):
        """Classify an uploaded radiograph on the pulmonary node."""
        raw = await file.read()
        if not raw:
            raise HTTPException(400, "empty upload")
        if len(raw) > 12 * 1024 * 1024:
            raise HTTPException(413, "image too large (12 MB limit)")
        try:
            payload = chiron_live.image_to_payload(raw)
        except RuntimeError as exc:
            raise HTTPException(501, str(exc))
        except Exception as exc:
            raise HTTPException(400, f"could not decode image: {exc}")
        elapsed = await hub.send_sample(NODE_TYPE_PULMO, payload,
                                        meta={"source": file.filename})
        return {"ok": True, "node_type": NODE_TYPE_PULMO,
                "bytes": len(payload), "upload_seconds": round(elapsed, 2),
                "filename": file.filename}

    @app.post("/api/live/audio")
    async def api_live_audio(seconds: int = Form(CAPTURE_DEFAULT_SECONDS),
                             normalise: str = Form("auto"),
                             file: UploadFile | None = File(None)):
        """Classify audio on the cardiac node.

        With no file, records from the MEMS microphone attached to this host.
        With a file, uses the supplied WAV instead, which is how a previously
        captured recording can be replayed through the same path.
        """
        if file is not None:
            raw = await file.read()
            if not raw:
                raise HTTPException(400, "empty upload")
            source = file.filename
        else:
            try:
                raw = await chiron_live.capture_wav(max(1, min(int(seconds), 30)))
            except Exception as exc:
                raise HTTPException(503, f"capture failed: {exc}")
            source = f"microphone {seconds}s"
        try:
            payload = chiron_live.audio_to_payload(raw, normalise=normalise)
        except Exception as exc:
            raise HTTPException(400, f"could not decode audio: {exc}")
        prov = chiron_live.signal_provenance(payload)
        prov["source"] = source
        # Every heart-sound path gets a spectrogram. `capture` marks microphone
        # input specifically: that is the one path where no class is asserted,
        # and the spectrogram alone can no longer signal which path we are on.
        prov["capture"] = (file is None)
        prov["spectrogram"] = chiron_live.spectrogram_png(payload)
        elapsed = await hub.send_sample(NODE_TYPE_CARDIO, payload, meta=prov)
        return {"ok": True, "node_type": NODE_TYPE_CARDIO,
                "provenance": prov,
                "bytes": len(payload), "upload_seconds": round(elapsed, 2),
                "source": source}

    @app.post("/api/purge")
    async def api_purge():
        """Right to erasure: wipe the local event store on demand."""
        n = store.purge()
        await hub.broadcast({"type": "purged", "deleted": n})
        return {"ok": True, "deleted": n}

    @app.get("/api/export.csv")
    async def api_export():
        rows = store.events(limit=100000)
        if not rows:
            return PlainTextResponse("", media_type="text/csv")
        cols = list(rows[0])
        lines = [",".join(cols)]
        for r in rows:
            lines.append(",".join("" if r[c] is None else str(r[c]) for c in cols))
        return PlainTextResponse("\n".join(lines), media_type="text/csv")

    @app.websocket("/ws")
    async def ws_endpoint(ws: WebSocket):
        await ws.accept()
        hub.clients.add(ws)
        try:
            await ws.send_json({"type": "hello", "simulate": simulate,
                                "nodes": [n.to_dict() for n in hub.nodes.values()]})
            while True:
                await ws.receive_text()
        except WebSocketDisconnect:
            pass
        finally:
            hub.clients.discard(ws)

    return app


async def retention_task(store: Store, days: int) -> None:
    while True:
        removed = store.enforce_retention(days)
        if removed:
            LOG.info("retention: removed %d events older than %d days", removed, days)
        await asyncio.sleep(3600)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--simulate", action="store_true",
                    help="replay exported golden vectors instead of using BLE")
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--retention-days", type=int, default=RETENTION_DAYS_DEFAULT)
    ap.add_argument("--log-level", default="INFO")
    args = ap.parse_args()

    logging.basicConfig(level=args.log_level,
                        format="%(asctime)s %(levelname)-7s %(message)s")

    store = Store(DB_PATH)
    hub = Hub(store)
    app = build_app(hub, store, args.simulate, args.retention_days)

    @app.on_event("startup")
    async def _startup():
        asyncio.create_task(retention_task(store, args.retention_days))
        if args.simulate:
            LOG.info("SIMULATION MODE -- replaying exported golden vectors")
            for nt, name in ((NODE_TYPE_CARDIO, "cardio"), (NODE_TYPE_PULMO, "pulmo")):
                mf = REPO_ROOT / "vectors" / f"chiron_{name}_vectors_manifest.json"
                if mf.exists():
                    asyncio.create_task(SimulatedNode(hub, nt, mf).run())
                else:
                    LOG.warning("no manifest for %s at %s", name, mf)
        else:
            for nt in (NODE_TYPE_CARDIO, NODE_TYPE_PULMO):
                asyncio.create_task(ble_node_task(hub, nt))

    import uvicorn
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
