# Chiron Gateway & Dashboard

BLE central, event store and local web dashboard for Project Chiron. One
asyncio process: `bleak` clients, SQLite and FastAPI all share the event loop,
so there is no IPC and nothing extra to fail during a demo.

```
[Cardio node BRD2608A] --BLE-->  \
                                  >--- Raspberry Pi 4 ---- http://<pi>:8000 --> browser
[Pulmo  node BRD2709A] --BLE-->  /     bleak + SQLite + FastAPI
```

## Run

```bash
pip install -r requirements.txt
```

**With hardware** — discovers `Chiron-Cardio` / `Chiron-Pulmo` by name and
connects to each independently:

```bash
python chiron_gateway.py
```

**Without hardware** — replays the exported golden-vector manifests:

```bash
python chiron_gateway.py --simulate
```

Then open <http://localhost:8000>.

### Why the simulator matters

`--simulate` is not a placeholder. It replays
`vectors/chiron_*_vectors_manifest.json`, which holds the **actual offline INT8
predictions, confidences and expected labels** for every held-out sample. The
dashboard therefore shows exactly what the hardware will show, so the whole UI
can be built, tested and rehearsed before a board is plugged in — and once the
boards arrive, any disagreement between simulated and real output is a genuine
firmware bug rather than a UI question.

## Endpoints

| Method | Path | Purpose |
|---|---|---|
| GET | `/` | Dashboard |
| GET | `/api/nodes` | Identity, connection state, class names, model/arena facts |
| GET | `/api/events?limit=&node=` | Event history, newest first |
| GET | `/api/stats` | Per-node accuracy tally and mean latency |
| POST | `/api/command/{node_type}` | `{"opcode":2,"param":0}` → GATT write |
| POST | `/api/purge` | Erase the local event store |
| GET | `/api/export.csv` | Full event log |
| WS | `/ws` | Live event stream |

Opcodes: `1` RUN_ONE(vector), `2` RUN_ALL, `3` SET_MODE(0/1/2),
`4` SET_INTERVAL(ms), `5` RESET_STATS.

## Design notes

**No framework, no CDN.** The dashboard is plain HTML/CSS/JS and the chart is
~40 lines of canvas. Nothing is fetched at runtime, so it works with no
internet — which is the point of an offline-first medical demonstrator, and
also removes a build step from the Pi.

**No hard-coded class names.** Labels come from each node's Class Names GATT
characteristic, so the same page renders a 5-class cardio node beside a 2-class
pulmo node and would survive a model change without edits.

**Ground truth is always on screen.** Every row shows expected vs predicted
with ✔/✘, and the accuracy tile is computed from those. The dashboard *proves*
results rather than just displaying them — hand a judge the Self-Test button
and the question "is this pre-recorded?" answers itself.

## Expected on-device vs offline accuracy

The device applies a **0.60 confidence floor** and reports `UNCERTAIN` below it,
which the offline evaluation does not. So the live tally is slightly lower by
construction:

| | offline | on device | difference |
|---|---|---|---|
| Cardio | 25/25 (100 %) | 24/25 (96.0 %) | vector 06 (AS) at 0.586 → UNCERTAIN |
| Pulmo | 22/24 (91.7 %) | 21/24 (87.5 %) | vector 03 (Normal) at 0.531 → UNCERTAIN |

That gap is the safety behaviour working, not a regression: the device declines
to answer rather than guessing. Both genuine pulmo misses (vectors 00 and 22)
are real model errors and appear in both columns.

## Deploy on the Pi

```bash
sudo apt install -y python3-venv bluez
python3 -m venv ~/chiron/.venv && ~/chiron/.venv/bin/pip install -r requirements.txt
sudo cp chiron-gateway.service /etc/systemd/system/
sudo systemctl enable --now chiron-gateway
```

## Troubleshooting

**"disconnected — reconnecting" in the footer, and Self-Test does nothing.**
Install `uvicorn[standard]`, not bare `uvicorn`. Plain uvicorn ships **no
WebSocket implementation**, so `/ws` answers `Unsupported upgrade request` and
every live update is silently dropped.

This one is nasty because almost everything still looks fine: the REST API
answers 200, the boards genuinely run (their LED blinks), and the results reach
`events.db` correctly. Only the *push* to the browser is missing, so the page
freezes at whatever it loaded with. Check the server log for:

```
WARNING:  No supported WebSocket library detected.
```

Verify the socket directly rather than trusting the REST API:

```bash
python -c "import websockets,asyncio,json; asyncio.run((lambda: (lambda ws: None))())" 2>/dev/null; python - <<'EOF'
import asyncio, json, websockets
async def m():
    async with websockets.connect("ws://127.0.0.1:8000/ws") as ws:
        print(json.loads(await ws.recv()))
asyncio.run(m())
EOF
```

A `{"type": "hello", ...}` frame means the socket is healthy.

**A characteristic you just added does not appear.** BlueZ has cached a stale
GATT database — `sudo rm -rf /var/lib/bluetooth/*` and reconnect.

**Both nodes show offline on the Pi while the laptop still has them.** A BLE
peripheral accepts **one central at a time**. Stop the laptop gateway before
starting the Pi one, or the two will fight over the connections.
