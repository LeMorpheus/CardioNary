"""Chiron BLE GATT protocol — the single source of truth for the wire format.

Mirrors firmware/chiron_*/src/chiron_ble.c exactly. Both sides must agree on
these UUIDs and byte layouts, so they live in one file rather than being
duplicated between the firmware and the gateway.

See docs/08_BLE_GATT_SPEC.md.
"""

from __future__ import annotations

import struct

from chiron_crypto import unseal, SEALED_LEN
from dataclasses import dataclass, asdict, field
from datetime import datetime, timezone

# ---------------------------------------------------------------------------
# UUIDs (generated once, must match config/btconf/gatt_configuration.btconf)
# ---------------------------------------------------------------------------
SERVICE_UUID      = "d29b1431-95b1-466b-88c4-e1f3221f8227"
CHAR_RESULT       = "54a5acfc-eb45-4f56-9f5f-241bc6673faf"
CHAR_COMMAND      = "01306e7a-d517-4ff8-b85c-3f4cc7d1bbfa"
CHAR_NODE_INFO    = "b7ae1031-57c6-4dc1-abe9-0dd370990326"
CHAR_STATUS       = "176a66a3-ef40-43ee-88df-bd8eb97a8659"
CHAR_CLASS_NAMES  = "c088e877-db71-472d-8b9d-32ecd8895b61"
CHAR_UPLOAD       = "9c4f2a18-3b7e-4c21-9d6a-5e8f0b21c743"

# ---------------------------------------------------------------------------
# A live (uploaded) sample is reported with this vector id and no expected
# class, so it can never be folded into the golden-vector accuracy tally.
UPLOADED_VECTOR_ID = 0xFFFF

NODE_TYPE_CARDIO = 0x01
NODE_TYPE_PULMO  = 0x02
NODE_NAMES = {NODE_TYPE_CARDIO: "Chiron-Cardio", NODE_TYPE_PULMO: "Chiron-Pulmo"}

CMD_RUN_ONE      = 0x01
CMD_RUN_ALL      = 0x02
CMD_SET_MODE     = 0x03
CMD_SET_INTERVAL = 0x04
CMD_RESET_STATS  = 0x05
# param = confidence floor x 1000 (e.g. 550 -> 0.55). Runtime-adjustable so the
# accuracy/safety trade-off can be demonstrated live rather than argued about.
CMD_SET_THRESHOLD = 0x06
CMD_RUN_UPLOADED  = 0x07

# Must match CHIRON_CONFIDENCE_FLOOR_DEFAULT in each node's inc/chiron_infer.h.
# The gateway pushes these on connect so the slider cannot display a floor the
# device is not actually applying.
#
# Cardio uses 0.50 because with 5 classes a softmax max CAN fall below it, so the
# gate is reachable. Pulmo needs 0.55 because with 2 classes the winning
# probability is ALWAYS >= 0.50, making a 0.50 floor impossible to trigger.
DEFAULT_THRESHOLD = {NODE_TYPE_CARDIO: 0.50, NODE_TYPE_PULMO: 0.55}

CLASS_UNCERTAIN = 0xFF

STATE_NAMES = {0: "boot", 1: "idle", 2: "running", 3: "error"}

# Result flag bits
FLAG_CONFIRMED   = 0x01   # above the confidence floor
FLAG_SELF_TEST   = 0x02   # part of a sweep
FLAG_GT_KNOWN    = 0x04   # expected_class is meaningful
FLAG_MATCH       = 0x08   # prediction == expected


@dataclass
class InferenceResult:
    """16-byte notification from the Inference Result characteristic."""

    node_type: int
    class_id: int
    confidence: float
    flags: int
    vector_id: int
    expected_class: int
    top2_class_id: int
    top2_confidence: float
    front_ms: int
    inference_ms: int
    sequence: int
    ts_utc: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    STRUCT = "<BBBBHBBBBHHH"          # 16 bytes

    @classmethod
    def unpack(cls, data: bytes) -> "InferenceResult":
        """Accepts both wire formats.

        16 bytes = legacy plaintext, 28 = sealed (counter || ciphertext || tag).
        Handling both means a firmware and a gateway that disagree about
        sealing still interoperate instead of the dashboard going silent.
        """
        if len(data) >= SEALED_LEN:
            opened = unseal(data[:SEALED_LEN])
            if opened is not None:
                data = opened

        if len(data) < 16:
            raise ValueError(f"result payload too short: {len(data)} bytes")
        (node_type, class_id, conf, flags, vector_id, expected, top2, top2c,
         _reserved, front_ms, inf_ms, seq) = struct.unpack(cls.STRUCT, data[:16])
        return cls(
            node_type=node_type,
            class_id=class_id,
            confidence=conf / 255.0,
            flags=flags,
            vector_id=vector_id,
            expected_class=expected,
            top2_class_id=top2,
            top2_confidence=top2c / 255.0,
            front_ms=front_ms,
            inference_ms=inf_ms,
            sequence=seq,
        )

    def pack(self) -> bytes:
        return struct.pack(
            self.STRUCT, self.node_type, self.class_id,
            int(round(self.confidence * 255)), self.flags, self.vector_id,
            self.expected_class, self.top2_class_id,
            int(round(self.top2_confidence * 255)), 0,
            self.front_ms, self.inference_ms, self.sequence,
        )

    @property
    def is_uncertain(self) -> bool:
        return self.class_id == CLASS_UNCERTAIN

    @property
    def ground_truth_known(self) -> bool:
        return bool(self.flags & FLAG_GT_KNOWN)

    @property
    def is_correct(self) -> bool | None:
        if not self.ground_truth_known:
            return None
        return bool(self.flags & FLAG_MATCH)

    def to_dict(self) -> dict:
        d = asdict(self)
        d.update(
            is_uncertain=self.is_uncertain,
            is_correct=self.is_correct,
            self_test=bool(self.flags & FLAG_SELF_TEST),
        )
        return d


@dataclass
class NodeInfo:
    """20-byte Node Info characteristic."""

    node_type: int
    class_count: int
    vector_count: int
    model_version: str
    arena_used_bytes: int
    model_size_bytes: int
    window_or_size: int
    sample_rate_khz: int
    build_id: str

    STRUCT = "<BBHIIHBBI"             # 20 bytes

    @classmethod
    def unpack(cls, data: bytes) -> "NodeInfo":
        if len(data) < 20:
            raise ValueError(f"node info too short: {len(data)} bytes")
        (nt, cc, vc, ver, arena, msize, win, sr, build) = struct.unpack(
            cls.STRUCT, data[:20])
        return cls(
            node_type=nt, class_count=cc, vector_count=vc,
            model_version=f"{(ver >> 16) & 0xFF}.{(ver >> 8) & 0xFF}.{ver & 0xFF}",
            arena_used_bytes=arena, model_size_bytes=msize * 16,
            window_or_size=win, sample_rate_khz=sr,
            build_id=f"{build:08x}",
        )

    def pack(self) -> bytes:
        maj, mnr, pch = (int(x) for x in self.model_version.split("."))
        ver = (maj << 16) | (mnr << 8) | pch
        return struct.pack(
            self.STRUCT, self.node_type, self.class_count, self.vector_count,
            ver, self.arena_used_bytes, self.model_size_bytes // 16,
            self.window_or_size, self.sample_rate_khz, int(self.build_id, 16))

    def to_dict(self) -> dict:
        d = asdict(self)
        # The image node has no sample rate; report the field it actually uses.
        d["modality"] = "image" if self.sample_rate_khz == 0 else "audio"
        if d["modality"] == "image":
            d["input_description"] = f"{self.window_or_size}x{self.window_or_size} grayscale"
        else:
            d["input_description"] = (f"{self.window_or_size * 100} ms @ "
                                      f"{self.sample_rate_khz} kHz")
        return d


@dataclass
class NodeStatus:
    """8-byte Status characteristic."""

    state: int
    error_code: int
    runs_completed: int
    runs_correct: int
    uptime_s: int

    STRUCT = "<BBHHH"                 # 8 bytes

    @classmethod
    def unpack(cls, data: bytes) -> "NodeStatus":
        if len(data) < 8:
            raise ValueError(f"status too short: {len(data)} bytes")
        st, err, done, ok, up = struct.unpack(cls.STRUCT, data[:8])
        return cls(state=st, error_code=err, runs_completed=done,
                   runs_correct=ok, uptime_s=up)

    def pack(self) -> bytes:
        return struct.pack(self.STRUCT, self.state, self.error_code,
                           self.runs_completed, self.runs_correct, self.uptime_s)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["state_name"] = STATE_NAMES.get(self.state, "unknown")
        return d


def pack_command(opcode: int, param: int = 0) -> bytes:
    """4-byte Command characteristic write."""
    return struct.pack("<BHB", opcode, param, 0)


def parse_class_names(data: bytes) -> list[str]:
    """Newline-separated labels, so the dashboard needs no hard-coded tables."""
    return [n for n in data.decode("utf-8", "replace").split("\n") if n]
