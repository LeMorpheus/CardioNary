"""Gateway side of the telemetry seal. Mirrors firmware/chiron_*/src/chiron_crypto.c.

Wire format:  counter(4, big-endian) || ciphertext(16) || tag(8)

  confidentiality  AES-CTR, keystream = AES-ECB(key, 'CHRN' || 0*8 || counter)
  authenticity     AES-CMAC over (counter || ciphertext), truncated to 8 bytes

Encrypt-then-MAC, so the tag is verified BEFORE anything is decrypted and a
forged payload is discarded without its plaintext ever being formed.

Replay: the counter is monotonic per node per boot. A counter not greater than
the last one seen from that node is rejected. It resets on reboot, which is why
the check is per-connection rather than persisted -- a stricter scheme needs a
session nonce negotiated at connect, which is the natural next step.
"""
from __future__ import annotations

import hmac
import logging

from cryptography.hazmat.primitives import cmac
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes

from chiron_secret import TELEMETRY_KEY

LOG = logging.getLogger("chiron.crypto")

PLAIN_LEN = 16
COUNTER_LEN = 4
TAG_LEN = 8
SEALED_LEN = COUNTER_LEN + PLAIN_LEN + TAG_LEN      # 28

_last_counter: dict[int, int] = {}


def _keystream(counter: int) -> bytes:
    """AES-ECB of the nonce block; identical construction to keystream_block()."""
    block = bytearray(16)
    block[0:4] = b"CHRN"
    block[12] = (counter >> 24) & 0xFF
    block[13] = (counter >> 16) & 0xFF
    block[14] = (counter >> 8) & 0xFF
    block[15] = counter & 0xFF
    enc = Cipher(algorithms.AES(TELEMETRY_KEY), modes.ECB()).encryptor()
    return enc.update(bytes(block)) + enc.finalize()


def unseal(payload: bytes, node_type: int | None = None) -> bytes | None:
    """Verify and decrypt one sealed result. Returns None if it is not authentic."""
    if len(payload) < SEALED_LEN:
        return None

    counter = int.from_bytes(payload[:COUNTER_LEN], "big")
    ct = payload[COUNTER_LEN:COUNTER_LEN + PLAIN_LEN]
    tag = payload[COUNTER_LEN + PLAIN_LEN:SEALED_LEN]

    # CMAC.verify() expects the full 16-byte tag; ours is truncated to 8, so
    # compute the full tag and compare the prefix in constant time.
    c = cmac.CMAC(algorithms.AES(TELEMETRY_KEY))
    c.update(payload[:COUNTER_LEN + PLAIN_LEN])
    if not hmac.compare_digest(c.finalize()[:TAG_LEN], tag):
        LOG.warning("telemetry tag failed verification - payload discarded")
        return None

    if node_type is not None:
        prev = _last_counter.get(node_type)
        if prev is not None and counter <= prev:
            LOG.warning("replayed or out-of-order counter %d (last %d)", counter, prev)
            return None
        _last_counter[node_type] = counter

    ks = _keystream(counter)
    return bytes(a ^ b for a, b in zip(ct, ks))


def seal(plain: bytes, counter: int) -> bytes:
    """Inverse of unseal. Used by the simulator and the tests."""
    ks = _keystream(counter)
    ct = bytes(a ^ b for a, b in zip(plain[:PLAIN_LEN], ks))
    head = counter.to_bytes(COUNTER_LEN, "big") + ct
    c = cmac.CMAC(algorithms.AES(TELEMETRY_KEY))
    c.update(head)
    return head + c.finalize()[:TAG_LEN]
