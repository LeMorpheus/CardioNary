"""Placeholder telemetry key. NOT the real key, and not secret.

The original project sealed every result the hardware sent over Bluetooth with a
pre-shared AES-128 key. That key is not published. This all-zeros placeholder
exists only so the backend imports and runs, which is all the software-only work
needs: simulate mode and uploads never use the key.

If you ever need a real key, generate your own and keep it out of Git:

    python -c "import os; print(list(os.urandom(16)))"
"""

TELEMETRY_KEY = bytes(16)
