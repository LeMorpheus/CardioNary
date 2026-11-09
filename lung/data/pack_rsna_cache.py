"""Pack the per-patient RSNA caches into one file in the same layout as the Kermany cache,
so lung.model.features (Harsh's script) can run on RSNA unchanged.

Usage:
    python -m lung.data.pack_rsna_cache
Writes data/cache/rsna_96.npz  (X, y, patient_id, fold)
"""
from __future__ import annotations

import numpy as np

from lung.eval.common import CACHE_DIR, load_rsna


def main() -> int:
    man, X = load_rsna()
    out = CACHE_DIR / "rsna_96.npz"
    np.savez_compressed(out, X=X, y=man.label.to_numpy(np.int64),
                        # Use a fixed-width Unicode array so consumers can load this
                        # archive with NumPy's safe default allow_pickle=False.
                        patient_id=man.patient_id.astype(str).to_numpy(dtype=str),
                        fold=man.fold.to_numpy(np.int64))
    print(f"packed {len(man)} RSNA images, shape {X.shape}, value range "
          f"[{X.min():.2f}, {X.max():.2f}] -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
