"""Fetch Project Chiron datasets from their official sources and verify checksums.

Datasets are NEVER committed to the repository. This script reproduces `data/raw/`
from scratch. See docs/04_DATASETS_RESEARCH.md and DATA_LICENSES.md.

Usage:
    python core/download_datasets.py --dataset kauh
    python core/download_datasets.py --dataset all
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import urllib.parse
import urllib.request
import zipfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = REPO_ROOT / "data" / "raw"

UA = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) Chiron-dataset-fetcher/1.0"

# Mendeley Data: "A dataset of lung sounds recorded from the chest wall using an
# electronic stethoscope" (Fraiwan et al.), DOI 10.17632/jwyy9np4gv.3, CC BY 4.0.
KAUH_API = (
    "https://data.mendeley.com/public-api/datasets/jwyy9np4gv/files"
    "?folder_id=root&version=3"
)

# Expected sha256 for the two files we need, taken from the Mendeley API response.
KAUH_EXPECTED = {
    "Audio Files.zip": "8093b608584c631b016086e1f2df38f89db5eae0234a42fe9ab5cdaac7121d7c",
}


def _get(url: str, timeout: int = 120) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": UA, "Accept": "*/*"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.read()


def _safe_url(url: str) -> str:
    """Percent-encode non-ASCII characters in the path (the Yaseen archives
    have Korean filenames, which http.client refuses to send raw)."""
    parts = urllib.parse.urlsplit(url)
    return urllib.parse.urlunsplit(
        (
            parts.scheme,
            parts.netloc,
            urllib.parse.quote(parts.path, safe="/%"),
            urllib.parse.quote(parts.query, safe="=&%"),
            parts.fragment,
        )
    )


def _download_to(url: str, dest: Path, timeout: int = 600) -> None:
    """Stream a URL to disk with coarse progress reporting."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    req = urllib.request.Request(_safe_url(url), headers={"User-Agent": UA, "Accept": "*/*"})
    tmp = dest.with_suffix(dest.suffix + ".part")
    with urllib.request.urlopen(req, timeout=timeout) as resp, open(tmp, "wb") as fh:
        total = int(resp.headers.get("Content-Length", 0))
        done = 0
        next_mark = 10
        while True:
            chunk = resp.read(1 << 16)
            if not chunk:
                break
            fh.write(chunk)
            done += len(chunk)
            if total:
                pct = done * 100 // total
                if pct >= next_mark:
                    print(f"    {pct:3d}%  ({done/1e6:.1f} / {total/1e6:.1f} MB)")
                    next_mark = pct - (pct % 10) + 10
    tmp.replace(dest)


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def fetch_kauh(force: bool = False) -> Path:
    """Download and extract the KAUH lung-sound dataset."""
    out_dir = RAW_DIR / "kauh"
    audio_dir = out_dir / "audio"

    if audio_dir.exists() and not force:
        n = len(list(audio_dir.rglob("*.wav")))
        if n > 0:
            print(f"[kauh] already present: {n} wav files in {audio_dir}")
            return out_dir

    out_dir.mkdir(parents=True, exist_ok=True)
    print("[kauh] querying Mendeley API ...")
    files = json.loads(_get(KAUH_API).decode())
    index = {f["filename"]: f for f in files}
    print(f"[kauh] dataset contains: {', '.join(sorted(index))}")

    # Save the file manifest for provenance.
    (out_dir / "mendeley_files.json").write_text(json.dumps(files, indent=2))

    # --- audio ---------------------------------------------------------------
    entry = index.get("Audio Files.zip")
    if entry is None:
        raise SystemExit("[kauh] 'Audio Files.zip' not found in the Mendeley listing")

    zip_path = out_dir / "Audio Files.zip"
    if not zip_path.exists() or force:
        size_mb = entry["size"] / 1e6
        print(f"[kauh] downloading Audio Files.zip ({size_mb:.1f} MB) ...")
        _download_to(entry["content_details"]["download_url"], zip_path)

    print("[kauh] verifying sha256 ...")
    digest = _sha256(zip_path)
    expected = KAUH_EXPECTED["Audio Files.zip"]
    if digest != expected:
        raise SystemExit(
            f"[kauh] CHECKSUM MISMATCH for Audio Files.zip\n"
            f"  expected {expected}\n  got      {digest}\n"
            "Refusing to continue: the upstream file changed or the download is corrupt."
        )
    print(f"[kauh] sha256 OK ({digest[:16]}...)")

    # --- annotation spreadsheet ---------------------------------------------
    ann = index.get("Data annotation.xlsx")
    if ann is not None:
        ann_path = out_dir / "Data annotation.xlsx"
        if not ann_path.exists() or force:
            print("[kauh] downloading Data annotation.xlsx ...")
            _download_to(ann["content_details"]["download_url"], ann_path)

    # --- extract -------------------------------------------------------------
    if audio_dir.exists():
        shutil.rmtree(audio_dir)
    audio_dir.mkdir(parents=True)
    print("[kauh] extracting ...")
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(audio_dir)

    wavs = list(audio_dir.rglob("*.wav"))
    print(f"[kauh] extracted {len(wavs)} wav files -> {audio_dir}")
    if not wavs:
        raise SystemExit("[kauh] no .wav files found after extraction")
    return out_dir


# ---------------------------------------------------------------------------
# Yaseen 2018 heart-sound corpus (5 classes, 200 clips each, 8 kHz mono).
# Distributed as five .rar archives in the paper's GitHub repository.
# Yaseen, Son & Kwon, Applied Sciences 8(12):2344, 2018.
# ---------------------------------------------------------------------------
YASEEN_API = (
    "https://api.github.com/repos/yaseen21khan/"
    "Classification-of-Heart-Sound-Signal-Using-Multiple-Features-/contents/"
)
YASEEN_CLASS_PREFIX = {"AS": "AS", "MR": "MR", "MS": "MS", "MVP": "MVP", "N": "N"}

UNRAR_CANDIDATES = [
    Path(r"C:\Program Files\WinRAR\UnRAR.exe"),
    Path(r"C:\Program Files (x86)\WinRAR\UnRAR.exe"),
    Path(r"C:\Program Files\7-Zip\7z.exe"),
    Path(r"C:\Program Files (x86)\7-Zip\7z.exe"),
]


def _find_unrar() -> Path:
    for c in UNRAR_CANDIDATES:
        if c.exists():
            return c
    found = shutil.which("unrar") or shutil.which("7z") or shutil.which("7za")
    if found:
        return Path(found)
    raise SystemExit(
        "No RAR extractor found. Install 7-Zip or WinRAR, or extract the .rar "
        "files in data/raw/yaseen/ manually into data/raw/yaseen/audio/."
    )


def fetch_yaseen(force: bool = False) -> Path:
    import subprocess

    out_dir = RAW_DIR / "yaseen"
    audio_dir = out_dir / "audio"
    if audio_dir.exists() and not force:
        n = len(list(audio_dir.rglob("*.wav")))
        if n > 0:
            print(f"[yaseen] already present: {n} wav files in {audio_dir}")
            return out_dir

    out_dir.mkdir(parents=True, exist_ok=True)
    print("[yaseen] querying GitHub API ...")
    req = urllib.request.Request(
        YASEEN_API, headers={"User-Agent": UA, "Accept": "application/vnd.github+json"}
    )
    with urllib.request.urlopen(req, timeout=90) as r:
        listing = json.loads(r.read().decode())

    rars = [e for e in listing if e["name"].lower().endswith(".rar")
            and not e["name"].lower().startswith("code")]
    print(f"[yaseen] {len(rars)} class archives found")

    for entry in rars:
        dest = out_dir / entry["name"]
        if not dest.exists() or force:
            print(f"[yaseen] downloading {entry['name']} ({entry['size']/1e6:.1f} MB) ...")
            _download_to(entry["download_url"], dest)

    audio_dir.mkdir(parents=True, exist_ok=True)
    tool = _find_unrar()
    print(f"[yaseen] extracting with {tool.name} ...")
    for entry in rars:
        src = out_dir / entry["name"]
        if tool.name.lower().startswith("7z"):
            cmd = [str(tool), "x", "-y", f"-o{audio_dir}", str(src)]
        else:
            cmd = [str(tool), "x", "-y", "-idq", str(src), str(audio_dir) + "\\"]
        res = subprocess.run(cmd, capture_output=True, text=True)
        if res.returncode != 0:
            print(f"[yaseen] WARNING extracting {entry['name']}: {res.stderr[:200]}")

    wavs = list(audio_dir.rglob("*.wav"))
    print(f"[yaseen] extracted {len(wavs)} wav files -> {audio_dir}")
    if not wavs:
        raise SystemExit("[yaseen] no .wav files found after extraction")
    return out_dir


DATASETS = {"kauh": fetch_kauh, "yaseen": fetch_yaseen}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", default="kauh", choices=[*DATASETS, "all"])
    ap.add_argument("--force", action="store_true", help="re-download even if present")
    args = ap.parse_args()

    names = list(DATASETS) if args.dataset == "all" else [args.dataset]
    for name in names:
        DATASETS[name](force=args.force)
    print("done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
