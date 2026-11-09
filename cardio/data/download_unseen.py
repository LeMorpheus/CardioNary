"""Fetch CirCor 1.0.3 via its public AWS mirror and verify published SHA256s.

Requires the AWS CLI, or uv to run it in an isolated tool environment.
No AWS account or credentials are needed. Existing downloads are resumed.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
from pathlib import Path
import shutil
import subprocess

ROOT = Path(__file__).resolve().parents[2]
SOURCE = 's3://physionet-open/circor-heart-sound/1.0.3/'


def verify(destination):
    destination = destination.resolve()
    def verify_line(line):
        digest, relative = line.split(maxsplit=1)
        path = destination / relative.lstrip('*')
        if not path.resolve().is_relative_to(destination):
            raise ValueError(f'Unsafe checksum path: {relative}')
        checksum = hashlib.sha256()
        with path.open('rb') as stream:
            while chunk := stream.read(1024 * 1024):
                checksum.update(chunk)
        if checksum.hexdigest() != digest:
            raise ValueError(f'Checksum mismatch: {path}; remove that file and rerun.')
        return 1
    lines = (destination / 'SHA256SUMS.txt').read_text().splitlines()
    with ThreadPoolExecutor(max_workers=16) as pool:
        return sum(pool.map(verify_line, lines))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=ROOT / 'data/raw/circor')
    parser.add_argument('--verify-only', action='store_true')
    args = parser.parse_args()
    if not args.verify_only:
        if shutil.which('aws'):
            command = ['aws']
        elif shutil.which('uv'):
            command = ['uv', 'tool', 'run', '--from', 'awscli', 'aws']
        else:
            parser.error('Install the AWS CLI or uv, then rerun.')
        print(f'Downloading/resuming CirCor 1.0.3 into {args.root}', flush=True)
        subprocess.run(command + ['s3', 'sync', '--no-sign-request', SOURCE,
                                  str(args.root), '--no-progress', '--only-show-errors'], check=True)
    print(f'SHA256 verified: {verify(args.root)} files', flush=True)


if __name__ == '__main__':
    main()
