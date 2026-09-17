#!/usr/bin/env python3
"""Chunked, resumable, multi-connection downloader.

The EB-NeRD S3 bucket throttles hard per connection (~30 KB/s) from this
network, so we fan out over many HTTP Range requests instead.  Progress is
checkpointed per chunk, so an interrupted run resumes where it stopped.

    python3 par_download.py <url> <dest> [--conns 24] [--chunk-mb 16]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor

UA = {"User-Agent": "curl/8.5.0"}


def head_size(url: str) -> int:
    req = urllib.request.Request(url, method="HEAD", headers=UA)
    with urllib.request.urlopen(req, timeout=60) as r:
        return int(r.headers["Content-Length"])


def fetch_chunk(url: str, fd: int, start: int, end: int, retries: int = 100) -> int:
    """GET [start, end] into fd at offset start. Returns bytes written."""
    for attempt in range(retries):
        try:
            req = urllib.request.Request(
                url, headers={**UA, "Range": f"bytes={start}-{end}"}
            )
            written, pos = 0, start
            with urllib.request.urlopen(req, timeout=120) as r:
                while True:
                    buf = r.read(1 << 20)
                    if not buf:
                        break
                    os.pwrite(fd, buf, pos)
                    pos += len(buf)
                    written += len(buf)
            if written == end - start + 1:
                return written
            start += written  # short read: resume the remainder
        except (urllib.error.URLError, OSError, TimeoutError):
            time.sleep(min(2 ** min(attempt, 5), 30))
    raise RuntimeError(f"chunk {start}-{end} failed after {retries} attempts")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("url")
    ap.add_argument("dest")
    ap.add_argument("--conns", type=int, default=24)
    ap.add_argument("--chunk-mb", type=int, default=16)
    a = ap.parse_args()

    dest, state_path = a.dest, a.dest + ".chunks.json"
    size = head_size(a.url)
    chunk = a.chunk_mb << 20
    n = (size + chunk - 1) // chunk

    if os.path.exists(dest) and os.path.getsize(dest) == size and not os.path.exists(state_path):
        print(f"[skip] {os.path.basename(dest)} already complete ({size/1e6:.0f} MB)")
        return 0

    done: set[int] = set()
    if os.path.exists(state_path):
        try:
            done = set(json.load(open(state_path))["done"])
        except Exception:
            done = set()

    fd = os.open(dest, os.O_RDWR | os.O_CREAT)
    os.ftruncate(fd, size)

    lock = threading.Lock()
    got = [len(done) * chunk]
    t0 = time.time()

    def save() -> None:
        json.dump({"size": size, "chunk": chunk, "done": sorted(done)}, open(state_path, "w"))

    def work(i: int) -> None:
        start, end = i * chunk, min((i + 1) * chunk, size) - 1
        w = fetch_chunk(a.url, fd, start, end)
        with lock:
            done.add(i)
            got[0] += w
            save()
            el = time.time() - t0
            print(
                f"\r[{os.path.basename(dest)}] {len(done)}/{n} chunks "
                f"{got[0]/1e6:7.0f}/{size/1e6:.0f} MB  {got[0]/1e3/max(el,1):6.0f} KB/s",
                end="", flush=True,
            )

    todo = [i for i in range(n) if i not in done]
    print(f"[{os.path.basename(dest)}] {size/1e6:.0f} MB, {n} chunks, {len(todo)} to go, {a.conns} conns")
    with ThreadPoolExecutor(max_workers=a.conns) as ex:
        list(ex.map(work, todo))
    os.close(fd)
    print()

    if os.path.getsize(dest) == size:
        os.remove(state_path)
        print(f"[done] {dest}")
        return 0
    print(f"[FAIL] size mismatch for {dest}", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
