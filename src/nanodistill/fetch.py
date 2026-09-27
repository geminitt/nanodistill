"""Copy trained students from the Modal volume to runs/students/, verified, for local evaluation.

    pixi run python -m nanodistill.fetch cot_full_s0 cot_full_s1 ...

Large downloads from the volume come back corrupted on this machine: runs of a few MiB replaced by zeros, at
random places, and now and then a download ends early. 12 downloads of the six 1.2 GB students with `modal volume get` all differed from each other,
and six streamed downloads of one student gave six different files. The corruption only ever zeroes bytes,
and trained bf16 weights never contain 64 KiB of zeros, so the file is downloaded three times at its full size and put
together block by block from the copies whose block is not all zero; copies must agree wherever both are
non-zero, and the result must contain no all-zero block.
"""

import hashlib
import os
import sys

import modal
import numpy as np

volume = modal.Volume.from_name("nanodistill")


def stream_hash(path: str, out: str | None = None) -> str:
    h = hashlib.sha256()
    f = open(out, "wb") if out else None
    for chunk in volume.read_file(path):
        h.update(chunk)
        if f:
            f.write(chunk)
    if f:
        f.close()
    return h.hexdigest()


def file_hash(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 24), b""):
            h.update(block)
    return h.hexdigest()


BLOCK = 1 << 16


def zero_blocks(data: np.ndarray) -> np.ndarray:
    full = len(data) // BLOCK * BLOCK
    return np.flatnonzero(~data[:full].reshape(-1, BLOCK).any(axis=1))


def merge(data: list[np.ndarray], weights: bool = True) -> np.ndarray:
    """One file from copies whose only damage is zeroed blocks."""
    if len({len(d) for d in data}) != 1:
        raise RuntimeError("downloads differ in length")
    out = data[0].copy()
    for b in sorted({int(x) for d in data for x in zero_blocks(d)}):
        good = [x for x in (d[b * BLOCK:(b + 1) * BLOCK] for d in data) if x.any()]
        if any((x != good[0]).any() for x in good[1:]):
            raise RuntimeError(f"copies disagree at block {b} with non-zero data")
        if good:
            out[b * BLOCK:(b + 1) * BLOCK] = good[0]
    for i, d in enumerate(data):     # outside its zeroed blocks every copy must equal the result
        keep = np.ones(len(d), bool)
        for b in zero_blocks(d):
            keep[b * BLOCK:(b + 1) * BLOCK] = False
        if (d[keep] != out[keep]).any():
            raise RuntimeError(f"copy {i} differs from the others outside its zeroed blocks")
    if weights and len(zero_blocks(out)):
        raise RuntimeError(f"{len(zero_blocks(out))} blocks are zero in every copy")
    return out


def download(src: str, dst: str, size: int, copies: int = 3, attempts: int = 10) -> None:
    parts = [f"{dst}.part{i}" for i in range(copies)]
    for part in parts:
        for _ in range(attempts):
            if stream_hash(src, part) == file_hash(part) and os.path.getsize(part) == size:
                break
        else:
            raise RuntimeError(f"{src}: no complete download in {attempts} attempts")
    merge([np.fromfile(part, dtype=np.uint8) for part in parts], dst.endswith(".safetensors")).tofile(dst)
    for part in parts:
        os.remove(part)


def fetch(name: str, root: str = "runs/students") -> str:
    os.makedirs(f"{root}/{name}", exist_ok=True)
    for entry in volume.listdir(f"/students/{name}"):
        src = entry.path if entry.path.startswith("/") else "/" + entry.path
        download(src, f"{root}/{name}/{os.path.basename(src)}", entry.size)
    return file_hash(f"{root}/{name}/model.safetensors")


if __name__ == "__main__":
    for name in sys.argv[1:]:
        print(name, fetch(name)[:16], flush=True)
