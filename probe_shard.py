#!/usr/bin/env python3
"""Print the structure of the first HDF5 shard to understand the actual layout.

Usage:
    python probe_shard.py /path/to/shard.h5
"""
import sys
import h5py
import numpy as np


def print_attrs(obj, indent):
    for k, v in obj.attrs.items():
        val = v.decode() if isinstance(v, (bytes, np.bytes_)) else v
        print(f"{indent}  @{k} = {repr(val)[:200]}")


def walk(obj, indent="", max_depth=7, max_items=3):
    keys = list(obj.keys()) if hasattr(obj, "keys") else []
    shown = 0
    for key in keys:
        if shown >= max_items:
            print(f"{indent}  ... ({len(keys) - shown} more)")
            break
        child = obj[key]
        if isinstance(child, h5py.Dataset):
            print(f"{indent}  [{key}]  shape={child.shape}  dtype={child.dtype}")
            print_attrs(child, indent + "  ")
        else:
            print(f"{indent}  {key}/")
            print_attrs(child, indent + "  ")
            if indent.count("  ") < max_depth:
                walk(child, indent + "  ", max_depth, max_items)
        shown += 1


path = sys.argv[1] if len(sys.argv) > 1 else \
    "/home/hpc/k_e06y/e06y0005/hackathon_test1/test_1/test_1_shard_000.h5"

print(f"File: {path}\n")
with h5py.File(path, "r") as f:
    print("Top-level keys:", list(f.keys()))
    print_attrs(f, "")
    print()
    walk(f)
