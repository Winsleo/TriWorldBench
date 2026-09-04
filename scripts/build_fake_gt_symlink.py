#!/usr/bin/env python3
"""Fast scaffold for large test_dataset (500+ ep) FAKE GT tree.

Same logic as build_fake_gt.py but:
- Uses a single shared black JPEG and symlinks all frames to it (instant).
- Parallelizes episode processing with multiprocessing (one process per ep).

Usage:
    python scripts/build_fake_gt_symlink.py \\
        --gen-root /path/to/gen_with_episode_dirs \\
        --gt-root  test_data/test/gt_dataset \\
        --frame-count 1500 \\
        --jobs 8

Each output file is a hardlink to /tmp/black_320x240.jpg (or a symlink if hardlink
fails across filesystems).  From the eval pipeline's POV each is a readable image.

For 500 episodes x 3 views x 1500 frames = 2.25M entries, this completes in
seconds (vs ~30 min with build_fake_gt.py's PIL-per-frame approach).
"""
from __future__ import annotations

import argparse
import multiprocessing as mp
import os
import re
import sys
from pathlib import Path

EP_RE = re.compile(r"^episode\d+$")
VIEW_NAMES = ("head", "left", "right")
FRAME_FMT = "frame_{:05d}.jpg"


def _make_black_frame() -> Path:
    """Create the single shared placeholder JPEG; return its path."""
    from PIL import Image
    p = Path("/tmp/black_320x240.jpg")
    if not p.exists():
        Image.new("RGB", (320, 240), color=(0, 0, 0)).save(p, format="JPEG", quality=85)
    return p


def _scaffold_one_ep(args):
    ep_name, gt_root, frame_count, black_image = args
    n_links = 0
    for view in VIEW_NAMES:
        video_dir = Path(gt_root) / ep_name / view / "video"
        video_dir.mkdir(parents=True, exist_ok=True)
        for i in range(frame_count):
            target = video_dir / FRAME_FMT.format(i)
            if target.exists() or target.is_symlink():
                continue
            try:
                os.link(black_image, target)  # hardlink (cheap, same fs)
            except OSError:
                os.symlink(black_image, target)  # fallback
            n_links += 1
    return ep_name, n_links


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gen-root", type=Path, required=True)
    parser.add_argument("--gt-root", type=Path, required=True)
    parser.add_argument("--frame-count", type=int, default=1500)
    parser.add_argument("--jobs", type=int, default=8,
                        help="Parallel workers (episodes in parallel)")
    args = parser.parse_args()

    if not args.gen_root.is_dir():
        sys.exit(f"gen-root not found: {args.gen_root}")
    args.gt_root.mkdir(parents=True, exist_ok=True)

    episodes = sorted(
        p.name for p in args.gen_root.iterdir()
        if p.is_dir() and EP_RE.match(p.name)
    )
    if not episodes:
        sys.exit(f"No episodeN dirs found in {args.gen_root}")

    black_image = str(_make_black_frame())
    print(f"[symlink_fake_gt] gen={args.gen_root} gt={args.gt_root}")
    print(f"[symlink_fake_gt] {len(episodes)} ep(s) x {args.frame_count} frames/view x {len(VIEW_NAMES)} views")
    print(f"[symlink_fake_gt] jobs={args.jobs} placeholder={black_image}")

    work = [(ep, str(args.gt_root), args.frame_count, black_image) for ep in episodes]
    total_links = 0
    with mp.Pool(args.jobs) as pool:
        for i, (ep, n) in enumerate(pool.imap_unordered(_scaffold_one_ep, work), 1):
            total_links += n
            if i % 50 == 0 or i == len(episodes):
                print(f"  [{i}/{len(episodes)}] {ep}: +{n} links")
    print(f"[symlink_fake_gt] done. {total_links} links written to {args.gt_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())