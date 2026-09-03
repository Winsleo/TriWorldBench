#!/usr/bin/env python3
"""Build a minimal GT tree for ``skip_auto_dependencies`` eval.

When running TriWorldBench on a dataset without real GT (e.g. test_dataset
without official GT frames), ``prepare_split_views`` still requires a GT tree
matching the generation tree's episode structure:

    <gt_root>/episodeN/{head,left,right}/video/frame_XXXXX.jpg

This script builds such a tree using a single black JPEG frame per view, just
enough to satisfy ``discover_episodes`` and ``prepare_split_views`` schema.
The 9 GT-free metrics (0, 1, 2, 3, 5, 7, 11, 17, 18) do not actually read
GT content, so placeholder frames are sufficient.

Usage::

    python scripts/build_fake_gt.py \
        --gen-root input/a2v_wan22_ti2v_5b_mv3 \
        --gt-root  test_data/gt_dataset_fake \
        --frame-width 320 --frame-height 240 \
        --frame-count 1

Re-running is idempotent: existing frames are not regenerated.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(SCRIPT_DIR / "lib"))

from metrics._common.frames import FRAME_EXT, frame_filename  # noqa: E402

VIEW_NAMES = ("head", "left", "right")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gen-root", type=Path, required=True,
                        help="Path to method input dir (input/<method>/)")
    parser.add_argument("--gt-root", type=Path, required=True,
                        help="Output GT dir to scaffold (becomes cfg gt_dataset)")
    parser.add_argument("--frame-width", type=int, default=320)
    parser.add_argument("--frame-height", type=int, default=240)
    parser.add_argument("--frame-count", type=int, default=121,
                        help="Number of placeholder frames per view. "
                             "Should match generation frame count to avoid "
                             "metric errors (e.g. background_consistency "
                             "divides by frame-deltas). Default 121 covers "
                             "A2V 121-frame output.")
    parser.add_argument("--limit", type=int, default=0,
                        help="If > 0, only process first N episodes")
    return parser.parse_args()


def make_black_frame(path: Path, width: int, height: int) -> None:
    """Write a 320x240 black JPEG (or skip if exists)."""
    if path.exists():
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        from PIL import Image
    except ImportError:
        sys.exit("Pillow not installed; cannot generate placeholder frames. "
                 "Run: pip install Pillow")
    img = Image.new("RGB", (width, height), color=(0, 0, 0))
    img.save(path, format="JPEG", quality=85)


def main() -> int:
    args = parse_args()
    if not args.gen_root.is_dir():
        sys.exit(f"gen-root not found: {args.gen_root}")
    args.gt_root.mkdir(parents=True, exist_ok=True)

    # Iterate episode dirs that look like episodeN
    import re
    ep_re = re.compile(r"^episode\d+$")
    episodes = sorted(
        p.name for p in args.gen_root.iterdir()
        if p.is_dir() and ep_re.match(p.name)
    )
    if args.limit > 0:
        episodes = episodes[: args.limit]

    if not episodes:
        sys.exit(f"No episodeN dirs found in {args.gen_root}")

    print(f"[build_fake_gt] gen={args.gen_root} gt={args.gt_root}")
    print(f"[build_fake_gt] {len(episodes)} episode(s), "
          f"{args.frame_count} frame(s)/view at {args.frame_width}x{args.frame_height}")

    for ep in episodes:
        for view in VIEW_NAMES:
            video_dir = args.gt_root / ep / view / "video"
            for frame_idx in range(args.frame_count):
                fname = frame_filename(frame_idx)
                make_black_frame(video_dir / fname,
                                 args.frame_width, args.frame_height)
        print(f"  [ok] {ep}: {VIEW_NAMES} x {args.frame_count} frames")

    print(f"[build_fake_gt] done. GT tree at: {args.gt_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())