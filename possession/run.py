"""The whole pipeline: tracking -> dead ball -> team in possession -> submission CSV.

    python possession/run.py                       # writes evaluation/results/Los_Egipcios.csv
    python possession/run.py --out some/file.csv
    python possession/run.py --refresh             # re-parse the ATD instead of using the cache

Prints aggregates only (counts, shares, scores, our own predicted segments) -- never data values.
Deterministic: no randomness anywhere, so two runs write the same bytes. The SHA-256 of the
file is printed so that can be checked.
"""

import argparse
import hashlib
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
import deadball  # noqa: E402  (also puts the repo root, starter/ and evaluation/ on the path)
import control  # noqa: E402
from games import GAME, half_video_blocks, source_frames  # noqa: E402

FPS = 25
TEAM = "Los_Egipcios"                     # the submission file is named after the team
DEFAULT_OUT = os.path.join(ROOT, "evaluation", "results", TEAM + ".csv")


def graded_mask(game=GAME):
    graded = np.zeros(source_frames("tactical", game), bool)
    for _, first, last in half_video_blocks(game):
        graded[first:last + 1] = True
    return graded


def build_labels(data_dir=None, cache_dir=None, refresh=False, stats=None):
    """(labels, dead) for every video frame."""
    graded = graded_mask()
    ball, px, py, ps, team = deadball.load_tracking(data_dir=data_dir, refresh=refresh,
                                                    cache_dir=cache_dir)
    p = dict(deadball.PARAMS)
    feats = deadball.build_features(ball, px, py, ps, p, data_dir=data_dir)
    dead = deadball.dead_mask(deadball.evidence(feats, p, graded), p)
    labels = control.decide(ball, px, py, team, dead=dead, graded=graded, stats=stats)
    return labels, dead


def write_submission(path, labels):
    """frame,label CSV with LF line ends on every platform, so Windows and Linux runs write
    the same bytes (the starter's writer gives CRLF on Windows)."""
    folder = os.path.dirname(path)
    if folder and not os.path.isdir(folder):
        os.makedirs(folder)
    with open(path, "w", newline="\n") as handle:
        handle.write("frame,label\n")
        handle.writelines("{},{}\n".format(i, label) for i, label in enumerate(labels))
    return path


def write(out, data_dir=None, cache_dir=None, refresh=False, stats=None):
    """Build the labels and write the submission. Returns the labels."""
    labels, _ = build_labels(data_dir, cache_dir, refresh, stats)
    write_submission(out, labels)
    return labels


def summary(labels, stats):
    """Aggregates worth watching on the full match, where there is no ground truth to score on."""
    graded = graded_mask()
    g = labels[graded]
    share = {k: 100.0 * float(np.mean(g == k)) for k in "HADX"}
    print("graded frames: H {H:.1f}%  A {A:.1f}%  D {D:.1f}%  X {X:.1f}%".format(**share))

    lengths = []
    for _, first, last in half_video_blocks(GAME):
        block = labels[first:last + 1]
        cut = np.flatnonzero(block[1:] != block[:-1]) + 1
        starts = np.concatenate(([0], cut))
        ends = np.concatenate((cut, [len(block)]))
        lengths += [e - s for s, e in zip(starts, ends) if block[s] in ("H", "A")]
    lengths = np.array(lengths, dtype=float) / FPS
    if len(lengths):
        print("possessions: {}   mean {:.1f} s   median {:.1f} s   under 1 s: {}".format(
            len(lengths), lengths.mean(), float(np.median(lengths)), int(np.sum(lengths < 1.0))))
        print("  (the public 5 minutes have 17 of mean 14.2 s -> expect roughly 300-400 here;"
              " far more means fragmenting, far fewer means merging)")
    print("ball tracked on {ball_tracked_pct:.1f}% of graded frames, {ball_after_fill_pct:.1f}% "
          "after filling short gaps".format(**stats))
    print("live frames: near a team {near_pct:.1f}%, controlled by a team {control_pct:.1f}%; "
          "{touches} touches; {unlabelled_live_frames} live frames left unlabelled".format(**stats))


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", default=DEFAULT_OUT)
    parser.add_argument("--data-dir", default=None, help="default: MTAH_DATA_DIR or ./data")
    parser.add_argument("--cache-dir", default=None, help="default: ./cache")
    parser.add_argument("--refresh", action="store_true", help="re-parse the ATD")
    args = parser.parse_args()

    stats = {}
    labels, dead = build_labels(args.data_dir, args.cache_dir, args.refresh, stats)
    write_submission(args.out, labels)
    print("wrote {}".format(os.path.relpath(args.out, ROOT)))
    with open(args.out, "rb") as handle:
        print("sha256 {}".format(hashlib.sha256(handle.read()).hexdigest()))
    summary(labels, stats)
    if args.data_dir is None:                       # the public window only means something on
        deadball.report(labels, dead, os.path.basename(args.out))   # the real match
    print("\nNext: python evaluation/validate_submission.py {}".format(
        os.path.relpath(args.out, ROOT)))


if __name__ == "__main__":
    main()
