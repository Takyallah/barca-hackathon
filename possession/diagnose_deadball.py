"""Which signals actually mark a dead ball? A diagnostic, not part of the pipeline.

    python possession/diagnose_deadball.py

For every candidate signal it prints how often the signal is on during TRUE dead-ball frames and
during TRUE live frames of the public 5 minutes, and how often over the whole match. Then, for
each true stoppage, when each signal first comes on relative to the stoppage start.

Output is percentages, counts and timings only -- no positions, speeds or other data values.
The public window has 4 stoppages: read it to see which signals carry information, not to set
weights to three decimals.
"""

import argparse
import collections
import math
import os
import sys
import warnings

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
import deadball  # noqa: E402  (also puts the repo root, starter/ and evaluation/ on the path)
import control  # noqa: E402
import load_smart_tagging  # noqa: E402
import possession_metrics as pm  # noqa: E402
from games import GAME, GAMES, half_video_blocks, source_frames  # noqa: E402

FPS = 25

# Params
ENERGY_QUANTILES = (0.15, 0.25, 0.35, 0.50)
GAP_SECONDS = (1, 3, 10)
BEFORE_SETPIECE_SECONDS = (10, 20, 30)
LINE_NEAR_M = 2.0            # ball this close to a line (or beyond it)
LOST_NEAR_M = 3.0            # last seen this close to a line before the tracking lost it
MIN_FRAMES_ON = 10           # a signal on for fewer window frames than this gets llr 0
TIMING_SIGNALS = (
    "ball not tracked",
    "ball lost near a line (whole gap)",
    "ball seen outside the lines",
    "waiting for a restart after going out (deadball.py)",
    "ball 8 m or more inside every line",
    "players slower than the match's 25th percentile",
    "players slower than the match's 50th percentile",
    "over half the players under 1 m/s",
    "before a set-piece kick (deadball.py)",
    "a team controls the ball (possession rule)",
)


def graded_mask(game=GAME):
    graded = np.zeros(source_frames("tactical", game), bool)
    for _, first, last in half_video_blocks(game):
        graded[first:last + 1] = True
    return graded


def run_length(mask):
    """For every frame, the length of the True run it sits in (0 where False)."""
    out = np.zeros(len(mask), np.int64)
    for s, e in zip(*deadball.runs(mask)):
        out[s:e] = e - s
    return out


def load_signals(data_dir=None, cache_dir=None, game=GAME):
    """Ordered {name: bool array over all video frames} of candidate dead-ball signals."""
    graded = graded_mask(game)
    ball, px, py, ps, team = deadball.load_tracking(game, data_dir, cache_dir=cache_dir)
    p = dict(deadball.PARAMS)
    feats = deadball.build_features(ball, px, py, ps, p, game, data_dir)
    events = load_smart_tagging.load(game, data_dir)
    n = len(ball)
    length, width = GAMES[game]["pitch"]
    bx, by = ball[:, 0].astype(np.float64), ball[:, 1].astype(np.float64)
    vis = ~np.isnan(bx)
    sig = collections.OrderedDict()

    # -- what deadball.py already uses
    sig["ball seen outside the lines"] = feats["out"]
    sig["waiting for a restart after going out (deadball.py)"] = feats["pending"] & ~feats["out"]
    sig["ball stationary (deadball.py)"] = feats["still"]
    sig["ball fast inside the pitch (deadball.py)"] = feats["inside_fast"]
    sig["before a set-piece kick (deadball.py)"] = feats["sp_pre"]
    sig["just after a set-piece kick (deadball.py)"] = feats["sp_post"]
    sig["goal until the kick-off (deadball.py)"] = feats["goal_dead"]
    sig["open-play tag (deadball.py)"] = feats["open_play"]

    # -- how fast the players are moving, against the match's own distribution
    energy = feats["energy"]
    reference = energy[graded & ~np.isnan(energy)]
    with np.errstate(invalid="ignore"):
        for q in ENERGY_QUANTILES:
            sig["players slower than the match's {:.0f}th percentile".format(100 * q)] = (
                energy < np.quantile(reference, q))
    with warnings.catch_warnings(), np.errstate(invalid="ignore"):
        warnings.simplefilter("ignore", category=RuntimeWarning)
        standing = deadball.centred_nanmean(
            np.nanmean(np.where(np.isnan(ps), np.nan, ps < 1.0), axis=1), 12)
        fastest = np.nanmax(ps, axis=1)
        sig["over half the players under 1 m/s"] = standing > 0.5
        sig["over 70% of the players under 1 m/s"] = standing > 0.7
        sig["no player above 4 m/s"] = fastest < 4.0
    sig["fewer than 18 players tracked"] = np.sum(~np.isnan(px), axis=1) < 18

    # -- is the ball tracked, and where was it lost
    missing = ~vis
    gap = run_length(missing)
    sig["ball not tracked"] = missing
    for seconds in GAP_SECONDS:
        sig["ball not tracked, in a gap of {} s or more".format(seconds)] = gap >= seconds * FPS
    with np.errstate(invalid="ignore"):
        to_line = np.minimum(np.minimum(bx, length - bx), np.minimum(by, width - by))
        sig["ball within {:.0f} m of a line, or beyond it".format(LINE_NEAR_M)] = (
            vis & (to_line < LINE_NEAR_M))
    lost_near_line = np.zeros(n, bool)
    for s, e in zip(*deadball.runs(missing)):
        if s > 0 and to_line[s - 1] < LOST_NEAR_M:
            lost_near_line[s:e] = True
    sig["ball lost near a line (whole gap)"] = lost_near_line
    with np.errstate(invalid="ignore"):
        sig["ball 8 m or more inside every line"] = vis & (to_line >= 8.0)
    sig["ball lost away from the lines (whole gap)"] = missing & ~lost_near_line

    # -- ball speed
    speed = control.ball_speed(ball.astype(np.float64))
    with np.errstate(invalid="ignore"):
        sig["ball speed under 1 m/s"] = speed < 1.0
        sig["ball speed 1-3 m/s"] = (speed >= 1.0) & (speed < 3.0)
        sig["ball speed 3-8 m/s"] = (speed >= 3.0) & (speed < 8.0)
        sig["ball speed over 8 m/s"] = speed >= 8.0

    # -- what the possession rule sees
    cp = dict(control.PARAMS)
    near, _ = control.proximity(ball.astype(np.float64), px, py, team, cp)
    with np.errstate(invalid="ignore"):
        carried = np.isnan(speed) | (speed <= cp["control_max_ball_speed"])
        d = np.hypot(px - ball[:, 0:1], py - ball[:, 1:2])
    nearest = np.where(np.isnan(d), np.inf, d).min(axis=1)
    sig["a team is near the ball (possession rule)"] = near != control.NOBODY
    sig["a team controls the ball (possession rule)"] = (near != control.NOBODY) & carried
    sig["ball over 5 m from every player"] = vis & (nearest > 5.0)

    # -- Smart Tagging
    in_phase = np.zeros(n, bool)
    in_setpiece = np.zeros(n, bool)
    after_start = np.zeros(n, bool)
    before = {s: np.zeros(n, bool) for s in BEFORE_SETPIECE_SECONDS}
    by_code = collections.OrderedDict()
    for ev in events:
        code = (ev["code"] or "").strip().upper()
        a, b = max(0, ev["start_frame"]), min(n, ev["end_frame"])
        by_code.setdefault(code, np.zeros(n, bool))[a:b] = True
        if code == "SET PIECES":
            in_setpiece[a:b] = True
            after_start[a:min(n, a + 5 * FPS)] = True
            for seconds, mask in before.items():
                mask[max(0, a - seconds * FPS):a] = True
        elif code != "GOALS":
            in_phase[a:b] = True
    sig["inside a Smart Tagging phase (any but set piece / goal)"] = in_phase
    sig["inside no Smart Tagging tag at all"] = ~in_phase & ~in_setpiece
    sig["inside a SET PIECES tag"] = in_setpiece
    sig["first 5 s of a SET PIECES tag"] = after_start
    for seconds in BEFORE_SETPIECE_SECONDS:
        sig["within {} s before a SET PIECES tag".format(seconds)] = before[seconds]
    for code, mask in by_code.items():
        sig["tag: {}".format(code)] = mask
    return sig


def rates(signals, truth):
    """[{name, dead_pct, live_pct, llr}] against a truth label array ('D', 'H', 'A', else ignored).

    llr = ln( P(on | dead) / P(on | live) ), lightly smoothed. Positive = points to dead ball.
    """
    truth = np.asarray(truth)
    dead = truth == "D"
    live = (truth == "H") | (truth == "A")
    n_dead, n_live = int(dead.sum()), int(live.sum())
    rows = []
    for name, mask in signals.items():
        on_dead, on_live = int(np.sum(mask & dead)), int(np.sum(mask & live))
        llr = math.log(((on_dead + 0.5) / (n_dead + 1.0)) / ((on_live + 0.5) / (n_live + 1.0)))
        rows.append(dict(
            name=name,
            dead_pct=100.0 * on_dead / max(n_dead, 1),
            live_pct=100.0 * on_live / max(n_live, 1),
            llr=llr if on_dead + on_live >= MIN_FRAMES_ON else 0.0,    # too rare to say anything
        ))
    return rows


def report(signals, truth, graded):
    truth = np.asarray(truth)
    window = truth != "X"
    dead = truth == "D"
    print("public window: {} dead frames, {} live frames".format(
        int(dead.sum()), int(((truth == "H") | (truth == "A")).sum())))
    print("\n{:<58} {:>7} {:>7} {:>6}   {:>7} {:>7}".format(
        "signal (sorted by strength)", "dead %", "live %", "llr", "window%", "match %"))
    for row in sorted(rates(signals, truth), key=lambda r: -abs(r["llr"])):
        mask = signals[row["name"]]
        print("{:<58} {:>7.1f} {:>7.1f} {:>+6.2f}   {:>7.1f} {:>7.1f}".format(
            row["name"][:58], row["dead_pct"], row["live_pct"], row["llr"],
            100.0 * float(np.mean(mask[window])), 100.0 * float(np.mean(mask[graded]))))
    print("\n  dead % / live %: share of true dead / live frames with the signal on")
    print("  llr > 0 points to dead ball, < 0 to live; |llr| under 0.5 is weak;"
          " 0.00 with both shares near 0 = hardly ever on in the window")
    print("  window % vs match %: if they differ a lot, the 5 minutes are not typical for it")

    print("\nper true stoppage: share of its frames with the signal on, and when the signal")
    print("first comes on relative to the true start (s; negative = before; -- = never)")
    for s, e in zip(*deadball.runs(dead)):
        print("\n  stoppage {}-{} ({:.1f} s)".format(s, e - 1, (e - s) / float(FPS)))
        lo = max(0, s - 3 * FPS)
        for name in TIMING_SIGNALS:
            mask = signals[name]
            on = np.flatnonzero(mask[lo:e])
            first = "{:+.1f}".format((lo + on[0] - s) / float(FPS)) if len(on) else "--"
            print("    {:<52} on {:>5.1f}%   first at {:>6}".format(
                name[:52], 100.0 * float(np.mean(mask[s:e])), first))


def match_checks(data_dir=None, cache_dir=None, game=GAME):
    """Whole-match checks that need no ground truth. Prints shares and timings only.

    1. Are the long ball gaps stoppages? If they are, players should be slow inside them (in the
       public window: slower than the match median on 78% of dead frames, 15% of live frames).
    2. How long is each GOALS tag, and how soon does its kick-off follow?
    """
    graded = graded_mask(game)
    ball, px, py, ps, team = deadball.load_tracking(game, data_dir, cache_dir=cache_dir)
    p = dict(deadball.PARAMS)
    events = load_smart_tagging.load(game, data_dir)
    feats = deadball.build_features(ball, px, py, ps, p, game, data_dir, events=events)
    dead = deadball.dead_mask(deadball.evidence(feats, p, graded), p)
    n = len(ball)

    missing = np.isnan(ball[:, 0])
    energy = feats["energy"]
    reference = energy[graded & ~np.isnan(energy)]
    with np.errstate(invalid="ignore"):
        slow25 = energy < np.quantile(reference, 0.25)
        slow50 = energy < np.quantile(reference, 0.50)
    setpiece = np.zeros(n, bool)
    for ev in events:
        if (ev["code"] or "").strip().upper() == "SET PIECES":
            setpiece[max(0, ev["start_frame"] - 10 * FPS):max(0, ev["end_frame"])] = True

    print("\nwhole match, graded frames, by ball tracking state (no ground truth involved)")
    print("  {:<24} {:>7} {:>10} {:>10} {:>10} {:>10} {:>9}".format(
        "", "share %", "slow<p25 %", "slow<p50 %", "open play%", "set piece%", "our D %"))
    rows = (("ball tracked", graded & ~missing),
            ("gap under 10 s", graded & missing & ~feats["long_gap"]),
            ("gap of 10 s or more", graded & feats["long_gap"]),
            ("all graded frames", graded))
    for name, mask in rows:
        k = max(int(mask.sum()), 1)
        print("  {:<24} {:>7.1f} {:>10.1f} {:>10.1f} {:>10.1f} {:>10.1f} {:>9.1f}".format(
            name, 100.0 * mask.sum() / graded.sum(),
            100.0 * np.sum(slow25 & mask) / k, 100.0 * np.sum(slow50 & mask) / k,
            100.0 * np.sum(feats["open_play"] & mask) / k, 100.0 * np.sum(setpiece & mask) / k,
            100.0 * np.sum(dead & mask) / k))
    print("  set piece % = inside a SET PIECES tag or within 10 s before one")
    gaps = [e - s for s, e in zip(*deadball.runs(feats["long_gap"] & graded))]
    if gaps:
        gaps = np.array(gaps, float) / FPS
        print("  long gaps: {}   median {:.0f} s   longest {:.0f} s   over 60 s: {}".format(
            len(gaps), float(np.median(gaps)), gaps.max(), int(np.sum(gaps > 60))))

    kickoffs = sorted(ev["start_frame"] for ev in events
                      if (ev["code"] or "").strip().upper() == "SET PIECES"
                      and (ev["tags"].get("Type") or "").strip().lower() == "kick off (after goal)")
    print("\nGOALS tags: length, when the ball is first seen out, when the next after-goal")
    print("kick-off tag starts (s from the tag start), and how much dead ball we force")
    for ev in events:
        if (ev["code"] or "").strip().upper() != "GOALS":
            continue
        s, e = max(0, ev["start_frame"]), min(n, ev["end_frame"])
        out = np.flatnonzero(feats["out"][max(0, s - 5 * FPS):e])
        first_out = "{:+.0f}".format((max(0, s - 5 * FPS) + out[0] - s) / float(FPS)) if len(out) else "--"
        later = [k for k in kickoffs if k > s]
        kick = "{:+.0f}".format((later[0] - s) / float(FPS)) if later else "--"
        forced = feats["goal_dead"][max(0, s - 5 * FPS):min(n, e + 30 * FPS)]
        print("  tag {:>5.0f} s   ball out at {:>4}   kick-off tag at {:>5}   forced dead {:>5.0f} s"
              .format((e - s) / float(FPS), first_out, kick, forced.sum() / float(FPS)))


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data-dir", default=None)
    parser.add_argument("--cache-dir", default=None)
    args = parser.parse_args()
    truth = pm.read_label_csv(deadball.PUBLIC_GT)["label"].values.astype(str)
    report(load_signals(args.data_dir, args.cache_dir), truth, graded_mask())
    match_checks(args.data_dir, args.cache_dir)


if __name__ == "__main__":
    main()
