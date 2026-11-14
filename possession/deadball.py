"""Dead-ball detection -- step 1 of the possession pipeline.

Decides, for every frame, whether the ball is ALIVE or DEAD, from the tactical ATD and the Smart
Tagging only. It also writes a test submission (D where dead, smoothed nearest-player team
elsewhere) so the effect of the dead-ball step can be scored on its own.

    python possession/deadball.py                # build, write submission, report on public window
    python possession/deadball.py --calibrate    # small grid on the public window (sanity check only)
    python possession/deadball.py --refresh      # re-parse the ATD instead of using the cache

The first run parses the ATD (a minute or two) and caches arrays in ./cache (gitignored).

Method
------
1. Per-frame evidence that the ball is dead, as a log-likelihood ratio (positive = dead):
   - ball seen outside the pitch lines             strong dead (throw-ins, goal line, goals)
   - after the ball went out, until a real kick    dead-leaning
     or until it is seen deep inside the pitch
   - players' mean speed vs the match median       slower = dead-leaning, faster = alive-leaning
   - ball untracked for 10 s or more               dead-leaning (a ball off the pitch has no track)
   - ball moving fast inside the pitch             alive-leaning
   - Smart Tagging open-play phase                 alive-leaning
   - Smart Tagging set piece, before the kick      dead-leaning; just after the kick strongly alive
   - GOALS tag until the kick-off after it         forced dead (only if one follows soon)
   Which signals to use, and roughly how strongly, was checked with diagnose_deadball.py.
2. A two-state Viterbi decode with a switching cost turns noisy evidence into clean stretches.
3. Fixed structure: 1st half alive from its kickoff (11185); 2nd half D until kickoff (113742).
"""

import argparse
import os
import sys
import warnings

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
for sub in ("", "starter", "evaluation"):
    sys.path.insert(0, os.path.join(ROOT, sub))
from games import GAME, GAMES, require, source_frames, half_video_blocks  # noqa: E402
from baseline import team_of_each_track, smooth, write_submission  # noqa: E402
import load_smart_tagging  # noqa: E402
import possession_metrics as pm  # noqa: E402

FPS = 25
# Smart Tagging phases that only happen with the ball in play (each has a mirror "defending" code).
OPEN_PLAY_CODES = ("progression", "defending in middle third", "build up",
                   "defending in attacking third", "attacking transition",
                   "defensive transition", "long ball")
CACHE_DIR = os.path.join(ROOT, "cache")
RESULTS_DIR = os.path.join(ROOT, "evaluation", "results")
PUBLIC_GT = os.path.join(ROOT, "evaluation", "gt", "BAR-ATH_possession_gt_public.csv")
SECOND_HALF_KICKOFF = 113742

# Params -- football priors, deliberately NOT tuned on the public window.
PARAMS = dict(
    out_margin_m=0.3,       # ball must be this far past a line to count as out (tracking noise)
    out_min_frames=3,       # ...on at least this many consecutive frames
    w_out=3.0,              # evidence per frame while the ball is seen out
    w_pending=1.5,          # after the ball went out, until a restart kick is seen
    pending_max_s=30,       # cap on how long "waiting for a restart" can last
    pending_infield_m=8.0,  # ball this far inside every line: play has restarted
    w_energy=2.0,           # player-speed evidence at full strength
    energy_q=0.50,          # quantile of match player speed used as the dead/alive pivot
    energy_scale_q=0.25,    # ...evidence is at full strength this far below the pivot
    energy_live_cap=0.75,   # fast players count this fraction of full strength towards alive
    gap_min_s=10,           # ball untracked this long without a break: a long gap
    w_gap=1.5,
    w_open_play=-1.0,       # inside an open-play Smart Tagging phase
    goal_max_s=300,         # a goal only forces dead ball if its kick-off comes within this
                            # (the GOALS tags run from the goal to the kick-off, up to 251 s)
    still_speed=1.0,        # m/s: ball counts as stationary below this
    fast_speed=3.0,         # m/s: ball counts as clearly moving above this
    w_fast=-1.0,
    kick_speed=6.0,         # m/s: a restart kick
    kick_quiet_frames=15,   # ball still or unseen this long before a kick
    kick_slack_frames=10,   # ...within this many extra frames (the ball needs a moment to
                            # cross the line, and speed smoothing blurs the last few frames)
    w_setpiece_pre=1.5,
    w_setpiece_post=-4.0,
    setpiece_post_frames=50,
    w_goal=10.0,
    switch_penalty=25.0,    # cost of switching alive<->dead; higher = fewer, longer stretches
)


# -- loading ------------------------------------------------------------------------------------

def load_tracking(game=GAME, data_dir=None, refresh=False, cache_dir=None):
    """ball (n,2), px/py/ps (n,slots) in metres and m/s, NaN where missing; plus team per slot.

    cache_dir: where the parsed arrays are kept (default ./cache). Tests on synthetic data pass
    their own, so a synthetic parse can never be mistaken for the real one.
    """
    cache_dir = cache_dir or CACHE_DIR
    meta_path, raw_path = require(["atd_meta", "atd_raw"], game, data_dir)
    team = team_of_each_track(meta_path)
    n, slots = source_frames("tactical", game), len(team)
    cache = os.path.join(cache_dir, "{}_tracking.npz".format(game))

    if os.path.isfile(cache) and not refresh:
        z = np.load(cache)
        return z["ball"], z["px"], z["py"], z["ps"], team

    length, width = GAMES[game]["pitch"]
    ball = np.full((n, 2), np.nan, np.float32)
    px = np.full((n, slots), np.nan, np.float32)
    py = np.full((n, slots), np.nan, np.float32)
    ps = np.full((n, slots), np.nan, np.float32)

    print("parsing {} (cached afterwards) ...".format(os.path.basename(raw_path)))
    with open(raw_path) as handle:
        for line in handle:
            parts = line.rstrip("\r\n").split(":")
            if len(parts) != 3:
                continue
            i = int(parts[0]) - 1                       # ATD counts from 1, video from 0
            if not 0 <= i < n:
                continue
            b = parts[2].split(",")
            if len(b) >= 2 and b[0][:1] != "N" and b[1][:1] != "N":
                ball[i, 0] = float(b[0]) * length
                ball[i, 1] = float(b[1]) * width
            for s, group in enumerate(parts[1].split(";")):
                if s >= slots or not group or group[0] == "N":
                    continue
                v = group.split(",")
                px[i, s] = float(v[0]) * length
                py[i, s] = float(v[1]) * width
                if len(v) > 2 and v[2][:1] != "N":
                    ps[i, s] = float(v[2])

    if not os.path.isdir(cache_dir):
        os.makedirs(cache_dir)
    np.savez(cache, ball=ball, px=px, py=py, ps=ps)
    return ball, px, py, ps, team


# -- small helpers ------------------------------------------------------------------------------

def centred_nanmean(x, half):
    """Mean over +/- half frames, ignoring NaN. NaN where the window holds nothing."""
    x = np.asarray(x, dtype=np.float64)
    ok = ~np.isnan(x)
    s = np.concatenate(([0.0], np.cumsum(np.where(ok, x, 0.0))))
    c = np.concatenate(([0], np.cumsum(ok)))
    i = np.arange(len(x))
    lo, hi = np.maximum(0, i - half), np.minimum(len(x), i + half + 1)
    count = c[hi] - c[lo]
    out = np.full(len(x), np.nan)
    m = count > 0
    out[m] = (s[hi] - s[lo])[m] / count[m]
    return out


def runs(mask):
    """(starts, ends) of True runs, ends exclusive."""
    d = np.diff(np.concatenate(([0], mask.astype(np.int8), [0])))
    return np.flatnonzero(d == 1), np.flatnonzero(d == -1)


def next_true(mask_idx, frame):
    """First index in the sorted array mask_idx that is >= frame, or None."""
    k = np.searchsorted(mask_idx, frame)
    return int(mask_idx[k]) if k < len(mask_idx) else None


# -- features -----------------------------------------------------------------------------------

def build_features(ball, px, py, ps, p, game=GAME, data_dir=None, events=None):
    """Per-frame signals. `events`: Smart Tagging instances (read from the file if None)."""
    n = len(ball)
    length, width = GAMES[game]["pitch"]
    bx, by = ball[:, 0].astype(np.float64), ball[:, 1].astype(np.float64)
    vis = ~np.isnan(bx)

    # Player movement: given speed where present, else position difference (slot swaps clipped).
    with np.errstate(invalid="ignore"):
        v = np.full(px.shape, np.nan, np.float32)
        v[1:] = np.hypot(np.diff(px, axis=0), np.diff(py, axis=0)) * FPS
        v[v > 12] = np.nan
        speed = np.where(np.isnan(ps), v, ps)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        energy = centred_nanmean(np.nanmean(speed, axis=1), 12)

    # Ball speed from consecutive positions.
    bv = np.full(n, np.nan)
    with np.errstate(invalid="ignore"):
        bv[1:] = np.hypot(np.diff(bx), np.diff(by)) * FPS
        bv[bv > 45] = np.nan
    ball_speed = centred_nanmean(bv, 3)

    # Ball outside the lines, for a minimum run length.
    m = p["out_margin_m"]
    with np.errstate(invalid="ignore"):
        out_raw = vis & ((bx < -m) | (bx > length + m) | (by < -m) | (by > width + m))
    out = np.zeros(n, bool)
    for s, e in zip(*runs(out_raw)):
        if e - s >= p["out_min_frames"]:
            out[s:e] = True

    with np.errstate(invalid="ignore"):
        inside_fast = vis & ~out & (ball_speed > p["fast_speed"])
        quiet = ~vis | (ball_speed < p["still_speed"])
        still = vis & (ball_speed < p["still_speed"])
        fast_kick = vis & ~out & (ball_speed > p["kick_speed"])

    # A kick: fast ball inside the pitch after the ball was quiet for a while -- at least
    # kick_quiet_frames quiet frames among the last kick_quiet_frames + kick_slack_frames.
    k, slack = p["kick_quiet_frames"], p["kick_slack_frames"]
    cq = np.concatenate(([0], np.cumsum(quiet)))
    i = np.arange(n)
    quiet_before = (cq[i] - cq[np.maximum(0, i - k - slack)]) >= k
    kick = fast_kick & quiet_before
    kick_idx = np.flatnonzero(kick)

    # Waiting for a restart after the ball went out: until a real kick, not just any fast ball
    # (during a stoppage the ball gets thrown and carried around). Restarts are taken at the
    # edge of the pitch, so a ball seen well inside every line also means play is back on.
    with np.errstate(invalid="ignore"):
        to_line = np.minimum(np.minimum(bx, length - bx), np.minimum(by, width - by))
        infield_idx = np.flatnonzero(vis & (to_line >= p["pending_infield_m"]))
    pending = np.zeros(n, bool)
    cap = int(p["pending_max_s"] * FPS)
    for s, e in zip(*runs(out)):
        stops = [j for j in (next_true(kick_idx, e), next_true(infield_idx, e)) if j is not None]
        pending[e:min(stops + [e + cap, n])] = True

    # Ball untracked for a long stretch.
    long_gap = np.zeros(n, bool)
    for s, e in zip(*runs(~vis)):
        if e - s >= p["gap_min_s"] * FPS:
            long_gap[s:e] = True

    # Smart Tagging: open play, set pieces and goals.
    if events is None:
        events = load_smart_tagging.load(game, data_dir)
    open_play = np.zeros(n, bool)
    for ev in events:
        if (ev["code"] or "").strip().lower() in OPEN_PLAY_CODES:
            open_play[max(0, ev["start_frame"]):max(0, ev["end_frame"])] = True
    sp_pre = np.zeros(n, bool)
    sp_post = np.zeros(n, bool)
    goal_dead = np.zeros(n, bool)
    restarts = []                                       # (restart_frame, type)
    for ev in events:
        if (ev["code"] or "").strip().lower() != "set pieces":
            continue
        s, e = ev["start_frame"], ev["end_frame"]
        j = next_true(kick_idx, max(0, s - FPS))
        r = j if j is not None and j <= e + 2 * FPS else (s + e) // 2
        sp_pre[max(0, s):r] = True
        sp_post[r:r + p["setpiece_post_frames"]] = True
        restarts.append((r, (ev["tags"].get("Type") or "").strip().lower()))

    kickoffs_after_goal = sorted(r for r, t in restarts if t == "kick off (after goal)")
    for ev in events:
        if (ev["code"] or "").strip().lower() != "goals":
            continue
        s, e = ev["start_frame"], ev["end_frame"]
        window = np.flatnonzero(out[max(0, s - 5 * FPS):e + 1])
        g = max(0, s - 5 * FPS) + int(window[0]) if len(window) else s
        later = [r for r in kickoffs_after_goal if g < r <= g + p["goal_max_s"] * FPS]
        if later:                                       # no kick-off soon after: not a real goal
            goal_dead[g:later[0]] = True

    return dict(energy=energy, out=out, pending=pending, still=still,
                inside_fast=inside_fast, sp_pre=sp_pre, sp_post=sp_post, goal_dead=goal_dead,
                long_gap=long_gap, open_play=open_play)


def evidence(f, p, graded):
    """Per-frame log-likelihood ratio, positive = dead."""
    e = f["energy"]
    eg = e[graded & ~np.isnan(e)]
    pivot = np.quantile(eg, p["energy_q"])
    scale = max(pivot - np.quantile(eg, p["energy_scale_q"]), 1e-3)
    energy_term = np.clip((pivot - e) / scale, -p["energy_live_cap"], 1.0) * p["w_energy"]
    energy_term = np.where(np.isnan(energy_term), 0.0, energy_term)

    llr = energy_term
    llr = llr + f["out"] * p["w_out"]
    llr = llr + (f["pending"] & ~f["out"]) * p["w_pending"]
    llr = llr + f["long_gap"] * p["w_gap"]
    llr = llr + f["open_play"] * p["w_open_play"]
    llr = llr + f["inside_fast"] * p["w_fast"]
    llr = llr + f["sp_pre"] * p["w_setpiece_pre"]
    llr = llr + f["sp_post"] * p["w_setpiece_post"]
    llr = llr + f["goal_dead"] * p["w_goal"]
    return llr


# -- decoding -----------------------------------------------------------------------------------

def decode_two_state(llr, penalty):
    """Viterbi over {alive, dead}, starting alive. Returns a bool array, True = dead."""
    n = len(llr)
    if n == 0:
        return np.zeros(0, bool)
    vals = llr.tolist()
    neg = -1e18
    alive, dead = 0.0, neg
    back = np.zeros((n, 2), bool)
    for i in range(1, n):
        a_sw, d_sw = dead - penalty, alive - penalty
        if a_sw > alive:
            new_alive = a_sw
            back[i, 0] = True
        else:
            new_alive = alive
        if d_sw > dead:
            new_dead = d_sw + vals[i]
            back[i, 1] = True
        else:
            new_dead = dead + vals[i]
        alive, dead = new_alive, new_dead
    state = 1 if dead > alive else 0
    path = np.empty(n, bool)
    for i in range(n - 1, -1, -1):
        path[i] = state == 1
        if back[i, state]:
            state = 1 - state
    return path


def dead_mask(llr, p, game=GAME):
    """True = dead, on graded frames only. Ungraded frames are False and handled as X later."""
    n = len(llr)
    dead = np.zeros(n, bool)
    for half, first, last in half_video_blocks(game):
        start = first
        if half == 2 and first < SECOND_HALF_KICKOFF:
            dead[first:SECOND_HALF_KICKOFF] = True
            start = SECOND_HALF_KICKOFF
        dead[start:last + 1] = decode_two_state(llr[start:last + 1], p["switch_penalty"])
    return dead


# -- test submission ----------------------------------------------------------------------------

def nearest_team_codes(ball, px, py, team):
    """0 = home, 1 = away, -1 = no ball."""
    d2 = (px - ball[:, 0:1]) ** 2 + (py - ball[:, 1:2]) ** 2
    d2 = np.where(np.isnan(d2), np.inf, d2)
    idx = np.argmin(d2, axis=1)
    best = d2[np.arange(len(d2)), idx]
    return np.where(np.isfinite(best), team[idx], -1).astype(np.int8)


def labels_from(dead, team_labels, graded):
    """D where dead; elsewhere the smoothed team, filled within each alive stretch; X off-grade."""
    lab = pd.Series(np.where(np.isin(team_labels, ["H", "A"]), team_labels, None), dtype=object)
    alive = graded & ~dead
    stretch = pd.Series(np.cumsum(~alive))              # constant within each alive stretch
    lab[~alive] = None
    lab = lab.groupby(stretch).transform(lambda s: s.ffill().bfill())
    out = np.full(len(dead), "X", dtype="<U1")
    out[alive] = lab[alive].fillna("X").values
    out[graded & dead] = "D"
    return out


# -- reporting ----------------------------------------------------------------------------------

def segments_of(mask, lo, hi):
    s, e = runs(mask[lo:hi + 1])
    return [(lo + a, lo + b - 1, (b - a) / FPS) for a, b in zip(s, e)]


def score(labels):
    gt = pm.label_codes(pm.read_label_csv(PUBLIC_GT))
    return pm.evaluate(gt, pm.label_codes(labels), bootstrap_n=0)


def report(labels, dead, name):
    gt_df = pm.read_label_csv(PUBLIC_GT)
    r = score(labels)
    d = r["per_class"]["D"]
    print("\n{}  (public 5-minute window -- a sanity check, not a leaderboard)".format(name))
    print("  alive/dead F1 {:.4f}   D precision {:.3f}  recall {:.3f}  F1 {:.3f}".format(
        r["alive_dead_f1"], d["precision"], d["recall"], d["f1"]))
    print("  macroF1 {:.4f}   spellF1 {:.4f}   possessions {} (truth {})".format(
        r["macro_f1"], r["spell_f1_mean"], r["n_pred_spells"], r["n_gt_spells"]))

    base = os.path.join(RESULTS_DIR, "baseline_smooth50.csv")
    if os.path.isfile(base):
        b = score(pm.read_label_csv(base)["label"].values)
        print("  baseline_smooth50 for comparison: alive/dead F1 {:.4f}  macroF1 {:.4f}  "
              "spellF1 {:.4f}".format(b["alive_dead_f1"], b["macro_f1"], b["spell_f1_mean"]))

    lo, hi = 11185, 18684
    gt_dead = gt_df["label"].values == "D"
    print("\n  dead-ball stretches in the public window (frame from-to, seconds)")
    print("    truth:     " + ", ".join("{}-{} ({:.1f}s)".format(*s)
                                      for s in segments_of(gt_dead, lo, hi)))
    print("    predicted: " + ", ".join("{}-{} ({:.1f}s)".format(*s)
                                      for s in segments_of(dead, lo, hi)))


# -- main ---------------------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--refresh", action="store_true", help="re-parse the ATD")
    parser.add_argument("--calibrate", action="store_true",
                        help="grid over switch penalty and energy pivot, public window only")
    parser.add_argument("--out", default=os.path.join(RESULTS_DIR, "deadball_v1.csv"))
    args = parser.parse_args()

    p = dict(PARAMS)
    n = source_frames("tactical", GAME)
    graded = np.zeros(n, bool)
    for _, first, last in half_video_blocks(GAME):
        graded[first:last + 1] = True

    ball, px, py, ps, team = load_tracking(refresh=args.refresh)
    feats = build_features(ball, px, py, ps, p)
    team_labels = smooth(nearest_team_codes(ball, px, py, team), 50)

    if args.calibrate:
        print("\npenalty  energy_q   alive/dead F1   D F1    spellF1   (public window, 4 stoppages)")
        for q in (0.35, 0.50, 0.65):
            for pen in (10, 25, 50, 100):
                pp = dict(p, energy_q=q, switch_penalty=pen)
                dead = dead_mask(evidence(feats, pp, graded), pp)
                r = score(labels_from(dead, team_labels, graded))
                print("  {:>5}   {:>6}     {:.4f}        {:.3f}   {:.4f}".format(
                    pen, q, r["alive_dead_f1"], r["per_class"]["D"]["f1"], r["spell_f1_mean"]))
        print("\nOnly 4 dead-ball stretches here. Prefer the defaults unless a setting is "
              "clearly broken.")
        return

    dead = dead_mask(evidence(feats, p, graded), p)
    labels = labels_from(dead, team_labels, graded)
    write_submission(args.out, labels)
    np.save(os.path.join(CACHE_DIR, "{}_dead.npy".format(GAME)), dead)

    print("wrote {}".format(os.path.relpath(args.out, ROOT)))
    print("dead on {:.1f}% of graded frames, {} dead-ball stretches".format(
        100.0 * dead[graded].mean(), len(runs(dead & graded)[0])))
    report(labels, dead, os.path.basename(args.out))
    print("\nNext: python evaluation/validate_submission.py {}".format(
        os.path.relpath(args.out, ROOT)))


if __name__ == "__main__":
    main()
