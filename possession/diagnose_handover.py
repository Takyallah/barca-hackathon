"""Why are hand-overs late or missed? A diagnostic, not part of the pipeline.

    python possession/diagnose_handover.py

Three tables, all against the public 5 minutes:
1. Frame level: for several radius / margin / speed-cap settings, on how many live frames the
   rule names a team, and how often that team is the true owner.
2. One row per true possession: how much control the rule sees for its own team and for the
   other team, and how long the own team needs to reach the 25 control frames.
3. A few variants of the hand-over rule, each run over the whole match.

Output is percentages, counts and timings only -- no positions, speeds or other data values.
Table 3 shows spellF1 on 17 possessions: far too noisy to pick a winner by. Use it to see which
way a change moves the possession count and the frame-level team accuracy.
"""

import argparse
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
import deadball  # noqa: E402  (also puts the repo root, starter/ and evaluation/ on the path)
import control  # noqa: E402
import possession_metrics as pm  # noqa: E402
from games import GAME, half_video_blocks, source_frames  # noqa: E402

FPS = 25

# Params
RADII = (1.0, 1.5, 2.0, 3.0, 4.0)
MARGINS = (0.0, 0.5, 1.0, 2.0)
SPEED_CAPS = (4.0, 8.0, 12.0, 1e9)
VARIANTS = (
    ("current", {}),
    ("no pass rule", dict(pass_min_dist_m=1e9)),
    ("0.6 s to switch", dict(switch_frames=15)),
    ("1.6 s to switch", dict(switch_frames=40)),
    ("radius 3 m", dict(radius_m=3.0)),
    ("speed cap 12 m/s", dict(control_max_ball_speed=12.0)),
)
SPELL_GAP_FRAMES = 12        # wrong-team control frames this close together are one spell
CROSS_M = 1.5                # two tracks this close could have swapped identities
LOOKBACK_S = 2               # ...looked for this long before the spell
JUMP_M = 2.0                 # a track moving this far in one frame was re-assigned
AGE_CAP_S = 60               # how far back the start of a track is searched for
YOUNG_S = 5.0                # a track younger than this is "new"
NEAR_BOUNDARY_S = 5.0        # this close to a true hand-over counts as a timing difference


def load_context(data_dir=None, cache_dir=None, game=GAME):
    """Everything the tables need: tracking arrays, team per slot, our dead mask, graded mask."""
    graded = np.zeros(source_frames("tactical", game), bool)
    for _, first, last in half_video_blocks(game):
        graded[first:last + 1] = True
    ball, px, py, ps, team = deadball.load_tracking(game, data_dir, cache_dir=cache_dir)
    p = dict(deadball.PARAMS)
    feats = deadball.build_features(ball, px, py, ps, p, game, data_dir)
    dead = deadball.dead_mask(deadball.evidence(feats, p, graded), p)
    return dict(ball=ball.astype(np.float64), px=px, py=py, team=team, dead=dead, graded=graded)


def window_of(truth):
    """(lo, hi) slice bounds covering every labelled truth frame, with a 2 s margin."""
    idx = np.flatnonzero(np.asarray(truth) != "X")
    return max(0, int(idx[0]) - 2 * FPS), min(len(truth), int(idx[-1]) + 1 + 2 * FPS)


def window_evidence(ctx, truth, params=None):
    """(lo, evidence dict) for the truth window, with nothing masked as dead."""
    lo, hi = window_of(truth)
    p = dict(control.PARAMS, **(params or {}))
    ev = control.frame_evidence(ctx["ball"][lo:hi], ctx["px"][lo:hi], ctx["py"][lo:hi],
                                ctx["team"], np.ones(hi - lo, bool), p)
    return lo, ev


def frame_grid(ctx, truth):
    """[{radius, margin, speed_cap, near_pct, near_acc, ctrl_pct, ctrl_acc}] on true live frames."""
    truth = np.asarray(truth)
    settings = [(r, m, 8.0) for r in RADII for m in MARGINS]
    settings += [(2.0, 1.0, v) for v in SPEED_CAPS if v != 8.0]
    settings += [(1e9, 0.0, 1e9)]                           # plain nearest player
    rows = []
    for radius, margin, cap in settings:
        lo, ev = window_evidence(ctx, truth, dict(radius_m=radius, margin_m=margin,
                                                  control_max_ball_speed=cap))
        t = truth[lo:lo + len(ev["near"])]
        live = (t == "H") | (t == "A")
        owner = np.where(t == "A", control.AWAY, control.HOME)
        row = dict(radius=radius, margin=margin, speed_cap=cap)
        for key, codes in (("near", ev["near"]), ("ctrl", ev["ctrl"])):
            said = live & (codes != control.NOBODY)
            row[key + "_pct"] = 100.0 * said.sum() / max(live.sum(), 1)
            row[key + "_acc"] = 100.0 * np.sum(said & (codes == owner)) / max(said.sum(), 1)
        rows.append(row)
    return rows


def possession_rows(ctx, truth, labels):
    """One dict per true possession in the window. Times in seconds from its first frame."""
    truth = np.asarray(truth)
    labels = np.asarray(labels)
    lo, ev = window_evidence(ctx, truth)
    limit = control.PARAMS["switch_frames"]
    rows = []
    for letter, code in (("H", control.HOME), ("A", control.AWAY)):
        for s, e in zip(*control.runs(truth == letter)):
            ctrl = ev["ctrl"][s - lo:e - lo]
            near = ev["near"][s - lo:e - lo]
            own, other = ctrl == code, ctrl == 1 - code
            own_at = np.flatnonzero(own)
            touches = [t for t in ev["touches"] if s - lo <= t[2] < e - lo]
            rows.append(dict(
                team=letter, start=int(s), seconds=(e - s) / float(FPS),
                ball_pct=100.0 * float(np.mean(~ev["missing"][s - lo:e - lo])),
                own_near_pct=100.0 * float(np.mean(near == code)),
                other_near_pct=100.0 * float(np.mean(near == 1 - code)),
                own_ctrl=int(own.sum()), other_ctrl=int(other.sum()),
                other_spells=len(control.runs(other)[0]),
                own_touches=sum(1 for t in touches if t[3] == code),
                other_touches=sum(1 for t in touches if t[3] != code),
                first_own_s=own_at[0] / float(FPS) if len(own_at) else None,
                reach_s=own_at[limit - 1] / float(FPS) if len(own_at) >= limit else None,
                right_pct=100.0 * float(np.mean(labels[s:e] == letter)),
            ))
    return sorted(rows, key=lambda r: r["start"])


def variant_rows(ctx, truth):
    """[{name, window possessions, team accuracy, spellF1, whole-match counts}] per variant."""
    truth = np.asarray(truth)
    gt = pm.label_codes(truth)
    rows = []
    for name, params in VARIANTS:
        stats = {}
        labels = control.decide(ctx["ball"], ctx["px"], ctx["py"], ctx["team"], dead=ctx["dead"],
                                graded=ctx["graded"], params=params, stats=stats)
        r = pm.evaluate(gt, pm.label_codes(labels), bootstrap_n=0)
        lengths = []
        for _, first, last in half_video_blocks(GAME):
            block = labels[first:last + 1]
            cut = np.flatnonzero(block[1:] != block[:-1]) + 1
            starts = np.concatenate(([0], cut))
            ends = np.concatenate((cut, [len(block)]))
            lengths += [b - a for a, b in zip(starts, ends) if block[a] in ("H", "A")]
        rows.append(dict(
            name=name, spells=r["n_pred_spells"], truth_spells=r["n_gt_spells"],
            team_acc=100.0 * r["team_acc_both_alive"], spell_f1=r["spell_f1_mean"],
            macro_f1=r["macro_f1"], match_spells=len(lengths),
            match_mean_s=float(np.mean(lengths)) / FPS if lengths else 0.0,
            by_count=stats["switches_by_count"], by_pass=stats["switches_by_pass"]))
    return rows


def _spells(mask, breaker):
    """[(first, last, frames)] of True frames in `mask`, joined across gaps of up to
    SPELL_GAP_FRAMES unless a `breaker` frame lies in between."""
    out = []
    for i in np.flatnonzero(mask).tolist():
        if out and i - out[-1][1] <= SPELL_GAP_FRAMES and not breaker[out[-1][1]:i].any():
            out[-1][1], out[-1][2] = i, out[-1][2] + 1
        else:
            out.append([i, i, 1])
    return out


def _track_features(ctx, ball_xy, frame, code):
    """What the controlling track did just before `frame`: (slot, crossed, jumped, age_s)."""
    px, py, team = ctx["px"], ctx["py"], np.asarray(ctx["team"])
    with np.errstate(invalid="ignore"):
        d = np.hypot(px[frame] - ball_xy[0], py[frame] - ball_xy[1]).astype(np.float64)
    d[np.isnan(d) | (team != code)] = np.inf
    slot = int(np.argmin(d))
    a = max(0, frame - LOOKBACK_S * FPS)
    tx, ty = px[a:frame + 1, slot].astype(np.float64), py[a:frame + 1, slot].astype(np.float64)
    with np.errstate(invalid="ignore"):
        gap = np.hypot(px[a:frame + 1][:, team != code] - tx[:, None],
                       py[a:frame + 1][:, team != code] - ty[:, None])
        step = np.hypot(np.diff(tx), np.diff(ty))
    gap, step = gap[~np.isnan(gap)], step[~np.isnan(step)]
    back = px[max(0, frame - AGE_CAP_S * FPS):frame + 1, slot]
    lost = np.flatnonzero(np.isnan(back))
    age = (len(back) - 1 - int(lost[-1])) / float(FPS) if len(lost) else None
    return (slot, bool(len(gap) and gap.min() <= CROSS_M), bool(len(step) and step.max() > JUMP_M),
            age)


def _control_split(ctx, truth):
    """(lo, evidence, truth slice, wrong mask, right mask) for the truth window."""
    truth = np.asarray(truth)
    lo, ev = window_evidence(ctx, truth)
    t = truth[lo:lo + len(ev["ctrl"])]
    live = (t == "H") | (t == "A")
    owner = np.where(t == "A", control.AWAY, control.HOME)
    said = live & (ev["ctrl"] != control.NOBODY)
    return lo, ev, t, said & (ev["ctrl"] != owner), said & (ev["ctrl"] == owner)


def wrong_control_spells(ctx, truth):
    """One dict per spell in which the rule sees the WRONG team controlling the ball.

    relation says whether it looks like a timing difference with the truth ("old owner still on
    it" just after a true hand-over, "next owner early" just before one) or sits in the middle
    of a possession. crossed / jumped / slot_age_s describe the controlling track just before
    the spell: a close crossing with an opponent, a one-frame jump, or a brand-new track are
    the signatures of the tracker having swapped or re-assigned an identity.
    """
    lo, ev, t, wrong, right = _control_split(ctx, truth)
    cut = np.flatnonzero(t[1:] != t[:-1]) + 1
    bounds = np.concatenate(([0], cut, [len(t)]))
    rows = []
    for s, e, frames in _spells(wrong, right):
        code = int(ev["ctrl"][s])
        letter = "A" if code == control.AWAY else "H"
        slot, crossed, jumped, age = _track_features(ctx, ev["filled"][s], lo + s, code)
        k = int(np.searchsorted(bounds, s, side="right")) - 1
        seg_start, seg_end = int(bounds[k]), int(bounds[k + 1])
        prev = t[seg_start - 1] if seg_start > 0 else None
        nxt = t[seg_end] if seg_end < len(t) else None
        after, before = (s - seg_start) / float(FPS), (seg_end - 1 - e) / float(FPS)
        if prev == letter and after <= NEAR_BOUNDARY_S:
            relation = "old owner still on it"
        elif nxt == letter and before <= NEAR_BOUNDARY_S:
            relation = "next owner early"
        else:
            relation = "mid-possession"
        slots = set()
        for i in np.flatnonzero(wrong[s:e + 1]).tolist():
            slots.add(_track_features(ctx, ev["filled"][s + i], lo + s + i, code)[0])
        rows.append(dict(start=lo + s, frames=frames, span_s=(e - s + 1) / float(FPS),
                         owner=str(t[s]), controller=letter, after_start_s=after,
                         before_end_s=before, relation=relation, slots=len(slots),
                         slot_age_s=age, crossed=crossed, jumped=jumped))
    return rows


def right_control_summary(ctx, truth):
    """The same track signatures for spells where the rule names the RIGHT team, as a baseline."""
    lo, ev, t, wrong, right = _control_split(ctx, truth)
    feats = [_track_features(ctx, ev["filled"][s], lo + s, int(ev["ctrl"][s]))
             for s, e, frames in _spells(right, wrong)]
    n = max(len(feats), 1)
    return dict(spells=len(feats),
                crossed_pct=100.0 * sum(f[1] for f in feats) / n,
                jumped_pct=100.0 * sum(f[2] for f in feats) / n,
                young_pct=100.0 * sum(f[3] is not None and f[3] < YOUNG_S for f in feats) / n)


def show(value, fmt="{:.1f}"):
    return "--" if value is None else fmt.format(value)


def report(ctx, truth, labels):
    print("1. frame level, true live frames of the public window")
    print("   named % = the rule names a team; right % = that team is the true owner")
    print("   {:>7} {:>7} {:>9}   {:>12} {:>8}   {:>12} {:>8}".format(
        "radius", "margin", "speed cap", "near: named%", "right %", "ctrl: named%", "right %"))
    for r in frame_grid(ctx, truth):
        radius = "any" if r["radius"] > 1e6 else "{:.1f}".format(r["radius"])
        cap = "none" if r["speed_cap"] > 1e6 else "{:.0f}".format(r["speed_cap"])
        print("   {:>7} {:>7.1f} {:>9}   {:>12.1f} {:>8.1f}   {:>12.1f} {:>8.1f}".format(
            radius, r["margin"], cap, r["near_pct"], r["near_acc"], r["ctrl_pct"], r["ctrl_acc"]))

    print("\n2. every true possession (frames = control frames the rule sees; s from its start)")
    print("   {:>4} {:>6} {:>6} {:>6} {:>9} {:>9} {:>7} {:>8} {:>8} {:>9} {:>9} {:>7}".format(
        "team", "start", "len s", "ball %", "own ctrl", "oth ctrl", "oth sp.", "own tch",
        "oth tch", "1st own s", "25th at s", "right %"))
    for r in possession_rows(ctx, truth, labels):
        print("   {:>4} {:>6} {:>6.1f} {:>6.0f} {:>9} {:>9} {:>7} {:>8} {:>8} {:>9} {:>9} {:>7.0f}"
              .format(r["team"], r["start"], r["seconds"], r["ball_pct"], r["own_ctrl"],
                      r["other_ctrl"], r["other_spells"], r["own_touches"], r["other_touches"],
                      show(r["first_own_s"]), show(r["reach_s"]), r["right_pct"]))
    print("   oth sp. = separate spells of control by the other team inside the possession")
    print("   right % = share of the possession our current output gives to the right team")

    print("\n3. variants of the hand-over rule (same dead-ball mask for all)")
    print("   {:<36} {:>9} {:>9} {:>8} {:>8}   {:>7} {:>7} {:>8} {:>8}".format(
        "", "poss", "team acc", "spellF1", "macroF1", "match n", "mean s", "by count",
        "by pass"))
    rows = variant_rows(ctx, truth)
    for r in rows:
        print("   {:<36} {:>9} {:>9.1f} {:>8.4f} {:>8.4f}   {:>7} {:>7.1f} {:>8} {:>8}".format(
            r["name"], r["spells"], r["team_acc"], r["spell_f1"], r["macro_f1"],
            r["match_spells"], r["match_mean_s"], r["by_count"], r["by_pass"]))
    print("   poss (truth has {}) / team acc / spellF1 / macroF1: public window. match n / mean s"
          " / by count /".format(rows[0]["truth_spells"]))
    print("   by pass: whole match (possessions, mean length, switches by each trigger).")
    print("   spellF1 here moves by ~0.06 per possession: do not pick a variant by it.")
    report_wrong_spells(ctx, truth)


def report_wrong_spells(ctx, truth):
    spells = wrong_control_spells(ctx, truth)
    print("\n4. spells where the rule sees the WRONG team in control ({} spells, {} frames)".format(
        len(spells), sum(r["frames"] for r in spells)))
    print("   {:>6} {:>6} {:>5} {:>5} {:>9} {:>9}  {:<22} {:>5} {:>7} {:>7} {:>6}".format(
        "start", "frames", "truth", "rule", "after st.", "bef. end", "looks like", "trks",
        "crossed", "jumped", "age s"))
    for r in spells:
        age = "{}+".format(AGE_CAP_S) if r["slot_age_s"] is None else "{:.0f}".format(
            r["slot_age_s"])
        print("   {:>6} {:>6} {:>5} {:>5} {:>9.1f} {:>9.1f}  {:<22} {:>5} {:>7} {:>7} {:>6}".format(
            r["start"], r["frames"], r["owner"], r["controller"], r["after_start_s"],
            r["before_end_s"], r["relation"], r["slots"], "yes" if r["crossed"] else "no",
            "yes" if r["jumped"] else "no", age))
    print("   after st. / bef. end = s after the true possession started / before it ended")
    print("   trks = tracks involved; crossed = that track was within {} m of an opponent in the"
          .format(CROSS_M))
    print("   {} s before; jumped = it moved over {} m in one frame; age = s since it appeared"
          .format(LOOKBACK_S, JUMP_M))
    if spells:
        n = float(len(spells))
        print("   wrong-team spells: crossed {:.0f}%  jumped {:.0f}%  track under {:.0f} s old "
              "{:.0f}%".format(100 * sum(r["crossed"] for r in spells) / n,
                               100 * sum(r["jumped"] for r in spells) / n, YOUNG_S,
                               100 * sum(r["slot_age_s"] is not None
                                         and r["slot_age_s"] < YOUNG_S for r in spells) / n))
    base = right_control_summary(ctx, truth)
    print("   right-team spells ({}): crossed {:.0f}%  jumped {:.0f}%  track under {:.0f} s old "
          "{:.0f}%".format(base["spells"], base["crossed_pct"], base["jumped_pct"], YOUNG_S,
                           base["young_pct"]))


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data-dir", default=None)
    parser.add_argument("--cache-dir", default=None)
    args = parser.parse_args()
    truth = pm.read_label_csv(deadball.PUBLIC_GT)["label"].values.astype(str)
    ctx = load_context(args.data_dir, args.cache_dir)
    labels = control.decide(ctx["ball"], ctx["px"], ctx["py"], ctx["team"], dead=ctx["dead"],
                            graded=ctx["graded"])
    report(ctx, truth, labels)


if __name__ == "__main__":
    main()
