"""Which team has the ball -- step 2 of the possession pipeline.

Given the tracking (ball, players, team of each track slot) and the dead-ball mask from
deadball.py, decide H or A for every live frame. Pure functions on arrays; reads no files.

The rule
--------
1. A frame is NEAR a team when that team's closest player is within `radius_m` of the ball and
   at least `margin_m` closer than the other team's closest player. Otherwise it is near nobody:
   ball in the air, contested, or not tracked.
2. A near frame is a CONTROL frame when the ball is also moving no faster than a player can carry
   it (`control_max_ball_speed`). A pass rolling past an opponent is near him but not controlled.
3. Possession stays with the last team until the other team shows control, in one of two ways:
     a. `switch_frames` control frames (1.0 s), added up across all its players. Frames near
        nobody pause the count; the original team controlling the ball again resets it.
     b. a completed pass: two touches by the other team, at least `pass_min_dist_m` apart, with
        no control by the original team in between. A touch is a spell near the ball in which the
        ball was held or its velocity changed -- a ball flying past untouched is not one.
4. The switch is backdated: to the first of the counted frames (a), or to the first touch (b).
   If the ball was untracked just before, the switch goes to the middle of that gap.
5. After a dead ball there is no wait: the first team near the ball has it from the restart.
"""

import math

import numpy as np

FPS = 25
HOME, AWAY, NOBODY = 0, 1, -1

# Params -- football priors, deliberately NOT tuned on the public window.
PARAMS = dict(
    radius_m=2.0,                 # area of influence around a player
    margin_m=1.0,                 # must be this much closer than the nearest opponent
    control_max_ball_speed=8.0,   # m/s: faster than this, nobody is carrying it
    switch_frames=25,             # control frames the other team needs (1.0 s)
    touch_min_frames=3,           # a touch is at least this long near the ball
    touch_dv=4.0,                 # m/s: velocity change that proves the ball was played
    touch_velocity_frames=3,      # frames used to measure velocity before/after a touch
    pass_min_dist_m=5.0,          # two touches this far apart are a pass, not one player twice
    fill_gap_frames=12,           # interpolate the ball across gaps up to this long (0.5 s)
)


# -- per-frame evidence --------------------------------------------------------------------------

def runs(mask):
    """(starts, ends) of True runs, ends exclusive."""
    d = np.diff(np.concatenate(([0], np.asarray(mask).astype(np.int8), [0])))
    return np.flatnonzero(d == 1), np.flatnonzero(d == -1)


def fill_gaps(ball, max_frames):
    """Copy of ball (n,2) with NaN gaps of up to max_frames linearly interpolated."""
    out = np.array(ball, dtype=np.float64)
    missing = np.isnan(out[:, 0]) | np.isnan(out[:, 1])
    out[missing] = np.nan
    for s, e in zip(*runs(missing)):
        if s == 0 or e == len(out) or e - s > max_frames:
            continue
        f = (np.arange(s, e) - (s - 1)) / float(e - (s - 1))
        out[s:e] = out[s - 1] + f[:, None] * (out[e] - out[s - 1])
    return out


def ball_speed(ball):
    """m/s per frame. Central difference over +/-2 frames where possible (robust to jitter),
    then +/-1, then one-sided. NaN where the ball has no tracked neighbour."""
    n = len(ball)
    speed = np.full(n, np.nan)

    def use(lo, hi, frames):
        """speed[i] from ball[i+hi] - ball[i+lo], wherever still unknown."""
        i = np.arange(max(0, -lo), n - max(0, hi))
        d = ball[i + hi] - ball[i + lo]
        v = np.hypot(d[:, 0], d[:, 1]) * FPS / frames
        take = np.isnan(speed[i]) & ~np.isnan(v)
        speed[i[take]] = v[take]

    use(-2, 2, 4.0)
    use(-1, 1, 2.0)
    use(-1, 0, 1.0)
    use(0, 1, 1.0)
    speed[np.isnan(ball[:, 0])] = np.nan
    return speed


def proximity(ball, px, py, team, p):
    """(near, dist): team near the ball per frame (HOME/AWAY/NOBODY), and how near, in metres."""
    with np.errstate(invalid="ignore"):
        d = np.hypot(px - ball[:, 0:1], py - ball[:, 1:2])
    d = np.where(np.isnan(d), np.inf, d)
    n = len(ball)
    dh = d[:, team == HOME].min(axis=1) if np.any(team == HOME) else np.full(n, np.inf)
    da = d[:, team == AWAY].min(axis=1) if np.any(team == AWAY) else np.full(n, np.inf)
    with np.errstate(invalid="ignore"):
        home = (dh <= p["radius_m"]) & (da - dh >= p["margin_m"])
        away = (da <= p["radius_m"]) & (dh - da >= p["margin_m"])
    near = np.where(home, HOME, np.where(away, AWAY, NOBODY)).astype(np.int8)
    return near, np.where(home, dh, np.where(away, da, np.inf))


def find_touches(near, ctrl, dist, ball, p):
    """[(start, end, frame, team, x, y)] for every spell near the ball that counts as a touch.

    `frame` is the closest approach, taken as the moment of the touch. A spell counts when the
    ball was held (control frames) or its velocity changed across it; a ball that flies past with
    the same velocity before and after was not touched.
    """
    n, k = len(near), p["touch_velocity_frames"]
    cut = np.flatnonzero(np.diff(near) != 0) + 1
    starts = np.concatenate(([0], cut))
    ends = np.concatenate((cut, [n]))
    out = []
    for s, e in zip(starts.tolist(), ends.tolist()):
        t = int(near[s])
        if t == NOBODY or e - s < p["touch_min_frames"]:
            continue
        held = int(np.count_nonzero(ctrl[s:e] == t)) >= p["touch_min_frames"]
        if not held:
            if s - 1 - k < 0 or e + k >= n:
                continue
            v_in = (ball[s - 1] - ball[s - 1 - k]) * FPS / k
            v_out = (ball[e + k] - ball[e]) * FPS / k
            dv = math.hypot(v_out[0] - v_in[0], v_out[1] - v_in[1])
            if not dv >= p["touch_dv"]:                 # also False when a velocity is NaN
                continue
        f = s + int(np.argmin(dist[s:e]))
        out.append((s, e - 1, f, t, float(ball[f, 0]), float(ball[f, 1])))
    return out


# -- the hand-over state machine -----------------------------------------------------------------

def owner_path(ctrl, near, touches, missing, p, prev_owner=None, stats=None):
    """Owner (HOME/AWAY, or NOBODY if never known) for each frame of one live stretch.

    ctrl, near: per-frame team codes. touches: (start, end, frame, team, x, y) inside the
    stretch. missing: True where the ball is untracked. prev_owner: who had it before this
    stretch, used only if nobody comes near the ball in the whole stretch. stats: optional dict,
    gets switches_by_count / switches_by_pass added to.
    """
    n = len(ctrl)
    out = np.full(n, NOBODY, np.int8)
    seen = np.flatnonzero(near != NOBODY)
    if len(seen):
        owner = int(near[seen[0]])
    elif prev_owner in (HOME, AWAY):
        owner = prev_owner
    else:
        return out

    switches = [(0, owner)]
    touch_ending_at = {t[1]: t for t in touches}
    state = dict(owner=owner, count=0, cand=None, first=None)

    def reset():
        state["count"], state["cand"], state["first"] = 0, None, None

    def switch(frame, why):
        if stats is not None:
            stats[why] = stats.get(why, 0) + 1
        if frame > 0 and missing[frame - 1]:            # ball untracked just before: split the gap
            g0 = frame - 1
            while g0 > 0 and missing[g0 - 1]:
                g0 -= 1
            frame = (g0 + frame) // 2
        frame = max(frame, switches[-1][0] + 1)
        state["owner"] = 1 - state["owner"]
        switches.append((frame, state["owner"]))
        reset()

    limit = p["switch_frames"]
    for i, c in enumerate(ctrl.tolist()):
        if c == state["owner"]:
            reset()
        elif c != NOBODY:
            if state["cand"] is None:
                state["cand"] = i
            state["count"] += 1
            if state["count"] >= limit:
                switch(state["cand"], "switches_by_count")

        touch = touch_ending_at.get(i)
        if touch is not None:
            if touch[3] == state["owner"]:
                reset()
            elif state["first"] is None:
                state["first"] = touch
            else:
                first = state["first"]
                if math.hypot(touch[4] - first[4], touch[5] - first[5]) >= p["pass_min_dist_m"]:
                    switch(first[2], "switches_by_pass")

    bounds = [f for f, _ in switches] + [n]
    for (f, team), nxt in zip(switches, bounds[1:]):
        out[f:nxt] = team
    return out


def frame_evidence(ball, px, py, team, alive, p):
    """What the rule sees on every frame: dict(filled, missing, speed, near, dist, ctrl, touches).

    near / ctrl are NOBODY outside `alive`, so nothing carries across a dead ball.
    """
    filled = fill_gaps(ball, p["fill_gap_frames"])
    speed = ball_speed(filled)
    near, dist = proximity(filled, px, py, np.asarray(team), p)
    near[~np.asarray(alive, bool)] = NOBODY
    with np.errstate(invalid="ignore"):
        carried = np.isnan(speed) | (speed <= p["control_max_ball_speed"])
    ctrl = np.where(carried, near, NOBODY).astype(np.int8)
    return dict(filled=filled, missing=np.isnan(filled[:, 0]), speed=speed, near=near, dist=dist,
                ctrl=ctrl, touches=find_touches(near, ctrl, dist, filled, p))


# -- public entry point --------------------------------------------------------------------------

def decide(ball, px, py, team, dead=None, graded=None, params=None, stats=None):
    """Label every frame: H / A where the ball is live, D where dead, X where ungraded.

    ball (n,2) and px, py (n,slots) in metres, NaN where missing; team (slots,) 0 = home,
    1 = away. dead and graded are bool arrays (default: nothing dead, everything graded).
    Pass a dict as `stats` to get counts back (aggregates only).
    """
    p = dict(PARAMS, **(params or {}))
    n = len(ball)
    dead = np.zeros(n, bool) if dead is None else np.asarray(dead, bool)
    graded = np.ones(n, bool) if graded is None else np.asarray(graded, bool)
    team = np.asarray(team)
    alive = graded & ~dead

    ev = frame_evidence(ball, px, py, team, alive, p)
    missing, near, ctrl, touches = ev["missing"], ev["near"], ev["ctrl"], ev["touches"]
    counts = {}

    owner = np.full(n, NOBODY, np.int8)
    prev = None
    k = 0
    for a, b in zip(*runs(alive)):
        while k < len(touches) and touches[k][0] < a:
            k += 1
        inside = []
        while k < len(touches) and touches[k][1] < b:
            s, e, f, t, x, y = touches[k]
            inside.append((s - a, e - a, f - a, t, x, y))
            k += 1
        path = owner_path(ctrl[a:b], near[a:b], inside, missing[a:b], p, prev, counts)
        owner[a:b] = path
        if path[-1] != NOBODY:
            prev = int(path[-1])

    # A live stretch nobody ever came near, before any known owner: take the next known one.
    unknown = np.flatnonzero(alive & (owner == NOBODY))
    known = np.flatnonzero(alive & (owner != NOBODY))
    if len(unknown) and len(known):
        nxt = np.searchsorted(known, unknown)
        ok = nxt < len(known)
        owner[unknown[ok]] = owner[known[nxt[ok]]]

    labels = np.full(n, "X", dtype="<U1")
    labels[graded & dead] = "D"
    labels[alive & (owner == HOME)] = "H"
    labels[alive & (owner == AWAY)] = "A"

    if stats is not None:
        stats.update(
            ball_tracked_pct=100.0 * float(np.mean(~np.isnan(np.asarray(ball)[graded, 0]))),
            ball_after_fill_pct=100.0 * float(np.mean(~missing[graded])),
            near_pct=100.0 * float(np.mean(near[alive] != NOBODY)) if alive.any() else 0.0,
            control_pct=100.0 * float(np.mean(ctrl[alive] != NOBODY)) if alive.any() else 0.0,
            touches=len(touches),
            unlabelled_live_frames=int(np.sum(alive & (owner == NOBODY))),
            switches_by_count=counts.get("switches_by_count", 0),
            switches_by_pass=counts.get("switches_by_pass", 0),
        )
    return labels
