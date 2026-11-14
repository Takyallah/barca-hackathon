"""Hand-built football situations with a known right answer, for the possession rules.

    python tests/scenarios.py            # table: PASS/FAIL, expected vs produced segments

Everything here is SYNTHETIC: a few static players and a scripted ball, in metres at 25 fps.
No match data is read. Each scenario states what should happen and why.
"""

import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "possession"))

FPS = 25
TOLERANCE = 3          # frames a boundary may be off (the scorer forgives 1, charges 2% at 3)


class Scene:
    """Static players, scripted ball. Build it step by step; every step appends frames."""

    def __init__(self, home, away):
        self.home, self.away = list(home), list(away)
        self.ball, self.dead = [], []
        self.pos = None

    @property
    def n(self):
        return len(self.ball)

    def _add(self, point, dead):
        self.ball.append(point)
        self.dead.append(dead)
        if point is not None:
            self.pos = point

    def hold(self, player, seconds, dead=False):
        """Ball kept at a player's feet: a 0.4 m circle around them at about 2 m/s."""
        cx, cy = player
        for k in range(int(round(seconds * FPS))):
            self._add((cx + 0.4 * math.cos(0.2 * k), cy + 0.4 * math.sin(0.2 * k)), dead)

    def pass_to(self, target, speed=15.0, dead=False):
        """Ball travels in a straight line from where it is to `target` at `speed` m/s."""
        x0, y0 = self.pos
        steps = max(1, int(round(math.hypot(target[0] - x0, target[1] - y0) / speed * FPS)))
        for k in range(1, steps + 1):
            f = k / float(steps)
            self._add((x0 + (target[0] - x0) * f, y0 + (target[1] - y0) * f), dead)

    def place(self, point, seconds, dead=False):
        """Ball lying still at a point."""
        for _ in range(int(round(seconds * FPS))):
            self._add(point, dead)

    def hide(self, seconds, dead=False):
        """Ball not tracked."""
        for _ in range(int(round(seconds * FPS))):
            self._add(None, dead)

    def arrays(self):
        n = self.n
        ball = np.full((n, 2), np.nan, np.float32)
        for i, point in enumerate(self.ball):
            if point is not None:
                ball[i] = point
        players = self.home + self.away
        px = np.tile(np.array([p[0] for p in players], np.float32), (n, 1))
        py = np.tile(np.array([p[1] for p in players], np.float32), (n, 1))
        team = np.array([0] * len(self.home) + [1] * len(self.away), np.int8)
        return ball, px, py, team, np.array(self.dead, bool)


def segments(labels):
    """[(label, first_frame, last_frame)] for maximal runs."""
    out, start = [], 0
    for i in range(1, len(labels) + 1):
        if i == len(labels) or labels[i] != labels[start]:
            out.append((str(labels[start]), start, i - 1))
            start = i
    return out


def compare(expected, got, tol=TOLERANCE):
    """(ok, reason). Same labels in the same order, every boundary within `tol` frames."""
    if [s[0] for s in expected] != [s[0] for s in got]:
        return False, "label order {} != {}".format([s[0] for s in got], [s[0] for s in expected])
    for (label, a, _), (_, b, _) in zip(expected, got):
        if abs(a - b) > tol:
            return False, "{} starts at {}, expected {} (+/-{})".format(label, b, a, tol)
    return True, ""


# -- scenarios: each returns (scene, expected segments) ------------------------------------------

def hold_keeps_possession():
    """A home player keeps the ball for 4 s. -> H throughout."""
    s = Scene(home=[(30, 30)], away=[(50, 30)])
    s.hold((30, 30), 4)
    return s, [("H", 0, s.n - 1)]


def clean_handover():
    """Home passes straight to an away player who keeps it 3 s. -> switch when he receives it."""
    s = Scene(home=[(30, 30)], away=[(50, 30)])
    s.hold((30, 30), 2)
    s.pass_to((50, 30))
    t = s.n
    s.hold((50, 30), 3)
    return s, [("H", 0, t - 1), ("A", t, s.n - 1)]


def brief_touch_does_not_switch():
    """Away player has it for 0.4 s, then home wins it back. -> H throughout."""
    s = Scene(home=[(10, 30), (30, 45)], away=[(30, 30)])
    s.hold((10, 30), 2)
    s.pass_to((30, 30))
    s.hold((30, 30), 0.4)
    s.pass_to((30, 45))
    s.hold((30, 45), 2)
    return s, [("H", 0, s.n - 1)]


def pass_through_two_opponents():
    """A home pass rolls within 1 m of two away players and reaches a team-mate. -> H throughout."""
    s = Scene(home=[(10, 30), (50, 30)], away=[(25, 31), (38, 29)])
    s.hold((10, 30), 2)
    s.pass_to((50, 30), speed=12.0)
    s.hold((50, 30), 2)
    return s, [("H", 0, s.n - 1)]


def deflection_does_not_switch():
    """An away player deflects a home pass, it still reaches a home player. -> H throughout."""
    s = Scene(home=[(10, 30), (50, 36)], away=[(30, 30.5)])
    s.hold((10, 30), 2)
    s.pass_to((30, 30))
    s.pass_to((50, 36))
    s.hold((50, 36), 2)
    return s, [("H", 0, s.n - 1)]


def control_adds_up_across_team_mates():
    """Away #1 has it 0.6 s, passes to away #2 who has it 0.6 s (1.2 s for the team), then home
    wins it back. -> A from #1's reception, H again from the home reception."""
    s = Scene(home=[(10, 30), (60, 30)], away=[(30, 30), (40, 30)])
    s.hold((10, 30), 2)
    s.pass_to((30, 30))
    t1 = s.n
    s.hold((30, 30), 0.6)
    s.pass_to((40, 30))
    s.hold((40, 30), 0.6)
    s.pass_to((60, 30))
    t2 = s.n
    s.hold((60, 30), 2)
    return s, [("H", 0, t1 - 1), ("A", t1, t2 - 1), ("H", t2, s.n - 1)]


def one_touch_pass_between_opponents():
    """Away #1 plays it first time to away #2 (12 m), who plays it first time to away #3. Nobody
    holds it for 1 s, but they completed a pass. -> A from #1's touch."""
    s = Scene(home=[(10, 30)], away=[(30, 30), (30, 42), (42, 42)])
    s.hold((10, 30), 2)
    s.pass_to((30, 30))
    t = s.n - 1
    s.pass_to((30, 42))
    s.pass_to((42, 42))
    s.hold((42, 42), 0.4)
    return s, [("H", 0, t - 1), ("A", t, s.n - 1)]


def contested_ball_stays_home():
    """Home has it, then the ball sits between the two for 3 s, slightly nearer the away player
    (0.9 m vs 1.1 m). Not clearly his. -> H throughout."""
    s = Scene(home=[(30, 30)], away=[(32, 30)])
    s.place((30.2, 30), 2)
    s.place((31.1, 30), 3)
    return s, [("H", 0, s.n - 1)]


def contested_ball_stays_away():
    """Same as above, mirrored: away had it first, then it sits slightly nearer the home
    player. -> A throughout (last team keeps it)."""
    s = Scene(home=[(30, 30)], away=[(32, 30)])
    s.place((31.8, 30), 2)
    s.place((30.9, 30), 3)
    return s, [("A", 0, s.n - 1)]


def equal_distance_keeps_last_team():
    """Ball exactly half-way between the two (2 m each) for 3 s after home had it. -> H."""
    s = Scene(home=[(30, 30)], away=[(34, 30)])
    s.place((30.3, 30), 2)
    s.place((32, 30), 3)
    return s, [("H", 0, s.n - 1)]


def ball_lost_inside_a_possession():
    """Home has it, the tracking loses the ball for 4 s, home still has it. -> H throughout."""
    s = Scene(home=[(30, 30)], away=[(50, 30)])
    s.hold((30, 30), 2)
    s.hide(4)
    s.hold((30, 30), 2)
    return s, [("H", 0, s.n - 1)]


def ball_lost_across_a_handover():
    """Home has it, ball untracked for 3 s, reappears with away for 3 s. -> switch in the
    middle of the gap, not when the ball reappears."""
    s = Scene(home=[(30, 30)], away=[(50, 30)])
    s.hold((30, 30), 2)
    g0 = s.n
    s.hide(3)
    g1 = s.n
    s.hold((50, 30), 3)
    mid = (g0 + g1) // 2
    return s, [("H", 0, mid - 1), ("A", mid, s.n - 1)]


def long_ball_to_a_team_mate():
    """40 m ball in the air, far from everyone, home to home. -> H throughout."""
    s = Scene(home=[(10, 30), (60, 30)], away=[(35, 50)])
    s.hold((10, 30), 2)
    s.pass_to((60, 30), speed=20.0)
    s.hold((60, 30), 2)
    return s, [("H", 0, s.n - 1)]


def pause_does_not_reset_the_count():
    """Away has it 0.6 s, ball untracked 0.8 s, away has it another 0.6 s. -> A from the first
    reception: untracked frames pause the count, they do not reset it."""
    s = Scene(home=[(10, 30)], away=[(30, 30)])
    s.hold((10, 30), 2)
    s.pass_to((30, 30))
    t = s.n
    s.hold((30, 30), 0.6)
    s.hide(0.8)
    s.hold((30, 30), 0.6)
    return s, [("H", 0, t - 1), ("A", t, s.n - 1)]


def restart_gives_the_ball_at_once():
    """Home kicks it out, 4 s dead ball, away throws it in to a team-mate. -> H, then D, then A
    from the very first live frame (no 1 s wait after a restart)."""
    s = Scene(home=[(30, 30)], away=[(40, 1), (40, 16)])
    s.hold((30, 30), 2)
    s.pass_to((40, -1))
    d0 = s.n
    s.place((40, 0.7), 4, dead=True)
    d1 = s.n
    s.pass_to((40, 16))
    s.hold((40, 16), 2)
    return s, [("H", 0, d0 - 1), ("D", d0, d1 - 1), ("A", d1, s.n - 1)]


SCENARIOS = [
    hold_keeps_possession,
    clean_handover,
    brief_touch_does_not_switch,
    pass_through_two_opponents,
    deflection_does_not_switch,
    control_adds_up_across_team_mates,
    one_touch_pass_between_opponents,
    contested_ball_stays_home,
    contested_ball_stays_away,
    equal_distance_keeps_last_team,
    ball_lost_inside_a_possession,
    ball_lost_across_a_handover,
    long_ball_to_a_team_mate,
    pause_does_not_reset_the_count,
    restart_gives_the_ball_at_once,
]


def run(build):
    """(ok, reason, expected, got) for one scenario."""
    import control
    scene, expected = build()
    ball, px, py, team, dead = scene.arrays()
    got = segments(control.decide(ball, px, py, team, dead=dead))
    ok, reason = compare(expected, got)
    return ok, reason, expected, got


def show(segs):
    return "  ".join("{} {}-{}".format(*s) for s in segs)


if __name__ == "__main__":
    failed = 0
    for build in SCENARIOS:
        ok, reason, expected, got = run(build)
        failed += not ok
        print("{}  {}".format("PASS" if ok else "FAIL", build.__name__))
        print("      {}".format(" ".join(build.__doc__.split())))
        print("      expected: {}".format(show(expected)))
        print("      produced: {}{}".format(show(got), "" if ok else "   <-- " + reason))
    print("\n{} of {} scenarios pass".format(len(SCENARIOS) - failed, len(SCENARIOS)))
    sys.exit(1 if failed else 0)
