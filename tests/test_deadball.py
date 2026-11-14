"""Unit tests for the dead-ball evidence in possession/deadball.py. Synthetic inputs only.

    python -m unittest tests.test_deadball -v
"""

import os
import sys
import unittest

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "possession"))

import deadball  # noqa: E402

FPS = 25
MID = (50.0, 30.0)
KEYS = ("out", "pending", "still", "inside_fast", "sp_pre", "sp_post", "goal_dead",
        "long_gap", "open_play")


def features(ball, events=()):
    """build_features on a scripted ball, two standing players, and hand-written tags."""
    ball = np.asarray(ball, np.float32)
    n = len(ball)
    px = np.tile(np.array([[30.0, 60.0]], np.float32), (n, 1))
    py = np.tile(np.array([[20.0, 40.0]], np.float32), (n, 1))
    ps = np.zeros((n, 2), np.float32)
    return deadball.build_features(ball, px, py, ps, dict(deadball.PARAMS), events=list(events))


def tag(code, start_s, end_s, **tags):
    return {"code": code, "start_frame": int(start_s * FPS), "end_frame": int(end_s * FPS),
            "start_s": start_s, "end_s": end_s, "tags": tags}


def blank(n, energy=None):
    """Features with every signal off and (by default) the same player speed everywhere."""
    f = {key: np.zeros(n, bool) for key in KEYS}
    f["energy"] = np.ones(n) if energy is None else np.asarray(energy, float)
    return f


def llr(f):
    return deadball.evidence(f, dict(deadball.PARAMS), np.ones(len(f["energy"]), bool))


def still_ball(n, missing=()):
    ball = np.tile(np.array([MID], np.float32), (n, 1))
    for a, b in missing:
        ball[a:b] = np.nan
    return ball


class PlayerSpeed(unittest.TestCase):

    def test_players_slower_than_the_match_median_point_to_dead_ball(self):
        e = llr(blank(1001, energy=np.linspace(0.0, 4.0, 1001)))     # frame k = k/10 th percentile
        self.assertGreater(e[400], 0.0)                              # 40th percentile: dead-leaning
        self.assertLess(e[600], 0.0)                                 # 60th percentile: live-leaning
        self.assertGreater(e[250], e[400])                           # slower = stronger


class BallGaps(unittest.TestCase):

    def test_a_12_s_gap_is_a_long_gap_and_a_5_s_gap_is_not(self):
        f = features(still_ball(1500, missing=[(100, 400), (600, 725)]))
        self.assertTrue(f["long_gap"][100:400].all())
        self.assertFalse(f["long_gap"][600:725].any())
        self.assertFalse(f["long_gap"][400:600].any())

    def test_a_long_gap_is_evidence_for_dead_ball(self):
        f = blank(100)
        base = llr(f)
        f["long_gap"][:] = True
        self.assertTrue((llr(f) > base).all())


class Tags(unittest.TestCase):

    def test_open_play_phases_are_marked_and_set_pieces_are_not(self):
        f = features(still_ball(1000), [tag("PROGRESSION", 4, 8),
                                        tag("SET PIECES", 20, 30, Type="Corner Kick")])
        self.assertTrue(f["open_play"][100:200].all())
        self.assertFalse(f["open_play"][:100].any() or f["open_play"][200:].any())

    def test_open_play_is_evidence_for_live_ball(self):
        f = blank(100)
        base = llr(f)
        f["open_play"][:] = True
        self.assertTrue((llr(f) < base).all())

    def test_goal_is_dead_until_the_kick_off_that_follows_it(self):
        no_ball = np.full((3000, 2), np.nan, np.float32)
        f = features(no_ball, [tag("GOALS", 10, 20),
                               tag("SET PIECES", 60, 65, Type="Kick Off (After Goal)")])
        self.assertTrue(f["goal_dead"][250:1500].all())
        self.assertFalse(f["goal_dead"][1700:].any())

    def test_a_four_minute_goal_stoppage_is_dead_until_its_kick_off(self):
        """The real GOALS tags run from the goal to the kick-off and last up to 251 s."""
        no_ball = np.full((8000, 2), np.nan, np.float32)
        f = features(no_ball, [tag("GOALS", 10, 260),
                               tag("SET PIECES", 259, 264, Type="Kick Off (After Goal)")])
        self.assertTrue(f["goal_dead"][250:6450].all())
        self.assertFalse(f["goal_dead"][6700:].any())

    def test_goal_with_no_kick_off_soon_after_forces_nothing(self):
        no_ball = np.full((16000, 2), np.nan, np.float32)
        f = features(no_ball, [tag("GOALS", 10, 20),
                               tag("SET PIECES", 600, 605, Type="Kick Off (After Goal)")])
        self.assertFalse(f["goal_dead"].any())


class Evidence(unittest.TestCase):

    def test_a_stationary_ball_is_not_evidence_either_way(self):
        f = blank(100)
        base = llr(f)
        f["still"][:] = True
        self.assertTrue(np.array_equal(llr(f), base))


class WaitingForARestart(unittest.TestCase):

    def test_ball_thrown_around_after_going_out_does_not_end_the_wait(self):
        """Out of play, thrown back in and along the touchline to the taker (fast, but no pause
        before it), placed for 1 s, then kicked. The wait ends at the kick, not at the first
        fast ball inside."""
        def line(a, b, frames):
            return [(a[0] + (b[0] - a[0]) * k / frames, a[1] + (b[1] - a[1]) * k / frames)
                    for k in range(1, frames + 1)]
        ball = [(50.0, 5.0)] * 50                                    # 0-49    held near the line
        ball += line((50, 5), (50, -0.5), 10)                        # 50-59   goes out
        ball += line((50, -0.5), (50, -4.5), 50)                     # 60-109  rolls away, 2 m/s
        ball += line((50, -4.5), (50, 2), 20)                        # 110-129 thrown back, 8 m/s
        ball += line((50, 2), (68, 2), 55)                           # 130-184 along the line
        ball += [(68.0, 2.0)] * 25                                   # 185-209 placed
        ball += line((68, 2), (68, 32), 50)                          # 210-259 kicked, 15 m/s
        ball += [(68.0, 32.0)] * 100
        f = features(ball)
        self.assertTrue(f["out"][65:105].all())
        self.assertTrue(f["pending"][130:205].all())
        self.assertFalse(f["pending"][225:].any())

    def test_ball_deep_inside_the_pitch_ends_the_wait_even_without_a_kick(self):
        """Out of play, then the ball comes back and keeps moving deep infield with no pause, so
        no kick is ever seen. A restart happens at the edge of the pitch, so a ball 8 m inside
        every line means play has restarted."""
        def line(a, b, frames):
            return [(a[0] + (b[0] - a[0]) * k / frames, a[1] + (b[1] - a[1]) * k / frames)
                    for k in range(1, frames + 1)]
        ball = [(50.0, 5.0)] * 50                                    # 0-49    held near the line
        ball += line((50, 5), (50, -0.5), 10)                        # 50-59   goes out
        ball += line((50, -0.5), (50, -4.5), 50)                     # 60-109  rolls away, 2 m/s
        ball += line((50, -4.5), (50, 19.5), 75)                     # 110-184 comes back, 8 m/s
        ball += line((50, 19.5), (70, 19.5), 100)                    # 185-284 moved about, 5 m/s
        ball += line((70, 19.5), (50, 19.5), 100)                    # 285-384
        f = features(ball)
        self.assertTrue(f["pending"][126:145].all())                 # still near the line
        self.assertFalse(f["pending"][155:].any())                   # 8 m inside from frame 150


    def test_a_throw_in_from_outside_the_line_ends_the_wait(self):
        """Ball lies still outside the pitch for 4 s, then is thrown in. The first frames of the
        throw are still outside the line; the wait must end as soon as the ball is back inside."""
        ball = [(50.0, 5.0)] * 50                                    # 0-49    live
        ball += [(50.0 + 0.0 * k, 5.0 - 0.6 * k) for k in range(1, 12)]   # 50-60   kicked out
        ball += [(50.0, -1.6)] * 100                                 # 61-160  lying outside
        ball += [(50.0, -1.6 + 0.3 * k) for k in range(1, 61)]       # 161-220 thrown in, 7.5 m/s
        ball += [(50.0, 16.4)] * 100
        f = features(ball)
        self.assertTrue(f["out"][65:160].all())
        self.assertFalse(f["pending"][175:].any())


if __name__ == "__main__":
    unittest.main()
