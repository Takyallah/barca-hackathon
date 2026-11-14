"""Unit tests for possession/control.py. Synthetic inputs only -- no match data.

    python -m unittest discover -s tests -v
"""

import os
import sys
import unittest

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "possession"))

import scenarios  # noqa: E402

H, A, NOBODY = 0, 1, -1


def owner_of(codes, **overrides):
    """Run the hand-over state machine on bare per-frame control codes."""
    import control
    ctrl = np.array(codes, np.int8)
    p = dict(control.PARAMS, **overrides)
    return control.owner_path(ctrl, ctrl, [], np.zeros(len(ctrl), bool), p).tolist()


class Scenarios(unittest.TestCase):
    """Every football situation in tests/scenarios.py gives the expected segments."""

    def test_scenarios(self):
        for build in scenarios.SCENARIOS:
            with self.subTest(build.__name__):
                ok, reason, expected, got = scenarios.run(build)
                self.assertTrue(ok, "{}\n  expected {}\n  produced {}".format(
                    reason, scenarios.show(expected), scenarios.show(got)))


class HandOverCount(unittest.TestCase):
    """The 1.0 s rule, frame-exact."""

    def test_24_frames_of_opponent_control_is_not_a_switch(self):
        self.assertEqual(owner_of([H] * 50 + [A] * 24 + [H] * 50), [H] * 124)

    def test_25_frames_switches_and_is_backdated_to_the_first_of_them(self):
        self.assertEqual(owner_of([H] * 50 + [A] * 25 + [NOBODY] * 10),
                         [H] * 50 + [A] * 35)

    def test_original_team_winning_it_back_resets_the_count(self):
        codes = [H] * 50 + [A] * 15 + [H] * 5 + [A] * 15 + [H] * 50
        self.assertEqual(owner_of(codes), [H] * len(codes))

    def test_nobody_frames_pause_the_count(self):
        codes = [H] * 50 + [A] * 15 + [NOBODY] * 30 + [A] * 15 + [NOBODY] * 10
        self.assertEqual(owner_of(codes), [H] * 50 + [A] * 70)

    def test_frames_before_the_first_control_go_to_the_first_team_seen(self):
        self.assertEqual(owner_of([NOBODY] * 10 + [A] * 40), [A] * 50)

    def test_threshold_is_a_parameter(self):
        self.assertEqual(owner_of([H] * 10 + [A] * 10 + [H] * 10, switch_frames=10),
                         [H] * 10 + [A] * 10 + [H] * 10)


class Labels(unittest.TestCase):

    def test_ungraded_frames_are_x_and_dead_frames_are_d(self):
        import control
        scene, _ = scenarios.hold_keeps_possession()
        ball, px, py, team, dead = scene.arrays()
        graded = np.ones(len(ball), bool)
        graded[:10] = False
        dead[20:30] = True
        labels = control.decide(ball, px, py, team, dead=dead, graded=graded)
        self.assertEqual(set(labels[:10]), {"X"})
        self.assertEqual(set(labels[20:30]), {"D"})
        self.assertEqual(set(labels[10:20]) | set(labels[30:]), {"H"})


if __name__ == "__main__":
    unittest.main()
