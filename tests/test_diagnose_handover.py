"""Unit test for the wrong-team-control table in possession/diagnose_handover.py. Synthetic only.

    python -m unittest tests.test_diagnose_handover -v
"""

import os
import sys
import unittest

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "possession"))

import diagnose_handover  # noqa: E402


def swap_scene():
    """An away player keeps the ball for 12 s. A home player runs up to him, and from frame 150
    the tracker has swapped the two identities: the home-labelled track now sits on the ball and
    the away-labelled track runs off."""
    n = 300
    ball = np.tile(np.array([[30.2, 30.0]]), (n, 1))
    px = np.zeros((n, 2), np.float32)
    py = np.full((n, 2), 30.0, np.float32)
    px[:150, 1] = 30.0                                   # slot 1 (away) on the ball
    px[:100, 0] = 40.0                                   # slot 0 (home) 10 m away...
    px[100:150, 0] = np.linspace(40.0, 31.0, 50)         # ...runs up to 1 m
    px[150:, 0] = 30.0                                   # swapped: home track on the ball
    px[150:, 1] = np.linspace(31.0, 45.0, 150)           # away track runs off
    return dict(ball=ball, px=px, py=py, team=np.array([0, 1], np.int8)), np.full(n, "A")


class WrongTeamControl(unittest.TestCase):

    def test_an_identity_swap_shows_up_as_a_spell_that_follows_a_close_crossing(self):
        ctx, truth = swap_scene()
        spells = diagnose_handover.wrong_control_spells(ctx, truth)
        self.assertEqual(len(spells), 1)
        spell = spells[0]
        self.assertAlmostEqual(spell["start"], 150, delta=30)
        self.assertGreater(spell["frames"], 100)
        self.assertTrue(spell["crossed"])
        self.assertEqual(spell["relation"], "mid-possession")

    def test_no_spells_when_the_rule_and_the_truth_agree(self):
        ctx, truth = swap_scene()
        ctx["px"][150:, 0] = 45.0                        # no swap: home track stays away,
        ctx["px"][150:, 1] = 30.0                        # away track stays on the ball
        self.assertEqual(diagnose_handover.wrong_control_spells(ctx, truth), [])


if __name__ == "__main__":
    unittest.main()
