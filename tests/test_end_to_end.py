"""The whole pipeline on a SYNTHETIC full-length match with a known answer (about a minute).

    python -m unittest tests.test_end_to_end -v

Writes the synthetic files to a temporary folder outside the repository and deletes it. The real
data and the real cache are never touched.
"""

import hashlib
import os
import shutil
import sys
import tempfile
import unittest

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
for sub in ("", "possession", "evaluation", "starter", "tests"):
    sys.path.insert(0, os.path.join(ROOT, sub))

import possession_metrics as pm  # noqa: E402
import synthetic_match  # noqa: E402
from games import GAME, half_video_blocks, source_frames  # noqa: E402


def sha256(path):
    with open(path, "rb") as handle:
        return hashlib.sha256(handle.read()).hexdigest()


class EndToEnd(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        import run
        cls.tmp = tempfile.mkdtemp(prefix="mtah_synthetic_")
        cls.data = os.path.join(cls.tmp, "data")
        cls.cache = os.path.join(cls.tmp, "cache")
        cls.truth = synthetic_match.write_match(cls.data)
        cls.first = os.path.join(cls.tmp, "first.csv")
        cls.second = os.path.join(cls.tmp, "second.csv")
        cls.labels = run.write(cls.first, data_dir=cls.data, cache_dir=cls.cache)      # parses
        run.write(cls.second, data_dir=cls.data, cache_dir=cls.cache)                  # from cache
        cls.n = source_frames("tactical", GAME)
        cls.graded = np.zeros(cls.n, bool)
        for _, first, last in half_video_blocks(GAME):
            cls.graded[first:last + 1] = True

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def test_file_is_a_valid_submission(self):
        self.assertIsNone(pm.validate_submission(pm.read_label_csv(self.first), self.n))

    def test_x_exactly_on_ungraded_frames(self):
        self.assertTrue(np.array_equal(self.labels == "X", ~self.graded))

    def test_second_half_is_dead_until_its_kickoff(self):
        first = half_video_blocks(GAME)[1][1]
        self.assertEqual(set(self.labels[first:synthetic_match.SECOND_HALF_KICKOFF]), {"D"})

    def test_possessions_match_the_known_answer(self):
        result = pm.evaluate(pm.label_codes(self.truth), pm.label_codes(self.labels),
                             bootstrap_n=0)
        self.assertGreaterEqual(result["spell_f1_mean"], 0.9, result["spell_f1_mean"])
        self.assertAlmostEqual(result["n_pred_spells"], result["n_gt_spells"],
                               delta=0.05 * result["n_gt_spells"])

    def test_two_runs_write_the_same_bytes(self):
        self.assertEqual(sha256(self.first), sha256(self.second))

    def test_handover_diagnostic_on_the_known_answer(self):
        """In the synthetic match the ball is always clearly at one player's feet or in flight,
        so the rule's per-frame verdict must agree with the true owner, and every true
        possession must show more control by its own team than by the other."""
        import diagnose_handover
        ctx = diagnose_handover.load_context(data_dir=self.data, cache_dir=self.cache)
        truth = self.truth.copy()
        truth[:11185] = "X"
        truth[18685:] = "X"                       # same 5-minute window as the public truth
        grid = diagnose_handover.frame_grid(ctx, truth)
        default = [r for r in grid if r["radius"] == 2.0 and r["margin"] == 1.0
                   and r["speed_cap"] == 8.0][0]
        self.assertGreater(default["ctrl_acc"], 95.0)
        self.assertGreater(default["ctrl_pct"], 50.0)
        rows = diagnose_handover.possession_rows(ctx, truth, self.labels)
        self.assertGreaterEqual(len(rows), 10)
        self.assertTrue(all(r["own_ctrl"] > r["other_ctrl"] for r in rows))
        variants = diagnose_handover.variant_rows(ctx, truth)
        self.assertEqual(variants[0]["name"], "current")
        self.assertGreater(variants[0]["spell_f1"], 0.9)

    def test_diagnostic_finds_the_signal_that_marks_dead_ball(self):
        """In the synthetic match the ball lies outside the lines during every stoppage, so that
        signal must fire on (nearly) all dead frames and (nearly) no live ones."""
        import diagnose_deadball
        rows = diagnose_deadball.rates(
            diagnose_deadball.load_signals(data_dir=self.data, cache_dir=self.cache), self.truth)
        by_name = {row["name"]: row for row in rows}
        out = by_name["ball seen outside the lines"]
        self.assertGreater(out["dead_pct"], 70.0)
        self.assertLess(out["live_pct"], 1.0)
        self.assertGreater(out["llr"], 3.0)
        for row in rows:
            self.assertTrue(0.0 <= row["dead_pct"] <= 100.0 and 0.0 <= row["live_pct"] <= 100.0)


if __name__ == "__main__":
    unittest.main()
