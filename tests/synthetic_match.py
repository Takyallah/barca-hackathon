"""Write a complete SYNTHETIC match in the documented file formats, with a known answer.

Same shape as the real bundle (199319 frames, same graded blocks) so the whole pipeline runs on
it unchanged, but the content is a scripted loop: home keep the ball and pass, away win it and
pass, away kick it out, home throw it in. Nothing here comes from the real data.

    truth = write_match(folder)     # folder/tactical/BAR-ATH_tactical_atd.{txt,xml}, _smart_tagging.xml

Always write into a temporary folder OUTSIDE the repository and delete it afterwards.
"""

import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)
from games import GAME, GAMES, asset_paths, half_video_blocks, source_frames  # noqa: E402
from scenarios import Scene  # noqa: E402

SECOND_HALF_KICKOFF = 113742

# Players in metres. The first few take part; the rest stand far from every ball path.
HOME = [(30, 20), (45, 30), (70, 0.8)] + [(8 + 8 * k, 62) for k in range(8)]
AWAY = [(60, 40), (75, 30)] + [(12 + 8 * k, 66) for k in range(9)]
OUT = (70, -1.5)        # where the ball lies while it is out of play


def one_cycle():
    """(ball positions, labels) for one loop of the script, starting with the throw-in."""
    s = Scene(HOME, AWAY)
    s.pos = OUT
    marks = []

    def step(label, action, *args, **kwargs):
        start = s.n
        action(*args, **kwargs)
        marks.append((label, start, s.n))

    step("H", s.pass_to, HOME[0])           # throw-in to home #1
    step("H", s.hold, HOME[0], 8)
    step("H", s.pass_to, HOME[1])
    step("H", s.hold, HOME[1], 6)
    step("H", s.pass_to, AWAY[0])           # intercepted: still home's until he receives it
    step("A", s.hold, AWAY[0], 8)
    step("A", s.pass_to, AWAY[1])
    step("A", s.hold, AWAY[1], 5)
    step("A", s.pass_to, OUT)               # kicked out
    step("D", s.place, OUT, 6)

    labels = np.empty(s.n, dtype="<U1")
    for label, a, b in marks:
        labels[a:b] = label
    ball = np.array(s.ball, dtype=np.float64)
    labels[(labels == "A") & (ball[:, 1] < 0)] = "D"     # dead from the moment it crosses the line
    return ball, labels


def write_match(folder, game=GAME):
    """Write the three tactical files under `folder`; return the true label per frame."""
    n = source_frames("tactical", game)
    length, width = GAMES[game]["pitch"]
    ball_cycle, label_cycle = one_cycle()

    ball = np.full((n, 2), np.nan)
    truth = np.full(n, "X", dtype="<U1")
    for half, first, last in half_video_blocks(game):
        start = first
        if half == 2:
            truth[first:SECOND_HALF_KICKOFF] = "D"
            start = SECOND_HALF_KICKOFF
        idx = np.arange(last + 1 - start) % len(ball_cycle)
        ball[start:last + 1] = ball_cycle[idx]
        truth[start:last + 1] = label_cycle[idx]

    paths = asset_paths(game, folder)
    os.makedirs(os.path.dirname(paths["atd_raw"]))

    players = ";".join("{:.6f},{:.6f},0.0".format(x / length, y / width) for x, y in HOME + AWAY)
    with open(paths["atd_raw"], "w") as handle:
        for i in range(n):
            if np.isnan(ball[i, 0]):
                b = "NaN,NaN,NaN"
            else:
                b = "{:.6f},{:.6f},0.0".format(ball[i, 0] / length, ball[i, 1] / width)
            handle.write("{}:{}:{}\n".format(i + 1, players, b))

    slots = ["T1"] * len(HOME) + ["T2"] * len(AWAY)
    with open(paths["atd_meta"], "w") as handle:
        handle.write("<main><Metadata><Teams>"
                     "<Team id=\"T1\"><Name>Synthetic Home</Name></Team>"
                     "<Team id=\"T2\"><Name>Synthetic Away</Name></Team></Teams><Players>")
        for k, team_id in enumerate(slots, 1):
            handle.write("<Player id=\"P{0}\" teamId=\"{1}\"><Name>Track {0}</Name>"
                         "<ShirtNumber>{0}</ShirtNumber></Player>".format(k, team_id))
        handle.write("</Players></Metadata></main>\n")

    with open(paths["smart_tagging"], "w") as handle:
        handle.write("<file><ALL_INSTANCES></ALL_INSTANCES></file>\n")
    return truth
