# Ball possession from tracking: a rule-based two-step pipeline

**Team Los Egipcios — More than a Hack 2027, Phase 1**

Our submission labels every frame in two steps: first decide whether the ball is in play, then
decide which team has it. Both steps are rules with football reasons behind them, not trained
models. On the public 5 minutes the file scores spellF1 0.5548 and macroF1 0.8780, against 0.3256
and 0.6207 for `baseline_smooth50`. With 17 possessions in that window the spellF1 gap is inside
the noise band, so we treat it as a sign that the pipeline works, not as a measurement of how
much better it is.

## 1. Approach, and why

**Why rules and not a model.** The only ground truth is 5 minutes: 17 possessions and 4
stoppages. That is too little to train on or to tune weights against, and the organisers show
that even one smoothing window tuned on it picks the wrong value. So every threshold is a round
number with a football reason, and we used the public window to check correctness and to see
which signals carry information, not to fit them.

**Why two steps.** The ranking metric matches whole possessions by temporal overlap, so it
rewards the right number of possessions with sharp boundaries. A dead ball always ends a
possession, and a missed stoppage merges two possessions into one. Getting the stoppages right
is therefore half the task, and it needs different evidence from the question of who has the
ball.

**Why not majority-vote smoothing.** Smoothing removes flicker by blurring the moment possession
changes, and the metric punishes a one-second offset heavily. Our rule waits for evidence that
the other team controls the ball and then moves the switch back to where that control began, so
flicker is removed without shifting the boundary.

### Step 1: is the ball in play? (`possession/deadball.py`)

Each frame gets a score from the signals below (positive points to dead ball). A two-state
Viterbi decode with a switching cost of 25 turns the scores into clean stretches, separately for
each half. The second half is dead until its kick-off.

| signal | score per frame |
|---|---|
| ball seen outside the lines (0.3 m margin, 3 frames or more) | +3.0 |
| waiting for a restart after the ball went out | +1.5 |
| players slower than the match median speed | up to +2.0 |
| players faster than the match median speed | down to −1.5 |
| ball untracked for 10 s or more | +1.5 |
| inside a Smart Tagging set-piece tag, before the kick | +1.5 |
| the 2 s after a set-piece kick | −4.0 |
| inside an open-play Smart Tagging phase | −1.0 |
| ball moving faster than 3 m/s inside the pitch | −1.0 |
| Smart Tagging goal, until the kick-off that follows | +10.0 |

Details that matter:

- **Player speed** is the mean speed of all tracked players, averaged over about one second, and
  compared with the match's own median. Using a quantile of the match instead of a fixed speed
  keeps the rule independent of how fast a particular match is played.
- **Waiting for a restart** starts when the ball is seen out and ends at a kick (a ball faster
  than 6 m/s after 0.6 s of being still or unseen), or when the ball is seen 8 m or more inside
  every line, or after 30 s. Restarts are taken at the edge of the pitch, so a ball deep infield
  means play is back on. A fast ball alone does not end the wait, because balls are thrown and
  carried around during stoppages.
- **Open-play phases** are Progression, Build Up, Attacking and Defensive Transition, Long Ball
  and their "defending" mirrors. They only happen with the ball in play.

### Step 2: which team has the ball? (`possession/control.py`)

1. A team is **near the ball** on a frame when its closest player is within 2 m of the ball and
   at least 1 m closer than the nearest opponent. Otherwise the frame belongs to nobody: ball in
   the air, contested, or not tracked.
2. A near frame is a **control frame** when the ball is also moving at 8 m/s or less, the fastest
   a player can carry it. A pass rolling past an opponent is near him but not controlled.
3. **Possession stays with the last team** until the other team shows control in one of two ways:
   - 25 control frames (1.0 s), added up across all its players. Frames that belong to nobody
     pause the count; the original team controlling the ball again resets it.
   - a completed pass: two touches by the other team, at least 5 m apart, with no control by the
     original team in between. A touch is a spell of 3 frames or more near the ball in which the
     ball was held or its velocity changed by 4 m/s or more.
4. **The switch is backdated** to the first counted frame, or to the first touch.
5. **After a dead ball** the first team near the ball has possession from the first live frame,
   with no one-second wait.

Graded frames are always `H`, `A` or `D`. We write `X` only on ungraded frames: on a graded frame
the ranking metric replaces `X` with our previous label and the tie-break metric counts it as a
miss, so abstaining can never score better than writing the label.

## 2. Assets

| asset | used for |
|---|---|
| tactical tracking (ATD), players and ball | everything in both steps |
| ATD metadata | team of each track slot |
| Smart Tagging | set pieces, goals and open-play phases, as evidence in step 1 |

**Not used:** the tactical video, the homography (only needed for video work), and the broadcast
video with its tracking and homography (not synchronised to the tactical timeline).

**Tried and dropped:**

- **A ball-state channel.** We checked the channel names in the ATD metadata: the ball has only
  x, y and speed, so dead ball and ownership have to be inferred.
- **"Ball stationary" as dead-ball evidence.** It is on for 6.5% of true dead frames and 6.3% of
  true live frames, so it carries nothing.
- **Three seconds to change possession**, our first idea. The public truth contains possessions
  of 0.8, 1.3 and 2.8 s; a 3 s rule cannot produce them and merges their neighbours.
- **A softer reset**, where control by the original team cancels the challenger's count frame for
  frame. It produced identical output over the whole match, so we removed it.
- **Other thresholds.** 0.6 s fragments possessions and lowers team accuracy; 1.6 s merges them;
  a 3 m radius and a 12 m/s speed cap change the counts without improving accuracy; removing the
  pass rule lowers team accuracy and merges possessions.
- **The Smart Tagging team of a set piece** as the restart team. Every throw-in is tagged with no
  team, so we use the first team near the ball instead.
- **Interpolating short ball gaps.** Implemented for gaps up to 0.5 s, but it has no effect here:
  almost every gap in this recording is longer than 3 s.

## 3. Missing and noisy tracking

**The ball is missing on 41% of graded frames, and the gaps are long.** Gaps of 10 s or more
cover 35% of graded frames (80 gaps, median 21 s). We checked what they are without any ground
truth: inside those gaps the players are slower than the match median on 87% of frames, against
28% when the ball is tracked, and an open-play tag is active on 5% of frames, against 55%. They
are stoppages with the ball off the pitch, which is why a long gap counts as dead-ball evidence.
The public window contains only one of the 80.

**Shorter gaps** are neutral evidence. Possession stays with the last team through them. If the
other team has control when the ball reappears, the switch is placed in the middle of the gap,
not at the frame where the ball comes back.

**Track slots are tracklets, not players.** We only ever use the team of a slot, never its
identity, so fragmented or duplicated tracks do no harm as long as the team is right.

**Jitter.** Ball speed is measured over a 5-frame span, which keeps position noise from looking
like movement.

## 4. Possession rules, case by case

| situation | what we do |
|---|---|
| loose ball, ball in the air | nobody is near it; possession stays with the last team |
| contested ball (both teams within the 1 m margin) | stays with the last team |
| single touch or deflection by an opponent | no change: under 1.0 s of control and no completed pass |
| pass rolling past opponents | no change: too fast to be control, and no velocity change so not a touch |
| opponent wins the ball and keeps it | switch after 1.0 s of control, backdated to its first frame |
| opponents play it first time between themselves | switch at the first touch (completed-pass rule) |
| original team wins it back before 1.0 s | no change; the opponent's count is reset |
| dead ball | `D`; no possession carries across it |
| restart | the first team near the ball has it from the first live frame |

## 5. Results and how we chose

Public 5 minutes (17 possessions, 4 stoppages), from `evaluation/run_evaluation.py --bootstrap 1000`:

| | spellF1 (5–95%) | macroF1 (5–95%) | dead-ball F1 | possessions |
|---|---|---|---|---|
| ours | 0.5548 (0.37–0.72) | 0.8780 (0.74–0.92) | 0.880 | 14 |
| `baseline_smooth50` | 0.3256 (0.21–0.45) | 0.6207 (0.46–0.74) | 0.323 | 26 |

Over the whole match the file has 249 possessions of mean 12.6 s, and 50.5% of graded frames are
dead ball.

We did not pick the version with the best public spellF1. An earlier version scored 0.5806 there
but labelled far less of the long ball gaps as dead. We kept the later one because the
no-ground-truth check above says those gaps are stoppages, and the public window, with one long
gap out of 80, cannot measure that.

## 6. Limitations

- **Stoppage edges.** In the public window our stoppages start 1.4–3.3 s late or 1.6 s early,
  because players only slow down after the whistle, and one ends 4 s late because the restart
  was not seen as a kick.
- **False stoppages.** Slow build-up play can look like a stoppage: there is one false 2.6 s
  dead ball inside a 42 s possession in the public window.
- **Short possessions.** Five of the 17 public possessions, all under 3.7 s, show no control
  frames at all for the team that owns them. No distance or time threshold can find them.
- **Hand-over timing.** About 9% of control frames name a different team from the truth at every
  radius and margin we tried. These sit next to true hand-overs and go both ways: sometimes the
  truth switches before control changes, sometimes after. We found one case that looks like a
  tracking identity error and no consistent offset to correct.
- **Weights are priors.** The scores in step 1 are round numbers. Their direction was checked
  against 4 stoppages, which is not enough to trust their exact size.
- **One match.** Speeds are compared with the match's own median, but distances and times are
  fixed values that have only seen this match.
- **A hard-coded frame.** The second-half kick-off frame is the one published in the starter
  code; the ATD metadata carries half start parameters that we did not use.
- **Reproduction** was verified on one machine: two runs, one parsing the tracking from scratch
  and one from the cache, wrote identical bytes.

**With more time:** fit the step 1 weights without labels (an HMM trained by EM on the whole
match); detect touches as events, which the short possessions suggest is closer to how the truth
defines ownership; use the video to find the ball when the tracking loses it; and take the
restart team from the geometry of the set piece.

## 7. Reproduction

Python 3.14.4 with the versions in `requirements-lock.txt` (numpy 2.5.4, pandas 3.1.1).

```bash
python -m venv .venv
.venv\Scripts\activate                 # Linux / macOS: source .venv/bin/activate
pip install -r requirements-lock.txt
# unpack the data bundle into data/ as described in DATA.md
python possession/run.py               # writes evaluation/results/Los_Egipcios.csv
python evaluation/validate_submission.py evaluation/results/Los_Egipcios.csv
```

The pipeline has no randomness. `run.py` prints the SHA-256 of the file it wrote; ours is
`ff47d8f9172af795721936c54a51cf1c221db2d01b7f274a538ed27f114e60ae`. The file is written with `\n`
line ends on every platform.

Tests need no match data: `python -m unittest discover -s tests -v` runs 29 tests on synthetic
data, including 15 hand-built football situations (`python tests/scenarios.py` prints them) and
a synthetic full-length match scored with the official scorer against its own known answer.

## 8. Use of an AI coding assistant

The code was written with an AI coding assistant (Claude Code). To respect the data terms, the
match data was never opened by it or shown to it: it worked from the documentation and tested the
code on synthetic data. The pipeline and the diagnostics were run on the real data by a team
member, and the assistant saw only the aggregate numbers those scripts print (scores, counts,
percentages, timings and our own predicted segments). The one exception is
`possession/atd_channels.py`, which the assistant ran itself: it prints the names of the channels
and metadata fields and the number of fields per line, and no values. The labels come from the
code in this repository; nothing was labelled or corrected by hand.
