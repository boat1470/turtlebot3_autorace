# Reference signs, cut from this simulator's own camera

`left.png` and `right.png` are crops of the arrow board as
`/camera/image_compensated` renders it, taken with the robot 0.22 m from the
board and square to its face, then scaled to 118 px wide to match the shape of
the images they replace.

## Why not use the ones in turtlebot3_autorace_detect/image

They do not work here. Measured against frames of the real board at 0.17,
0.22, 0.28 and 0.35 m:

| reference | inliers, summed over four frames |
|---|---|
| `turtlebot3_autorace_detect/image/left.png` | **0** |
| `turtlebot3_autorace_detect/image/right.png` | **0** |
| this `left.png` | 15, 6, 6 and its own frame |

Ratio-passing matches were not the problem - those reached 8 or 9. Not one of
them survived RANSAC, which is what "0 inliers" means: the matches were
scattered rather than describing one rigid sign. `left.png` is also a
turquoise sign where the simulator renders a navy one, and carries 32
descriptors against `right.png`'s 56, so on a left sign the right reference
would often outscore the left one.

## Why 0.22 m and not the biggest view

Cutting the reference from each distance in turn and grading it on the frames
it was *not* cut from:

| left from | right from | correct | wrong | no call |
|---|---|---|---|---|
| 0.35 m | 0.45 m | 7 | **1** | 0 |
| **0.22 m** | **0.22 m** | 6 | **0** | 2 |

The pair that is right most often is not the one to take. Confirming the wrong
arrow ends the mission at the junction; confirming nothing leaves
mission_control's give-up timer to carry on, which costs a great deal less.
Eight scenes is too small a sample to trust one extra correct call, and not
small enough to ignore a wrong one.

## These are for the simulator

They are photographs of this renderer. The board in Daegu will be lit
differently and printed at a different size, so expect to cut a new pair on
practice day, from the robot's own camera, the same way:

    python3 notes/sign_probe.py --save runs/signs

then crop the board out of a frame where it is 40-70 px across.
`sign.image_dir` selects which set the detector loads, so both can live side
by side.
