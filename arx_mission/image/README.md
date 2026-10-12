# Reference signs

`left.png` and `right.png` are the real robot's, cut 2026-10-07 from
`/camera/image_compensated` with the camera on `ae=false exposure_us=8000
gain=2.0`. `construction.png` and `parking.png` are still the simulator's.
The simulator's four are kept whole in `../image_sim/`, with the notes that
were written for them; everything below this paragraph is about those.

The real sign is not the shape the simulator renders. Here it is a blue disc
on a white board, where the simulator draws a board that is blue all over, so
the old pair could never have matched whatever the lighting. detect_sign runs
SIFT over the whole frame with no colour step before it, so the change of
shape costs nothing but the references.

Cut to the blue disc and a little white, and deliberately not down to the
black post below it: both arrows carry the same post, so its descriptors
would be shared between the two references and dilute the only part that
tells them apart.

    left.png    blue disc 55 px across in frame,  29 descriptors
    right.png   blue disc 91 px across in frame,  32 descriptors

Thinner than the simulator's 56 and 59, and that is the sign itself rather
than the camera: a crisp two-colour print gives SIFT corners at the arrow and
very little else, where the renderer's surface noise gave it more. It does not
cost anything in the comparison, because the published score is inliers over
the reference's own descriptor count.

Measured with `notes/sign_probe.py --image-dir`, at the distance each was cut
from:

    in front of the left sign    left 0.29-0.48   right 0.05-0.15   all LEFT
    in front of the right sign   left 0.13-0.29   right 0.42-0.64   all RIGHT

Still to do: grade both on frames they were *not* cut from, which is the test
that matters and the one the section below explains. The margin in front of
the right sign is 2.3x where the left sign's is 3.5x - the two arrows share a
disc, a white surround and a mirrored glyph, so they have a great deal in
common for SIFT to agree on.

---

# Below: the simulator's set, and how it was arrived at

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

## construction.png

Cut the same way, from 0.28 m west of the board. It matches far more
strongly than the arrows do - 10 to 33 inliers at every distance from 0.22
to 0.45 m and from either side, against the arrows' 4 to 17 - because it is
a bold red triangle on white rather than a thin white glyph on navy.

Cross-checked the same way, grading each candidate on the frames it was not
cut from:

  cut from     total inliers on the other five frames
  west 0.28 m                                      95
  east 0.28 m                                      88
  east 0.22 m                                      86
  west 0.35 m                                      66

The board is textured identically on both broad faces, so east and west
views are the same size at the same distance and either will do.

## parking.png

Cut the same way, from 0.28 m west of the board, and graded the same way -
each candidate scored on the frames it was not cut from:

  cut from     total inliers on the other three frames
  0.28 m                                            24
  0.35 m                                            23
  0.22 m                                            16
  0.17 m                                            14

0.28 m by a nose over 0.35, and the two are close enough that the choice
hardly matters. The blue board carries far less for SIFT to hold on to than
the construction triangle does - 33 descriptors against 150 - which is why
the scores here are 5 to 10 inliers where the construction board manages 10
to 33. The published score is inliers over the reference's own descriptor
count, so a thin reference is not penalised for being thin; but the margin
over min_inliers is smaller, and that is worth remembering if this sign
starts being missed.
