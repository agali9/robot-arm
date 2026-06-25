# Mechanical Overview

6-DOF collaborative arm built mostly from printed PETG-CF, steel tube axles, and off-the-shelf bearings and belts. J1 and J2 use repurposed hoverboard hub motors; J3–J6 use serial-bus servos.

## Targets

| | |
| --- | --- |
| Reach | ~600 mm to tool center point (589 mm to flange face) |
| Payload | 1.5 kg rated, 2.0 kg torque-capable |
| Mass | ~16.1 kg (CAD estimate) |
| Structure | Printed PETG-CF + purchased tubes, bearings, belts |

## Joint limits (firmware)

| Joint | Range |
| --- | --- |
| J1 (base yaw) | ±270° |
| J2 (shoulder) | −60° / +125° |
| J3 (elbow) | ±115° |
| J4 (forearm roll) | ±150° |
| J5 (wrist pitch) | ±100° |
| J6 (tool roll) | ±160° |

J2 and J3 limits are tighter than early sketches because belt wrap and neighboring parts physically block more travel. The ±150° figures on those joints are crash backstops, not the working range.

## Kinematics (home pose, mm)

World frame: Y up. Key points measured from the Fusion model:

| | Position |
| --- | --- |
| J1 axis | X 248, Z 0 — vertical |
| J2 axis | (248, 306) — along Z |
| J3 axis | (548, 306) |
| J4 roll axis | Y 306, bearing mid ~X 676 |
| J5 axis | (778, 306) |
| Tool flange face | X 837 |

Link lengths: J2→J3 = 300 mm, J3→J5 = 230 mm. Tool axis sits 5.5 mm below the J4 axis — a known offset; fix in software or raise the wrist pitch block before printing.

## Actuation

| Joints | Motor | Drive |
| --- | --- | --- |
| J1 | Hoverboard BLDC | 40T/80T HTD-5M, 2:1 |
| J2 | Hoverboard BLDC | 40T/120T HTD-5M, 3:1 |
| J3 | 45–60 kg·cm servo | 15T/60T HTD-5M, 4:1 |
| J4 | 30 kg·cm servo | 16T/32T HTD-3M, 2:1 |
| J5 | 45 kg·cm servo | Direct drive |
| J6 | 20 kg·cm servo | Direct drive, flange on bearing |

## Frozen interfaces

These bolt patterns and faces are fixed — later modules were designed against them:

- **J1 top plate:** Ø155, 8× M5 on Ø120 BC, 2× Ø5 dowels
- **J2 arm root:** 84×40 boss at X 310, 6× M5 + 2 dowels
- **J3 forearm mount:** face X 597.2, 4× M4 + 2× Ø4 dowels
- **J4 wrist mount:** face X 760, Ø46, 4× M4 + 2 dowels, Ø14 cable pass
- **Tool flange:** X 837, 4× M4 on Ø31.5 BC, Ø4 dowel, Ø10 center

## Before you spend money

1. Caliper the motor axles (Ø, flat depth, stub length) — bores are sized for Ø17 with one flat.
2. Buy the four servos, then finalize horn/coupling details.
3. Fix the 5.5 mm wrist offset in CAD if you want zero calibration constant.
4. Print a small HTD tooth test coupon before the J2 drum.
5. Run a fillet pass and check print orientations (pinch slits must cross layers).

## CAD status

All six joints modeled in Fusion (77 bodies, zero interference at last check). Physical assembly through J2 is underway — not a show-piece build; a few parts were printed with cosmetic or tolerance issues but used as-is where structure wasn't affected.
