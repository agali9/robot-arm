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
