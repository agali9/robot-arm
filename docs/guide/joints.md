# Joints

Per-joint breakdown for the 6-DOF arm. 

**Quick reference**


| Joint | DOF            | Plain English    | Actuator                 | Reduction            |
| ----- | -------------- | ---------------- | ------------------------ | -------------------- |
| J1    | Base yaw       | Turntable spin   | Hoverboard BLDC          | 40T/80T HTD-5M, 2:1  |
| J2    | Shoulder pitch | Arm lifts/lowers | Hoverboard BLDC          | 40T/120T HTD-5M, 3:1 |
| J3    | Elbow pitch    | Forearm folds    | 45–60 kg·cm serial servo | 15T/60T HTD-5M, 4:1  |
| J4    | Forearm roll   | Twist along arm  | 30 kg·cm serial servo    | 16T/32T HTD-3M, 2:1  |
| J5    | Wrist pitch    | Tilt the wrist   | 45 kg·cm serial servo    | Direct (1:1)         |
| J6    | Tool roll      | Spin the tool    | 20 kg·cm serial servo    | Direct (1:1)         |


**Terms used below**


| Term             | Meaning                                                              |
| ---------------- | -------------------------------------------------------------------- |
| Ø                | Diameter (mm). Ø35×Ø25 tube = 35 mm OD, 25 mm ID.                    |
| DOF              | Degree of freedom - one independent axis of motion.                  |
| BLDC             | Brushless DC motor (the hoverboard hub motors).                      |
| HTD-5M / HTD-3M  | Timing belt profile; number is tooth pitch in mm.                    |
| 40T/80T          | Pulley tooth counts; ratio sets gear reduction.                      |
| Dead axle        | Shaft fixed to the frame; the link rotates on bearings around it.    |
| Serial-bus servo | Position-controlled servo with digital feedback (Feetech STS class). |


---



## J1 — Base rotation (yaw)

**What it does:** Spins the entire arm left and right, like a lazy Susan. This is the only joint with a vertical axis at the base.

**How it works:** A steel tube (Ø35×Ø25, ~160 mm) is the rotating shaft. Two 6307 bearings (35/80/21) carry the load in the printed base tower. A hoverboard hub motor under the belt plane drives a 40T→80T HTD-5M belt for 2:1 reduction which leads to high torque at low speed, which is what you want for a heavy base.

The motor clamps onto the axle with a split PETG-CF boss (no permanent modification to the purchased motor). Torque transmits through a D-profile bore that mates the axle's single machined flat; clamp friction on the round sections handles axial retention. Cables exit the motor groove and route up through the hollow shaft with a service loop for ±270° travel.

**Travel:** ±270° firmware limit.

**Hands off to J2 at:** Ø155 top plate — 8× M5 on Ø120 bolt circle, 2× Ø5 dowel pins.

---



## J2 — Shoulder (pitch)

**What it does:** Lifts and lowers the arm in the vertical plane — the motion people picture when a robot "reaches up."

**How it works:** A **dead axle** (Ø25×Ø19 steel tube, fixed in the shoulder fork) defines the pitch axis. A printed drum with an integral 120T pulley rotates on two 6305 bearings (25/62/17). The second hoverboard motor sits in a boxed pylon (`J2_MotorPylon`) with a 40T pulley; a 700-5M-15 belt gives 3:1 reduction to the drum.

Belt tension is set with printed shims at the pylon-to-plate joint — tension loads go into the fork plate in compression, not slot friction. Motor retention uses the same D-bore + split-pinch pattern as J1 (39 mm grip, 2× M5 cross bolts).

**Travel:** −60° / +125° firmware. Wider motion is blocked by belt wrap and motor envelope before the mechanical hard stops

**Hands off to J3 at:** 84×40 mm boss at X 310 — 6× M5 + 2 dowels. Counterbalance anchor points exist on drum and fork (optional at 1.5 kg payload).

---



## J3 — Elbow (pitch)

**What it does:** Bends the forearm relative to the upper arm — the "elbow" joint.

**How it works:** 300 mm printed box beam (PETG-CF, 80×50 section) from the J2 drum to a clevis elbow. J3 reuses the J2 pattern at smaller scale: dead Ø15×Ø10 steel axle, pinch-clamped in the clevis, rotating hub on 2× 6902-2RS bearings with an integral printed 60T HTD-5M pulley.

The 45–60 kg·cm servo mounts at the **arm root** (not at the elbow) and drives 15T→60T via a 625-5M-15 belt (4:1). Proximal placement keeps ~180 g of actuator mass near J2, which cuts gravity torque on the shoulder compared to hanging a servo at the elbow.

**Travel:** ±115° firmware; ±150° clevis stops are crash backstops only (forearm contacts the beam earlier).

**Hands off to J4 at:** Boss face X 597.2 — 4× M4 + 2× Ø4 dowels. (Face moved +15 mm from the original X 582 because the 60T teeth intruded into the bolt pattern.)

---



## J4 — Forearm roll

**What it does:** Rotates the wrist assembly about the forearm's long axis — like twisting your forearm palm-up / palm-down while keeping the elbow angle fixed.

**How it works:** Offset servo at the forearm root (keeps the center bore clear for cables). 30 kg·cm serial-bus servo drives 16T→32T on HTD-3M-15 (2:1). Roll shaft is Ø25×Ø19 aluminum tube on 2× 6805-2RS bearings in a printed housing — aluminum instead of steel to save mass at a long lever arm.

Cables run through the Ø19 bore with a **torsional twist zone** (±150° rated) — same routing philosophy as the J1/J2 hollow shafts, so nothing external gets wrapped or pinched as J4 spins.

**Travel:** ±150° firmware.

**Hands off to wrist at:** Face X 760, Ø46 disc — 4× M4, 2 dowels, Ø14 cable pass-through. Module mass ~0.35 kg.

---



## J5 + J6 — Wrist (pitch + tool roll)

**What they do:** J5 tilts the tool (pitch). J6 spins the tool flange (roll). Together they orient a gripper in 3D — standard offset-wrist layout similar to UR-style arms.

**How it works:** Both joints are **direct-drive** serial servos — no wrist belts, which saves mass and length at the highest-leverage point on the arm.

- **J5 (pitch):** 45 kg·cm servo. Output is supported on both sides: horn on the servo pod and a Ø5 stub in a 625ZZ idler bearing on the base arm, so pitch bending moments don't load the servo output bearing alone.
- **J6 (roll):** 20 kg·cm servo. Tool flange mounts through a 6706-2RS thin-section bearing in the pitch block — side loads from the gripper go to the bearing, not the servo gearbox.

Gripper cable routes to a fixed connector on the wrist block (external short loop), avoiding a through-flange slip ring.

**Known offset:** Tool rotation axis sits **5.5 mm below** the J4 roll axis (Z −5.5 at the 6706). Kinematics and sim need this constant, or raise the pitch block in CAD before printing to eliminate it.

**Hands off to gripper at:** Tool flange face X 837 — 4× M4 on Ø31.5 BC, Ø4 dowel, Ø10 center bore (ISO 9409-inspired pattern). Wrist module ~0.25 kg.

**Travel:** J5 ±100°, J6 ±160°.

---



## Design themes across joints

1. **Heavy joints (J1/J2):** Repurposed hoverboard BLDC + HTD timing belts for torque density on a hobby budget.
2. **Mid joints (J3/J4):** Serial-bus servos with belt reduction — off-the-shelf actuators, printed pulleys, validated torque margins.
3. **Wrist (J5/J6):** Direct drive where torques are low but mass sensitivity is highest.
4. **Cable routing:** Hollow shafts and internal twist zones at every roll/pitch junction — no exposed harness wrapping around joints.
5. **Serviceability:** Every motor, belt, and bearing is replaceable without stripping upstream modules. Longest job: J2 belt change (~10 min, axle extraction).

