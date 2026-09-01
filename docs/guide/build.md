# Build Guide

What to buy, how to print, and assembly order. **This is an estimate and a build plan**; not a finalized shopping list. Prices are commodity ranges (AliExpress low / name-brand high).

**Build status:** J1 and J2 are assembled and moving on the physical arm. Some printed parts have visible quality issues (layer lines, minor dimensional slop) but nothing that affects structure or joint function — they were kept rather than reprinted. **J3 and above exist in CAD only** — servo choices, pulley sizes, and wrist geometry could still change before those modules are printed and tested.

## Bill of materials

### Hoverboard (~$80–150)


| Qty | Item            | What you get                                  |
| --- | --------------- | --------------------------------------------- |
| 1   | Used hoverboard | 2× hub motors (J1, J2), 36 V battery, charger |


One board gives you both shoulder/base motors plus the power source — no separate battery pack to buy. Caliper each motor axle (Ø, flat depth/width, stub length) before printing any motor-mount bores.

### Bearings (~$35–50)


| Qty | Item     | Spec                 | Joint                |
| --- | -------- | -------------------- | -------------------- |
| 2   | 6307-2RS | 35/80/21             | J1                   |
| 2   | 6305-2RS | 25/62/17             | J2                   |
| 2   | 6902-2RS | 15/28/7              | J3 *(planned)*       |
| 2   | 6805-2RS | 25/37/7              | J4 *(planned)*       |
| 1   | 6706-2RS | 30/37/4 thin-section | J6 *(planned)*       |
| 1   | 625ZZ    | 5/16/5               | J5 idler *(planned)* |


### Metal (~$35–65)


| Qty | Item          | Spec                                                   | Joint                     |
| --- | ------------- | ------------------------------------------------------ | ------------------------- |
| 1   | Steel tube    | Ø35×Ø25, 160 mm (DOM/4130)                             | J1 shaft                  |
| 1   | Steel tube    | Ø25, 3 mm wall, 145 mm                                 | J2 dead axle              |
| 1   | Steel tube    | Ø15×Ø10, 80 mm                                         | J3 dead axle *(planned)*  |
| 1   | Aluminum tube | 6061 Ø25×Ø19, 60 mm (file flat for pulley grub screws) | J4 roll shaft *(planned)* |


### Belts (~$45–55)


| Qty | Spec          | Joint          |
| --- | ------------- | -------------- |
| 1   | 800-5M-15 HTD | J1             |
| 1   | 700-5M-15 HTD | J2             |
| 1   | 625-5M-15 HTD | J3 *(planned)* |
| 1   | 150-3M-15 HTD | J4 *(planned)* |


### Pulleys

**All pulleys are printed** — nothing to buy off the shelf. Includes 40T, 80T (J1), 40T, 120T (J2), 60T, 15T (J3), 32T, 16T (J4). Print a tooth coupon and verify belt mesh before committing to large pulleys like the J2 drum.

### Servos (~$115–165) *(planned — specs not locked)*

Serial-bus, position feedback, metal gears (Feetech STS class or equivalent). Final ratings may change after J3 is built and load-tested:


| Qty | Rating      | Joint          |
| --- | ----------- | -------------- |
| 1   | 45–60 kg·cm | J3 *(planned)* |
| 1   | 30 kg·cm    | J4 *(planned)* |
| 1   | 45 kg·cm    | J5 *(planned)* |
| 1   | 20 kg·cm    | J6 *(planned)* |


Buy servos before finalizing horn bores on printed pulleys — or wait until J3 CAD is frozen.

### Fasteners (~$25–40)


| Item                                 | Qty (approx.) |
| ------------------------------------ | ------------- |
| M5×30 SHCS (pinch bolts)             | 4             |
| M5 square nuts                       | 4             |
| M5 cup-point + flat-point set screws | 4             |
| M4×12 / M4×16 SHCS                   | ~30           |
| M3×8 / M3×12 SHCS                    | ~40           |
| M5 spring / Belleville washers       | 6             |
| General M4/M5 SHCS assortment        | spare stock   |


### Heat-set inserts (~$12–18)


| Size | Qty (approx.) |
| ---- | ------------- |
| M5   | ~14           |
| M4   | ~28           |
| M3   | ~24           |


Brass, standard length. Install with soldering iron at 240–250 °C before assembly.

### Dowel pins (~$6–10)


| Item     | Qty |
| -------- | --- |
| Ø5×16 mm | 4   |
| Ø4×12 mm | 6   |


### Filament (~$80–120)

~5.6 L printed volume (~4.4 kg net). Buy ~6 kg including waste and test prints:


| Material | Amount | Use                            |
| -------- | ------ | ------------------------------ |
| PETG-CF  | ~4 kg  | Structural parts (≥40% infill) |
| PETG     | ~2 kg  | Lids, covers, shims            |


### Electronics (~$100–180)


| Item                        | Notes                                                    |
| --------------------------- | -------------------------------------------------------- |
| 2× VESC                     | Hoverboard motor control over CAN (J1, J2)               |
| Battery + charger           | From the hoverboard — reuse as-is                        |
| Serial-bus servo controller | Feetech/Dynamixel-style adapter for J3–J6 *(when built)* |
| Wiring, connectors, fuses   | Route through base grommets                              |
| E-stop                      | Hardware kill for first power-on                         |


### Miscellaneous (~$25–40)

Loctite 638 (bearing retention) · rubber bumpers M3/M5 ×4 · cable grommets Ø10/14/18/25 · 8-pin tool connector · zip ties/clips · printed shim set (0.2/0.5/1.0 mm for J2 belt tension)

### Tools (not in part cost, but required)

Hex keys 2 / 2.5 / 3 / 4 mm · torque driver · calipers · reamer or scraper for bearing pockets · soldering iron (inserts) · 3D printer with ≥450×230 mm bed (or split strategy for Base_Frame)

### Cost summary

Rough totals if you built the full arm as currently modeled in CAD. J3–J6 line items are **planned**, not validated on hardware.


| Category                                                   | Low       | High      |
| ---------------------------------------------------------- | --------- | --------- |
| Hoverboard (motors + battery + charger)                    | $80       | $150      |
| Bearings + metal + belts                                   | $115      | $175      |
| Servos *(planned)*                                         | $115      | $165      |
| Fasteners, inserts, dowels, misc                           | $45       | $70       |
| Filament                                                   | $80       | $120      |
| **Mechanical subtotal**                                    | **~$435** | **~$680** |
| Electronics (VESC, wiring, E-stop; battery included above) | $100      | $180      |
| **Full prototype estimate**                                | **~$535** | **~$860** |


Biggest line items: hoverboard, servos, and filament. Only J1/J2 costs are proven on a real build so far.

## Printing

- **Material:** PETG-CF, ≥40% infill, 3+ perimeters on structural parts. Plain PETG is fine for lids, covers, and all pulleys.
- **Pulleys:** every timing pulley is printed (40T, 80T, 120T, 60T, 15T, 32T, 16T). Print teeth vertically for best belt mesh. Run a tooth coupon before the full J2 drum.
- **Bearing pockets:** print −0.15 mm, Loctite 638 on outer races.
- **Heat-set inserts:** 240–250 °C, check square afterward.
- **Print order:** base → shoulder → upper arm → forearm → wrist. J3+ parts are CAD-only until those modules are finalized.
- **Largest part:** Base_Frame (~3 L, ~30 h). Confirm bed fit (450×230 mm may need diagonal placement).


| Part                     | Notes                                         |
