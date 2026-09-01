"""Tests for the REAL hardware pieces: wire protocols, URDF FK, and the real factory."""

from __future__ import annotations

import struct
import sys
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve()
sys.path.insert(0, str(_HERE.parents[1]))          # deploy/
_PROJECT = _HERE.parents[2]

from robotarm_deploy import contract as C  # noqa: E402
from robotarm_deploy.hardware import motors as M  # noqa: E402
from robotarm_deploy.hardware.real import HardwareConfig, make_real_hardware  # noqa: E402
from robotarm_deploy.kinematics import UrdfKinematicsProvider  # noqa: E402

_URDF = str(_PROJECT / "urdf" / "robot_arm.urdf")
_BASE = str(_PROJECT / "configs" / "base_transform.json")


# --- VESC / Feetech wire protocols ---------------------------------------------------
def test_vesc_set_pos_frame():
    arb, data = M.vesc_set_pos_frame(vesc_id=2, position_deg=90.0)
    assert arb == (2 | (M.VESC_CAN_PACKET_SET_POS << 8))
    assert struct.unpack(">i", data)[0] == int(round(90.0 * 1e6))


def test_feetech_write_and_read_roundtrip():
    pkt = M.feetech_write_pos_packet(servo_id=5, ticks=2500)
    assert pkt[:2] == b"\xff\xff" and pkt[2] == 5 and pkt[4] == M.STS_INSTR_WRITE
    assert pkt[5] == M.STS_ADDR_GOAL_POSITION
    assert pkt[6] | (pkt[7] << 8) == 2500                     # little-endian ticks
    # checksum valid
    body = pkt[2:-1]
    assert pkt[-1] == (~sum(body)) & 0xFF
    # read request + parse a synthetic response
    req = M.feetech_read_pos_packet(servo_id=5)
    assert req[4] == M.STS_INSTR_READ and req[5] == M.STS_ADDR_PRESENT_POSITION
    resp = bytes([0xFF, 0xFF, 5, 4, 0, 0xC4, 0x09, 0x00])     # 2500 = 0x09C4
    assert M.feetech_parse_pos_response(resp) == 2500


# --- URDF forward kinematics ---------------------------------------------------------
def test_fk_chain_and_finiteness():
    fk = UrdfKinematicsProvider(_URDF)
    assert [c[0] for c in fk._chain] == list(C.JOINT_NAMES)
    for _ in range(50):
        q = np.random.default_rng().uniform(C.JOINT_LOWER * 0.5, C.JOINT_UPPER * 0.5)
        p = fk.ee_position(q)
        assert p.shape == (3,) and np.all(np.isfinite(p))


def test_fk_base_transform_flips_yz():
    base = UrdfKinematicsProvider(_URDF)
    env = UrdfKinematicsProvider.from_config(_URDF, _BASE)   # Rx(180) about x
    pb = base.ee_position(np.zeros(6))
    pe = env.ee_position(np.zeros(6))
    assert abs(pe[0] - pb[0]) < 1e-4          # x preserved
    assert abs(pe[1] + pb[1]) < 1e-4          # y flipped
    assert abs(pe[2] + pb[2]) < 1e-4          # z flipped


# --- real hardware factory (must build WITHOUT opening buses / touching hardware) ----
def test_real_factory_builds_and_is_safe_by_default():
    backend, motors, encoders, cal, monitor = make_real_hardware(HardwareConfig(), dry_run=True)
    # nothing opened; safe by default
    assert motors.dry_run is True
    assert monitor.output_enabled is False
    # connect() must not energize motors or open buses
    backend.connect()
    assert all(not m.enabled for m in motors._motors.values())
    # sending targets in dry-run must NOT require a real bus (logged, not transmitted)
    backend.send_joint_targets(np.zeros(C.NUM_JOINTS, dtype=np.float32))   # no exception
    backend.disconnect()


def test_real_encoder_read_without_bus_raises_cleanly():
    _, _, encoders, _, _ = make_real_hardware(HardwareConfig(), dry_run=True)
    # reading a real encoder before its bus is open should raise a clear error, not crash
    try:
        encoders.read_joint_positions()
        raised = False
    except Exception:
        raised = True
    assert raised


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for fn in fns:
        fn(); print(f"  {fn.__name__} PASSED")
    print(f"hardware-real: all {len(fns)} tests PASSED")
