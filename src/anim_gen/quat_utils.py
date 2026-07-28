import math
from typing import Any

import numpy as np
from pxr import Gf


def make_quat_from_euler(euler_angles: np.ndarray[Any, Any]) -> Gf.Quatf:
    rx, ry, rz = euler_angles
    qd = (
        Gf.Rotation(Gf.Vec3d(1, 0, 0), rx) * Gf.Rotation(Gf.Vec3d(0, 1, 0), ry) * Gf.Rotation(Gf.Vec3d(0, 0, 1), rz)
    ).GetQuat()
    i = qd.imaginary
    q = Gf.Quatf(qd.real, i[0], i[1], i[2])
    q.Normalize()
    return q


def make_euler_from_quat(q: Gf.Quatf) -> np.ndarray[Any, Any]:
    w, x, y, z = q.GetReal(), q.GetImaginary()[0], q.GetImaginary()[1], q.GetImaginary()[2]

    sqw = w * w
    sqx = x * x
    sqy = y * y
    sqz = z * z
    unit = sqx + sqy + sqz + sqw
    test = x * y + z * w

    if test > 0.499 * unit:
        heading = 2 * math.atan2(x, w)
        attitude = math.pi / 2
        bank = 0.0
        return np.degrees(np.array([bank, heading, attitude]))

    if test < -0.499 * unit:
        heading = -2 * math.atan2(x, w)
        attitude = -math.pi / 2
        bank = 0.0
        return np.degrees(np.array([bank, heading, attitude]))

    # for some reason, introduces slight errors for heading and bank, attitude seems to be correct
    heading = math.atan2(2 * y * w - 2 * x * z, sqx - sqy - sqz + sqw)
    attitude = math.asin(2 * test / unit)
    bank = math.atan2(2 * x * w - 2 * y * z, -sqx + sqy - sqz + sqw)

    heading = math.degrees(heading)
    attitude = math.degrees(attitude)
    bank = math.degrees(bank)

    return np.array([bank, heading, attitude])
