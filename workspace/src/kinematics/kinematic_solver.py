"""Forward and inverse kinematics of the PiPER peg tip (peg_tcp), from its URDF.

Hardware-free. FK and the Jacobian are Pinocchio's, through the same
AgxPinocchio wrapper the impedance controller uses. IK is closed form by Pieper
decoupling: the joint 4, 5, 6 axes meet at a point, so the wrist centre depends
only on q1..q3 and the remaining orientation only on q4..q6, which gives up to
eight exact branches instead of the one basin a seeded Newton iteration finds.
"""

import numpy as np
import pinocchio as pin

from core.agx_pinocchio import AgxPinocchio

FLANGE_FRAME = "link6"
TCP_FRAME = "peg_tcp"

# Default IK seed, rad. Not the zero pose: q = 0 sits on the joint 2 and 3
# limits and j5 = 0 is a wrist singularity, the worst possible seeds.
Q_READY = np.array([0.0, 1.70, -1.30, 0.0, 0.70, 0.0])

SINGULAR_TOL = 1e-9

# The closed form solves an idealised arm 88 um from the real one, so a pose on
# the workspace boundary or a joint limit lands just outside; the slack admits
# it and the clamp plus the final acceptance check keep the result honest.
REACH_TOL = 1e-2
LIMIT_TOL = 1e-2


def _rotation_z(angle: float) -> pin.SE3:
    return pin.SE3(pin.utils.rotate("z", angle), np.zeros(3))


def _wrap(q: np.ndarray) -> np.ndarray:
    # Every PiPER joint range fits inside [-pi, pi].
    return (q + np.pi) % (2.0 * np.pi) - np.pi


def _idealise(M: pin.SE3, angle_tol: float = 1e-3, length_tol: float = 1e-4) -> pin.SE3:
    """Snap a joint placement to the exact geometry the closed form assumes.

    The URDF writes right angles as 1.5708 and carries an 88 um x-offset on
    joint 6 that stops the wrist being exactly spherical; both are CAD export
    artefacts.
    """
    rpy = pin.rpy.matrixToRpy(M.rotation)
    right_angles = np.round(rpy / (np.pi / 2)) * (np.pi / 2)
    rpy = np.where(np.abs(rpy - right_angles) < angle_tol, right_angles, rpy)
    t = np.where(np.abs(M.translation) < length_tol, 0.0, M.translation)
    return pin.SE3(pin.rpy.rpyToMatrix(rpy), t)


class KinematicSolver:
    def __init__(self, urdf_path: str):
        self.pin_model = AgxPinocchio(urdf_path)
        model = self.pin_model.robot.model
        self.q_min = model.lowerPositionLimit.copy()
        self.q_max = model.upperPositionLimit.copy()
        # Indexed by joint number; only the closed form uses these, FK and the
        # polish always use the real geometry.
        self.ideal = [None] + [_idealise(model.jointPlacements[j]) for j in range(1, 7)]
        # Both frames hang off joint 6, so this placement is constant.
        flange, tcp = (model.frames[self.pin_model.frame_id(f)] for f in (FLANGE_FRAME, TCP_FRAME))
        self.tcp_M_flange = tcp.placement.actInv(flange.placement)
        self._set_pieper_constants()

    def forward_kinematics(self, q: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """TCP position (3,) m and rotation (3, 3) for q (6,) rad."""
        return self.pin_model.forward_kinematics(q, TCP_FRAME)

    def jacobian(self, q: np.ndarray) -> np.ndarray:  # (6, 6), LOCAL_WORLD_ALIGNED
        return self.pin_model.jacobian(q, TCP_FRAME)

    def inverse_kinematics(
        self,
        p: np.ndarray,  # (3,) m
        R: np.ndarray,  # (3, 3)
        q_init: np.ndarray = Q_READY,  # (6,) rad
        tol: tuple[float, float] = (1e-3, 1e-3),  # (m, rad)
    ) -> np.ndarray | None:
        """Joint angles (6,) rad placing the TCP at (p, R), or None if unreachable.

        Of the branches that reach the pose within tol, the one closest to q_init
        wins, so passing the previous solution keeps successive targets on the
        same elbow and wrist configuration.
        """
        target = pin.SE3(np.asarray(R, dtype=float), np.asarray(p, dtype=float))
        # Clamp rather than reject: a branch can land just outside a limit, and
        # the acceptance check throws out any that clamping actually broke.
        flange_target = target * self.tcp_M_flange
        branches = [self._clamp(q) for q in self._closed_form_branches(flange_target)]
        candidates = self._accept(target, branches, tol)
        if not candidates:
            # Near j5 = 0 the 88 um of idealisation error is amplified into
            # degrees of orientation error, and every branch misses.
            candidates = self._accept(target, [self._polish(target, q) for q in branches], tol)
        if not candidates:
            return None
        return min(candidates, key=lambda q: np.linalg.norm(q - q_init))

    def _clamp(self, q: np.ndarray) -> np.ndarray:
        return np.clip(q, self.q_min, self.q_max)

    def _set_pieper_constants(self) -> None:
        # Derived from the placements rather than a transcribed DH table, so the
        # numbers cannot drift out of sync with the URDF.
        M1, M2, M3, M4, M6 = (self.ideal[j] for j in (1, 2, 3, 4, 6))
        self.t1 = M1.translation  # base to the joint 1 origin
        self.R2 = M2.rotation
        self.a2 = M3.translation[0]  # upper arm
        self.t4 = M4.translation  # forearm vector, in plane
        self.l3 = np.linalg.norm(self.t4)
        self.d6 = np.linalg.norm(M6.translation)  # wrist centre to flange
        # Constant angles folded into the planar 2R solve: the yaw baked into
        # M3 and the direction of the offset forearm vector.
        self.q3_offset = np.arctan2(M3.rotation[1, 0], M3.rotation[0, 0])
        self.alpha4 = np.arctan2(self.t4[1], self.t4[0])
        self._check_pieper_assumptions()

    def _check_pieper_assumptions(self) -> None:
        # Without this a changed URDF would silently give confidently wrong angles.
        M1, M2, M3, M4, M5, M6 = (self.ideal[j] for j in range(1, 7))
        assumptions = (
            (not np.allclose(M1.rotation, np.eye(3)), "joint 1 placement is not axis-aligned"),
            (not np.allclose(M2.translation, 0.0), "joints 1 and 2 do not intersect"),
            (abs(M2.rotation[2, 2]) > 1e-9, "joint 2 axis is not perpendicular to joint 1"),
            (abs(M3.rotation[2, 2] - 1.0) > 1e-9, "joints 2 and 3 are not parallel"),
            (abs(M4.translation[2]) > 1e-9, "joint 4 origin is out of the joint 3 plane"),
            (not np.allclose(M5.translation, 0.0), "joints 4 and 5 do not intersect"),
            (abs(M6.translation[0]) > 1e-9, "joints 5 and 6 do not intersect"),
            (not np.allclose(M6.rotation, M5.rotation.T), "wrist rotations are not a ZYZ pair"),
        )
        bad = [reason for violated, reason in assumptions if violated]
        if bad:
            raise ValueError("URDF does not match the Pieper structure: " + "; ".join(bad))

    def _closed_form_branches(self, target: pin.SE3) -> list[np.ndarray]:
        """Up to eight: two base rotations x two elbow bends x two wrist flips."""
        # The wrist centre sits d6 back along the flange z axis.
        p_w = target.translation - self.d6 * target.rotation[:, 2]
        branches = []
        for q1 in self._solve_q1(p_w):
            for q2, q3 in self._solve_q2_q3(p_w, q1):
                for q4, q5, q6 in self._solve_wrist(target, q1, q2, q3):
                    q = _wrap(np.array([q1, q2, q3, q4, q5, q6]))
                    if np.all(q >= self.q_min - LIMIT_TOL) and np.all(q <= self.q_max + LIMIT_TOL):
                        branches.append(q)
        return branches

    def _solve_q1(self, p_w: np.ndarray) -> list[float]:
        # Joint 2 sweeps the plane normal to its axis, so q1 must bring the wrist
        # centre into it: A cos(q1) + B sin(q1) + C = 0.
        u = p_w - self.t1
        n = self.R2[:, 2]
        A = n[0] * u[0] + n[1] * u[1]
        B = n[0] * u[1] - n[1] * u[0]
        C = n[2] * u[2]
        radius = np.hypot(A, B)
        if radius < SINGULAR_TOL or abs(C) > radius * (1.0 + REACH_TOL):
            return []  # wrist centre on the joint 1 axis, or out of reach
        phi = np.arctan2(B, A)
        delta = np.arccos(np.clip(-C / radius, -1.0, 1.0))
        return [phi + delta, phi - delta]

    def _solve_q2_q3(self, p_w: np.ndarray, q1: float) -> list[tuple[float, float]]:
        # A 2R arm of lengths a2 and l3 in the joint 2 plane, except the forearm
        # vector is bent by alpha4 relative to its own link.
        w = self.R2.T @ (_rotation_z(-q1).rotation @ (p_w - self.t1))
        L = np.hypot(w[0], w[1])
        cos_elbow = (L * L - self.a2**2 - self.l3**2) / (2.0 * self.a2 * self.l3)
        if abs(cos_elbow) > 1.0 + REACH_TOL:
            return []
        elbow = np.arccos(np.clip(cos_elbow, -1.0, 1.0))
        solutions = []
        for theta3 in (elbow - self.alpha4, -elbow - self.alpha4):
            reach = np.array([self.a2, 0.0, 0.0]) + _rotation_z(theta3).rotation @ self.t4
            q2 = np.arctan2(w[1], w[0]) - np.arctan2(reach[1], reach[0])
            solutions.append((q2, theta3 - self.q3_offset))
        return solutions

    def _solve_wrist(
        self, target: pin.SE3, q1: float, q2: float, q3: float
    ) -> list[tuple[float, float, float]]:
        # Concurrent wrist axes with inverse-pair joint 5 and 6 placements make
        # the residual rotation a plain ZYZ Euler triple.
        ideal = self.ideal
        oM3 = ideal[1] * _rotation_z(q1) * ideal[2] * _rotation_z(q2) * ideal[3] * _rotation_z(q3)
        W = (oM3.rotation @ ideal[4].rotation).T @ target.rotation
        sin_q5 = np.hypot(W[0, 2], W[1, 2])
        if sin_q5 < SINGULAR_TOL:
            # Singular: only q4 + q6 matters, so pin q4 at zero.
            if W[2, 2] > 0.0:
                return [(0.0, 0.0, np.arctan2(W[1, 0], W[0, 0]))]
            return [(0.0, np.pi, -np.arctan2(-W[1, 0], -W[0, 0]))]
        q5 = np.arctan2(sin_q5, W[2, 2])
        q4 = np.arctan2(W[1, 2], W[0, 2])
        q6 = np.arctan2(W[2, 1], -W[2, 0])
        return [(q4, q5, q6), (q4 + np.pi, -q5, q6 + np.pi)]

    def _accept(
        self, target: pin.SE3, branches: list[np.ndarray], tol: tuple[float, float]
    ) -> list[np.ndarray]:
        accepted = []
        for q in branches:
            p, R = self.forward_kinematics(q)
            error = pin.log6(pin.SE3(R, p).actInv(target)).vector
            if np.linalg.norm(error[:3]) <= tol[0] and np.linalg.norm(error[3:]) <= tol[1]:
                accepted.append(q)
        return accepted

    def _polish(self, target: pin.SE3, q: np.ndarray, iters: int = 30) -> np.ndarray:
        """Damped Gauss-Newton onto the target, damping scaled by the squared residual.

        The damping vanishes with the error, which is what lets this converge in
        a few steps although the Jacobian is near-singular where it is used.
        """
        for _ in range(iters):
            p, R = self.forward_kinematics(q)
            error = np.concatenate([target.translation - p, R @ pin.log3(R.T @ target.rotation)])
            if np.linalg.norm(error) < 1e-6:
                break
            J = self.jacobian(q)
            damping = 1e-10 + 1e-2 * error.dot(error)
            q = self._clamp(q + np.linalg.solve(J.T @ J + damping * np.eye(6), J.T @ error))
        return q
