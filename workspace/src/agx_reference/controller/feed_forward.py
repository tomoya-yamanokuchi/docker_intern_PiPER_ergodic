import numpy as np

from core.agx_pinocchio import AgxPinocchio, helper

# Friction of this arm's joints, from identify_friction.py. The viscous term is
# zero because its fit does not repeat between runs; the others do.
#
# The static level is the breakaway torque measured by a ramp from standstill.
# It is 2 to 3 times the sliding level, which is why a joint asked to move slowly
# stops and then lurches. It scatters by about 30 percent between runs, because
# it depends on where the gears are meshed and every run parks the arm somewhere
# else, so these are not precise numbers and there is no point tuning the last
# digit. Joints 1-3 are the set measured at 11:40 on 2026-09-18, which cut their
# velocity reversals from 6.1/6.3/2.0 to 0.4/2.1/2.6 per second at 0.2 rad/s.
#
# The Stribeck velocity is NOT the measured physical one. Physically it is about
# 0.2 rad/s on joints 4-6, but a compensation that falls off that slowly acts as
# negative damping right across the working speed range: set there, it made those
# joints 30 to 90 percent rougher rather than smoother. It is a compensation band
# instead -- wide enough to get a standing joint moving, gone once it slides. What
# bounds the trouble it can cause is the energy it injects per crossing, which
# goes as (f_static - f_coulomb) * v_s; these values hold joints 4-6 below the
# level joints 1-3 have been running at safely.
FRICTION_VISCOUS = np.zeros(6)  # (6,) N*m*s/rad
# The sliding level is measured over 0.07-0.25 rad/s, the speeds the task runs
# at, not over the fast end of the sweeps: friction still falls with speed across
# that whole range, so calibrating it fast under-compensates where it matters.
# Joints 1-3 are still the fast-band values because that set is the one validated
# on the arm; the same criterion would put them at [0.424, 0.501, 0.512].
FRICTION_COULOMB = np.array([0.395, 0.4066, 0.5001, 0.1241, 0.1122, 0.0887])  # (6,) N*m
FRICTION_STATIC = np.array([1.1635, 1.0701, 1.0406, 0.18, 0.162, 0.143])  # (6,) N*m
STRIBECK_VELOCITY = np.array([0.005, 0.005, 0.0102, 0.02, 0.02, 0.02])  # (6,) rad/s


class FeedForward:
    """Inertial and friction feedforward torque, added on top of an impedance law.

    The impedance controllers already compensate gravity and the velocity terms
    from the measured state (nle). This class covers what they leave out:

        f(qd_cur) = f_coulomb + (f_static - f_coulomb) * exp(-(qd_cur / v_s)^2)
        tau_ff    = M(q) * qdd_des + f_viscous * qd_des + f(qd_cur) * tanh(qd_des / eps)

    The direction of the friction term comes from the desired motion, so nothing
    is applied while the target stands still, whatever the arm itself is doing.
    Its magnitude comes from the measured speed, because that is what friction
    actually depends on: a joint that has stopped needs the static level to get
    going again, and a joint already sliding needs only the sliding level. Using
    the desired speed for the magnitude too would apply the static level to a
    joint that is already moving and drive it away from its target.
    """

    def __init__(
        self,
        urdf_path: str,
        dofs: int,
        coulomb_eps: float = 0.02,
    ):
        self.dofs = int(dofs)
        if self.dofs <= 0:
            raise ValueError("dofs must be a positive integer")
        if coulomb_eps <= 0.0:
            raise ValueError("coulomb_eps must be positive")

        self.pin_model = AgxPinocchio(urdf_path)
        self.coulomb_eps = float(coulomb_eps)
        self.set_friction_params(
            f_viscous=FRICTION_VISCOUS,
            f_coulomb=FRICTION_COULOMB,
            f_static=FRICTION_STATIC,
            stribeck_velocity=STRIBECK_VELOCITY,
        )

    def set_friction_params(
        self,
        f_viscous: np.ndarray,  # (dofs,) N*m*s/rad
        f_coulomb: np.ndarray,  # (dofs,) N*m, while sliding
        f_static: np.ndarray,  # (dofs,) N*m, breakaway from standstill
        stribeck_velocity: np.ndarray,  # (dofs,) rad/s, decay of static to sliding
    ):
        """Set the friction coefficients per joint."""
        self.f_viscous = helper.as_vec(f_viscous, self.dofs, "f_viscous")
        self.f_coulomb = helper.as_vec(f_coulomb, self.dofs, "f_coulomb")
        self.f_static = helper.as_vec(f_static, self.dofs, "f_static")
        self.stribeck_velocity = helper.as_vec(stribeck_velocity, self.dofs, "stribeck_velocity")
        if np.any(self.stribeck_velocity <= 0.0):
            raise ValueError("stribeck_velocity must be positive")

    def compute_torque(
        self,
        q_cur: np.ndarray,  # (dofs,) rad
        qd_cur: np.ndarray,  # (dofs,) rad/s, measured
        qd_des: np.ndarray,  # (dofs,) rad/s
        qdd_des: np.ndarray,  # (dofs,) rad/s^2
    ) -> np.ndarray:  # (dofs,) N*m
        """Feedforward torque at the current state for the desired motion."""
        q_cur = helper.as_vec(q_cur, self.dofs, "q_cur")
        qd_cur = helper.as_vec(qd_cur, self.dofs, "qd_cur")
        qd_des = helper.as_vec(qd_des, self.dofs, "qd_des")
        qdd_des = helper.as_vec(qdd_des, self.dofs, "qdd_des")

        zero = np.zeros(self.dofs)
        # M(q) @ qdd as the difference of two rnea calls, so no extra model access.
        inertia_torque = self.pin_model.inverse_dynamics(
            q_cur, zero, qdd_des
        ) - self.pin_model.nonlinear_effects(q_cur, zero)

        stribeck = np.exp(-((qd_cur / self.stribeck_velocity) ** 2))
        f_sliding = self.f_coulomb + (self.f_static - self.f_coulomb) * stribeck
        friction_torque = self.f_viscous * qd_des + f_sliding * np.tanh(qd_des / self.coulomb_eps)
        return inertia_torque + friction_torque


def fit_friction(qd: np.ndarray, tau: np.ndarray) -> tuple[float, float]:
    """Least-squares fit of tau = f_viscous * qd + f_coulomb * sign(qd) for one joint.

    Both inputs are (N,) samples of a single joint moving at steady velocity.
    There is no intercept: with samples symmetric in the sign of qd, a constant
    error of the gravity model is even in qd and so drops out of an odd fit.
    """
    qd = np.asarray(qd, dtype=float).reshape(-1)
    tau = np.asarray(tau, dtype=float).reshape(-1)
    if qd.shape != tau.shape:
        raise ValueError("qd and tau must have the same shape")

    regressors = np.column_stack([qd, np.sign(qd)])
    coefficients, *_ = np.linalg.lstsq(regressors, tau, rcond=None)
    return float(coefficients[0]), float(coefficients[1])
