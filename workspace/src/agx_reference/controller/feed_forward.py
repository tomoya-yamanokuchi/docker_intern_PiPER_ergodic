import numpy as np

from core.agx_pinocchio import AgxPinocchio, helper

# Friction of this arm's joints, pooled over the two identify_friction.py runs of
# 2026-09-24 (friction_20260924_151518.npz and _174438.npz), four poses between
# them. The viscous term is zero because its fit does not repeat between runs.
#
# Those two runs also measure how well the rest repeats, which bounds how much
# any of this is worth arguing about. Between them the sliding level agreed to
# 3-5 percent on joints 1-3 and the static level to 5-21 percent -- so the less
# repeatable constant is the one that governs most of a task's time, since an arm
# running the ergodic pipeline stands still about 87 percent of it. Joint 6 is
# below the floor entirely: it swung 44 percent between runs and its values
# straddle the 0.051 N*m t_ff quantum, so treat its numbers as a placeholder.
#
# Both levels are averaged over the two directions of travel rather than pooled.
# Each joint carries a torque offset that is even in the velocity -- so not
# friction -- and on joint 2 it is 0.185 N*m against a friction of 0.53. It
# tracks that joint's modelled gravity torque at a steady 8 percent across poses
# 2.7 times apart in load, which makes it a gravity model error, not something
# this feedforward can or should absorb. Averaging the directions cancels it;
# the run's own printed median does not, and reads joint 2 as 0.558.
#
# The sliding level is the median of the friction-versus-measured-speed curve
# over 0.07-0.2 rad/s, the speed the task runs at. It is not flat: joint 3 falls
# from 0.71 N*m at 0.007 rad/s to 0.48 by 0.2, so a single constant is a choice
# of operating point, not a property.
#
# The static level is the breakaway torque measured by a ramp from standstill,
# a median over 3 repeats x 2 directions x 4 poses. It is 2 to 3 times the
# sliding level, which is why a joint asked to move slowly stops and then
# lurches. Single readings scatter enormously -- joint 3's span the whole range
# 0.70 to 1.53 N*m -- part of which is the 0.251 N*m t_ff quantum on joints 1-3
# rather than the gear mesh. A single reading says almost nothing, which is why
# these are medians and why there is no point tuning the last digit.
#
# That spread is also the answer to a question worth not asking twice. An ergodic
# run was seen holding joint 3 motionless at 1.27 N*m, above the 1.04 N*m median
# breakaway, which looked like stiction growing with time at rest. It is not:
# identify_friction.py's dwell sweep held joints 3 and 5 still for 1, 5, 20 and
# 60 s before each ramp and found no monotone trend, with the whole four-dwell
# range sitting inside the spread of three repeats at a single dwell. 1.27 N*m is
# simply an ordinary draw from the distribution above.
#
# Friction itself varies with pose, but less than it appears: joint 2 reads 0.436
# against 0.817 N*m between two poses, and most of that is the gravity offset
# above growing with load rather than friction changing. Joint 3 varies 3 percent
# across the same poses.
#
# The Stribeck velocity is NOT the measured physical one. Physically it is about
# 0.2 rad/s on joints 4-6, but a compensation that falls off that slowly acts as
# negative damping right across the working speed range: set there, it made those
# joints 30 to 90 percent rougher rather than smoother. It is a compensation band
# instead -- wide enough to get a standing joint moving, gone once it slides. What
# bounds the trouble it can cause is the energy it injects per crossing, which
# goes as (f_static - f_coulomb) * v_s; these values hold joints 4-6 below the
# level joints 1-3 have been running at safely.
# Joint 2's modelled gravity torque is about 7.8 percent low. It shows up as the
# part of the friction estimate that is even in the velocity, which cannot be
# friction: -9.0, -7.3, -8.8 and -6.3 percent of the modelled gravity torque over
# four poses spanning 0.87 to 7.27 N*m of it, in two independent runs. A ratio
# that steady over an 8-fold range of load is a mass or centre-of-mass error.
# Joint 3's is -0.7 percent and joints 4-6 have no gravity torque to speak of, so
# it is local to joint 2 and no single rescaling of the distal links explains it
# -- which is why the correction lives here rather than in the URDF, which is
# shared with the viewer, the IK and everything else.
#
# Joint 1 also carries a steady offset, about 0.088 N*m over the same four poses,
# but the model gives joint 1 zero gravity torque and is right to: its axis is
# vertical. That one is not gravity and is left alone.
GRAVITY_SCALE = np.array([1.0, 1.078, 1.0, 1.0, 1.0, 1.0])  # (6,) x the modelled gravity

FRICTION_VISCOUS = np.zeros(6)  # (6,) N*m*s/rad
FRICTION_COULOMB = np.array([0.391, 0.531, 0.498, 0.086, 0.094, 0.052])  # (6,) N*m
FRICTION_STATIC = np.array([1.059, 1.132, 1.038, 0.176, 0.196, 0.172])  # (6,) N*m
STRIBECK_VELOCITY = np.array([0.005, 0.005, 0.01, 0.01, 0.01, 0.005])  # (6,) rad/s


class FeedForward:
    """Inertial, friction and gravity-model feedforward, added on top of an impedance law.

    The impedance controllers already compensate gravity and the velocity terms
    from the measured state (nle). This class covers what they leave out:

        f(qd_cur) = f_coulomb + (f_static - f_coulomb) * exp(-(qd_cur / v_s)^2)
        tau_ff    = M(q) * qdd_des + f_viscous * qd_des + f(qd_cur) * tanh(qd_des / eps)
                    + (gravity_scale - 1) * g(q)

    The last term corrects the measured error in the model's own gravity torque
    (GRAVITY_SCALE), which the impedance law would otherwise leave standing. It
    costs nothing extra: g(q) is the nonlinear_effects call the inertia term
    already makes.

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
        self.gravity_scale = helper.as_vec(GRAVITY_SCALE, self.dofs, "gravity_scale")
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
        # At zero velocity nonlinear_effects is the gravity torque alone, and it
        # serves twice: M(q) @ qdd as the difference of two rnea calls, and the
        # gravity-model correction, so neither needs an extra model access.
        gravity = self.pin_model.nonlinear_effects(q_cur, zero)
        inertia_torque = self.pin_model.inverse_dynamics(q_cur, zero, qdd_des) - gravity
        gravity_correction = (self.gravity_scale - 1.0) * gravity

        stribeck = np.exp(-((qd_cur / self.stribeck_velocity) ** 2))
        f_sliding = self.f_coulomb + (self.f_static - self.f_coulomb) * stribeck
        friction_torque = self.f_viscous * qd_des + f_sliding * np.tanh(qd_des / self.coulomb_eps)
        return inertia_torque + friction_torque + gravity_correction
