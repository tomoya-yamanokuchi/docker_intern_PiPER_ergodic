---
name: numeric-check
description: Write an independent offline check for a numerical function so correctness is proven by a command rather than by inspection. Use this whenever new kinematics, Jacobian, gradient, cost, or controller code has been written, whenever the user says "does this work" about anything numerical, and before any code is considered done. Applies to FK, IK, Jacobians, ergodic metrics, and optimisation steps.
argument-hint: "[function or module to check]"
---

Write a check for $ARGUMENTS in `tests/`. The check must not reuse the implementation it is verifying — an assertion computed by the code under test proves nothing.

Pick the strongest applicable check, in this order:

1. **Closed form.** A configuration where the answer is known analytically. Assert to a stated tolerance.
2. **Round trip.** Compose the function with its inverse and assert identity. For IK: `fk(ik(T)) ≈ T` over random reachable poses, with a documented pose tolerance in metres and radians. Assert on the pose, not on the joint angles — IK has multiple valid solutions and comparing `q` will produce false failures.
3. **Finite difference.** Any analytic derivative gets checked against a central difference of the function it differentiates. Report max relative error and assert a threshold. Choose the step size and say why.
4. **Invariant.** Something that must hold for all inputs: rotation matrices orthonormal with determinant 1, probability densities integrating to 1, a cost decreasing monotonically along the optimiser's own steps, symmetry under a symmetry of the problem.
5. **Degenerate case.** Zero horizon, single mode, uniform target, singular configuration. State what should happen — including "raises" if that is the correct behaviour.

## Property-based testing

When the check is of the round-trip, invariant, or finite-difference kind, prefer Hypothesis over a hand-picked list of inputs. A generator explores configurations you would not have thought to try, and on failure it shrinks to the smallest input that still breaks — which is usually the near-singular case.

```python
from hypothesis import given, settings, strategies as st
import hypothesis.extra.numpy as hnp

joints = hnp.arrays(
    dtype=float, shape=(6,),
    elements=st.floats(-3.0, 3.0, allow_nan=False, allow_infinity=False),
)

@settings(max_examples=200, deadline=None)
@given(q=joints)
def test_fk_ik_round_trip(q):
    ...
```

Two rules when using it:

- Generate over the **valid domain**, not over all floats. Sample joint angles inside the arm's actual limits; a failure outside them tells you nothing.
- When Hypothesis finds a counterexample, add it as its own explicit regression test with the concrete values. The generated test proves the property; the pinned test stops that exact bug returning.

Requirements for the check you write:

- Fixed random seed, stated in the test.
- Tolerances are named constants with a comment saying where the number came from. Do not tune a tolerance until the test passes; if it fails, the implementation is the suspect.
- The test runs with no hardware attached and finishes in under a few seconds.
- Run it. Paste the actual output. If it fails, report the failure rather than adjusting the test.

Finish by stating, in one line, what this check does **not** cover.
