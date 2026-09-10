---
name: paper-to-spec
description: Turn an academic paper or a section of one into an implementable spec with explicit notation, dimensions, and a verification plan, before any code is written. Use this whenever the user points at a paper, an arXiv link, a PDF, or an equation they want implemented — including ergodic control, optimal control, or kinematics papers. Use it even if the user just says "implement equation 7" or "can you code this method up".
disable-model-invocation: true
context: fork
argument-hint: "[paper path or arXiv id] [section or equations]"
---

Read $ARGUMENTS and write a spec to `notes/<short-name>-spec.md`. Write no implementation code in this turn.

Work in this order:

## 1. Notation table

Every symbol used in the equations you are asked to implement. Columns: symbol, meaning, dimension/shape, units, where it comes from (input, state, hyperparameter, derived). If the paper reuses a symbol with two meanings, flag it — this is where implementations usually go wrong.

## 2. The equations, restated

Rewrite each relevant equation in the paper's own notation, then again in the code's notation. State the index conventions explicitly (0- or 1-based, row- or column-major, which index runs over time vs. dimension).

## 3. Assumptions the paper makes and we do not satisfy

Be specific and honest. Discretisation, domain shape, actuation model, whether the paper assumes a point agent while we have a 6-DoF arm. This section is the one that saves weeks.

## 4. Algorithm

Numbered steps, each one a line. Mark which steps are per-timestep and which are precomputed. Note the cost of each step in terms of the sizes in the notation table.

## 5. Verification plan

Ordered from cheapest to most expensive. Prefer checks that do not need the robot:

- Closed-form or published value we can reproduce exactly
- Invariant that must hold (conservation, symmetry, normalisation, monotonicity of a cost)
- Finite-difference check of any analytic gradient
- Degenerate case with a known answer (single mode, uniform target, zero horizon)
- Behaviour on the paper's own toy example, if it has one

For each, say what the pass criterion is as a number, not as "looks reasonable".

## 6. Open questions

Things the paper does not pin down. Ask them; do not resolve them by guessing.

Finish by telling the user the spec is written and asking which section to implement first. Do not start implementing.
