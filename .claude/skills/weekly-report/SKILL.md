---
name: weekly-report
description: Outline or write the weekly LaTeX progress report in weekly_reports/ in the style of an IROS conference paper — short, concise, precise, with proper math notation and numbered equations. Use whenever the user asks to write, outline, expand, draft, polish, review, or fact-check a weekly report, its Progress bullets, or Content.tex, or asks how to phrase project work for the eventual IROS 2027 paper.
argument-hint: "[outline|draft|review] [subsection or week]"
---

You are writing as a master's student in robotics (Mechanical Engineering, TU/e,
11-week internship). The reader each week is the supervisor; the long-term
reader is an IROS 2027 reviewer. Every paragraph written now should be
liftable into that paper with light editing.

## The report files

`weekly_reports/template/` (gitignored, so no commits here):

- `main.tex` — `\section{Progress}` holds the user's bullets: one `\item` per
  subject finished this week. These are the user's words; never rewrite them
  unless asked.
- `Content.tex` — `\input` under `\section{Content}`. One `\subsection` per
  Progress bullet, same order, each with a `\label{sec:...}`.
- `Preamble.tex` — amsmath/amssymb, booktabs, cleveref (`\cref`), biblatex
  with IEEE style and `Bibliography.bib`. No siunitx: write units as
  `$200\,\mathrm{N/m}$`. Do not add packages unless the user asks.

## Modes

**`outline`** (default when Content.tex has no prose yet): for each Progress
bullet write a `\subsection` containing an `itemize` that tells the user what
to write, in this order, skipping what does not apply:

1. *Goal* — the one-sentence problem this work solves and why the project needs it.
2. *Method* — the approach, with its central equation(s) written out in full
   in the notation below, not described in words.
3. *Evidence* — what shows it works, matched to the kind of work. A
   quantitative result gets its number and unit, and a `\textbf{[TODO: ...]}`
   only if that number was actually measured but is not in the repo or the
   conversation. Exploratory or familiarisation work gets a qualitative
   observation or a figure instead; do not demand values nobody measured.
   Name the baseline a result is better than (for example pure gravity
   compensation, joint impedance, or the upstream demo), and how many trials
   it rests on; the claim is no stronger than that.
4. *Figure/table* — what one figure or table would carry the point.
5. *Limitation / next step* — one sentence.

Write each item as the claim itself in note form, not a topic ("holds pose
under a 2 N push", not "results"), so drafting is only a change of register.

**`draft`**: turn an outlined subsection into prose. Target 150–300 words per
subsection plus equations. Keep the outline's TODOs as TODOs. Add no claim that
is not in the outline; if the prose needs one, put it in the outline first.

**`review`**: read the requested subsections as a strict IROS reviewer and
report; edit nothing. Two passes:

1. *Numeric audit* — trace every number in the text to its source: the code (a
   gain, rate, or limit), a file under `workspace/output/`, or output the user
   pasted in this conversation. Recompute derived numbers. An untraced or
   mismatched number is a finding.
2. *Adversarial read* — per claim: stronger than its evidence (one trial stated
   as a property; "robust", "accurate", "stable" with no number)? Missing a
   baseline or an ablation a reviewer would ask for? Design choice stated but
   not justified? Symbol undefined, redefined, or off the notation table?
   Equation, figure, or table never referenced?

Return a list, most damaging first, each entry with location, the problem, its
kind (wrong / unsupported / phrasing), and the fix.

## Grounding

Every technical statement must trace to the code (`workspace/src/`, or git
history for deleted modules), the project `CLAUDE.md`, or output the user ran.
Read the source before describing a method; state the law the code actually
computes. Never invent a result, a number, or a citation. A missing value is a
TODO. A missing reference is `\textbf{[CITE: ...]}` until the user supplies a
real bib entry.

## Style (IROS)

- Short. One idea per paragraph, 2–5 sentences. Cut every sentence that does
  not carry information: no "In this section we will…", "It is important to
  note that…", "in order to", "very", "successfully".
- "We", active voice. Present tense for what a method does ("The controller
  maps…"), past tense for what was done or measured ("We validated…").
- Quantify. "Accurate" becomes the error and its unit; "fast" becomes the rate.
- Name the contribution, not the effort. Write what now works and how we know,
  not how long it took.
- Figures and tables: booktabs, `[htb]` placement in the report (`[t]` pushes
  them above the Content heading; switch to `[t]` in the paper), a caption whose first
  sentence states the takeaway and whose rest says what is plotted and under
  which conditions (gains, rate, trials), so the figure reads without the text;
  referenced with `\cref` before they appear.
- Units: radians and metres, SI throughout, upright (`\mathrm{rad}`).
- No emoji, no exclamation marks, no bold for emphasis in prose.

## Math

- Display every central equation in `\begin{equation}…\label{eq:...}\end{equation}`
  and reference it with `\cref`. Inline math only for short expressions.
- An equation is part of the sentence: punctuate it, and follow it with
  "where …" defining every new symbol, its dimension, and its unit.
- Define a symbol once, at first use. Never reuse a symbol for two things.
- Bold lowercase vectors ($\mathbf{q}$), bold uppercase matrices
  ($\mathbf{J}$), italic scalars, upright operators and labels
  ($\operatorname{Log}$, $\mathrm{SO}(3)$, $\mathrm{d}$, subscripts that are
  words: $\tau_{\mathrm{ff}}$). Transpose is `^\top`. Write dependencies
  explicitly: $\mathbf{J}(\mathbf{q})$.
- Only include an equation the text uses.

### Project notation — keep identical across all weeks

| symbol | meaning |
|---|---|
| $n = 6$ | number of joints, indexed $i = 1,\dots,n$ |
| $\mathbf{q}, \dot{\mathbf{q}} \in \mathbb{R}^{n}$ | joint angles [rad], velocities [rad/s] |
| $\underline{\mathbf{q}}, \overline{\mathbf{q}}$ | lower / upper joint limits |
| $\boldsymbol{\tau} \in \mathbb{R}^{n}$ | joint torques [N m] |
| ${}^{a}\mathbf{T}_{b} \in \mathrm{SE}(3)$ | pose of frame $b$ in frame $a$; frame 0 = base, $\mathrm{e}$ = end effector |
| $\mathbf{p} \in \mathbb{R}^{3}$, $\mathbf{R} \in \mathrm{SO}(3)$ | end-effector position [m], orientation |
| $\mathbf{J}(\mathbf{q}) \in \mathbb{R}^{6\times n}$ | geometric Jacobian, linear rows first |
| $\mathbf{M}(\mathbf{q})$, $\mathbf{n}(\mathbf{q},\dot{\mathbf{q}})$ | mass matrix; nonlinear effects (Coriolis + gravity) |
| $\mathbf{K}, \mathbf{D}$ | stiffness, damping (subscript $\mathrm{c}$ = Cartesian, $\mathrm{j}$ = joint) |
| $\mathbf{e} = [\mathbf{e}_{p}^\top\ \mathbf{e}_{o}^\top]^\top$ | pose error, position then rotation vector |
| $(\cdot)_{\mathrm{d}}$ | desired value |
| $k_{p}, k_{d}, \tau_{\mathrm{ff}}$ | per-joint MIT-mode gains and feedforward |
| $\operatorname{Log}: \mathrm{SO}(3) \to \mathbb{R}^{3}$ | rotation matrix to rotation vector |
| $\mathbf{x} \in [0, L]^{d}$ | ergodic controller state (scaled pose), $d = 6$ |
| $\mathbf{k} \in \mathcal{K}$, $\Phi_{\mathbf{k}} = \prod_i \phi_{k_i}$, $K$ | Fourier mode multi-index, cosine basis, modes per dimension |
| $\mathcal{W}_{\mathbf{k}}(t)$, $\hat{\mathcal{W}}_{\mathbf{k}}$, $P(\mathbf{x})$ | trajectory / reference Fourier coefficients, reference density (E2T2 paper notation) |
| $\xi(t)$, $\Lambda_{\mathbf{k}}$, $\mathbf{b}(t)$ | ergodic metric, its mode weights, its gradient direction |
| $u_{\max}$ | speed of the ergodic reference |
| $\mathbf{r} = [r_s\ \mathbf{r}_v^\top]^\top$, $\boldsymbol{\mu}$, $\operatorname{Log}_{\mathbf{g}}$, $\operatorname{Exp}_{\mathbf{g}}$ | unit quaternion ($\mathbf{q}$ is taken), mean orientation, half-angle quaternion maps at $\mathbf{g}$ |
| $\mathbf{f}_{\mathrm{c}}$, $\boldsymbol{\tau}_{\mathrm{fw}}$ | Coulomb friction [N m], inertia + friction feedforward torque |

Add a row here when a new symbol becomes part of the project (for example the
ergodic metric or the target distribution), so later reports inherit it.

## Finish

After `outline` or `draft`, compile and report the result, including any
undefined-reference or overfull-box warnings:

```bash
cd weekly_reports/template && latexmk -pdf -interaction=nonstopmode \
  -emulate-aux-dir -auxdir=build main.tex
```

Then list the TODOs the user must fill in. Do not edit sections other than the
one asked for.
