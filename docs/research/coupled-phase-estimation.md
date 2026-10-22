# Coupled phase estimation: how orientation can count as progress only where the position allows it

Research for issue #3 of map #1. The question: which established methods estimate the
**phase** along one **master** from the tool pose alone (**sensorless**), such that a turn
of the key only counts as progress at the lock depth? Vocabulary is `final/GLOSSARY.md`.

## Short answer

None of the alignment methods fixes this on its own. DTW, ProMP phase estimation, the HMM
forward pass and contouring progress all **rank** candidate phases against each other, and
a pose off the master still lands on whichever phase ranks highest. What the literature adds,
and what `PhaseFilter` lacks, is an **absolute test of being on the master**: the predictive
likelihood checked against a progress-dependent threshold (HMM execution monitoring), or an
inside/outside test of a tube around the demonstration (canal surfaces). A phase-dependent
covariance makes that tube narrow in position where the master turns, which is the coupling
the question asks for. But only the absolute test turns it into "a turn counts only at depth".

## Why `PhaseFilter` reports false progress (checked offline)

`PhaseFilter.update` (`final/src/method/law.py:112`) is an HMM forward pass. It has one state
per master sample, an isotropic Gaussian emission `N(y; m_i, h^2 I)` in the 6-D metric
(`final/src/method/pose.py`), a symmetric arc-length transition, and **renormalisation every
step** (`log_belief - logsumexp(log_belief)`). Across the master's turn the position barely
moves (x 415.8 -> 420.3 mm, y 0.4 mm, z 1.2 mm on `task_20261021_205106`). So a position error
adds almost the same term `|p - p_turn|^2 / 2h^2` to every turn state, and the
renormalisation cancels it. Only the rotation then tells the turn states apart, so a turned
key reads as the end of the turn wherever it is.

The check runs the real `PhaseFilter` on the door task's master, with h = 2.16 mm. The key
goes along the master to a point short of the lock, then turns 0 -> 47 deg there
(scratch script, run in the container with `final/src` on the path):

```
short of depth    0 mm: phase 0.509, nearest master   0.0 mm =  0.0 h, log evidence     -0.4
short of depth   10 mm: phase 0.493, nearest master  12.8 mm =  5.9 h, log evidence    -18.4
short of depth   30 mm: phase 0.482, nearest master  31.7 mm = 14.7 h, log evidence   -108.2
```

Insertion ends at phase 0.37 on this master. A key turned 30 mm short of the lock reads
0.48, almost the 0.51 of a correct turn, while it sits 14.7 h from the master. The signal that
would reject it is the predictive log evidence `logsumexp(log_lik + log_belief)`, which falls
from -0.4 to -108. The filter computes everything needed for it and normalises it away. This
matches the arm: the ceiling poses were 14-35 mm from the master (issue #1).

## The methods

| method | needs | sensorless? | detects being off the master | coupling of orientation to position |
|---|---|---|---|---|
| DTW, online DTW | 1 reference | yes (pose only) | no: alignment is relative; the accumulated cost exists but is not tested | none beyond the metric |
| ProMP / Interaction ProMP phase estimation | several demos (Gaussian over weights) | yes (kinematic observations) | predictive density available, not used as a gate in the sources | yes, through the full weight covariance |
| Bayesian Interaction Primitives (EKF) | several demos | yes | not as a test; the phase is a state | yes, through the joint state covariance |
| HMM / HSMM forward variable | several demos for the covariances; 1 is possible with fixed widths | yes | yes, with a progress-dependent likelihood threshold (Park et al.) | yes, with a full covariance per state |
| Canal surface / generalised cylinder | several demos (examples use 3-5) | yes | yes: inside/outside the cross-section | position only, no orientation |
| TP-GMM | several demos in varied frames | needs the frames, i.e. the lock pose | no | yes, per frame |
| Riemannian GMM/GMR | several demos | yes | not by itself | yes: conditioning orientation on position, done properly on R^3 x S^3 |
| Contouring progress / reactive phase | 1 path | yes | no: projection onto the path | none |

### DTW and monotone phase alignment

DTW aligns two sequences by dynamic programming under monotonicity, continuity and slope
constraints [Sakoe & Chiba 1978]. On-line time warping does the same incrementally, in
linear time, as one series arrives [Dixon 2005]. Interaction Primitives used DTW for phase
estimation, and Campbell & Ben Amor list its costs: it is hard to run in real time in high
dimensions, and it aggregates errors when run incrementally [Campbell & Ben Amor 2017, §2,
§3.2].

- *Needs*: one reference trajectory, so the master is enough.
- *Off the master*: it has no test. DTW returns the best monotone alignment however poor it is.
- *For `PhaseFilter`*: the useful part is the monotone, slope-bounded transition, a
  left-to-right kernel in place of the symmetric `arc2` one. That stops backward jumps but
  not the false forward drift above, which advances smoothly. It also conflicts with the
  **retry**, which needs the phase to fall back (issue #1: phase back to 0.00, 10-12 times
  per run). Not a fix here.

### Phase-dependent covariance: the tube that narrows where the master turns

Canal surfaces, and their generalisation TLGC, encode a skill as a spine and a cross-section
at each point along it. The spine is the axis-wise mean of the demonstrations; the simplest
cross-section is a circle whose radius is the distance to the farthest corresponding
demonstration point [Ahmadzadeh & Chernova 2018, §4.1.1-4.1.2]. A new start is related to
the boundary by the ratio `eta = |p0 - c0| / |g0 - c0|` [§4.2]. The method encodes position
only, not orientation [§4.1].

- *Needs*: several demonstrations for the cross-sections. Here the phase-labelled
  **datapoints** can stand in for them: the spread of the datapoints labelled near master
  sample i gives a covariance `Sigma_i` for that sample, with no new demonstration.
- *Off the master*: yes. Inside or outside the cross-section is an absolute test, the very
  thing `PhaseFilter` lacks.
- *Coupling*: where the master turns, the operator held the key in the lock, so the
  position part of `Sigma_i` is tight and the rotation part wide. A key off the keyhole line
  is then outside the tube at every turn state.
- *For `PhaseFilter`*: emission `N(y; m_i, Sigma_i)` instead of `N(y; m_i, h^2 I)`. On its
  own this still renormalises. Pair it with the gate under HMM/HSMM.

### ProMP and phase estimation in learning from demonstration

A ProMP keeps a distribution over trajectories [Paraschos et al. 2013]. Its weights are
Gaussian with mean and covariance fitted by maximum likelihood over **multiple
demonstrations**, and the phase replaces absolute time [Campbell & Ben Amor 2017, §3.1].
Maeda et al. estimate the phase of a partly observed movement with Interaction ProMPs, to
recognise the action and generate the robot's trajectory before the human finishes
[Maeda et al. 2017, abstract]. Bayesian Interaction Primitives put the phase and the phase
velocity into an EKF state next to the weights, as SLAM localisation against a map
[Campbell & Ben Amor 2017, §3.2].

They also note a failure that matches the turn: "if one or more of the trajectory degrees of
freedom is static for a period of time ... the phase won't impact the predicted value - then
it is not possible to estimate the change in phase". Their fix is a constant-velocity phase
model [§3.2]. During the master's turn, position is that static DoF.

- *Needs*: several demonstrations. One master cannot fit `Sigma_w`.
- *Sensorless*: yes, the observations are kinematic.
- *Off the master*: the predictive Gaussian gives a likelihood, but the sources use it to
  condition, not to reject.
- *For `PhaseFilter`*: not a drop-in. A phase-velocity state would stop the phase racing
  ahead on rotation alone, but it brings a speed prior that a stalled key violates.

### HMM / HSMM progress

`PhaseFilter` already is the HMM forward variable,
`alpha_i,n = (sum_j alpha_j,n-1 a_j,i) N(x_n; mu_i, Sigma_i)`, normalised
[Calinon et al. 2011, eq. 4]. Calinon et al. note that this forward variable "exponentially
biases position information over transition information". The HSMM adds an explicit
duration model per state to balance the two [§II-B, §II-C]. Each state's Gaussian can carry
a full covariance with "variation and correlation information" [§II-B], which is the
phase-dependent covariance above.

For being off the model, Park et al. train an HMM on non-anomalous executions and flag an
anomaly when the log-likelihood falls below a threshold **that varies with the progress of
execution** [Park et al. 2016]. Their observations are multimodal, including force. The
gating mechanism itself needs only the likelihood.

- *Needs*: several demonstrations to learn the covariances and durations. With one master,
  use fixed widths (h, as now) or the datapoint covariances above.
- *Off the master*: yes, through the likelihood threshold.
- *For `PhaseFilter`*: the smallest change of all. Keep the forward pass. Each step compute
  the predictive log evidence `logsumexp(log_lik + log_prior)`, which is already there before
  the normalisation. Count an **advance** only while it stays above a threshold.
  Equivalently, add an explicit off-master state with a flat likelihood that holds no phase.
  The threshold can come from the model rather than be tuned: with `Sigma_i` (or `h^2 I`) in
  6-D, `d^2` is chi-square with 6 DoF, whose 99 % quantile is 16.8, so `d <= 4.1` in units of
  h or Mahalanobis. That puts the 10 mm case above (5.9 h) and the 30 mm case (14.7 h) off
  the master.

### Task-parameterised GMM

TP-GMM adapts a movement to new situations given as frames of reference, such as object or
landmark poses [Calinon 2016]. It needs demonstrations in several frames, and online it
needs the frame, here the lock's pose. The method is sensorless and the lock offset is
unknown, so it does not apply without a pose estimate of the lock.

### Orientation on the right manifold

Zeestraten et al. extend Gaussian conditioning, the Gaussian product and GMR (and with them
TP-GMM) to Riemannian manifolds, including the robot pose on `R^3 x S^3`
[Zeestraten et al. 2017, abstract, §III]. Conditioning `p(orientation | position, phase)`
is the formal version of "orientation counts only where the position allows". It needs a
joint covariance, so several demonstrations or the datapoints. The current metric
(rotation vector from R0, scaled by ROTATION_LENGTH) is a single tangent space. That is
adequate for a 50 deg turn about one axis, and it is not what is failing.

### Contouring progress and reactive phase

Model predictive contouring control adds a virtual path parameter, the arc length, to the
state [Lam, Manzie & Good 2010]. Englert & Toussaint update a phase variable from actual task
progress instead of time [Englert & Toussaint 2014, abstract]. Both project the state onto
the reference, so both have the relative-ranking problem. They show that a progress state
is standard, not how to reject off-path poses.

## What this suggests for `PhaseFilter` (for the map, not decided)

1. **Gate the advance on the predictive evidence.** This is the established mechanism
   [Park et al. 2016], it needs no new data, and the check above shows it separates the
   cases. The phase may still be reported; the **stall** simply does not reset on an
   off-master advance, so the **retry** keeps firing.
2. **Optionally, phase-dependent covariance from the datapoints** [Calinon et al. 2011;
   Ahmadzadeh & Chernova 2018], so the gate is tight in position at the turn and loose in
   rotation. It needs enough datapoints per master sample. Check that before relying on it.
3. Not recommended here: a monotone DTW transition (it breaks the retry), ProMP/BIP (several
   demonstrations, and the speed prior conflicts with stalls), TP-GMM (needs the lock frame).

## Not verified

- Maeda et al. 2017 and Englert & Toussaint 2014: the full texts were unreachable (both
  hosts refused connections). The claims rest on the abstracts. Campbell & Ben Amor's
  description of IP phase estimation via DTW is the primary statement used for IPs.
- Park et al. 2016: from the abstract on the CMU and Georgia Tech pages, not the full text.
- The chi-square threshold assumes the emission model is right. Real datapoint spread is not
  Gaussian, so the 4.1 is a starting point to check against `final/output/run_*.npz`, not a
  derived constant for the arm.
- The offline check uses a synthetic short-of-depth path built from the master. It does not
  replay an arm log.

## Sources

- H. Sakoe, S. Chiba, "Dynamic programming algorithm optimization for spoken word
  recognition", IEEE TASSP 26(1), 1978. <https://www.ece.ucsb.edu/Faculty/Rabiner/ece259/Reprints/132_dtw.pdf>
- S. Dixon, "Live tracking of musical performances using on-line time warping", DAFx 2005.
  <https://dafx.de/paper-archive/details/o-kAKlRZ1QVrnYxd28nF0g>
- A. Paraschos, C. Daniel, J. Peters, G. Neumann, "Probabilistic movement primitives",
  NeurIPS 2013. <https://papers.nips.cc/paper/5177-probabilistic-movement-primitives>
- G. Maeda, M. Ewerton, G. Neumann, R. Lioutikov, J. Peters, "Phase estimation for fast
  action recognition and trajectory generation in human-robot collaboration", IJRR
  36(13-14), 2017, doi:10.1177/0278364917693927. <https://publikationen.uni-tuebingen.de/xmlui/handle/10900/87625>
- J. Campbell, H. Ben Amor, "Bayesian Interaction Primitives: a SLAM approach to
  human-robot interaction", CoRL 2017, PMLR 78. <http://proceedings.mlr.press/v78/campbell17a/campbell17a.pdf>
- S. Calinon, A. Pistillo, D. G. Caldwell, "Encoding the time and space constraints of a
  task in explicit-duration hidden Markov model", IROS 2011. <https://calinon.ch/papers/Calinon-IROS2011.pdf>
- S. Calinon, "A tutorial on task-parameterized movement learning and retrieval",
  Intelligent Service Robotics 9(1), 2016. <https://calinon.ch/paper4018.htm>
- M. Zeestraten, I. Havoutis, J. Silverio, S. Calinon, D. G. Caldwell, "An approach for
  imitation learning on Riemannian manifolds", RA-L 2017. <https://calinon.ch/papers/Zeestraten-RAL2017.pdf>
- S. R. Ahmadzadeh, S. Chernova, "Trajectory-based skill learning using generalized
  cylinders", Frontiers in Robotics and AI, 2018. <https://pmc.ncbi.nlm.nih.gov/articles/PMC7805842/>
- D. Park, Z. Erickson, T. Bhattacharjee, C. C. Kemp, "Multimodal execution monitoring for
  anomaly detection during robot manipulation", ICRA 2016. <https://www.ri.cmu.edu/publications/multimodal-execution-monitoring-for-anomaly-detection-during-robot-manipulation>
- D. Lam, C. Manzie, M. Good, "Model predictive contouring control" (path progress as a
  virtual state); known only through search summaries, venue and text not checked.
- P. Englert, M. Toussaint, "Reactive phase and task space adaptation for robust motion
  execution", IROS 2014. <https://argmin.lis.tu-berlin.de/papers/14-englert-IROS.pdf>
