# Sequential ergodic exploration

The final implementation of ergodic exploration (E2T2, tensor-train) for sequential contact tasks, extended with an expanding target and a forgetting statistic.

## Task

**Sequential task**:
A task whose later stage is physically impossible until an earlier stage has succeeded, such as pushing a block only through its slot after insertion.

**Peg-slide**:
The 2-D sequential task: insert the peg tip into the slot of a block from above, then push the block to the goal.

**Clearance**:
The play the peg tip has in the slot on either side of its centre.
_Avoid_: tolerance, gap

**Offset**:
How far the block starts from where it stood when the task was taught. It is drawn once per sequence.
_Avoid_: displacement, perturbation, shift

**Taught world**:
The world at offset 0, where the demonstrations were made.

**Dead end**:
A region where the tip can never progress while the barrier stands: beside the block below its top, or against the barrier.
_Avoid_: trap, stuck region

**Slip-out**:
A failed trial in which the tip was inserted (the barrier was removed) but the goal was not reached.
_Avoid_: drop-out, loss of contact

**Never inserted**:
A failed trial in which the barrier still stands at the end. Together with slip-out, it accounts for every failure.

## Demonstration

**Master**:
The one taught motion, resampled at equal arc length, against which progress is measured.
_Avoid_: reference trajectory, nominal path

**Datapoint**:
One taught state, labelled with the phase of its nearest master sample. The target is built from the datapoints.
_Avoid_: sample, demo point

**Phase**:
How far along the master the task has come, from 0 at its start to 1 at its end.
_Avoid_: progress, stage index

**Front**:
The phase up to which the expanding target holds datapoints: the current phase plus one resolvable width.

## Method

**Target**:
The distribution the law tries to match, as coefficients.
_Avoid_: reference distribution, goal distribution

**Expanding target**:
A target holding only the datapoints up to the front, so it grows as the phase rises and shrinks if it falls.

**Statistic**:
The record of where the robot has been, as coefficients.
_Avoid_: coverage, history

**Forgetting horizon**:
The half-life with which the statistic forgets: the time of one pass along the master at full speed.
_Avoid_: memory length, window

## Experiment

**Trial**:
One attempt, from the master's start until the goal or the step limit.
_Avoid_: run, episode, attempt

**Sequence**:
Consecutive trials in one world at one offset. Only the state and the block are reset between trials; the statistic is handed on.
_Avoid_: session, series

**Success**:
A trial that reaches the goal within the step limit. A trial that ends in a slip-out or a dead end is a failure.
