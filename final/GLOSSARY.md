# Sequential ergodic exploration

The final implementation of ergodic exploration (E2T2, tensor-train) for sequential contact tasks, extended with a sliding target and a forgetting statistic.

## Task

**Sequential task**:
A task whose later stage is physically impossible until an earlier stage has succeeded, such as pushing a block only through its slot after insertion.

**Peg-slide**:
The 2-D sequential task: insert the peg tip into the slot of a block from above, then push the block to the goal.

**Locked door**:
The 6-DoF sequential task on the arm: insert the key into the lock, turn it to unlock, then push the door open.

**Tool**:
What the arm holds and moves through the task: the peg or the key.

**Clearance**:
The play the peg tip has in the slot on either side of its centre.
_Avoid_: tolerance, gap

**Offset**:
How far the block starts from where it stood when the task was taught. It is drawn once per sequence.
_Avoid_: displacement, perturbation, shift

**Taught world**:
The world at offset 0, where the demonstrations were made.

**Tilt**:
The peg's angle from vertical. It is part of the state the law explores.
_Avoid_: orientation, rotation

**Jam**:
The peg wedged in the slot: tilted too far for its depth, so it cannot go deeper until it tilts back or withdraws. The deeper the peg, the less tilt it tolerates.
_Avoid_: stuck, wedge

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

**Exploration distribution**:
The datapoints taught for a task, one confirmed pose at a time. The target is built from it.
_Avoid_: reference distribution, cloud

**Datapoint**:
One taught state of the exploration distribution, labelled with the phase of its nearest master sample, nearest in position and orientation together.
_Avoid_: sample, demo point

**Labelling**:
Giving every datapoint the phase of its nearest master sample, in position and orientation together. It needs only the master and the datapoints, never the arm.

**Phase**:
How far along the master the task has come, from 0 at its start to 1 at its end. It is estimated from the tool's pose alone, position and orientation together, so in a world that deviates from the taught one it can be wrong: a pose can look like a later stage although an earlier one never succeeded.
_Avoid_: progress, stage index

**False progress**:
The phase rising although an earlier stage has not succeeded, such as a key turned beside the lock: it is free to turn, and the turn has no effect.
_Avoid_: fake progress, wrong phase

**Tube**:
The region around the master in which a pose is credible for some stage of it: as wide as the datapoints taught at that stage spread, so narrow in the lock, and never narrower than the offset from the taught world that must still read as progress. Outside it the phase does not move.
_Avoid_: canal, envelope, corridor

**Front**:
The phase up to which the sliding target holds datapoints: the current phase plus two resolvable widths.

## Method

**E2T2**:
Ergodic exploration of the whole exploration distribution at once, its statistic kept across the trials of a sequence, as it was proposed. The baseline every method is compared against.
_Avoid_: regular ergodic control, plain ergodic control, vanilla

**Sensorless**:
Knowing only the robot's joint angles, and so the tool's pose: no force, contact or vision sensing. Everything the method knows about the world, including the phase, comes from these.

**Target**:
The distribution the law tries to match, as coefficients.
_Avoid_: reference distribution, goal distribution

**Sliding target**:
A target holding only the datapoints from the phase up to two resolvable widths ahead of it, nothing behind, so it moves along the master as the phase rises or falls.

**Advance**:
The phase rising one resolvable width above the lowest phase it has reached since the last advance.

**Stall**:
The time since the last advance.

**Retry**:
The sliding target's back edge retreating during a stall at the arm's full speed along the master, so that earlier stages return to the target as fast as the arm can go back to them. A key that missed the keyhole, read as turning, is recovered this way within the trial.

**Statistic**:
The record of where the robot has been, as coefficients.
_Avoid_: coverage, history

**Forgetting horizon**:
The half-life with which the statistic forgets: the time of one pass along the master at full speed.
_Avoid_: memory length, window

## Experiment

**Trial**:
One attempt, from the master's start until the goal or the time limit.
_Avoid_: run, episode, attempt

**Sequence**:
Consecutive trials in one world at one offset. Only the state and the block are reset between trials. E2T2 hands its statistic on, as it was proposed; the forgetting methods start every trial with a fresh one, so their trials are independent.
_Avoid_: session, series

**Success**:
A trial that reaches the goal within the time limit: on the arm, the phase passing 0.95, so an opening that stalls below it is a failure; in 2-D, the block reaching the goal, so a slip-out or a dead end is a failure.

**Unassisted success**:
A success in which the operator never touches the arm during the trial. A trial the operator helped home, such as by wiggling the key the last centimetre in, is assisted, whatever it ended in.
_Avoid_: good run, clean run
