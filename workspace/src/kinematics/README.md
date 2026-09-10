# How the IK works

Forward kinematics is the easy direction: angles in, tool pose out. Inverse is the
hard one — you know where you want the tool, you need the six joint angles.

## The trick: the wrist is a ball joint

Joints 4, 5 and 6 all pass through a single point (to within 88 micrometres, which
is a rounding artefact in the vendor's CAD export). That splits one hard 6-D
problem into two easy 3-D ones:

- joints 1, 2, 3 decide **where** that point sits
- joints 4, 5, 6 decide **which way** the tool points

## Step 1 — find the wrist centre

The wrist centre is always 91 mm back along the tool axis from the flange. Subtract
that from the target pose and you have a plain point in space. No angles involved.

## Step 2 — reach for it

Now it is just a shoulder and two links reaching a point.

- Joint 1 is the compass bearing: `atan2(y, x)`.
- Joints 2 and 3 form a triangle with two known side lengths, so the law of cosines
  gives the elbow angle. Elbow up or elbow down both work.

## Step 3 — twist the wrist

Whatever rotation is left over after step 2 is what the wrist has to supply. Three
axes meeting at right angles is a textbook Euler-angle extraction. There are two
solutions, the second being the wrist flipped over.

## That gives eight answers

2 shoulder × 2 elbow × 2 wrist flip. Most get discarded because they would need a
joint past its stop — joints 2 and 3 only turn one way, so they rule out a lot. Of
what survives we take the one nearest to where the arm already is, so it does not
reconfigure itself halfway through a move.

## How accurate is that?

The maths above is solved on an idealised arm, the one where that 88 µm gap is
exactly zero, so the answer lands about 90 µm off. We leave it there. The arm's own
position commands are quantised at roughly 190 µm, so refining below that would be
invisible on the hardware.

## The one exception: a floppy wrist

If joint 5 is near zero, joints 4 and 6 line up and only their *sum* matters — how
you split a rotation between them becomes almost arbitrary. There the same 90 µm of
geometry error gets amplified, and can throw the tool's *orientation* by several
degrees even though its position is still fine.

So when no branch lands on target, we take the closest one and nudge it downhill
with a few Newton steps until it does. That happens on about 3 poses in 5000. The
repair is there for that one bad case, not for accuracy.
