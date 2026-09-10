#!/usr/bin/env python3
# -*-coding:utf8-*-
"""
Draw the arm in 3D from its own STL meshes, in the host browser.

meshcat renders with three.js in the browser and ships only geometry and
transforms over ZMQ, so the container needs no X display, no OpenGL and no
rebuild. run.sh uses --net=host, so the address url() prints opens directly in
the host browser.

The pose comes from this project's own forward kinematics, which makes the
picture a check on the model rather than a second rendering of the URDF: the
round-trip test compares FK against FK and cannot catch a frame convention that
is wrong but self-consistent, whereas a mesh hung on the wrong frame is obvious.

Which meshes exist is read from the URDF, not listed here, so a link with no
<visual> is simply not drawn. That is what makes the zero-gripper-mass variant
show the arm as it is actually built: its gripper links carry no visual.

    from visualization.meshcat_view import MeshcatArm
    view = MeshcatArm(kin)
    print(view.url())
    view.set_configuration(q)
"""

import os
from xml.etree import ElementTree

import meshcat
import numpy as np
from kinematics.fk import joint_frames
from kinematics.transform import Transform, rpy_to_matrix
from meshcat import geometry

MESH_DIR = os.path.normpath(
    os.path.join(os.path.dirname(__file__), "..", "..", "robot_description", "meshes")
)

# Which joint frame each link rides on. The URDF does not state this directly,
# and it is structural rather than a choice: joint j drives link j, so link j
# sits on joint_frames() entry j - 1. base_link is the world frame, because
# base_to_dummy is a fixed joint with an identity origin. gripper_base is
# rigidly fixed to link6 and shares its frame.
#
# The gripper fingers link7 and link8 are absent on purpose. They are prismatic,
# locked out of the model, and each carries its own joint origin, so drawing
# them would need that offset applied rather than link6's frame. If a gripper
# goes back on, that is the piece to add.
LINK_FRAMES = {
    "base_link": None,
    "link1": 0,
    "link2": 1,
    "link3": 2,
    "link4": 3,
    "link5": 4,
    "link6": 5,
    "gripper_base": 5,
}

ARM_COLOR = 0x9DA5B4
TRIAD_LENGTH = 0.12


def _floats(text):
    return np.array([float(value) for value in text.split()])


def read_visuals(urdf_path, mesh_dir=MESH_DIR):
    """
    The mesh each link draws itself with, as {link: (mesh path, placement)}.

    The placement is the <visual> origin relative to the link frame. It is
    identity for every link of this arm, but reading it costs three lines and
    removes an assumption that would fail silently.
    """
    visuals = {}

    for link in ElementTree.parse(urdf_path).getroot().findall("link"):
        mesh = link.find("visual/geometry/mesh")

        if mesh is None:
            continue

        if mesh.get("scale") is not None:
            raise ValueError(
                f"{link.get('name')} scales its mesh, which this viewer does not apply"
            )

        origin = link.find("visual/origin")
        placement = Transform()

        if origin is not None:
            placement = Transform(
                rpy_to_matrix(*_floats(origin.get("rpy", "0 0 0"))),
                _floats(origin.get("xyz", "0 0 0")),
            )

        path = os.path.join(mesh_dir, os.path.basename(mesh.get("filename")))

        visuals[link.get("name")] = (path, placement)

    return visuals


class MeshcatArm:
    """
    A meshcat scene holding the arm, moved by set_configuration().

    The meshes are uploaded once, in __init__. An update afterwards is one 4x4
    matrix per link, which is cheap enough to drive from a live telemetry
    stream.
    """

    def __init__(self, kin, mesh_dir=MESH_DIR, zmq_url=None):

        self.kin = kin
        self.viewer = meshcat.Visualizer(zmq_url)

        # Clear first, so re-running against an already open browser tab
        # replaces the arm instead of leaving the previous one behind.
        self.viewer["arm"].delete()

        material = geometry.MeshLambertMaterial(color=ARM_COLOR)
        self.placements = {}

        for name, (path, placement) in read_visuals(kin.urdf_path, mesh_dir).items():
            if name not in LINK_FRAMES:
                print(f"INFO: {name} has a visual but no frame here, skipping")
                continue

            self.viewer["arm"][name].set_object(geometry.StlMeshGeometry.from_file(path), material)

            self.placements[name] = placement

    def set_configuration(self, q):
        """
        Place every link for the joint angles q [rad].
        """
        frames = joint_frames(self.kin, q)

        for name, placement in self.placements.items():
            index = LINK_FRAMES[name]
            frame = Transform() if index is None else frames[index]

            self.viewer["arm"][name].set_transform((frame * placement).homogeneous)

    def set_target(self, target, name="target", length=TRIAD_LENGTH):
        """
        Draw a pose as an RGB axis triad -- the pose an IK call was given, say.

        If the solve is right the arm's own flange lands on top of it.
        """
        self.viewer[name].set_object(geometry.triad(length))
        self.viewer[name].set_transform(target.homogeneous)

    def url(self):
        """
        The address to open in the host browser.
        """
        return self.viewer.url()
