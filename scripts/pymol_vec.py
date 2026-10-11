"""PyMOL arrows from the .npy files of `vesper orig -v DIR`.

run scripts/pymol_vec.py
showvectors DIR/ref_map_coords.npy, DIR/ref_map_vecs.npy, ref
"""

import os
from math import sqrt

import numpy as np

try:
    from pymol import cmd
    from pymol.cgo import CONE, CYLINDER
except ImportError:  # the arrow geometry is testable without PyMOL
    cmd = None
    CYLINDER, CONE = 9.0, 27.0


def arrow_parts(coord, vec, cut=0.0, head_length=0.7, notail=False):
    """Return (tail, head) for one vector, or None if it has zero length.

    tail is (start, end) of the cylinder or None; head is (base, tip) of the cone.
    The tip is coord + (1 - cut) * vec; the head is clamped to the vector length.
    """
    start = np.asarray(coord, dtype=float)
    delta = (1.0 - cut) * np.asarray(vec, dtype=float)
    length = sqrt(float(delta @ delta))
    if length == 0:
        return None
    t = 1.0 - min(head_length, length) / length
    tip = start + delta
    base = start + t * delta
    if notail or t == 0:
        return None, (base, tip)
    return (start, start + (t + 0.01) * delta), (base, tip)


def showvectors(
    coords_path,
    vecs_path,
    outname="vectors",
    head=0.2,
    tail=0.1,
    head_length=0.7,
    headrgb="1.0,1.0,1.0",
    tailrgb="1.0,1.0,1.0",
    cut=0.0,
    notail=0,
):
    if not os.path.exists(coords_path):
        print("Usage: coordinate file does not exist")
        return
    if not os.path.exists(vecs_path):
        print("Usage: vector file does not exist")
        return

    coords = np.load(coords_path)
    vecs = np.load(vecs_path)

    # PyMOL passes every argument as a string
    arrow_head_radius = float(head)
    arrow_tail_radius = float(tail)
    head_length = float(head_length)
    cut = float(cut)
    notail = int(notail)
    objectname = outname.strip('"[]()')

    headrgb = headrgb.strip('" []()')
    tailrgb = tailrgb.strip('" []()')
    hr, hg, hb = list(map(float, headrgb.split(",")))
    tr, tg, tb = list(map(float, tailrgb.split(",")))

    arrow = []
    for coord, vec in zip(coords, vecs):
        parts = arrow_parts(coord, vec, cut, head_length, notail)
        if parts is None:
            continue
        tail_pts, (base, tip) = parts
        if tail_pts is not None:
            arrow.extend([CYLINDER, *tail_pts[0], *tail_pts[1]])
            arrow.extend([arrow_tail_radius, tr, tg, tb, tr, tg, tb])
        arrow.extend([CONE, *base, *tip, arrow_head_radius, 0.0])
        arrow.extend([hr, hg, hb, hr, hg, hb, 1.0, 1.0])

    cmd.delete(objectname)
    cmd.load_cgo(arrow, objectname)


if cmd is not None:
    cmd.extend("showvectors", showvectors)
