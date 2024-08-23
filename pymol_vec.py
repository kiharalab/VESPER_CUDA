from __future__ import print_function
from pymol.cgo import *  # get constants
from math import *
from pymol import cmd
import numpy as np
import os


def showvectors(coords_path, vecs_path, outname="vectors", head=0.2, tail=0.1, head_length=0.7, headrgb="1.0,1.0,1.0",
                tailrgb="1.0,1.0,1.0", cut=0.0, notail=0):
    if not os.path.exists(coords_path):
        print("Usage: coordinate file does not exist")
        return
    if not os.path.exists(vecs_path):
        print("Usage: vector file does not exist")
        return

    coords = np.load(coords_path)
    vecs = np.load(vecs_path)

    arrow_head_radius = float(head)
    arrow_tail_radius = float(tail)
    arrow_head_length = float(head_length)
    cut = float(cut)
    objectname = outname
    objectname = objectname.strip('"[]()')

    headrgb = headrgb.strip('" []()')
    tailrgb = tailrgb.strip('" []()')
    hr, hg, hb = list(map(float, headrgb.split(',')))
    tr, tg, tb = list(map(float, tailrgb.split(',')))

    arrow = []

    for coord, vec in zip(coords, vecs):
        vectorx, vectory, vectorz = vec
        vectorz = vec[2]
        length = 1
        t = 1.0 - cut
        x1, y1, z1 = coord
        x2 = x1 + t * vectorx
        y2 = y1 + t * vectory
        z2 = z1 + t * vectorz
        vectorx = x2 - x1
        vectory = y2 - y1
        vectorz = z2 - z1
        length = sqrt(vectorx ** 2 + vectory ** 2 + vectorz ** 2)
        d = arrow_head_length  # Distance from arrow tip to arrow base
        t = 1.0 - (d / length)
        if notail:
            t = 0
        tail = [
            # Tail of cylinder
            CYLINDER, x1, y1, z1 \
            , x1 + (t + 0.01) * vectorx, y1 + (t + 0.01) * vectory, z1 + (t + 0.01) * vectorz \
            , arrow_tail_radius, tr, tg, tb, tr, tg, tb  # Radius and RGB for each cylinder tail
        ]
        if notail == 0:
            arrow.extend(tail)

        x = x1 + t * vectorx
        y = y1 + t * vectory
        z = z1 + t * vectorz
        dx = x2 - x
        dy = y2 - y
        dz = z2 - z
        seg = d / 100
        head = [
            CONE, x, y, z, x + d * dx, y + d * dy, z + d * dz, arrow_head_radius, 0.0, hr, hg, hb, hr, hg, hb, 1.0, 1.0]
        arrow.extend(head)

    cmd.delete(objectname)
    cmd.load_cgo(arrow, objectname)


cmd.extend("showvectors", showvectors)
