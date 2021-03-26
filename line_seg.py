import pygame
from vector import *
import math
from numba import njit
import numpy as np

@njit
def seg_cross_seg(l1, l2):
    # Precompute the relative vectors.
    (ux, uy) = (l1[1,0]-l1[0,0], l1[1,1]-l1[0,1])
    (vx, vy) = (l2[1,0]-l2[0,0], l2[1,1]-l2[0,1])
    (rx, ry) = (l2[0,0]-l1[0,0], l2[0,1]-l1[0,1])

    # Precompute the vector cross products.
    uXv = ux*vy - uy*vx
    rXu = rx*uy - ry*ux
    rXv = rx*vy - ry*vx

    # Check the intersection.
    if (uXv > 0):
        return ((rXu > 0) and (rXu < uXv) and (rXv > 0) and (rXv < uXv))
    else:
        return ((rXu < 0) and (rXu > uXv) and (rXv < 0) and (rXv > uXv))

@njit
def seg_midpoint(l1):
    return (l1[0] + l1[1])/2

@njit
def seg_get_line(l1):
    if l1[0][0] - l1[1][0] == 0:
        slope = 99999
    else:
        slope = (l1[0][1] - l1[1][1]) / (l1[0][0] - l1[1][0])
    y_int = l1[0][1] - slope * l1[0][0]
    return slope, y_int

@njit
def seg_intersection(l1, l2):
    m1, b1 = seg_get_line(l1)
    m2, b2 = seg_get_line(l2)

    if m1 - m2 == 0:
        if b1 == b2:
            return l2[0]
        else:
            return np.array([0.0, 0.0])

    if m1 == 99999:
        x = l1[0, 0]
        y = m2 * x + b2
    elif m2 == 99999:
        x = l2[0, 0]
        y = m1 * x + b1
    else:
        x = (b2 - b1) / (m1 - m2)
        y = m1 * x + b1
    return np.array([x, y])


def seg_show(gameDisplay, seg, color=(255, 100, 100)):
    pygame.draw.line(gameDisplay, color, (seg[0, 0], seg[0, 1]),
                     (seg[1, 0], seg[1, 1]))

class Line_seg:
    def __init__(self, *args):
        if len(args) == 2 and isinstance(args[0], Vector2d) and isinstance(args[1], Vector2d):
            self.p1 = args[0]
            self.p2 = args[1]
        else:
            self.p1 = Vector2d()
            self.p2 = Vector2d()

    def __add__(self, other):
        return Line_seg(self.p1 + other.p1, self.p2 + other.p2)

    def __str__(self):
        return "{p1} to {p2}".format(p1=self.p1, p2=self.p2)

    def numpy(self):
        return np.array([self.p1.numpy(), self.p2.numpy()])

    def mag(self):
        return mag(self.p1.numpy()-self.p2.numpy())

    def angle(self):
        return (self.p2 - self.p1).angle()

    def get_line(self):
        if self.p1.x - self.p2.x == 0:
            slope = 99999
        else:
            slope = (self.p1.y - self.p2.y)/(self.p1.x - self.p2.x)
        y_int = self.p1.y - slope * self.p1.x
        return slope, y_int

    def midpoint(self):
        return Vector2d(int((self.p1.x + self.p2.x)/2), int((self.p1.y + self.p2.y)/2))

    def intersection(self, seg):
        m1, b1 = self.get_line()
        m2, b2 = seg.get_line()

        if m1-m2 == 0:
            return seg.p1 if b1 == b2 else Vector2d(-1, -1)

        if m1 == 99999:
            x = self.p1.x
            y = m2 * x + b2
            return Vector2d(x, y)
        elif m2 == 99999:
            x = seg.p1.x
            y = m1 * x + b1
            return Vector2d(x, y)
        else:
            x = (b2-b1)/(m1-m2)
            y = m1 * x + b1
            return Vector2d(x, y)

    def is_intersecting_with(self, seg):
        #inter = self.intersection(seg)
        #if inter == None:
        #    return False

        #if (inter.x >= self.p1.x and inter.x <= self.p2.x) or (inter.x <= self.p1.x and inter.x >= self.p2.x):
        #    if (inter.x >= seg.p1.x and inter.x <= seg.p2.x) or (inter.x <= seg.p1.x and inter.x >= seg.p2.x):
        #        return True
        #return False
        return seg_cross_seg(self.numpy(), seg.numpy())

    def show(self, gameDisplay, color=(255, 100, 100)):
        pygame.draw.line(gameDisplay, color, (self.p1.x, self.p1.y),
                         (self.p2.x, self.p2.y))