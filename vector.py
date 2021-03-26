import math
import pygame
from numba import njit, float64
import numpy as np

@njit()
def init_vector(a, b):
    return np.array([a, b])

@njit(float64[:](float64[:], float64[:], float64))
def rot(vec, cvec, angle):
    new_x = (vec[0] - cvec[0]) * math.cos(angle / 180 * math.pi) - (
            vec[1] - cvec[1]) * math.sin(angle / 180 * math.pi) + cvec[0]
    new_y = (vec[0] - cvec[0]) * math.sin(angle / 180 * math.pi) + (
            vec[1] - cvec[1]) * math.cos(angle / 180 * math.pi) + cvec[1]
    return np.array([new_x, new_y])

@njit(float64(float64[:]))
def angle(vec):
    if vec[0] == 0:
        return 90 * vec[1] / abs(vec[1])
    angl = math.atan(vec[1] / vec[0]) / math.pi * 180
    if angl < 0:
        angl += 360
    return angl

@njit
def mag(vec):
    return np.linalg.norm(vec)

class Vector2d:
    def __init__(self, *args):
        if len(args) == 1 and isinstance(args[0], tuple):
            self.x = args[0][0]
            self.y = args[0][1]
        elif len(args) == 2 and isinstance(args[0], int) and isinstance(args[1], int):
            self.x = args[0]
            self.y = args[1]
        elif len(args) == 2 and isinstance(args[0], float) and isinstance(args[1], float):
            self.x = args[0]
            self.y = args[1]
        else:
            self.x = 0
            self.y = 0

    def __add__(self, other):
        if isinstance(other, Vector2d):
            return Vector2d(self.x + other.x, self.y + other.y)
        elif isinstance(other, int) or isinstance(other, float):
            return Vector2d(self.x + other, self.y + other)
        else:
            return Vector2d(self.x + other.x, self.y + other.y)

    def __sub__(self, other):
        if isinstance(other, Vector2d):
            return Vector2d(self.x - other.x, self.y - other.y)
        elif isinstance(other, int) or isinstance(other, float):
            return Vector2d(self.x - other, self.y - other)
        else:
            return Vector2d(self.x - other.x, self.y - other.y)

    def __mul__(self, other):
        if isinstance(other, Vector2d):
            return Vector2d(self.x * other.x, self.y * other.y)
        elif isinstance(other, int) or isinstance(other, float):
            return Vector2d(self.x * other, self.y * other)
        else:
            return Vector2d(self.x * other, self.y * other)


    def __str__(self):
        return "({x}, {y})".format(x=self.x, y=self.y)

    def __repr__(self):
        return self.__str__()

    def numpy(self):
        return np.array([self.x, self.y]).astype(float)

    def get_tuple(self):
        return (self.x, self.y)

    def mag(self):
        return mag(self.numpy())

    def angle(self):
        return angle(self.numpy())

    def rot(self, center, angle):
        vec = rot(self.numpy(), center.numpy(), angle)
        return Vector2d(int(vec[0]), int(vec[1]))

    def show(self, gameDisplay):
        pygame.draw.circle(gameDisplay, (0, 0, 0), (int(self.x), int(self.y)), 5, 1)