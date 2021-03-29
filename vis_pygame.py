import pygame
from car import *

def seg_show(gameDisplay, seg, color=(255, 100, 100)):
    pygame.draw.line(gameDisplay, color, (seg[0, 0], seg[0, 1]),
                     (seg[1, 0], seg[1, 1]))


def car_show(gameDisplay, car):
    points = car_get_points(car)

    segs = np.zeros((4, 2, 2))
    segs[0] = points[0:2]
    segs[1] = points[1:3]
    segs[2] = points[2:4]
    segs[3, 0] = points[3]
    segs[3, 1] = points[0]

    for i in range(4):
        pygame.draw.line(gameDisplay, (255, 100, 100), (segs[i][0, 0], segs[i][0, 1]),(segs[i][1, 0], segs[i][1, 1]))
