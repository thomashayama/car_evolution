from collections import deque
from line_seg import *
from vector import *
#import math
#import pygame
import numpy as np
from numba import njit

# car stored as [posx, posy, velx, vely, accx, accy, angle, ang_vel, length, width, sensors...]
@njit
def init_car(center, length, width, angle):
    return np.array([center[0], center[1], 0.0, 0.0, 0.0, 0.0, angle, 0.0, length, width, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

@njit
def car_pos(car):
    return car[0:2]

@njit
def car_vel(car):
    return car[2:4]

@njit
def car_acc(car):
    return car[4:6]

@njit
def car_angle(car):
    return car[6]

@njit
def car_ang_vel(car):
    return car[7]

@njit
def car_length(car):
    return car[8]

@njit
def car_width(car):
    return car[9]

@njit
def car_is_done(car):
    return car[10]

@njit
def car_sensors(car):
    return car[11:]

@njit
def set_car_pos(car, pos):
    car[0:2] = pos

@njit
def set_car_vel(car, vel):
    car[2:4] = vel

@njit
def set_car_acc(car, acc):
    car[4:6] = acc

@njit
def set_car_angle(car, angle):
    car[6] = angle


def set_car_ang_vel(car, ang_vel):
    car[7] = ang_vel

@njit
def set_car_length(car, length):
    car[8] = length

@njit
def set_car_width(car, width):
    car[9] = width

@njit
def set_car_is_done(car, is_done):
    car[10] = is_done

@njit
def set_car_sensors(car, sensors):
    car[11:] = sensors

@njit
def car_get_points(car):
    l = car_length(car)/2
    w = car_width(car)/2
    pos = car_pos(car)
    angle = car_angle(car)
    points = np.zeros((4, 2))
    points[0, :] = rot(np.array([pos[0]+l, pos[1]-w]), pos, angle)
    points[1, :] = rot(np.array([pos[0]+l, pos[1]+w]), pos, angle)
    points[2, :] = rot(np.array([pos[0]-l, pos[1]+w]), pos, angle)
    points[3, :] = rot(np.array([pos[0]-l, pos[1]-w]), pos, angle)
    return points

@njit
def car_get_sensors(car, segs, points):
    anchors = np.zeros((8, 2))
    for i in range(4):
        anchors[i*2, :] = seg_midpoint(segs[i])
        anchors[i*2+1, :] = points[(i + 1) % 4]

    sensors = np.zeros((8, 2, 2))
    for i in range(len(anchors)):
        sensors[i, 0] = anchors[i, :]
        sensors[i, 1] = rot(anchors[i, :] + np.array([200, 0]), anchors[i, :], car_angle(car) + i*360/8)
    return sensors

@njit
def car_get_hits(car, sensors, road):
    hits = np.zeros(len(sensors))
    for i in range(len(sensors)):
        short_dist = car_length(car) * 10
        s = sensors[i]
        for j in range(len(road)):
            inter = seg_intersection(s, road[j])
            if (inter[0] >= road[j,0,0] and inter[0] <= road[j,1,0])or(inter[0] <= road[j,0,0] and inter[0] >= road[j,1,0]):
                dist = rot(inter, s[0], -i*360/8 - car_angle(car))[0] - s[0,0]
                if dist < short_dist and dist >= 0:
                    short_dist = dist
        hits[i] = short_dist
    return hits

@njit
def car_tick(dt, car, max_vel, drag_coef, road, barrier):
    set_car_pos(car, car_pos(car) + car_vel(car) * dt)
    set_car_vel(car, car_vel(car) + dt * (car_acc(car) - drag_coef * car_vel(car) - 20 * np.sign(car_vel(car))))

    norm = mag(car_vel(car))

    if norm == 0:
        norm = 1
    coeff = max(min(norm, max_vel), -max_vel) / norm
    set_car_vel(car, car_vel(car) * coeff)

    set_car_angle(car, (dt * car_ang_vel(car) * norm) + car_angle(car) % 360)

    points = car_get_points(car)

    segs = np.zeros((4, 2, 2))
    segs[0] = points[0:2]
    segs[1] = points[1:3]
    segs[2] = points[2:4]
    segs[3, 0] = points[3]
    segs[3, 1] = points[0]

    hits = car_get_hits(car, car_get_sensors(car, segs, points), road)
    set_car_sensors(car, hits/car_length(car))

    if car_is_done(car) == 0.0:
        for e in range(len(road)):
            for s in range(len(segs)):
                if seg_cross_seg(segs[s], road[e]):
                    set_car_is_done(car, 1.0)
                    break

    for s in range(len(segs)):
        if seg_cross_seg(segs[s], barrier):
            return True

    return False

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


gray = (100, 100, 100)
white = (255, 255, 255)
black = (0, 0, 0)
red = (255, 100, 100)
green = (100, 255, 100)

class Car:
    def __init__(self, center, length, width, angle=0.0):
        self.length = length
        self.width = width

        self.pos = center
        self.vel = Vector2d(0, 0)
        self.acc = Vector2d(0, 0)
        self.angle = angle
        self.ang_vel = 0.0
        self.drag_coef = 2

        self.max_vel = 200

        self.last_seen = np.array([0]*8)
        self.is_done = False

    def get_points(self):
        l = int(self.length/2)
        w = int(self.width/2)
        points = deque([
            Vector2d((self.pos.x+l, self.pos.y-w)).rot(self.pos, self.angle),
            Vector2d((self.pos.x+l, self.pos.y+w)).rot(self.pos, self.angle),
            Vector2d((self.pos.x-l, self.pos.y+w)).rot(self.pos, self.angle),
            Vector2d((self.pos.x-l, self.pos.y-w)).rot(self.pos, self.angle)
        ])
        return points

    def get_sensors(self, anchors):
        sensors = [Line_seg(anchor, (anchor + Vector2d(200, 0)).rot(anchor, self.angle + i*360/8))
                   for i, anchor in enumerate(anchors)]
        return sensors

    def get_hits(self, anchors, road):
        sensors = self.get_sensors(anchors)
        dists = np.zeros(len(anchors))
        hits = np.array([])
        for i, s in enumerate(sensors):
            shortest = Vector2d(-1, -1)
            short_dist = 2000
            for edge in road:
                inter = s.intersection(edge)
                if (inter.x >= edge.p1.x and inter.x <= edge.p2.x)or(inter.x <= edge.p1.x and inter.x >= edge.p2.x):
                    dist = inter.rot(s.p1, -i*360/8 - self.angle).x - s.p1.x
                    if dist < short_dist and dist >=0:
                        short_dist = dist
                        shortest = inter
            dists[i] = short_dist
            hits = np.append(hits, shortest)
        return hits, dists

    def tick(self, dt, gameDisplay, road, barrier, show=True, all_vis=False, color=red):
        self.pos = self.pos + self.vel * dt
        self.vel = Vector2d(self.vel.x + dt * (self.acc.x - self.drag_coef * self.vel.x - 20*np.sign(self.vel.x)),
                            self.vel.y + dt * (self.acc.y - self.drag_coef * self.vel.y - 20*np.sign(self.vel.y)))
        #self.vel = self.vel + self.acc * dt
        norm = self.vel.mag()

        if norm == 0:
            norm = 1
        coeff = max(min(norm, self.max_vel), -self.max_vel)/norm
        self.vel = self.vel * coeff

        self.angle = ((dt * self.ang_vel * norm) + self.angle)%360

        points = self.get_points()

        segs = [
            Line_seg(points[0], points[1]),
            Line_seg(points[1], points[2]),
            Line_seg(points[2], points[3]),
            Line_seg(points[3], points[0])
        ]

        sensor_anchors = np.array([])
        for i in range(4):
            if show:
                segs[i].show(gameDisplay, color=color)
            sensor_anchors = np.append(sensor_anchors, segs[i].midpoint())
            sensor_anchors = np.append(sensor_anchors, points[(i + 1) % 4])
        #sensor_anchors = np.append(sensor_anchors, Line_seg(segs[0].midpoint(), segs[0].p1).midpoint())
        #sensor_anchors = np.append(sensor_anchors, Line_seg(segs[0].midpoint(), segs[0].p2).midpoint())

        hits, dists = self.get_hits(sensor_anchors, road)
        #self.last_seen = [hit.mag()/self.length for hit in hits]
        self.last_seen = dists/self.length
        #print(self.last_seen)
        #for hit in hits:
        #    pygame.draw.circle(gameDisplay, (0, 0, 0), (int(hit.x), int(hit.y)), 5, 5)
        if show:
            for i, hit in enumerate(hits):
                if all_vis:
                    Line_seg(hit, sensor_anchors[i]).show(gameDisplay)
                    hit.show(gameDisplay)

        if self.is_done == False:
            for edge in road:
                for seg in segs:
                    if seg.is_intersecting_with(edge):
                        self.is_done = True
                        break

        for seg in segs:
            if seg.is_intersecting_with(barrier):
                return True
        return False