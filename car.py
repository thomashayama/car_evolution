from line_seg import Line_seg
from vector import Vector2d
import math
import pygame
from collections import deque
import numpy as np

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