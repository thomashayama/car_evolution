from line_seg import Line_seg
from vector import Vector2d
from car import Car
import math
import torch
import torch.nn as nn
import numpy as np

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.fc1 = nn.Linear(11, 6)
        nn.init.xavier_uniform_(self.fc1.weight)
        self.fc2 = nn.Linear(6, 2)
        nn.init.xavier_uniform_(self.fc2.weight)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.sigmoid(x) * 2 - 1
        return x

gray = (100, 100, 100)
white = (255, 255, 255)
black = (0, 0, 0)
red = (255, 100, 100)
green = (100, 255, 100)

car_width = 20
car_length = 40
base_acc = 200

class car_ai_instance:
    def __init__(self, pos=Vector2d(100, 100), angle=0):
        self.key_pressed = {
            'right': False,
            'left': False,
            'up': False,
            'down': False,
            'space': False
        }

        self.car = Car(pos, car_length, car_width, angle=angle)
        self.brain = Net()
        self.score = 0
        self.curr_goal = 0
        self.color = red
        self.pedal = 0.0
        self.since_goal = 0.0

    def think(self):
        with torch.no_grad():
            decisions = self.brain(torch.tensor(
                np.append(self.car.last_seen,
                          [math.cos(self.car.angle / 180 * math.pi),
                           math.sin(self.car.angle / 180 * math.pi),
                           self.car.vel.mag()])).float())
            #self.pedal = max(min(decisions[0]-.5, 5), -.5)
            #self.steer = max(min(decisions[1]-.5, 1), -1)
            self.pedal = decisions[0].numpy()
            self.steer = decisions[1].numpy()

        self.car.ang_vel = self.steer
        '''
        if self.key_pressed['up']:
            self.car.vel.x = speed * math.cos(self.car.angle / 180 * math.pi)
            self.car.vel.y = speed * math.sin(self.car.angle / 180 * math.pi)
        elif self.key_pressed['down']:
            self.car.vel.x = -speed * math.cos(self.car.angle / 180 * math.pi)
            self.car.vel.y = -speed * math.sin(self.car.angle / 180 * math.pi)
        else:
            self.car.vel.x = 0
            self.car.vel.y = 0'''

        self.car.acc.x = base_acc * math.cos(self.car.angle / 180 * math.pi) * self.pedal
        self.car.acc.y = base_acc * math.sin(self.car.angle / 180 * math.pi) * self.pedal

    def tick(self, dt, gameDisplay, road, goals, show=True):
        if self.car.is_done == False:
            self.score += dt
            self.since_goal += dt
            self.score += dt * self.car.vel.mag() * .1
            self.think()

            goal = goals[self.curr_goal % len(goals)]
            if self.car.tick(dt, gameDisplay, road, goal, show=show, color=self.color):
                self.curr_goal += 1
                self.score += 250 * self.curr_goal
                self.since_goal = 0.0

            if self.since_goal > 3:
                self.car.is_done = True

