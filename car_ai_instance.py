from line_seg import Line_seg
from vector import *
from car import *
import math
import torch
import torch.nn as nn
import numpy as np
from numba import njit

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

def model_init():
    return np.random.randn(11, 6), np.random.randn(6, 2)

def forward(l1, l2, x):
    y = np.dot(x, l1)
    y = np.maximum(y, 0)
    y = np.dot(y, l2)
    y = 1/(np.exp(-y)+1)
    return y

gray = (100, 100, 100)
white = (255, 255, 255)
black = (0, 0, 0)
red = (255, 100, 100)
green = (100, 255, 100)

car_width = 20
car_length = 40
base_acc = 200

class car_ai_instance:
    def __init__(self, pos=np.array([100.0, 100.0]), angle=0.0):
        self.key_pressed = {
            'right': False,
            'left': False,
            'up': False,
            'down': False,
            'space': False
        }

        self.car = init_car(pos, car_length, car_width, angle)
        self.brain = Net()
        self.score = 0
        self.curr_goal = 0
        self.color = red
        self.pedal = 0.0
        self.since_goal = 0.0

    def think(self):
        with torch.no_grad():
            decisions = self.brain(torch.tensor(
                np.append(car_sensors(self.car),
                          [math.cos(car_angle(self.car) / 180 * math.pi),
                           math.sin(car_angle(self.car) / 180 * math.pi),
                           mag(car_vel(self.car))])).float())
            self.pedal = decisions[0].numpy()
            self.steer = decisions[1].numpy()

        set_car_ang_vel(self.car, self.steer)
        new_acc = np.array([base_acc * math.cos(car_angle(self.car) / 180 * math.pi) * self.pedal,
                            base_acc * math.sin(car_angle(self.car) / 180 * math.pi) * self.pedal])

        set_car_acc(self.car, new_acc)

    def tick(self, dt, gameDisplay, road, goals, show=True):
        if car_is_done(self.car) == 0.0:
            self.score += dt
            self.since_goal += dt
            self.score += dt * mag(car_vel(self.car)) * .1
            self.think()

            goal = goals[self.curr_goal % len(goals)]
            if car_tick(dt, self.car, 200, 2, road, goal):
                self.curr_goal += 1
                self.score += 250 * self.curr_goal
                self.since_goal = 0.0

            if self.since_goal > 3:
                set_car_is_done(self.car, 1.0)
        if show:
            car_show(gameDisplay, self.car)

