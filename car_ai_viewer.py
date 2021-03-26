import pygame
import math
import roads
from car_ai_instance import car_ai_instance
import random
import numpy as np
import torch
import time
import os

pygame.init()

gray = (100, 100, 100)
white = (255, 255, 255)
black = (0, 0, 0)
red = (255, 100, 100)
green = (100, 255, 100)

class dataloader:
    def __init__(self, data, flip_every=-1):
        self.flip_every = flip_every
        self.data = data
        self.counter = 0
        self.curr_data = 0

    def step(self):
        self.counter += 1
        if self.counter >= self.flip_every:
            self.counter = 0
            self.curr_data = (self.curr_data + 1) % len(self.data)

    def get_data(self):
        return self.data[self.curr_data]

if __name__== "__main__":
    display_width = 800
    display_height = 600
    gen_size = 100
    batch_size = 10
    new_random = 10
    mutation_r = .1
    max_run_time = 5
    sim_realtime = False
    fps = 30
    dir = "./brains/2021-03-23 22_43_15.973427"

    gameDisplay = pygame.display.set_mode((display_width, display_height))
    clock = pygame.time.Clock()
    pygame.display.set_caption('Car')
    font = pygame.font.SysFont('arialblack', 18)

    crashed = False
    running_time = 0.0
    batch_floor = 0
    batch_roof = batch_size
    gen = 0
    best_brain = [car_ai_instance().brain, 0.0]

    data = dataloader([roads.road2, roads.road3], 500)
    files = os.listdir(dir)
    def get_gen(x):
        s = x.split('s')[0]
        return int(s[3:])
    files.sort(key=get_gen)

    agent = car_ai_instance(pos=data.get_data().pos, angle=data.get_data().angle)
    agent.brain = torch.load(dir + '/' + files.pop(0))['model']
    print(torch.load(dir + '/' + files[0])['model'])

    road = data.get_data().data
    goals = data.get_data().goals

    key_pressed = {
        'right': False,
        'left': False,
        'up': False,
        'down': False,
        'space': False
    }

    while not crashed:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                crashed = True

            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_s:
                    running_time += 99999

        gameDisplay.fill(gray)

        agent.tick(clock.get_time()/1000, gameDisplay, road, goals, show=True)
        for seg in road:
            seg.show(gameDisplay, white)
        for goal in goals:
            goal.show(gameDisplay, green)

        running_time += clock.get_time()/1000
        if agent.car.is_done or running_time >= max_run_time:
            running_time = 0
            gen += 1
            if gen >= len(files):
                break
            agent = car_ai_instance(pos=data.get_data().pos, angle=data.get_data().angle)
            agent.brain = torch.load(dir + '/' + files[gen])['model']
            max_run_time += math.pow(2, -max_run_time/10)
            data.step()
            if data.counter == 0:
                max_run_time = 5


        text1 = font.render('gen ' + str(gen), True, white)
        gameDisplay.blit(text1, (1, 1))

        pygame.display.update()
        clock.tick(fps)

    pygame.quit()
    quit()