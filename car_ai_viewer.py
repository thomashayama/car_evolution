import pygame
import math
import roads
from car_ai_instance import *
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
    fps = 30
    dir = "./brains/2021-03-25 22_46_46.231196"

    gameDisplay = pygame.display.set_mode((display_width, display_height))
    clock = pygame.time.Clock()
    pygame.display.set_caption('Car')
    font = pygame.font.SysFont('arialblack', 18)

    crashed = False
    running_time = 0.0
    gen = 0

    data = dataloader([roads.road2, roads.road3], 50)
    files = os.listdir(dir)
    def get_gen(x):
        s = x.split('s')[0]
        return int(s[3:])
    files.sort(key=get_gen)

    agent = car_ai_instance(pos=data.get_data().pos, angle=data.get_data().angle)
    agent.brain = torch.load(dir + '/' + files[0])['model']
    print('Showing:', files[0])

    key_pressed = {
        'right': False,
        'left': False,
        'up': False,
        'down': False,
        'space': False
    }

    while not crashed:
        gameDisplay.fill(gray)

        agent.tick(clock.get_time()/1000, gameDisplay, data.get_data().data, data.get_data().goals, show=True)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                crashed = True

            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_s:
                    agent.car.is_done = True

        for seg in data.get_data().data:
            seg.show(gameDisplay, white)
        for goal in data.get_data().goals:
            goal.show(gameDisplay, green)

        running_time += clock.get_time()/1000
        if agent.car.is_done:
            running_time = 0
            gen += 1
            if gen >= len(files):
                break
            agent = car_ai_instance(pos=data.get_data().pos, angle=data.get_data().angle)
            agent.brain = torch.load(dir + '/' + files[gen])['model']
            print('Showing:', files[gen])
            data.step()


        text1 = font.render('gen ' + str(gen), True, white)
        text2 = font.render('score: ' + str(round(agent.score)), True, white)
        gameDisplay.blit(text1, (1, 1))
        gameDisplay.blit(text2, (1, 15))

        pygame.display.update()
        clock.tick(fps)

    pygame.quit()
    quit()