import pygame
import roads_np as roads
from car_ai_instance import *
from car_ai_utils import *
from vis_pygame import *
import numpy as np
import torch
import datetime
import os

GRAY = (100, 100, 100)
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 100, 100)
GREEN = (100, 255, 100)

pygame.init()

if __name__== "__main__":
    display_width = 800
    display_height = 600
    gen_size = 100
    batch_size = 100
    new_random = 10
    mutation_r = .1
    max_run_time = 50
    sim_realtime = False
    fps = 30

    brain_dir = "./brains/" + str(datetime.datetime.now()).replace(':', '_') + "/"
    if not os.path.exists(brain_dir):
        os.mkdir(brain_dir)

    gameDisplay = pygame.display.set_mode((display_width, display_height))
    clock = pygame.time.Clock()
    pygame.display.set_caption('Car')
    font = pygame.font.SysFont('arialblack', 18)
    show_queue = [] # Queue of car state_dicts to show

    crashed = False
    running_time = 0.0
    batch_floor = 0
    batch_roof = batch_size
    gen = 0
    best_brain = (car_ai_instance().brain, 0.0)

    data = dataloader([roads.road2, roads.road3], 50)
    #agents = [car_ai_instance(pos=data.get_data().pos, angle=data.get_data().angle) for i in range(gen_size)]
    agents = load('brains/gen45s-1238393', gen_size, new_random, data, mutation_r=.1)

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
                    tm = "car_" + get_time() + ".pt"
                    print("Model ", tm)
                    torch.save({
                        'model': best_brain[0],
                        'model_state_dict': best_brain[0].state_dict(),
                    }, ".\\brains\\" + tm)

        gameDisplay.fill(gray)

        all_dead = True
        for agent in agents[batch_floor:batch_roof]:
            agent.tick(clock.get_time()/1000, gameDisplay, data.get_data().data, data.get_data().goals, show=True)
            if car_is_done(agent.car) == 0.0:
                all_dead = False
        for seg in data.get_data().data:
            seg_show(gameDisplay, seg, white)
        for goal in data.get_data().goals:
            seg_show(gameDisplay, goal, green)

        running_time += clock.get_time()/1000
        if all_dead or running_time >= max_run_time:
            running_time = 0
            if batch_roof >= gen_size:
                data.step()
                agents, best_brain = new_gen(agents, data, gen_size=gen_size, mutation_r=mutation_r, new_random=new_random)
                print("Gen " + str(gen) + " done - Score: " + str(best_brain[1]))
                torch.save({
                    'model': best_brain[0],
                }, brain_dir + '/gen' + str(gen) + 's-' + str(
                    round(best_brain[1])))
                gen += 1
                batch_floor = 0
                batch_roof = batch_size
                max_run_time += 1#math.pow(2, -max_run_time/10)
            else:
                batch_floor += batch_size
                batch_roof = min((gen_size, batch_floor+batch_size))


        text1 = font.render('Simulating gen ' + str(gen), True, white)
        gameDisplay.blit(text1, (1, 1))

        pygame.display.update()
        clock.tick(fps)

    pygame.quit()
    quit()