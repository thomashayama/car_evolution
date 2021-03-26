import pygame
import math
import roads
from car_ai_instance import car_ai_instance
import random
import numpy as np
import torch
import time
import datetime
import os

pygame.init()

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

def mutate(brain, mutation_r):
    with torch.no_grad():
        for param in brain.parameters():
            param.add_(torch.randn(param.size()) * mutation_r)

def reproduce(p1, p2, data):
    new_dict = {}
    w1 = p1.brain.state_dict()
    w2 = p2.brain.state_dict()
    for layer in w1:
        l1 = w1[layer].view(-1)
        l2 = w2[layer].view(-1)
        new_dict[layer] = torch.zeros(len(l1))
        for i in range(len(l1)):
            if random.randrange(100) < mutation_r*100:
                new_dict[layer][i] = random.randrange(1000)/1000
            else:
                #new_dict[layer][i] = random.choice([l1[i],l2[i]]) * (1 + random.randrange(2/mutation_r)*mutation_r - mutation_r)
                new_dict[layer][i] = (l1[i] + l2[i])/2
        new_dict[layer] = new_dict[layer].view(w1[layer].shape)
    ret = car_ai_instance(pos=data.get_data().pos, angle=data.get_data().angle)
    ret.brain.load_state_dict(new_dict)
    return ret

def load(fname, gen_size, new_random, data):
    load_brain = torch.load(fname)['model']
    p1 = car_ai_instance(pos=data.get_data().pos, angle=data.get_data().angle)
    p2 = car_ai_instance(pos=data.get_data().pos, angle=data.get_data().angle)
    p1.brain.load_state_dict(load_brain.state_dict().copy())
    p2.brain.load_state_dict(load_brain.state_dict().copy())
    agents = [p1] * gen_size
    for i in range(10):
        agents[i] = p1

    # Generate new generation
    for i in range(10, gen_size - 11 - new_random):
        agents[i] = reproduce(p1, p2, data)

    for i in range(gen_size - 11 - new_random, gen_size - 1):
        agents[i] = car_ai_instance(pos=data.get_data().pos, angle=data.get_data().angle)
    return agents

def get_time():
    tm = time.gmtime(time.time())
    ans = ''
    for elem in tm:
        ans += str(elem) + '-'
    return '-'.join(ans.split('-')[0:-4])

def new_gen(agents):
    global best_brain, data

    '''
    resolution = .1
    choice_lst = np.array([])
    tot_score = np.array([agent.score for agent in agents]).sum()

    best = car_ai_instance(pos=data.get_data().pos, angle=data.get_data().angle)
    for agent in agents:
        choice_lst = np.append(choice_lst, [agent]*int(100*agent.score/tot_score/resolution))
        if best.score < agent.score:
            best = agent

    if best.score > best_brain[1]:
        best_brain = (best.brain, best.score)

    bestcar = car_ai_instance(pos=data.get_data().pos, angle=data.get_data().angle)
    bestcar.brain = best.brain
    bestcar.color = green
    print(best.score)

    new_agents = [bestcar] * gen_size
    for i in range(10):
        new_agents[i] = reproduce(best, best, data)

    for i in range(10, gen_size - 11 - new_random):
        parent1 = random.choice(choice_lst)
        parent2 = random.choice(choice_lst)
        new_agents[i] = reproduce(parent1, parent2, data)

    for i in range(gen_size - 11 - new_random, gen_size - 1):
        new_agents[i] = car_ai_instance(pos=data.get_data().pos, angle=data.get_data().angle)
    '''
    agents.sort(key=lambda x: x.score, reverse=True)
    best_brain = [agents[0].brain, agents[0].score]
    new_agents = []
    for i in range(10):
        new_agent = car_ai_instance(pos=data.get_data().pos, angle=data.get_data().angle)
        new_agent.brain = agents[i].brain
        new_agents.append(new_agent)
        for j in range(8):
            new_agent = car_ai_instance(pos=data.get_data().pos, angle=data.get_data().angle)
            new_agent.brain.load_state_dict(agents[i].brain.state_dict().copy())
            mutate(new_agent.brain, mutation_r)
            new_agents.append(new_agent)

    for i in range(10):
        new_agents.append(car_ai_instance(pos=data.get_data().pos, angle=data.get_data().angle))
    return new_agents

if __name__== "__main__":
    display_width = 800
    display_height = 600
    gen_size = 100
    batch_size = 20
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

    gray = (100, 100, 100)
    white = (255, 255, 255)
    black = (0, 0, 0)
    red = (255, 100, 100)
    green = (100, 255, 100)

    crashed = False
    running_time = 0.0
    batch_floor = 0
    batch_roof = batch_size
    gen = 0
    best_brain = (car_ai_instance().brain, 0.0)

    data = dataloader([roads.road2, roads.road3], 50)
    agents = [car_ai_instance(pos=data.get_data().pos, angle=data.get_data().angle) for i in range(gen_size)]
    #agents = load('brains/car_30503.496.pt', gen_size, new_random, data)

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
                    tm = "car_" + get_time() + ".pt"
                    print("Model ", tm)
                    torch.save({
                        'model': best_brain[0],
                        'model_state_dict': best_brain[0].state_dict(),
                    }, ".\\brains\\" + tm)

        gameDisplay.fill(gray)

        all_dead = True
        for agent in agents[batch_floor:batch_roof]:
            agent.tick(clock.get_time()/1000, gameDisplay, road, goals, show=True)
            if agent.car.is_done == False:
                all_dead = False
        for seg in road:
            seg.show(gameDisplay, white)
        for goal in goals:
            goal.show(gameDisplay, green)

        running_time += clock.get_time()/1000
        if all_dead or running_time >= max_run_time:
            running_time = 0
            if batch_roof >= gen_size:
                agents = new_gen(agents)
                print("Gen " + str(gen) + " done - Score: " + str(best_brain[1]))
                torch.save({
                    'model': best_brain[0],
                }, brain_dir + '/gen' + str(gen) + 's-' + str(
                    round(best_brain[1])))
                gen += 1
                batch_floor = 0
                batch_roof = batch_size
                max_run_time += 1#math.pow(2, -max_run_time/10)
                data.step()
            else:
                batch_floor += batch_size
                batch_roof = min((gen_size, batch_floor+batch_size))


        text1 = font.render('Simulating gen ' + str(gen), True, white)
        text2 = font.render('Showing gen ' + str(gen), True, white)
        gameDisplay.blit(text1, (1, 1))
        gameDisplay.blit(text2, (1, 16))

        pygame.display.update()
        clock.tick(fps)

    pygame.quit()
    quit()