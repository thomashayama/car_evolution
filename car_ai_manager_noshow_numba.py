import math
import roads
from car_ai_instance_numba import *
import random
import numpy as np
import torch
import time
import datetime
import os

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

def load(fname, gen_size, new_random, data):
    load_brain = torch.load(fname)['model']
    p1 = car_ai_instance(pos=data.get_data().pos_np, angle=data.get_data().angle)
    p1.brain.load_state_dict(load_brain.state_dict().copy())
    agents = [car_ai_instance(pos=data.get_data().pos_np, angle=data.get_data().angle) for _ in range(gen_size)]
    agents[0] = p1

    for i in range(1, gen_size-new_random):
        agents[i].brain.load_state_dict(p1.brain.state_dict().copy())
        mutate(agents[i].brain, mutation_r)

    return agents

def get_time():
    tm = time.gmtime(time.time())
    ans = ''
    for elem in tm:
        ans += str(elem) + '-'
    return '-'.join(ans.split('-')[0:-4])

def new_gen(agents):
    global best_brain, data

    agents.sort(key=lambda x: x.score, reverse=True)
    best_brain = [agents[0].brain, agents[0].score]
    new_agents = []
    for i in range(1):
        new_agent = car_ai_instance(pos=data.get_data().pos_np, angle=data.get_data().angle)
        new_agent.brain = agents[i].brain
        new_agents.append(new_agent)
        for j in range(8):
            new_agent = car_ai_instance(pos=data.get_data().pos_np, angle=data.get_data().angle)
            new_agent.brain.load_state_dict(agents[i].brain.state_dict().copy())
            mutate(new_agent.brain, mutation_r)
            new_agents.append(new_agent)

    for i in range(new_random):
        new_agents.append(car_ai_instance(pos=data.get_data().pos_np, angle=data.get_data().angle))
    return new_agents

def to_np(x):
    y = np.zeros((len(x), 2, 2))
    for i, seg in enumerate(x):
        y[i,:,:] = np.array([[seg.p1.x, seg.p1.y], [seg.p2.x, seg.p2.y]])
    return y


if __name__== "__main__":
    display_width = 800
    display_height = 600
    gen_size = 100
    new_random = 10
    mutation_r = .1
    max_run_time = 50
    fps = 30
    fps_noise = 5

    brain_dir = "./brains/" + str(datetime.datetime.now()).replace(':', '_') + "/"
    if not os.path.exists(brain_dir):
        os.mkdir(brain_dir)

    crashed = False
    running_time = 0.0
    gen = 0
    best_brain = (car_ai_instance().brain, 0.0)

    data = dataloader([roads.road2, roads.road3], 50)
    #agents = [car_ai_instance(pos=data.get_data().pos, angle=data.get_data().angle) for i in range(gen_size)]
    agents = load('brains/gen45s-1238393', gen_size, new_random, data)

    road = to_np(data.get_data().data)
    goals = to_np(data.get_data().goals)

    while not crashed:
        noise = np.random.randn()*fps_noise
        dt = 1 / (fps+noise)
        all_dead = True
        for agent in agents:
            agent.tick(dt, None, road, goals, show=False)
            if car_is_done(agent.car) == 0.0:
                all_dead = False

        running_time += dt
        if all_dead or running_time >= max_run_time:
            running_time = 0
            agents = new_gen(agents)
            print("Gen " + str(gen) + " done - Score: " + str(best_brain[1]))
            torch.save({
                'model': best_brain[0],
            }, brain_dir + '/gen' + str(gen) + 's-' + str(round(best_brain[1])))
            gen += 1
            max_run_time += 1#math.pow(2, -max_run_time/10)
            data.step()
            if data.counter == 0:
                max_run_time = 5
