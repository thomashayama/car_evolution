import torch
import numpy as np
from car_ai_instance import *
import time
import random

GRAY = (100, 100, 100)
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 100, 100)
GREEN = (100, 255, 100)

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

def reproduce(p1, p2, data, mutation_r=0):
    new_dict = {}
    w1 = p1.brain.state_dict()
    w2 = p2.brain.state_dict()
    for layer in w1:
        l1 = w1[layer].view(-1)
        l2 = w2[layer].view(-1)
        new_dict[layer] = torch.zeros(len(l1))
        for i in range(len(l1)):
            if np.random.rand() < mutation_r:
                new_dict[layer][i] = random.randrange(1000)/1000
            else:
                #new_dict[layer][i] = random.choice([l1[i],l2[i]]) * (1 + random.randrange(2/mutation_r)*mutation_r - mutation_r)
                new_dict[layer][i] = (l1[i] + l2[i])/2
        new_dict[layer] = new_dict[layer].view(w1[layer].shape)
    ret = car_ai_instance(pos=data.get_data().pos, angle=data.get_data().angle)
    ret.brain.load_state_dict(new_dict)
    return ret

def new_gen(agents, data, gen_size = 1, mutation_r=.05, new_random=0):
    agents.sort(key=lambda x: x.score, reverse=True)
    best_brain = [agents[0].brain, agents[0].score]
    new_agents = []
    made = 0
    for i in range(int((gen_size-new_random)/10)):
        made += 1
        new_agent = car_ai_instance(pos=data.get_data().pos, angle=data.get_data().angle)
        new_agent.brain = agents[i].brain
        new_agents.append(new_agent)
        for j in range(int((gen_size-new_random-int((gen_size-new_random)/10))/10)):
            made += 1
            new_agent = car_ai_instance(pos=data.get_data().pos, angle=data.get_data().angle)
            new_agent.brain.load_state_dict(agents[i].brain.state_dict().copy())
            mutate(new_agent.brain, mutation_r)
            new_agents.append(new_agent)

    for i in range(gen_size - made):
        new_agents.append(car_ai_instance(pos=data.get_data().pos, angle=data.get_data().angle))
    return new_agents, best_brain

def get_time():
    tm = time.gmtime(time.time())
    ans = ''
    for elem in tm:
        ans += str(elem) + '-'
    return '-'.join(ans.split('-')[0:-4])

def mutate(brain, mutation_r):
    with torch.no_grad():
        for param in brain.parameters():
            param.add_(torch.randn(param.size()) * mutation_r)

def load(fname, gen_size, new_random, data, mutation_r=.1):
    load_brain = torch.load(fname)['model']
    p1 = car_ai_instance(pos=data.get_data().pos, angle=data.get_data().angle)
    p1.brain.load_state_dict(load_brain.state_dict().copy())
    agents = [car_ai_instance(pos=data.get_data().pos, angle=data.get_data().angle) for _ in range(gen_size)]
    agents[0] = p1

    for i in range(1, gen_size-new_random):
        agents[i].brain.load_state_dict(p1.brain.state_dict().copy())
        mutate(agents[i].brain, mutation_r)

    return agents