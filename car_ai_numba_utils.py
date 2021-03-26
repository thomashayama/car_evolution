import torch
import numpy as np
from car_ai_instance_numba import *
import time

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