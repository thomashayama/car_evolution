import math
import roads_np as roads
from car_ai_instance import *
from car_ai_numba_utils import *
import numpy as np
import torch
import time
import datetime
import os


def new_gen(agents, data, gen_size = 1, mutation_r=.05, new_random=0):
    global best_brain

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
    return new_agents

if __name__== "__main__":
    fps = 30
    fps_noise = 5       # Prevents deterministic NNs

    # Hyperparameters
    gen_size = 110
    new_random = 10
    mutation_r = .05
    max_run_time = 30

    brain_dir = "./brains/" + str(datetime.datetime.now()).replace(':', '_') + "/"
    if not os.path.exists(brain_dir):
        os.mkdir(brain_dir)

    crashed = False
    running_time = 0.0
    gen = 0
    best_brain = (car_ai_instance().brain, 0.0)

    data = dataloader([roads.road2, roads.road3], 50)
    #agents = [car_ai_instance(pos=data.get_data().pos, angle=data.get_data().angle) for i in range(gen_size)]
    agents = load('brains/gen151s-4538154', gen_size, new_random, data, mutation_r=.1)

    while not crashed:
        noise = np.random.randn()*fps_noise
        dt = 1 / (fps+noise)
        all_dead = True
        for agent in agents:
            agent.tick(dt, None, data.get_data().data, data.get_data().goals, show=False)
            if car_is_done(agent.car) == 0.0:
                all_dead = False

        running_time += dt
        if all_dead or running_time >= max_run_time:
            running_time = 0
            agents = new_gen(agents, data, gen_size=gen_size, mutation_r=mutation_r, new_random=new_random)
            print("Gen " + str(gen) + " done - Score: " + str(best_brain[1]))
            torch.save({
                'model': best_brain[0],
            }, brain_dir + '/gen' + str(gen) + 's-' + str(round(best_brain[1])))
            gen += 1
            max_run_time += 1   #math.pow(2, -max_run_time/10)
            data.step()
            if data.counter == 0:
                max_run_time = 5
