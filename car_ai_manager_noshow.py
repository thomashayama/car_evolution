import math
import roads_np as roads
from car_ai_instance import *
from car_ai_utils import *
import numpy as np
import torch
import datetime
import os
import multiprocessing
import concurrent.futures

if __name__== "__main__":
    start = time.perf_counter()
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
    processes = []

    data = dataloader([roads.road2, roads.road3], 50)
    #agents = [car_ai_instance(pos=data.get_data().pos, angle=data.get_data().angle) for i in range(gen_size)]
    agents = load('brains/gen45s-1238393', gen_size, new_random, data, mutation_r=.1)

    while not crashed:
        noise = np.random.randn()*fps_noise
        dt = 1 / (fps+noise)
        all_dead = True
        for agent in agents:
            agent.tick(dt, None, data.get_data().data, data.get_data().goals, False)
        #with concurrent.futures.ThreadPoolExecutor() as executor:
        #    for agent in agents:
        #        executor.submit(agent.tick, (dt, None, data.get_data().data, data.get_data().goals, False))

            #results = [executor.submit(agent.tick, (dt, None, data.get_data().data, data.get_data().goals, False) for agent in agents]
        #for agent in agents:
        #    p = multiprocessing.Process(target=agent.tick, args=(dt, None, data.get_data().data, data.get_data().goals, False))
        #    processes.append(p)
        #    p.start()

        #for p in processes:
        #    p.join()
        #p = []

        for agent in agents:
            if car_is_done(agent.car) == 0.0:
                all_dead = False
                break

        running_time += dt
        if all_dead or running_time >= max_run_time:
            data.step()
            running_time = 0
            agents, best_brain = new_gen(agents, data, gen_size=gen_size, mutation_r=mutation_r, new_random=new_random)
            print("Gen " + str(gen) + " done - Score: " + str(best_brain[1]))
            torch.save({
                'model': best_brain[0],
            }, brain_dir + '/gen' + str(gen) + 's-' + str(round(best_brain[1])))
            gen += 1
            max_run_time += 1   #math.pow(2, -max_run_time/10)
            if gen > 50:
                break

    finish = time.perf_counter()
    print(f'Finished in {round(finish-start, 2)} seconds')
