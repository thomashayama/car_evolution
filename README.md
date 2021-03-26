# car_evolution
Using an evolution based algorithm to optimize a neural network to control a 2d car

PREP:
  Update scipy, numpy, pygame, torch, and numba

USE: 
  - Run car_ai_manager.py to see every generation while it learns or car_ai_manager_noshow.py 
    to run without visualizing (much faster)
  - Each generation's best brain for the run is stored in ./brains under a directory named after
    the timestamp for when the run started
  - Run car_ai_viewer to see the best brain in each generation run the course (for the selected 
    brain directory)
  
