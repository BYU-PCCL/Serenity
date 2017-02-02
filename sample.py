
import numpy as np
import world
import copter
import intruder
import prob_mass
import time


XDIM = 1000
YDIM = 1000

MESSY_WORLD = True
NUM_TREATS = 3
INTRUDER_MOMENTUM=4

#A world with cookies for grabbing, based on the Bremen point cloud
w = world.World(XDIM, YDIM, NUM_TREATS, "bremen")

#An intruder who likes cookies and walks linearly from waypoint to waypoint
i = intruder.Thief(w, MESSY_WORLD, "cookies", INTRUDER_MOMENTUM)

#use the actual intruder as our model, for now
i_model = i

c=copter.Copter(w)
pm = prob_mass.prob_mass(w, i, i_model, c)
DENSITY_MAP = pm.PRIORS
c.set_initial_path(DENSITY_MAP)

while True:
    c.step(DENSITY_MAP)
    i.step()
    DENSITY_MAP = pm.step()
    time.sleep(500)
    print DENSITY_MAP[500:600,500:600]
    print DENSITY_MAP.shape


