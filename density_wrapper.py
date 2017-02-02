
import numpy as np
import world
import copter
import intruder
import prob_mass


MESSY_WORLD = True
INTRUDER_MOMENTUM=4
DOWNSAMPLE_FACTOR=2


class density_wrapper:

    def __init__(self, XDIM=1000, YDIM=1000, MODE=1, NUM_TREATS=3, MAP="bremen"):

        #A world with cookies for grabbing, based on the Bremen point cloud
        self.w = world.World(XDIM, YDIM, NUM_TREATS, "bremen")

        #An intruder who likes cookies and walks linearly from waypoint to waypoint
        self.i = intruder.Thief(self.w, MESSY_WORLD, "cookies", INTRUDER_MOMENTUM)

        #use the actual intruder as our model, for now
        self.i_model = self.i

        self.c=copter.Copter(self.w)
        self.pm = prob_mass.prob_mass(self.w, self.i, self.i_model, self.c, MODE)
	self.DENSITY_MAP = self.pm.PRIORS
        self.c.set_initial_path(self.DENSITY_MAP)

    def step(self):
	self.c.step(self.DENSITY_MAP)
        self.i.step()
    	self.DENSITY_MAP = self.pm.step()
	return self.DENSITY_MAP

