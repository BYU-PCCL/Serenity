import math
from math import floor
import numpy as np
import random as rand
import pygame

import world

from viewer.my_rrt import run_rrt

MAX_MOMENTUM = 5
MARGIN = 0.001

class Copter:

    def __init__(self, my_world, priors=[], start_x=-1, start_y=-1):
        self.xdim = my_world.xdim
        self.ydim = my_world.ydim
	self.my_world = my_world
        if start_x == -1:
           self.x = rand.randint(my_world.movement_bounds[0], my_world.movement_bounds[1])
        else:
           self.x = start_x
        if start_y == -1:
           self.y = rand.randint(my_world.movement_bounds[2], my_world.movement_bounds[3])
        else:
           self.y = start_y
        #self.waypoint = (rand.randint(0, self.xdim), rand.randint(0,self.ydim))
	#self.RRT_path = self.generate_path_to_point(self.waypoint)
	self.path = self.generate_path(priors)
	self.path_waypoint = self.path[0]
        self.momentum_x = 1
        self.momentum_y = 1
        self.hearing_range = 80  
        self.sight_range = 50
        self.orientation = 0
        #c.icon = ??? #we'll do this later
        self.hearing_square = pygame.Surface((self.hearing_range, self.hearing_range))
        self.hearing_square.set_alpha(70)
        self.hearing_square.fill((255, 255, 255)) #white
	self.last_x = self.x
	self.last_y = self.y

    def select_waypoint(self, priors=[]):
        #if no priors were passed in, 
        #choose a random waypoint
        if priors == []:
            waypoint = (rand.randint(0, self.xdim), rand.randint(0, self.ydim))
            #print "RANDOM WAYPOINT SELECTED: " + str(waypoint)
            return waypoint

        #select the waypoint that maximizes the probability
        #of spotting the thief from that location
        SEARCH_RADIUS = self.hearing_range - 1 #was 100-1
        SEARCH_STEP_SIZE = 25 #was 50
        boundary = [0, SEARCH_RADIUS, 0, SEARCH_RADIUS]
        most_likely_region = boundary
        best_probability = np.sum(priors[boundary[0]:boundary[1], boundary[2]:boundary[3]])
        for x in range(0, self.xdim-SEARCH_RADIUS, SEARCH_STEP_SIZE):
            for y in range(0, self.ydim - SEARCH_RADIUS, SEARCH_STEP_SIZE):
                boundary = [x, x+SEARCH_RADIUS, y, y+SEARCH_RADIUS]
                prob = np.sum(priors[boundary[0]:boundary[1], boundary[2]:boundary[3]])
                if abs(best_probability - prob) < MARGIN:
                    #if two regions seem mostly equal,
                    #then take the closest one
                    #print "There's a tie! Taking nearest region..."
                    dist1 = abs(self.x - (most_likely_region[0] + most_likely_region[1])/2)
                    dist1 += abs(self.y - (most_likely_region[2] + most_likely_region[3])/2)
                    dist2 = abs(self.x - (boundary[0] + boundary[1])/2)
                    dist2 += abs(self.y - (boundary[2] + boundary[3])/2)
                    if dist2 < dist1:
                        best_probability = prob
                        most_likely_region = boundary
                elif prob > best_probability:
                    #print boundary
                    #print "Found new best probability: " + str(prob)
                    best_probability = prob
                    most_likely_region = boundary

        #select a waypoint at the center of the best region
        waypoint = ((most_likely_region[0] + most_likely_region[1])/2, (most_likely_region[2] + most_likely_region[3])/2)
        # print "NEW WAYPOINT SELECTED: " + str(waypoint)
        return waypoint

    def RRT_step(self, priors=[]):
	if len(self.RRT_path) == 0:
	    self.waypoint = self.select_waypoint(priors)
	    self.RRT_path = self.generate_path_to_point(self.waypoint)
	else:
	    self.x = self.RRT_path[0][0]
	    self.y = self.RRT_path[0][1]
	    self.RRT_path = self.RRT_path[1:]

    def generate_path(self, priors):
	NUM_WAYPOINTS = 3
	
	path = [(self.x, self.y)]

	#for now, just pick random points
#	for i in range(NUM_WAYPOINTS):
#	    path.append((rand.randint(0, self.xdim), rand.randint(0,self.ydim)))


	#for even later: select a region to search
	#then search within that region...

        if priors == []:
	    print "No probabilities available. Could not plan copter path."
            return path

        SEARCH_RADIUS = 49 #was 100...
        SEARCH_STEP_SIZE = 15 
	NUM_SEARCH_REGIONS = 10

	#scan immediate vicinity...
	start_x = max(self.my_world.movement_bounds[0], self.x - NUM_SEARCH_REGIONS*SEARCH_STEP_SIZE)
	start_y = max(self.my_world.movement_bounds[2], self.y - NUM_SEARCH_REGIONS*SEARCH_STEP_SIZE)
	stop_x = min(self.my_world.movement_bounds[1], self.x + NUM_SEARCH_REGIONS*SEARCH_STEP_SIZE)
	stop_y = min(self.my_world.movement_bounds[3], self.y + NUM_SEARCH_REGIONS*SEARCH_STEP_SIZE)

	threshhold = np.mean(priors)
	probs = []
	locs = []
	#print "search boundaries are %i, %i, %i, %i" % (start_x, start_y, stop_x, stop_y)
        for x in range(start_x, stop_x-SEARCH_RADIUS, SEARCH_STEP_SIZE):
	    #print "x is %i" % (x)
            for y in range(start_y, stop_y - SEARCH_RADIUS, SEARCH_STEP_SIZE):
		#print "y is %i" % (y)
                boundary = [x, x+SEARCH_RADIUS, y, y+SEARCH_RADIUS]
                prob = np.mean(priors[boundary[0]:boundary[1], boundary[2]:boundary[3]])
                if prob > threshhold:
		    #print "prob of %f exceeds threshhold of %f" % (prob, threshhold)
                    probs.append(prob)
                    locs.append( (x + SEARCH_RADIUS/2, y + SEARCH_RADIUS/2) )

	#take the top one
	if len(probs) > 0:
	    i = probs.index(max(probs))
	    path.append(locs[i])

	#order found regions according to probability
	#select the first n
	#order those n such that the path is minimized
	#print "path is:"
	#print path
	return path

    def move_toward_point(self, point):
	self.last_x = self.x
	self.last_y = self.y

	delta_x = point[0] - self.x
	delta_y = point[1] - self.y
	distance = math.sqrt(delta_x*delta_x + delta_y*delta_y)
	num_steps = distance/MAX_MOMENTUM
	x_amt = delta_x/num_steps
	y_amt = delta_y/num_steps

	self.x += x_amt
	self.y += y_amt

        if abs(self.x - point[0]) < MAX_MOMENTUM and abs(self.y - point[1]) < MAX_MOMENTUM:
	    self.x = point[0]
	    self.y = point[1]


    def path_step(self, priors):
	if (self.x, self.y) == self.path_waypoint:
	    if len(self.path) > 1:
		self.path = self.path[1:]
		self.path_waypoint = self.path[0]
	    else:
		self.path = self.generate_path(priors)
		self.path_waypoint = self.path[0]
	else:
	   self.move_toward_point(self.path_waypoint)

	

    def step(self, priors=[]):
	#self.waypoint_step()
	#self.RRT_step(priors)
	self.path_step(priors)

    def waypoint_step(self, priors=[]):
	self.last_x = self.x
	self.last_y = self.y
        self.momentum_x += 1
        self.momentum_y += 1
        self.momentum_x = min(MAX_MOMENTUM, self.momentum_x)
        self.momentum_y = min(MAX_MOMENTUM, self.momentum_y)
        
        if abs(self.x - self.waypoint[0]) < self.momentum_x and abs(self.y - self.waypoint[1]) < self.momentum_y:
           #pick a new waypoint 
           self.waypoint = self.select_waypoint(priors)
           self.momentum_x = 1
           self.momentum_y = 1

        if self.waypoint[0] > self.x:
            self.x += self.momentum_x
            self.orientation = 1
        if self.waypoint[0] < self.x:
            self.x -= self.momentum_x
            self.orientation = 3
        if self.waypoint[1] > self.y:
            self.y += self.momentum_y
            self.orientation = 2
        if self.waypoint[1] < self.y:
            self.y -= self.momentum_y
            self.orientation = 0

    def movement_vector(self):
	return (self.x-self.last_x,self.y-self.last_y)

    def get_points_in_hearing_range(self):
        return_val = [] 
        for x in range(max(0, self.x - self.hearing_range/2), min(self.xdim-1, self.x + self.hearing_range/2)):
            for y in range(max(0, self.y - self.hearing_range/2), min(self.ydim-1, self.y + self.hearing_range/2)):
                return_val.append([x,y])
        return return_val

    def get_points_in_line_of_sight(self):
        pass

    def get_hearing_boundaries(self):
        boundaries = []
        boundaries.append(max(0, self.x-self.hearing_range/2))                #xmin
        boundaries.append(min(self.xdim, self.x+self.hearing_range/2))        #xmax
        boundaries.append(max(0, self.y-self.hearing_range/2))                #ymin
        boundaries.append(min(self.ydim, self.y+self.hearing_range/2))        #ymax
        return boundaries

    def get_sight_boundaries(self):
        boundaries=[]
        boundaries.append((self.x, self.y))
        if self.orientation == 0:
            #facing up
            boundaries.append([max(0, self.x-self.sight_range), max(0, self.y-self.sight_range)])
            boundaries.append([min(self.xdim, self.x+self.sight_range), max(0, self.y - self.sight_range)])
        if self.orientation == 1:
            #facing right
            boundaries.append([min(self.xdim, self.x+self.sight_range), max(0, self.y-self.sight_range)])
            boundaries.append([min(self.xdim, self.x+self.sight_range), min(self.ydim, self.y + self.sight_range)])
        if self.orientation == 2:
            #facing down
            boundaries.append([max(0, self.x-self.sight_range), max(0, self.y+self.sight_range)])
            boundaries.append([min(self.xdim, self.x+self.sight_range), max(0, self.y + self.sight_range)])
        if self.orientation == 3:
            #facing left
            boundaries.append([max(0, self.x-self.sight_range), max(0, self.y-self.sight_range)])
            boundaries.append([max(0, self.x-self.sight_range), min(self.ydim, self.y + self.sight_range)])
        return boundaries

    def generate_path_to_point(self, target):

        print('Generating RRT path...')
        current_location = np.atleast_2d([self.x, self.y])

        rough_path = np.floor(run_rrt(
            current_location,
            np.atleast_2d(target),
            self.my_world.contours,
            1000,
            step_limit=30000))[1:]

        exact_path = [[self.x, self.y]]

        for next_x, next_y in rough_path:
            current_x, current_y = exact_path[-1]
            number_steps_between = floor(max(abs(next_x - current_x), abs(next_y - current_y)) / MAX_MOMENTUM)

            intermediate_xs = np.atleast_2d(np.floor(np.linspace(current_x, next_x, number_steps_between))).T
            intermediate_ys = np.atleast_2d(np.floor(np.linspace(current_y, next_y, number_steps_between))).T

            intermediate_points = np.hstack([intermediate_xs, intermediate_ys])

            exact_path += [x.astype(int) for x in intermediate_points]

	print self.waypoint
	print exact_path
        print('len(exact_path)', len(exact_path))
        return exact_path


