import math
import numpy as np
import random as rand
import pygame

import world

MAX_MOMENTUM = 5
MARGIN = 0.001

class Copter:

    def __init__(self, my_world, start_x=-1, start_y=-1):
	self.xdim = my_world.xdim
	self.ydim = my_world.ydim
	if start_x == -1:
	   self.x = rand.randint(my_world.movement_bounds[0], my_world.movement_bounds[1])
	else:
	   self.x = start_x
	if start_y == -1:
	   self.y = rand.randint(my_world.movement_bounds[2], my_world.movement_bounds[3])
	else:
	   self.y = start_y
	self.waypoint = (rand.randint(0, self.xdim), rand.randint(0,self.ydim))
	self.momentum_x = 1
	self.momentum_y = 1
	self.hearing_range = 50  
	self.sight_range = 50
	self.orientation = 0
	#c.icon = ??? #we'll do this later
	self.hearing_square = pygame.Surface((self.hearing_range, self.hearing_range))
	self.hearing_square.set_alpha(70)
	self.hearing_square.fill((255, 255, 255)) #white

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
	print "NEW WAYPOINT SELECTED: " + str(waypoint)
	return waypoint

    def step(self, priors=[]):
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
	boundaries.append(max(0, self.x-self.hearing_range/2))		#xmin
	boundaries.append(min(self.xdim, self.x+self.hearing_range/2))	#xmax
	boundaries.append(max(0, self.y-self.hearing_range/2))		#ymin
	boundaries.append(min(self.ydim, self.y+self.hearing_range/2))	#ymax
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

