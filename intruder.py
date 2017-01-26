from __future__ import division
import math
from math import floor
import random as rand

import numpy as np
import cv2

from viewer.my_rrt import run_rrt_poly
INTRUDER_TYPE = 1 #0 = momentum, 1 = waypoints

class Intruder:

    def __init__(self, my_world, MESSY_WORLD=True, target="", max_speed=1, start_x=-1, start_y=-1, use_rrt=True):
        print('Initializing intruder...')
        self.xdim = my_world.xdim
        self.ydim = my_world.ydim
        self.my_world = my_world
        self.target = target
        self.MAX_SPEED = max_speed
        self.use_rrt = use_rrt

        if start_x == -1:
           self.x = rand.randint(0, self.xdim)
        else:
           self.x = start_x

        if start_y == -1:
           self.y = rand.randint(0, self.ydim)
        else:
           self.y = start_y
        
        while not self.my_world.is_valid(self.x, self.y):
            self.x = rand.randint(0, self.xdim)
            self.y = rand.randint(0, self.ydim)

        self.momentum_x = 0
        self.momentum_y = 0
        self.MESSY_WORLD = MESSY_WORLD
        self.count = 0
        self.waypoint = self.select_waypoint()
        self.path_progress = 0

    def generate_path_to_point(self, target):
        print('Generating RRT path...')
        current_location = np.atleast_2d([self.x, self.y])

        rough_path = np.floor(run_rrt_poly(
            current_location,
            np.atleast_2d(target),
            self.my_world.contours,
            scale=1000,
            step_limit=30000))[1:]

        exact_path = [[self.x, self.y]]

        print('self.MAX_SPEED', self.MAX_SPEED)

        for next_x, next_y in rough_path:
            current_x, current_y = exact_path[-1]
            number_steps_between = floor(max(abs(next_x - current_x), abs(next_y - current_y))) 

            intermediate_xs = np.atleast_2d(np.floor(np.linspace(current_x, next_x, number_steps_between/self.MAX_SPEED))).T
            intermediate_ys = np.atleast_2d(np.floor(np.linspace(current_y, next_y, number_steps_between/self.MAX_SPEED))).T

            intermediate_points = np.hstack([intermediate_xs, intermediate_ys])

            print ''
            print intermediate_points

            exact_path += [x for x in intermediate_points]

        return exact_path

    def random_location(self):
        pass

    def momentum_step(self):
        self.momentum_x += rand.randint(-1, 1)
        self.momentum_y += rand.randint(-1, 1)
        if abs(self.momentum_x) > self.MAX_SPEED:
            self.momentum_x = 0
        if abs(self.momentum_y) > self.MAX_SPEED:
            self.momentum_y = 0
        
        new_x = self.x
        new_y = self.y
        
        new_x = max(0, new_x + self.momentum_x)
        new_y = max(0, new_y + self.momentum_y)
        new_x = min(self.xdim-1, new_x)
        new_y = min(self.ydim-1, new_y)
        
        if self.my_world.is_valid(new_x, new_y):
            self.x = new_x
            self.y = new_y
        else:
            self.momentum_x = 0
            self.momentum_y = 0

    def select_waypoint(self):
        print('Selecting waypoint...')

        self.path_progress = 0

        if (self.target == "" or self.my_world.num_treats == 0) or \
           (self.MESSY_WORLD == True and rand.randint(0,1) == 0 ):
            waypoint = (rand.randint(0, self.xdim-1), rand.randint(0, self.ydim-1))
        #select a targeted waypoint the other half of the time
        elif self.target=="cookies":
            index = rand.randint(0,self.my_world.num_treats-1)
            waypoint = self.my_world.cookies[index]
        #if self.target=="popcorn":
        #        index = rand.randint(0,self.my_world.num_treats-1)
        #        waypoint = self.my_world.popcorn[index]
        #if self.target=="truffles":
        #        index = rand.randint(0,self.my_world.num_treats-1)
        #        waypoint = self.my_world.truffles[index]
        
        if not self.my_world.is_valid(waypoint[0], waypoint[1]):
            self.select_waypoint()

        if self.use_rrt:
            self.path = self.generate_path_to_point(waypoint)

            if len(self.path) < 5:
                self.select_waypoint()

            self.count = 0
            self.path_progress = 0

        return waypoint

    def waypoint_step(self):
        if abs(self.x - self.waypoint[0]) < 10 and abs(self.y - self.waypoint[1]) < 10:
           #pick a new waypoint 
           self.waypoint = self.select_waypoint()
           self.momentum_x = 1
           self.momentum_y = 1

        print('self.path_progress', self.path_progress)
        print('self.waypoint, self.x, self.y', self.waypoint, self.x, self.y)

        if self.use_rrt:

            if self.path_progress >= len(self.path):
                self.select_waypoint()

            new_x = self.path[self.path_progress][0]
            new_y = self.path[self.path_progress][1]
            self.path_progress = self.path_progress + 1

            self.x = new_x
            self.y = new_y

        else:
            self.momentum_x += 1
            self.momentum_y += 1
            self.momentum_x = min(self.MAX_SPEED, self.momentum_x)
            self.momentum_y = min(self.MAX_SPEED, self.momentum_y)
            

            jitter = 0
            if self.MESSY_WORLD == True:
                jitter = rand.randint(0,2) - 1

            new_x = self.x
            new_y = self.y

            if self.waypoint[0] > self.x:
                new_x = self.x + (self.momentum_x+jitter) 
            if self.waypoint[0] < self.x:
                new_x = self.x - (self.momentum_x+jitter) 
            if self.waypoint[1] > self.y:
                new_y = self.y + (self.momentum_y+jitter) 
            if self.waypoint[1] < self.y:
                new_y = self.y - (self.momentum_y+jitter) 

            if self.my_world.is_valid(new_x, new_y):
                self.x = new_x
                self.y = new_y
            else:
                self.count += 1
                if self.count > 5:
                    self.waypoint = self.select_waypoint()


    def step(self):
        if INTRUDER_TYPE == 0:
            self.momentum_step()
        else:
            self.waypoint_step()

    def select_random_location(self):
        x1,x2,y1,y2 = self.my_world.movement_bounds[0], self.my_world.movement_bounds[1], self.my_world.movement_bounds[2], self.my_world.movement_bounds[3]
        self.x = rand.randint(x1,x2)
        self.y = rand.randint(x1,x2)

        while not self.my_world.is_valid(self.x,self.y):
            self.x = rand.randint(x1,x2)
            self.y = rand.randint(x1,x2)

    def simple_movement_kernel(self):
        return self.simple_kernel

    def complex_movement_kernel(self):
        return self.kernel
    
    def movement_kernal(self,x,y):
        return self.kernal[x][y]


class Trespasser(Intruder):

    def __init__(self, my_world, MESSY_WORLD=True, target="", max_speed=1, start_x=-1, start_y=-1):
        Intruder.__init__(self, my_world, MESSY_WORLD, target, max_speed, start_x, start_y)
