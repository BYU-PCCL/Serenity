import sys
import math
import random as rand
import numpy as np

import cv2
from scipy.misc import imread

import isovist

SAFETY_MARGIN = 3   #boundary around obstacles
		    #used in self.is_valid()


class World:

    def __init__(self, xdim, ydim, num_treats, world_type = "bremen", using_viewer = False, num_obstacles=10, min_obst_dim=10, max_obst_dim=100):
        self.xdim = xdim
        self.ydim = ydim
	self.using_viewer = using_viewer

	self.polygon_segments = []

	print("Creating world of type '" + world_type + "'")
	if world_type == "bremen":
            self.movement_bounds = [int(0.3*xdim),int(0.82*xdim),int(0.28*ydim),int(0.8*ydim)]
            #self.movement_bounds = [0,xdim-1,0,ydim-1]
	    self.polygon_segments = []
            self.read_terrain_image('point_clouds/bremen_altstadt_final.png')
	else:
            self.movement_bounds = [0,xdim-1,0,ydim-1]
	    self.polygon_segments = []
            self.polygon_segments += self.initialize_terrain(num_obstacles, min_obst_dim, max_obst_dim)
       
        ###DEPRECATED###
	#self.create_valid_squares()

        self.initialize_treats(num_treats)
	self.polygon_map = self.dw_load_polygons("./paths.txt", self.using_viewer)
	self.polygon_segments = self.create_segments_from_polygons(self.polygon_map)
	
        self.isovist = isovist.Isovist(self.polygon_segments)

        print('  Creating validity map...')
        self.validity_map = self.create_validity_map()

        #print('  Storing contours...')
        #self.contours = self.detect_contours()

    def create_segments_from_polygons(self, polygons):

	polygon_segments = []

	for polygon in polygons:
	    segments = []
	    for i in range(len(polygon)-1):
	        segment = [polygon[i], polygon[i+1]]
		segments.append(map(tuple,segment))
	    segment = [polygon[-1], polygon[0]]
	    segments.append(map(tuple,segment))
            polygon_segments.append(segments)

	return polygon_segments

        """
	min_obst_dim = 10
	max_obst_dim = 100
        for i in range(10):
            start_x = rand.randint(0,self.xdim-min_obst_dim)
            start_y = rand.randint(0,self.ydim-min_obst_dim)
            end_x = start_x + rand.randint(min_obst_dim, max_obst_dim)
            if end_x > self.xdim-1:
                end_x = self.xdim-1
            end_y = start_y + rand.randint(min_obst_dim, max_obst_dim)
            if end_y > self.ydim-1:
                end_y = self.ydim
            self.terrain[start_x:end_x, start_y:end_y] = 1
	    point1 = (end_x, start_y)
	    point2 = (end_x, end_y)
	    point3 = (start_x, end_y)
	    point4 = (start_x, start_y)
	    segments = [[point1, point2], [point2,point3], [point3, point4], [point4,point1]]
	    polygon_segments.append(segments)
	#print polygon_segments
	return polygon_segments
        """

    def dw_load_polygons( self, fn="./paths.txt", using_viewer = False ):
        bdata = []
        for x in open( fn ):
            tmp = np.fromstring( x, dtype=float, sep=' ' )
#            tmp = np.reshape( tmp/1000.0, (-1,2) )
            tmp = np.reshape( tmp, (-1,2) )
            tmp = np.vstack(( np.mean(tmp, axis=0, keepdims=True), tmp, tmp[0,:] ))
#            tmp[:,1] = 1.0 - tmp[:,1]  # flip on the y axis

            if using_viewer:
		pass
                #tmp[:,1] = 1000.0 - tmp[:,1]  # flip on the y axis
            bdata.append( tmp )
        return bdata

    def initialize_treats(self, num_treats):
        self.num_treats = num_treats
        self.cookies = []
        self.popcorn = []
        self.truffles = []
        for k in range(num_treats):
            coords = [rand.randint(0,self.xdim-1), rand.randint(0,self.ydim-1)]
            while(not self.is_valid(coords[0], coords[1])):
                coords = [rand.randint(0,self.xdim-1), rand.randint(0,self.ydim-1)]
            self.cookies.append(coords)
        #for p in range(num_treats):
        #    coords = [rand.randint(0,xdim), rand.randint(0,ydim)]
        #    self.popcorn.append(coords)
        #for t in range(num_treats):
        #    coords = [rand.randint(0,xdim), rand.randint(0,ydim)]
        #    self.truffles.append(coords)

    def add_terrain_frame(self):
	#adds a 1-pixel boundary around the traversable regions
	#of the world.
        self.terrain[self.movement_bounds[0]] = np.ones([self.ydim])
        self.terrain[self.movement_bounds[1]] = np.ones([self.ydim])
        self.terrain.T[self.movement_bounds[2]] = np.ones([self.xdim])
        self.terrain.T[self.movement_bounds[3]] = np.ones([self.xdim])

	#adds matching boundary polygon segments for isovist calculations
	point1 = (self.movement_bounds[1], self.movement_bounds[2])
	point2 = (self.movement_bounds[1], self.movement_bounds[3])
	point3 = (self.movement_bounds[0], self.movement_bounds[3])
	point4 = (self.movement_bounds[0], self.movement_bounds[2])
	segments = [[point1, point2], [point2,point3], [point3, point4], [point4,point1]]
	self.polygon_segments.append(segments)

    def initialize_terrain(self, num_obstacles, min_obst_dim, max_obst_dim):

	#simple world with rectangular obstacles
        self.terrain = np.zeros([self.xdim, self.ydim])

	#boundary to define traversible space
        self.add_terrain_frame()

	#matching polygons for isovist calculations
	polygon_segments = []
        for i in range(num_obstacles):
            start_x = rand.randint(0,self.xdim-min_obst_dim)
            start_y = rand.randint(0,self.ydim-min_obst_dim)
            end_x = start_x + rand.randint(min_obst_dim, max_obst_dim)
            if end_x > self.xdim-1:
                end_x = self.xdim-1
            end_y = start_y + rand.randint(min_obst_dim, max_obst_dim)
            if end_y > self.ydim-1:
                end_y = self.ydim
            self.terrain[start_x:end_x, start_y:end_y] = 1
	    point1 = (end_x, start_y)
	    point2 = (end_x, end_y)
	    point3 = (start_x, end_y)
	    point4 = (start_x, start_y)
	    segments = [[point1, point2], [point2,point3], [point3, point4], [point4,point1]]
	    polygon_segments.append(segments)
	#print polygon_segments
	#sys.exit()
	return polygon_segments


    def read_terrain_image(self, filename):
	
	#read image from file
        img = imread(filename)

        #use the red color channel for obstacles
        terrain = img[:,:,0] 
        terrain = terrain > 100
        self.terrain = terrain.T[:self.xdim, :self.ydim]
        
	#self.polygon_segments = self.isovist.getPolygonSegments()

	self.add_terrain_frame()

        #use the green color channel for sniper locations
        self.sniper_locations = img[:,:,1].T


    def within_movement_bounds(self, x, y):
        return (x>self.movement_bounds[0]) and (x<self.movement_bounds[1]) and (y>self.movement_bounds[2]) and (y<self.movement_bounds[3])


    def create_validity_map(self):
        validity_map = np.zeros((self.terrain.shape[0] + 1, self.terrain.shape[1] + 1))

        for y in range(self.terrain.shape[0]):
            for x in range(self.terrain.shape[1]):
                validity_map[y, x] = self.is_valid(x, y)

        return validity_map

    def is_valid(self, x, y):
        #a square is valid IFF if it is not too near the edge
	#of an obstacle or the edge of the screen.
        
        #original version; we're trying to be faster
        try:
            return self.validity_map[y, x]
        except AttributeError:
            return self.within_movement_bounds(x,y) and not np.any(self.terrain[x-SAFETY_MARGIN:x+SAFETY_MARGIN,y-SAFETY_MARGIN:y+SAFETY_MARGIN])
        
