import sys
import math
import random as rand
import numpy as np

import cv2
from scipy.misc import imread

import isovist

SAFETY_MARGIN = 3

def sigmoid(x):
    return 1.0 / (1.0 + np.exp( -x ) )

def crop(array, left_bound, right_bound, lower_bound, upper_bound):
    arr = []
    for i in range(array.shape[0]):
        if array[i][0] > left_bound and array[i][0] < right_bound:
            if array[i][1] > lower_bound and array[i][1] < upper_bound:
                arr.append(array[i])
    return np.array(arr)

def shift_and_scale(array, xdim, ydim):
    #shifts all xyz coords into a positive frame of reference
    #scales xy coords to fall within [xdim, ydim]
    x=array.T[0]
    y=array.T[1]
    z=array.T[2]

    x += -1*np.min(x)
    y += -1*np.min(y)
    z += -1*np.min(z)

    x *= xdim/np.max(x)
    y *= ydim/np.max(y)
    z *= ydim/np.max(z)         #scale between 1-ydim for now

def horizontal_slice(array, min_height=200, max_height=1000):
    #returns a horizontal slice including all points in the
    #desired altitude range
    arr = []
    for i in range(array.shape[0]):
        if array[i][2] > min_height and array[i][2] < max_height:
            arr.append(array[i])
    return np.array(arr)


class World:

    def __init__(self, xdim, ydim, num_treats, num_obstacles, min_obst_dim=10, max_obst_dim=100):
        self.xdim = xdim
        self.ydim = ydim
        self.movement_bounds = [0,xdim-1,0,ydim-1]
        #self.movement_bounds = [150,xdim-151,150,ydim-151]
        #self.movement_bounds = [int(0.3*xdim),int(0.82*xdim),int(0.28*ydim),int(0.8*ydim)]
        self.polygon_segments = self.initialize_terrain(num_obstacles, min_obst_dim, max_obst_dim)
        #self.load_point_cloud('point_clouds/final_xyz.npy')
        #self.read_point_cloud_image('point_clouds/bremen_altstadt_final.png')
        self.create_valid_squares()
        self.initialize_treats(num_treats)
        #self.isovist = isovist.Isovist(self.terrain)
	#self.polygon_map, self.polygon_segments = self.create_polygon_map()
	#self.polygon_segments = self.create_polygon_map()
        self.isovist = isovist.Isovist(self.polygon_segments)

        print('  Creating validity map...')
        self.validity_map = self.create_validity_map()
        print('  self.terrain', self.terrain)

        print('  Storing contours...')
        self.contours = self.detect_contours()

    def detect_contours(self):
        terrain = self.terrain.astype(np.uint8)
        terrain = cv2.erode(terrain, np.ones((2, 2)), iterations=1)
        terrain = cv2.dilate(terrain, np.ones((3, 3)), iterations=2)
        print('np.mean(terrain)', np.mean(terrain))
        terrain = cv2.cvtColor(cv2.cvtColor(terrain, cv2.COLOR_GRAY2BGR), cv2.COLOR_BGR2GRAY)
        contours, hierarchy = cv2.findContours(terrain, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        contour_image = np.zeros((1000, 1000, 3))
        cv2.drawContours(contour_image, contours, -1, (255, 255, 0), 1)
        cv2.imwrite('contours.png', np.rot90(np.flipud(contour_image), k=3))

        fixed_contours = []

        for contour in contours:
            fixed_contour = np.array([x[0] for x in contour])
            fixed_contours.append(fixed_contour)

        return fixed_contours

    def create_valid_squares(self):
        self.valid_squares = np.zeros([self.xdim,self.ydim])
        self.valid_squares[self.movement_bounds[0]:self.movement_bounds[1], self.movement_bounds[2]:self.movement_bounds[3]] = 1 
        self.valid_squares += self.terrain
        self.valid_squares = 1 - self.valid_squares

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
        self.terrain[self.movement_bounds[0]] = np.ones([self.ydim])
        self.terrain[self.movement_bounds[1]] = np.ones([self.ydim])
        self.terrain.T[self.movement_bounds[2]] = np.ones([self.xdim])
        self.terrain.T[self.movement_bounds[3]] = np.ones([self.xdim])

    def initialize_terrain(self, num_obstacles, min_obst_dim, max_obst_dim):
        #For now: simple obstacles
        #(later we'll use a point cloud)
        self.terrain = np.zeros([self.xdim, self.ydim])
        self.add_terrain_frame()
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
	    segments = [[point1, point2], [point3, point4]]
	    polygon_segments.append(segments)
	#print polygon_segments
	#sys.exit()
	return polygon_segments

    def create_polygon_map(self):
	return [[[(0, 0), (840, 0)], [(840, 0), (840, 360)], [(840, 360), (0, 360)], [(0, 360), (0, 0)]], [[(100, 150), (120, 50)], [(120, 50), (200, 80)], [(200, 80), (140, 210)], [(140, 210), (100, 150)]], [[(100, 200), (120, 250)], [(120, 250), (60, 300)], [(60, 300), (100, 200)]], [[(200, 260), (220, 150)], [(220, 150), (300, 200)], [(300, 200), (350, 320)], [(350, 320), (200, 260)]], [[(540, 60), (560, 40)], [(560, 40), (570, 70)], [(570, 70), (540, 60)]], [[(650, 190), (760, 170)], [(760, 170), (740, 270)], [(740, 270), (630, 290)], [(630, 290), (650, 190)]], [[(600, 95), (780, 50)], [(780, 50), (680, 150)], [(680, 150), (600, 95)]]]

	"""polygons = []
	polygon_segments = []

	for x in open('point_clouds/building_polygons.txt'):
	    polygon_points = np.fromstring(x, dtype=float, sep=' ')
	    #polygon_points[:,1] = 1000-polygon_points[:,1] #flip on the y axis
	    segment_list = []
	    for i in range(len(polygon_points)-3):
		segment_list.append([(polygon_points[i], polygon_points[i+1]), (polygon_points[i+2],polygon_points[i+3])])
	    segment_list.append([(polygon_points[-2], polygon_points[-1]), (polygon_points[0], polygon_points[1])])
	    polygon_segments.append(segment_list)
	    polygons.append(polygon_points)
	return polygons, polygon_segments"""

    def load_point_cloud(self, filename):
        print "LOADING POINT CLOUD..."
        xyz = np.load(filename)

        #clip out all of the ground points
        gnd_pts = xyz[:,2] <= 0.0

        # normalize and clip out the interesting bits
        xy = xyz[ ~gnd_pts, 0:2 ]

        xy -= np.min(xy)
        xy /= np.max(xy)

        xy[ xy[:,0] < 0.30, 0 ] = 0.30
        xy[ xy[:,0] > 0.65, 0 ] = 0.65

        xy[ xy[:,1] < 0.4, 1 ] = 0.4
        xy[ xy[:,1] > 0.8, 1 ] = 0.8

        xy -= np.min(xy,axis=0)
        xy /= np.max(xy)

        # now all interesting points are in [0,1]x[0,1]

        #
        # generate an image sized to match the world
        #

        xy *= [1000.0, 1000.0]

        inds = xy[:,0].astype(int) + 1000*(xy[:,1].astype(int))
        inds = inds.astype( int )

        # compute a density
        cnts = np.bincount( inds, minlength=1000*1000 )
        cnts = cnts[0:1000*1000]

        img = np.reshape( cnts, (1000,1000) )
        img = np.flipud( img )

        """x_max = np.max(xyz.T[0])
        y_max = np.max(xyz.T[1])
        x_min = np.min(xyz.T[0])
        y_min = np.min(xyz.T[1])
        #xyz = crop(xyz, 0.6*x_min, 0.5*x_max, 0.5*y_min, 0.5*y_max)
        shift_and_scale(xyz, self.xdim-1, self.ydim-1)
        xyz = horizontal_slice(xyz, 100, 200)"""

        """region_map = np.zeros([self.xdim, self.ydim])
        for i in range(xyz.shape[0]):
                 region_map[int(xyz[i][0])][(self.ydim-int(xyz[i][1])-1)] = 1

        #drop any "obstacles" that don't have a minimum mass (?)"""

        self.terrain = np.copy(img)
        #self.terrain = np.copy(sigmoid(-5.0 + img/10.0))
        self.add_terrain_frame()

    def read_point_cloud_image(self, filename):
        img = imread(filename)

        #use the red color channel for obstacles
        terrain = img[:,:,0] 
        terrain = terrain > 100
        self.terrain = terrain.T[:self.xdim, :self.ydim]
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
        #a square is valid IFF if it is not too near the
        #edge or pixels in the terrain/point cloud
        
        #original version; we're trying to be faster
        try:
            return self.validity_map[y, x]
        except AttributeError:
            return self.within_movement_bounds(x,y) and not np.any(self.terrain[x-SAFETY_MARGIN:x+SAFETY_MARGIN,y-SAFETY_MARGIN:y+SAFETY_MARGIN])
        
        #(no distinct speedup found using the function below, so it was dropped)
        #return self.within_movement_bounds(x,y) and (self.terrain[x][y] == 1)
        
        #by far the fasted way to update
        #return self.valid_squares[x][y]
