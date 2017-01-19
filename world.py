import math
import numpy as np
import random as rand
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
	#self.movement_bounds = [0,xdim-1,0,ydim-1]
	#self.movement_bounds = [150,xdim-151,150,ydim-151]
	self.movement_bounds = [int(0.3*xdim),int(0.82*xdim),int(0.28*ydim),int(0.8*ydim)]
	self.initialize_terrain(num_obstacles, min_obst_dim, max_obst_dim)
	#self.load_point_cloud('point_clouds/final_xyz.npy')
	#self.read_point_cloud_image('point_clouds/bremen_altstadt_final.png')
	self.create_valid_squares()
	self.initialize_treats(num_treats)
	self.isovist = isovist.Isovist(self.terrain)

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

    def is_valid(self, x, y):
        #a square is valid IFF if it is not too near the
	#edge or pixels in the terrain/point cloud
	
	#original version; we're trying to be faster
	return self.within_movement_bounds(x,y) and (np.sum(self.terrain[x-SAFETY_MARGIN:x+SAFETY_MARGIN,y-SAFETY_MARGIN:y+SAFETY_MARGIN]) == 0)
	
	#(no distinct speedup found using the function below, so it was dropped)
	#return self.within_movement_bounds(x,y) and (self.terrain[x][y] == 1)
	
	#by far the fasted way to update
	#return self.valid_squares[x][y]