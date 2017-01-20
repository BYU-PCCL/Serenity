import pygame
from pygame.locals import *
import random as rand
import numpy as np
import RRT 
import sys
import math



class DeathEater:

	def __init__(self, terrain, screen=None, clock=None):
		self.terrain = terrain
		self.xdim = len(self.terrain[0])
		self.ydim = len(self.terrain)
		self.screen = screen
		self.clock = clock

	'''
	Returns a list of newly rotated points for a polygon
	'''
	def RotateRect(self, points, rotation):
		rotPoints = []
		for p in points:
			x = p[0]
			y = p[1]

			rotx = x*math.cos(rotation) - y*math.sin(rotation)
			roty = x*math.sin(rotation) + y*math.cos(rotation)
			rotPoints.append(((rotx), (roty)))
		#print "Points:", points
		#print "Rotated Points: ", rotPoints
		return rotPoints

	'''
	Sample Map

	Returns an array of array of points
	each array in the array has the points for a polygon

	'''
	def SampleMap(self, Q):

		''' number of obstacles (min and max) '''
		min_obst_num = 200
		max_obst_num = 1000

		''' obstacles dimenstions (min and max) '''
		min_obst_dim= 10
		max_obst_dim= 85

		''' Currently storing polygons (currently rectangles) '''
		polygons = []
		
		num_obstacles = Q.randint(min_obst_num, max_obst_num)

		print num_obstacles
		for i in range(num_obstacles):
			''' Find random starting points'''
			start_x = Q.randint(0, self.xdim-min_obst_dim)
			start_y = Q.randint(0, self.ydim-min_obst_dim)

			'''Find random ending points (Within the min and max dimenstions)'''
			end_x = start_x + Q.randint(min_obst_dim, max_obst_dim)
			if end_x > self.xdim-1:
				end_x = self.xdim-1

			end_y = start_y + Q.randint(min_obst_dim, max_obst_dim)
			if end_y > self.ydim-1:
				end_y = self.ydim

			points = [(start_x, start_y),(start_x, end_y),(end_x, end_y),(end_x, start_y)]

			''' Rotate points '''
			rotation = Q.randint(-180, 180)
			rotated_points = self.RotateRect(points, math.radians(rotation))

			'''Add points into polygon list '''
			polygons.append(rotated_points)

		self.ShowPolygons(polygons)
		return polygons

	'''
	Draws the polygons onto the screen
	'''
	def ShowPolygons(self, polygons):
		if not self.screen == None:
			black = 20, 20, 40
			self.screen.fill(black)
			for p in polygons:
				#print "p:", p
				pygame.draw.polygon(self.screen, (255,255,255), p)
			pygame.display.update()

'''
Init Screen

Creates a pygame display

Returns a screen and a clock

'''

def InitScreen(xdim, ydim):
	pygame.init()
	pygame.font.init()

	size = (xdim, ydim)
	screen = pygame.display.set_mode(size)

	pygame.display.set_caption("Map Sampling")
	clock = pygame.time.Clock()

	return screen, clock


'''
	Collides returns true if pos ( a point )
	is within one of the polygons in the list of polygons

    This code is patterned after [Franklin, 2000]
    http://www.geometryalgorithms.com/Archive/algorithm_0103/algorithm_0103.htm
    Tells us if the point is in this polygon

'''
def Collides(pos, polygons):
    for points in polygons:
        
        cn = 0  # the crossing number counter
        pts = points[:]
        pts.append(points[0])
        for i in range(len(pts) - 1):
            if (((pts[i][1] <= pos[1]) and (pts[i+1][1] > pos[1])) or ((pts[i][1] > pos[1]) and (pts[i+1][1] <= pos[1]))):
                    if (pos[0] < pts[i][0] + float(pos[1] - pts[i][1]) / (pts[i+1][1] - pts[i][1]) * (pts[i+1][0] - pts[i][0])):
                            cn += 1
        if bool(cn % 2)==1:
            return True
    return False

'''
	returns a random point that does
	not collide with any obstacles

	usually used to find a random 
	starting point and random end point
	for RRT

'''

def GetGoodPoint(xdim, ydim, obs):
    while True:
        p = int(rand.random()*xdim), int(rand.random()*ydim)
        noCollision = Collides(p, obs)
        if noCollision == False:
            return p

'''
	Returns a random start and end point
	(for RRT's later use)
'''
def CreateRandPath(xdim, ydim, obstacles):
	start = GetGoodPoint(xdim, ydim, obstacles)
	end = GetGoodPoint(xdim, ydim, obstacles)

	#print "start: ", start, " end: ", end
	return start, end

'''
	Runs RRT with a random start and end point.
	This function simply prints the path found by RRT
'''

def RunARandSimulation(r, xdim, ydim, obstacles):
	start, end = CreateRandPath(xdim, ydim, obstacles)
	path = r.run(start, end, obstacles)
	print "PATH:", path


'''
	By default, this will randomly generate 10 sampled maps
	and show them one after the other on the pygame screen

	Meant to give an idea of how the sample maps look like
	in general
'''
def RunSampleMapSimulation(d, clock, Q, runSampleMapAmount=10):
	for i in xrange(runSampleMapAmount):
		obstacles = d.SampleMap(Q)
		Update()
		clock.tick(4)

'''
	Updates the pygame screen
	and allows for exiting of the pygame screen
'''

def Update():
	pygame.display.update()
	for e in pygame.event.get():
		if e.type == QUIT or (e.type == KEYUP and e.key == K_ESCAPE):
			sys.exit("Exiting")


'''
	main function

	paint = True : when you want a pygame visualization to appear ( is slower when simulating RRT)
	runRRT = True : when you want RRT to run. Otherwise, it won't find random points and find a path
	runMap = True : when you don't really want to run RRT, just want to run a map sampling. Meant to Map Sampling testing

'''
def main():

	'''
	Q function current set to rand

	
	'''
	Q = rand

	paint = True
	runRRT = False
	runMap = True

	'''
	xdim and ydim of the pygame screen 

	We start off with screen and clock = None in class 
	we don't want to paint to a screen
	'''
	xdim = 1000
	ydim = 1000
	array = np.zeros([xdim, ydim])
	screen = None
	clock = None

	if paint:
		screen, clock = InitScreen(xdim, ydim)

	d = DeathEater(array, screen=screen, clock=clock)

	'''
	We immediately create some obstacles. 
	'''

	obstacles = d.SampleMap(Q)

	'''
	If we want to run several samplings of maps and see them
	'''
	if runMap:
		RunSampleMapSimulation(d, clock, Q, runSampleMapAmount=10)

	'''
	If we want to see several runs of RRT
	Uses same obstacles
	Uses different start and end points for the search
	'''
	if runRRT:
		r = RRT.RRT(d.screen, d.clock, len(array), len(array[0]), paint=paint)

		runSimulationAmount = 1
		for i in xrange(runSimulationAmount):
			RunARandSimulation(r, xdim, ydim, obstacles)
	
	'''
	allows exiting of the pygame screen
	'''
	if paint:
		while True:
			Update()

if __name__ == '__main__':
    main()


