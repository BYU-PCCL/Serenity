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


	# Sample Map
	#
	# Returns an array pygame rectangles
	def SampleMap_old(self, max_obst_num = 15, min_obst_dim=10, max_obst_dim=70):

		# Currently storing rectangles
		polygons = []

		terrain = np.zeros([self.xdim, self.ydim])
		
		num_obstacles = rand.randint(1, max_obst_num)
		for i in range(num_obstacles):
			# Find random starting points
			start_x = rand.randint(0, self.xdim-min_obst_dim)
			start_y = rand.randint(0, self.ydim-min_obst_dim)

			# Find random ending points (Within the min and max dimenstions)
			end_x = start_x + rand.randint(min_obst_dim, max_obst_dim)
			if end_x > self.xdim-1:
				end_x = self.xdim-1

			end_y = start_y + rand.randint(min_obst_dim, max_obst_dim)
			if end_y > self.ydim-1:
				end_y = self.ydim

			obstacle = pygame.Rect((start_x, start_y),(end_x, end_y))

			polygons.append(obstacle)
			#print start_x, start_y, " & ",  start_x, end_y, " & ", end_x, end_y, " & ", end_x, start_y
			#polygons.append(( (start_x, start_y),(start_x, end_y),(end_x, end_y),(end_x, start_y) ))
		
		self.ShowRectangles(polygons)
		return polygons

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


	# Sample Map
	#
	# Returns an array pygame rectangles
	def SampleMap(self, min_obst_num = 200, max_obst_num = 1000, min_obst_dim=10, max_obst_dim=70):

		# Currently storing rectangles
		polygons = []

		terrain = np.zeros([self.xdim, self.ydim])
		
		num_obstacles = rand.randint(min_obst_num, max_obst_num)
		print num_obstacles
		for i in range(num_obstacles):
			# Find random starting points
			start_x = rand.randint(0, self.xdim-min_obst_dim)
			start_y = rand.randint(0, self.ydim-min_obst_dim)

			# Find random ending points (Within the min and max dimenstions)
			end_x = start_x + rand.randint(min_obst_dim, max_obst_dim)
			if end_x > self.xdim-1:
				end_x = self.xdim-1

			end_y = start_y + rand.randint(min_obst_dim, max_obst_dim)
			if end_y > self.ydim-1:
				end_y = self.ydim

			#print start_x, start_y, " & ",  start_x, end_y, " & ", end_x, end_y, " & ", end_x, start_y
			#polygons.append(( (start_x, start_y),(start_x, end_y),(end_x, end_y),(end_x, start_y) ))
			
			points = [(start_x, start_y),(start_x, end_y),(end_x, end_y),(end_x, start_y)]
			rotation = rand.randint(-180, 180)
			rotated_points = self.RotateRect(points, math.radians(rotation))
			polygons.append(rotated_points)

		self.ShowRectangles(polygons)
		return polygons

	def ShowRectangles(self, polygons):
		if not self.screen == None:
			black = 20, 20, 40
			self.screen.fill(black)
			for p in polygons:
				#print "p:", p
				pygame.draw.polygon(self.screen, (255,255,255), p)
			pygame.display.update()

def InitScreen(xdim, ydim):
	pygame.init()
	pygame.font.init()

	size = (xdim, ydim)
	screen = pygame.display.set_mode(size)

	pygame.display.set_caption("Rectangles")
	clock = pygame.time.Clock()

	return screen, clock


def Collides(pos,polygons):
    for points in polygons:
        """
        This code is patterned after [Franklin, 2000]
        http://www.geometryalgorithms.com/Archive/algorithm_0103/algorithm_0103.htm
        Tells us if the point is in this polygon
        """
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

def GetGoodPoint(xdim, ydim, obs):
    while True:
        p = int(rand.random()*xdim), int(rand.random()*ydim)
        noCollision = Collides(p, obs)
        if noCollision == False:
            return p

def CreateRandPath(xdim, ydim, obstacles):
	start = GetGoodPoint(xdim, ydim, obstacles)
	end = GetGoodPoint(xdim, ydim, obstacles)


	print "start: ", start, " end: ", end
	return start, end

def RunARandSimulation(r, xdim, ydim, obstacles):
	start, end = CreateRandPath(xdim, ydim, obstacles)
	path = r.run(start, end, obstacles)
	print "PATH:", path

def RunSampleMapSimulation(d, clock, runSampleMapAmount=10):
	for i in xrange(runSampleMapAmount):
		obstacles = d.SampleMap(min_obst_dim=10, max_obst_dim=85)
		Update()
		clock.tick(4)

def Update():
	pygame.display.update()
	for e in pygame.event.get():
		if e.type == QUIT or (e.type == KEYUP and e.key == K_ESCAPE):
			sys.exit("Exiting")

def main():
	paint = True
	runRRT = True
	runMap = False
	xdim = 1000
	ydim = 1000
	array = np.zeros([xdim, ydim])
	screen = None
	clock = None

	if paint:
		screen, clock = InitScreen(xdim, ydim)

	d = DeathEater(array, screen=screen, clock=clock)

	obstacles = d.SampleMap( min_obst_dim=10, max_obst_dim=85)

	if runMap:
		RunSampleMapSimulation(d, clock, runSampleMapAmount=10)

	if runRRT:
		r = RRT.RRT(d.screen, d.clock, len(array), len(array[0]), paint=paint)

		runSimulationAmount = 1
		for i in xrange(runSimulationAmount):
			RunARandSimulation(r, xdim, ydim, obstacles)
	

	if paint:
		while True:
			Update()

if __name__ == '__main__':
    main()


