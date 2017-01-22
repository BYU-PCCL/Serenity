import pygame
from pygame.locals import *
import random as rand
import numpy as np
import sys
import math



class Isovist:

	def __init__(self, screen=None, clock=None):
		self.screen = screen
		self.clock = clock

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

def getPolygonSegments():
	polygonSegments = []
	
	'''border'''
	polygonSegments.append([ 
		[ (0,0),(840,0) ], 
		[ (840,0),(840,360) ],
		[ (840,360), (0,360)],
		[ (0,360), (0,0) ]
		])

	'''polygon #1'''
	polygonSegments.append([ 
		[ (100,150),(120,50) ], 
		[ (120,50),(200,80) ],
		[ (200,80), (140,210)],
		[ (140,210), (100,150) ]
		])

	'''polygon #2'''
	polygonSegments.append([ 
		[ (100,200),(120,250) ], 
		[ (120,250),(60,300) ],
		[ (60,300), (100,200)]
		])

	'''polygon #3'''
	polygonSegments.append([ 
		[ (200,260),(220,150) ], 
		[ (220,150),(300,200) ],
		[ (300,200), (350,320)],
		[ (350,320), (200,260) ]
		])

	'''polygon #4'''
	polygonSegments.append([ 
		[ (540,60),(560,40) ], 
		[ (560,40),(570,70) ],
		[ (570,70), (540,60)]
		])

	'''polygon #5'''
	polygonSegments.append([ 
		[ (650,190),(760,170) ], 
		[ (760,170),(740,270) ],
		[ (740,270), (630,290)],
		[ (630,290), (650,190) ]
		])

	'''polygon #6'''
	polygonSegments.append([ 
		[ (600,95),(780,50) ], 
		[ (780,50),(680,150) ],
		[ (680,150), (600,95)]
		])

	return polygonSegments


def GetIntersection(ray, segment):
	
	# RAY in parametric: Point + Direction * T1
	r_px = ray[0][0]
	r_py = ray[0][1]

	# direction
	r_dx = ray[1][0] - ray[0][0]
	r_dy = ray[1][1] - ray[0][1]

	# SEGMENT in parametric: Point + Direction*T2
	s_px = segment[0][0]
	s_py = segment[0][1]

	# direction
	s_dx = segment[1][0] - segment[0][0]
	s_dy = segment[1][1] - segment[0][1]

	
	r_mag = math.sqrt(r_dx ** 2 + r_dy ** 2)
	s_mag = math.sqrt(s_dx ** 2 + s_dy ** 2)



	# PARALLEL - no intersection
	if (r_dx/r_mag) == (s_dx/s_mag):
		if (r_dy/r_mag) == (s_dy/s_mag):
			return None, None
	

	denominator = float( -s_dx*r_dy + r_dx*s_dy )
	if denominator == 0:
		return None, None

	T1 = (-r_dy * (r_px - s_px) + r_dx * ( r_py - s_py)) / denominator
	T2 = (s_dx * ( r_py - s_py) - s_dy * ( r_px - s_px)) / denominator


	if T1 >= 0 and T1 <= 1 and T2 >= 0 and T2 <= 1:
		#Return the POINT OF INTERSECTION
		x = r_px+r_dx*T2
		y = r_py+r_dy*T2
		param = T2
		return (int(x),int(y)), param

	return None, None


def GetUniquePoints(polygons):
	points = []
	for polygon in polygons:
		for segment in polygon:
			if segment[0] not in points:
				points.append(segment[0])
			if segment[1] not in points:
				points.append(segment[1])
	return points

def GetUniqueAngles(uniquePoints, center):
	uniqueAngles = []
	for point in uniquePoints:
		angle = math.atan2(point[1]-center[1], point[0]-center[0])
		uniqueAngles.append(angle)
		#uniqueAngles.append(angle-0.00001)
		#uniqueAngles.append(angle+0.00001)
	return uniqueAngles

def SortIntoPolygonPoints(points):
	#print "Points:", points
	points.sort(compare)
	return points


def compare(a, b):
	mouse = pygame.mouse.get_pos()

	a_row = a[0]
	a_col = a[1]

	b_row = b[0]
	b_col = b[1]

	a_vrow = a_row - mouse[0]
	a_vcol = a_col - mouse[1]

	b_vrow = b_row - mouse[0]
	b_vcol = b_col - mouse[1]

	a_ang = math.degrees(math.atan2(a_vrow, a_vcol))
	b_ang = math.degrees(math.atan2(b_vrow, b_vcol))

	if a_ang < b_ang:
	    return -1

	if a_ang > b_ang:
	    return 1

	return 0 


def main():

	

	paint = True

	'''
	xdim and ydim of the pygame screen 

	We start off with screen and clock = None in class 
	we don't want to paint to a screen
	'''
	xdim = 841
	ydim = 361

	array = np.zeros([xdim, ydim])
	screen = None
	clock = None


	
	screen, clock = InitScreen(xdim, ydim)
	#bg = pygame.image.load("background.png")
	#screen.blit(bg, (0, 0))
	polygonSegments = getPolygonSegments()

	# Clear canvas
	screen.fill((255,255,255))

	s = pygame.Surface((xdim,ydim))  # the size of your rect
	s.set_alpha(0)                # alpha level
	s.fill((255,255,255))           # this fills the entire surface
	screen.blit(s, (0,0))

	# Draw segments
	for polygon in polygonSegments:
		for segment in polygon:
			pygame.draw.line(screen, (0, 0, 0), segment[0], segment[1],2)

	

	Update()
	center = (420,180)

	''' Get Unique Points '''
	uniquePoints = GetUniquePoints(polygonSegments)
	

	while True:
		for e in pygame.event.get():
			if e.type == QUIT or (e.type == KEYUP and e.key == K_ESCAPE):
				sys.exit("Exiting")

			# Clear canvas
			screen.fill((255,255,255))

			s = pygame.Surface((xdim,ydim))  # the size of your rect
			s.set_alpha(0)                # alpha level
			s.fill((255,255,255))           # this fills the entire surface
			screen.blit(s, (0,0))

			# Draw segments
			for polygon in polygonSegments:
				for segment in polygon:
					pygame.draw.line(screen, (0, 0, 0), segment[0], segment[1],2)

			
			
			mouse = pygame.mouse.get_pos()
			#center = mouse

			# Get unique angles based on unique points
			uniqueAngles = GetUniqueAngles(uniquePoints, center)
			#print "Unique Angles:", uniqueAngles

			#pygame.draw.line(screen, (100, 100, 100), center, mouse)
			pygame.draw.circle(screen, (100,100,100), mouse, 5)
			
			intersections = []
			for angle in uniqueAngles:

				# Calculate dx & dy from angle
				dx = math.cos(angle) * 2000
				dy = math.sin(angle) * 2000

				# Ray from center of screen to mouse
				ray = [ mouse , (mouse[0]+dx, mouse[1]+dy) ]

				# Find CLOSEST intersection
				closestIntersect = None
				closestParam = 10000000

				for polygon in polygonSegments:
					for segment in polygon:
						intersect, param = GetIntersection(ray, segment)
						
						if intersect != None:
							if closestIntersect == None or param < closestParam:
								closestIntersect = intersect
								closestParam = param
				
				if closestIntersect != None:
					intersections.append(closestIntersect)

			for inter in intersections:
				pygame.draw.line(screen, (255, 0, 0), mouse, inter,2)
				pygame.draw.circle(screen, (0,0,0), inter, 5)

			points = SortIntoPolygonPoints(intersections)

			isovist_surface = pygame.Surface((xdim,ydim)) 
			isovist_surface.set_alpha(80)
			pygame.draw.polygon(isovist_surface, (255,0,0), points)
			screen.blit(isovist_surface, isovist_surface.get_rect())


			Update()
			pygame.time.delay(10)

		                    

if __name__ == '__main__':
    main()


