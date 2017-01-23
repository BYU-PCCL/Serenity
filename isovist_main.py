import math
import pygame
from pygame.locals import *
import random as rand
import numpy as np
import sys
import isovist as iso
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

	pygame.display.set_caption("Isovist")
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




'''
	main function

'''

def main():

	'''
	xdim and ydim of the pygame screen 
	'''
	xdim = 841
	ydim = 361

	array = np.zeros([xdim, ydim])
	screen, clock = InitScreen(xdim, ydim)
	polygonSegments = getPolygonSegments()

	# Clear canvas
	screen.fill((255,255,255))

	s = pygame.Surface((xdim,ydim))  	# the size of your rect
	s.set_alpha(0)                		# alpha level
	s.fill((255,255,255))           	# this fills the entire surface
	screen.blit(s, (0,0))

	# Draw segments
	for polygon in polygonSegments:
		for segment in polygon:
			pygame.draw.line(screen, (0, 0, 0), segment[0], segment[1],2)

	Update()

	isovist = iso.Isovist(polygonSegments)
	center = (420, 180)
	while True:
		for e in pygame.event.get():
			if e.type == QUIT or (e.type == KEYUP and e.key == K_ESCAPE):
				sys.exit("Exiting")

			# Clear canvas
			screen.fill((255,255,255))
			s = pygame.Surface((xdim,ydim))  # the size of your rect
			s.set_alpha(0)                   # alpha level
			s.fill((255,255,255))            # this fills the entire surface
			screen.blit(s, (0,0))

			# Draw segments ( map )
			for polygon in polygonSegments:
				for segment in polygon:
					pygame.draw.line(screen, (0, 0, 0), segment[0], segment[1],2)

			mouse = pygame.mouse.get_pos()
			pygame.draw.circle(screen, (100,100,100), center, 5)
			pygame.draw.circle(screen, (100,100,100), mouse, 5)

			# getting directions
			dirx = mouse[0] - center[0]
			diry = mouse[1] - center[1]
			direction = (dirx, diry)

			RRTPath = [mouse]
			UAVLocation = center
			UAVForwardVector = direction

			isIntruderFound = isovist.IsIntruderSeen(RRTPath, UAVLocation, UAVForwardVector, UAVFieldOfVision = 45)
			
			intruderColor = (255,0,0)
			if isIntruderFound:
				intruderColor = (0,255,0)

			# Draw Polygon for intersections (isovist)
			isovist_surface = pygame.Surface((xdim,ydim)) 
			isovist_surface.set_alpha(80)

			# JUST for drawing the isovist
			intersections = isovist.GetIsovistIntersections(UAVLocation, UAVForwardVector)
			pygame.draw.polygon(isovist_surface, intruderColor, intersections)
			screen.blit(isovist_surface, isovist_surface.get_rect())

			Update()
			pygame.time.delay(10)

		                    

if __name__ == '__main__':
    main()


