import math
import pygame
from pygame.locals import *
import random as rand
import numpy as np
import sys
import isovist as iso
from numpy import atleast_2d

from my_rrt import *
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
		if e.type == MOUSEBUTTONDOWN:
		    print('mouse down', pygame.mouse.get_pos())
		    return pygame.mouse.get_pos()

def load_polygons_here( fn="./paths.txt" ):
    bdata = []
    for x in open( fn ):
        tmp = np.fromstring( x, dtype=float, sep=' ' )
        tmp = np.reshape( tmp/1000, (-1,2) )
        tmp = np.vstack(( np.mean(tmp, axis=0, keepdims=True), tmp, tmp[0,:] ))
        #tmp[:,1] = 1.0 - tmp[:,1]  # flip on the y axis
        bdata.append( tmp )
    return bdata

def load_polygons( fn="./paths.txt" ):
	polygonSegments = []
	for line in open( fn ):
		line = line.strip('\n')
		toList = line.split(' ')
		toList = [(float(x)/1000) for x in toList]
		
		it = iter(toList)
		toList = [toList[i:i+2] for i in range(0, len(toList), 2)]

		for pair in toList:
			#pair[1] = 1.0 - pair[1]
			pair[0] = int (pair[0] *500)
			pair[1] = int (pair[1] *500)

		#toList = [toList[i:i+2] for i in range(0, len(toList), 2)]
		#toList[-1].insert(0, toList[0][0])
		temp = []
		for i in xrange(1,len(toList)):
			pair = (toList[i-1], toList[i])
			temp.append(pair)
		temp.append((toList[0],toList[-1]))
		
		polygonSegments.append(temp)

	dim = 500

	'''border'''
	polygonSegments.append([ 
		[ (0,0),(1000,0) ], 
		[ (1000,0),(1000,1000) ],
		[ (1000,1000), (0,1000)],
		[ (0,1000), (0,0) ]
		])
        #print "toList:", toList
	# for p in polygonSegments:
	# 	print "\n", p
	return polygonSegments


'''
	main function

'''

def main():

	

	'''
	xdim and ydim of the pygame screen 
	'''
	xdim = 500
	ydim = 500
	backgroundFileName = "./cnts.png"
	background = pygame.image.load(backgroundFileName)
	background = pygame.transform.scale(background, (xdim, ydim))
	backgroundRect = background.get_rect()


	

	array = np.zeros([xdim, ydim])
	screen, clock = InitScreen(xdim, ydim)
	polygonSegments = load_polygons()

	# Clear canvas
	screen.fill((255,255,255))

	s = pygame.Surface((xdim,ydim))  	# the size of your rect
	s.set_alpha(0)                		# alpha level
	s.fill((255,255,255))           	# this fills the entire surface
	screen.blit(s, (0,0))

	screen.blit(background, backgroundRect)


	#### RRT STUFF
	start_paint = (int(0.1 *500),int(0.1 *500))
	end_paint = (int(0.9*500), int(0.9 *500))

	start = np.atleast_2d( [(0.1 ) ,(0.1 )] )
	end = np.atleast_2d( [(0.9 ),(0.9 )] )
	X1, Y1, X2, Y2 = polygons_to_segments(load_polygons_here())
	#print X1, Y1, X2, Y2
	# path = run_rrt( start, end, X1, Y1, X2, Y2)
	# print path
	# print "HERE"
	# for point in path:
	# 	print "POINT", point
	# 	point = (int(point[0]*500), int(point[1]*500))
	# 	pygame.draw.circle(screen, (255,100,255), point, 5)

	####

	# Draw segments
	for polygon in polygonSegments:
		for segment in polygon:
			pygame.draw.line(screen, (225, 225, 225), segment[0], segment[1] ,1)


	#RRTPath = [(200, 120), ( 450, 200)]
	#Draw hard coded RRT Path
	#pygame.draw.line(screen, (0, 0, 255), RRTPath[0], RRTPath[1],2)

	# for point in RRTPath:
	# 	pygame.draw.circle(screen, (255,100,255), point, 5)

	Update()

	isovist = iso.Isovist(polygonSegments)
	#(313, 115)
	agentx = 313
	agenty = 215
	UAVLocation = (agentx,agenty)
	mouseClick = None

	

	while True:
		if mouseClick != None:
			UAVLocation = mouseClick

		

		# Clear canvas
		# screen.fill((255,255,255))
		# s = pygame.Surface((xdim,ydim))  # the size of your rect
		# s.set_alpha(0)                   # alpha level
		# s.fill((255,255,255))            # this fills the entire surface
		# screen.blit(s, (0,0))
		screen.blit(background, backgroundRect)
		# Draw segments ( map )
		for polygon in polygonSegments:
			for segment in polygon:
				pygame.draw.line(screen, (225, 225, 225), segment[0], segment[1],1)


		mouse = pygame.mouse.get_pos()

		# for i in range(0, X1.shape[0]):
		# 	pygame.draw.line(screen, (225, 0, 0), [ X1[i], Y1[i] ], [ X2[i], Y2[i] ],1 )
		#print UAVLocation
		
		path = run_rrt( start, end, X1, Y1, X2, Y2)
		# print path
		# # print "HERE"
		# for point in path:
		# 	point = (int(point[0]*500), int(point[1]*500))
		# 	pygame.draw.circle(screen, (255,100,255), point, 5)
		readable_path = []
		for i in xrange(1, len(path)):
			s_point = path[i-1]
			s_point = (int(s_point[0]*500), int(s_point[1]*500))

			e_point = path[i]
			e_point = (int(e_point[0]*500), int(e_point[1]*500))
			pygame.draw.line(screen, (225, 225, 0), s_point, e_point, 1)

			readable_path.append(s_point)

		# getting directions
		dirx = mouse[0] - UAVLocation[0]
		diry = mouse[1] - UAVLocation[1]
		direction = (dirx, diry)

		#Draw hard coded RRT Path
		# pygame.draw.line(screen, (0, 0, 255), RRTPath[0], RRTPath[1],2)
		# for point in RRTPath:
		# 	pygame.draw.circle(screen, (255,100,255), point, 5)

		#pygame.draw.circle(screen, (255,255,255), start_paint, 15)
		#pygame.draw.circle(screen, (255,255,255), end_paint, 15)

		pygame.draw.circle(screen, (0,255,0), start_paint, 10)
		pygame.draw.circle(screen, (255,0,0), end_paint, 10)
		
		UAVForwardVector = direction

		isIntruderFound = isovist.IsIntruderSeen(readable_path, UAVLocation, UAVForwardVector, UAVFieldOfVision = 45)
		
		intruderColor = (255,0,0)
		if isIntruderFound:
			intruderColor = (0,255,0)

		# Draw Polygon for intersections (isovist)
		isovist_surface = pygame.Surface((xdim,ydim)) 
		isovist_surface.set_alpha(80)

		# JUST for drawing the isovist
		intersections = isovist.GetIsovistIntersections(UAVLocation, UAVForwardVector)
		if intersections != []:
			pygame.draw.polygon(isovist_surface, intruderColor, intersections)
			screen.blit(isovist_surface, isovist_surface.get_rect())

		pygame.draw.circle(screen, (100,100,100), UAVLocation, 5)
		#pygame.draw.circle(screen, (100,100,100), mouse, 5)

		#isox,isoy = UpdateMovement(isox,isoy)
		mouseClick = Update()
			
		pygame.time.delay(10)

		                    

if __name__ == '__main__':
    main()


