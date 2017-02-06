import math
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse import lil_matrix
from scipy.misc import imread, imsave, imresize
import pygame
import random as rand
import sys
from scipy import signal
import time

#DEPRECATED#
#import tensorflow as tf
#import pickle
#import os

import world
import copter
import intruder
import prob_mass

#HEADLESS
#if you're telnetting from home from a non-linx
#machine, set HEADLESS to True in order to view
#error messages
HEADLESS = False
#HEADLESS = True

#WORLD_TYPE = "bremen"
WORLD_TYPE = "blocks"

MODE = 1                 #0 = simple kernel, 1 = complex kernel
TREATS = 3                 #number of cookies/truffles/etc
MESSY_WORLD = True

#INTRUDER_TYPE = "RRT_INTRUDER"
INTRUDER_TYPE = "THIEF"
#INTRUDER_TYPE = "TRESPASSER"

if INTRUDER_TYPE == "RRT_INTRUDER":
    #RRT is too slow for a full policy rollout,
    ROLLOUT_EPOCHS = 1000
    ROLLOUT_TIME_STEPS = 1000

    #DEPRECATED#
    #UNIFORM_PRIOR = True
else:
    ROLLOUT_EPOCHS = 1000
    ROLLOUT_TIME_STEPS = 1000

    #DEPRECATED#
    #UNIFORM_PRIOR = False

#COLOR DEFINITIONS
BLACK    = (   0,   0,   0)
WHITE    = ( 255, 255, 255)
GREEN    = (   0, 255,   0)
DRK_GREEN = (   0, 100,   0)
RED      = ( 255,   0,   0)
BROWN      = ( 255,   200,  200)
BLUE     = (   0,   0, 255)
GRAY     = (  50,  50,  50)
DARK_GRY = (  15,  15,  15)
LITE_GRY = ( 100, 100, 100)

#GLOBAL CONSTANTS 
MAX_MOMENTUM = 5
INTRUDER_MOMENTUM = 3 	#sum of momentum plus intruder's jitter 
			#must be less than KERNEL_SIZE/2
KERNEL_SIZE = 11 	#must be an odd number 
ICON_SIZE = 10

if WORLD_TYPE == "bremen":
    XDIM=1000
    YDIM=1000
    OBSTACLES = 0	#not used, but needs to be declared
else:
    XDIM=500
    YDIM=500
    OBSTACLES = 15              

#DEPRECATED
#INTRUDER_TYPE = 1        #0 = momentum, 1 =waypoints

PAUSE_BETWEEN_TIME_STEPS = 0         #-1 prompts for input between steps
#USE_VECTOR_MATRIX_MULTIPLY = False
DOWNSAMPLE = 1                         #downsample factor for prior updates with complex kernel

SHOW_SIMPLE_KERNEL = False
SHOW_COMPLEX_KERNEL = False        #shows kernel at intruder's curreny xy coords

SHOW_INITIAL_PROBABILITY_MAP = False
SHOW_ISOVIST = True
SHOW_WORLD_TERRAIN = True
SHOW_POLYGONS = False
SHOW_COPTER_PATH = True

#COLOR_SCALE = XDIM*YDIM*2.5
COLOR_SCALE = XDIM*YDIM*10
KERNEL_COLOR_SCALE = 1200

#P_HEARD_SOMETHING_IF_NO_INTRUDER = 1e-5
#P_HEARD_SOMETHING_IF_INTRUDER = 0.999
P_HEARD_SOMETHING_IF_NO_INTRUDER = 0.0                #probablity of false pos.
P_HEARD_SOMETHING_IF_INTRUDER = 1.0                #probability of hearing intr.
P_SAW_SOMETHING_IF_NO_INTRUDER = 0.0                 #probability of false pos.
P_SAW_SOMETHING_IF_INTRUDER = 1.0                #probability of seeing intr.

IMAGE_CAPTURE_RATE = 1 #capture an image to disk ever n steps


def show_priors(priors):
    screen.fill(BLACK)
    img = pygame.surfarray.make_surface((priors*COLOR_SCALE).astype(int))
    screen.blit(img, img.get_rect())
    pygame.display.flip()
    raw_input("Press enter to continue")

def paint_to_screen(PRIORS, w, c, i, filename = "", isovist_color=WHITE):
    #UPDATE SCREEN DISPLAY 
    if HEADLESS != True:
        screen.fill(BLACK)
        
        #PROBABILITY MASS
        img = pygame.surfarray.make_surface((PRIORS*COLOR_SCALE).astype(int))
        if SHOW_INITIAL_PROBABILITY_MAP == True:
            print "DISPLAYING INITIAL PROBABILITY CONFIGURATION"
            img = pygame.surfarray.make_surface((i.geographic_probability_map*COLOR_SCALE).astype(int))
        screen.blit(img, img.get_rect())
        
        #WORLD TERRAIN
	if SHOW_WORLD_TERRAIN == True:
            img = pygame.surfarray.make_surface((w.terrain*1000).astype(int))
            img.set_alpha(70)
            screen.blit(img, img.get_rect())
        
	#POLYGON_MAP
	if SHOW_POLYGONS == True:
	    for p in w.polygon_map:
	        pygame.draw.polygon(screen, BROWN, p)

	#COPTER PATH
	if SHOW_COPTER_PATH == True:
 	    if len(c.path) > 0:
		pygame.draw.lines(screen, WHITE, False, [(c.x,c.y)] + c.path)

        #ISOVIST
        if SHOW_ISOVIST == True:
             #drone_isovist = w.isovist.FindIsovistForAgent(c.x,c.y)
             drone_isovist = w.isovist.GetIsovistIntersections((c.x,c.y), c.movement_vector(), c.isovist_angle)
             for point in drone_isovist:
                 #print point
                 x = point[0]
                 y = point[1]
                 pygame.draw.rect(screen, DRK_GREEN, [x - ICON_SIZE/4, y - ICON_SIZE/4, ICON_SIZE/2, ICON_SIZE/2])
             if len(drone_isovist) > 2:
                 isovist_surface = pygame.Surface((XDIM,YDIM))
                 isovist_surface = pygame.surfarray.make_surface((PRIORS*COLOR_SCALE).astype(int))
                 isovist_surface.set_alpha(80)
                 pygame.draw.polygon(isovist_surface, isovist_color, drone_isovist)
                 #pygame.draw.polygon(isovist_surface, c.isovist_color, drone_isovist)
                 screen.blit(isovist_surface, isovist_surface.get_rect())

        #TREATS
        #for k in w.cookies:
        #    img = pygame.image.load("imgs/cookie.png")
        #    img = pygame.transform.scale(img, (24,24))
        #    screen.blit(img, (k[0]-12, k[1]-12))

        #(I haven't implemented popcorn and truffles yet,
        #but if I did, here's some code for displaying
        #their location...)

        #for p in w.popcorn:
        #    pygame.draw.circle(screen, WHITE, [p[0], p[1]], 10)
        #for t in w.truffles:
        #    pygame.draw.circle(screen, BLUE, [t[0], t[1]], 10)
        
        #display the quadcopter's "hearing range"
        #screen.blit(c.hearing_square, (c.x - c.hearing_range/2, c.y - c.hearing_range/2))
        
        #COPTER AND INTRUDER
        pygame.draw.rect(screen, GREEN, [c.x - ICON_SIZE/2, c.y - ICON_SIZE/2, ICON_SIZE, ICON_SIZE])
        pygame.draw.rect(screen, RED, [i.x - ICON_SIZE/2, i.y - ICON_SIZE/2, ICON_SIZE, ICON_SIZE])


        #DEBUGGING TOOL: show the active kernel
        if SHOW_SIMPLE_KERNEL == True:
            kernel = i.simple_movement_kernel()
            kern = np.copy(kernel)
            kern = np.repeat(np.repeat(kern, 10, axis=0), 10, axis=1)
            SAVED_KERNEL = np.stack((kern, kern, kern), axis=-1)
            img = pygame.surfarray.make_surface((SAVED_KERNEL*KERNEL_COLOR_SCALE).astype(int))
            screen.blit(img, (10,10), img.get_rect())
        if SHOW_COMPLEX_KERNEL == True:
            kern = np.copy(i.movement_kernel(i.x, i.y))
            kern = np.repeat(np.repeat(kern, 10, axis=0), 10, axis=1)
            SAVED_KERNEL = np.stack((kern, kern, kern), axis=-1)
            img = pygame.surfarray.make_surface((SAVED_KERNEL*KERNEL_COLOR_SCALE).astype(int))
            screen.blit(img, (10,10), img.get_rect())


	if filename != "":
	    pygame.image.save(screen, filename)


        #FLIP THE DISPLAY
        #This must be done after all objects have been
        #drawn to the screen. Otherwise they won't be
        #visible.
        pygame.display.flip()


#MAIN EXECUTION LOOP
if HEADLESS != True:
    pygame.init()
    pygame.font.init()

print('Initializing world...')
w = world.World(XDIM, YDIM, TREATS, WORLD_TYPE, OBSTACLES)

print('Intruder Type: ' + INTRUDER_TYPE)
if INTRUDER_TYPE == "RRT_INTRUDER":
    i = intruder.Intruder(w, MESSY_WORLD, "cookies", INTRUDER_MOMENTUM)
    i_model = intruder.Thief(w, MESSY_WORLD, "cookies", INTRUDER_MOMENTUM)
elif INTRUDER_TYPE == "TRESPASSER":
    i = intruder.Trespasser(w, MESSY_WORLD, "cookies", INTRUDER_MOMENTUM)
    i_model = i
elif INTRUDER_TYPE == "THIEF": 
    i = intruder.Thief(w, MESSY_WORLD, "cookies", INTRUDER_MOMENTUM)
    i_model = i
else:
    print "UNKNOWN INTRUDER_TYPE"
    print INTRUDER_TYPE
    sys.exit()

c = copter.Copter(w)

print("Initializing priors")
pm = prob_mass.prob_mass(w,i,i_model,c,MODE,KERNEL_SIZE,ROLLOUT_EPOCHS,ROLLOUT_TIME_STEPS,DOWNSAMPLE)
#PRIORS = pb.PRIORS
c.set_initial_path(pm.PRIORS)

#print('Rolling out policy...')
#policy_rollout([i], ROLLOUT_EPOCHS, ROLLOUT_TIME_STEPS)
#PRIORS = i.geographic_probability_map
##PRIORS = np.ones([XDIM,YDIM])
##PRIORS = PRIORS/np.sum(PRIORS)


if HEADLESS != True:
    size = (XDIM, YDIM)
    screen = pygame.display.set_mode(size)
    pygame.display.set_caption("Heat Map")
    clock = pygame.time.Clock()

#while we haven't caught the intruder...
done = False

print('Entering main loop...')
counter = 0
while done != True:
    if HEADLESS != True:
        for event in pygame.event.get(): #User did something
            if event.type == pygame.QUIT: #User clicked close
                print("User asked to quit.")
                sys.exit()
#            elif event.type == pygame.KEYDOWN:
#               print("User pressed a key.")
#            elif event.type == pygame.KEYUP:
#                print("User let go of a key.")
#            elif event.type == pygame.MOUSEBUTTONDOWN:
#                print("User pressed a mouse button")

    c.step(pm.PRIORS)
    i.step()
    pm.step()

    #print "ISOVIST DEBUGGING:"
    #print WHITE
    #print pm.isovist_color

    counter += 1
    if counter % IMAGE_CAPTURE_RATE == 0:
	filename = "screen_captures/img"  + str(counter) + ".jpeg"
        paint_to_screen(pm.PRIORS, w, c, i, filename) 
    else:
	paint_to_screen(pm.PRIORS, w, c, i)
     
    if HEADLESS != True:
        clock.tick(60) #60 frames per second
        
    if PAUSE_BETWEEN_TIME_STEPS > 0:
        time.sleep(PAUSE_BETWEEN_TIME_STEPS)
    elif PAUSE_BETWEEN_TIME_STEPS == -1:
        raw_input("Press <ENTER> to continue")


#loop a bit after the intruder is caught
if HEADLESS != True:
    while True:
        pass
