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
import tensorflow as tf

import world
import copter
import intruder2

#HEADLESS
#if you're telnetting from home from a non-linx
#machine, set HEADLESS to True in order to view
#error messages
HEADLESS = False
#HEADLESS = True

#COLOR DEFINITIONS
BLACK    = (   0,   0,   0)
WHITE    = ( 255, 255, 255)
GREEN    = (   0, 255,   0)
DRK_GREEN = (   0, 100,   0)
RED      = ( 255,   0,   0)
BROWN      = ( 150,   50,  50)
BLUE     = (   0,   0, 255)
GRAY     = (  50,  50,  50)
DARK_GRY = (  15,  15,  15)
LITE_GRY = ( 100, 100, 100)

#GLOBAL CONSTANTS 
MAX_MOMENTUM = 10 
INTRUDER_MOMENTUM = 3 #sum of momentum plus intruder's jitter must be less than KERNEL_SIZE/2
KERNEL_SIZE = 11 #must be an odd number 
ICON_SIZE = 10

#XDIM = 500
#YDIM = 500
#XDIM = 1400
#YDIM = 1000
XDIM=500
YDIM=500

MODE = 0                 #0 = simple kernel, 1 = complex kernel
TREATS = 3                 #number of cookies/truffles/etc
OBSTACLES = 10                #number of obstacles on the world map
INTRUDER_TYPE = 1        #0 = momentum, 1 =waypoints
MESSY_WORLD = True

UNIFORM_PRIOR = False		#if true, movement kernels will not be built
ROLLOUT_EPOCHS = 1000
ROLLOUT_TIME_STEPS = 1000
PAUSE_BETWEEN_TIME_STEPS = 0         #-1 prompts for input between steps
USE_VECTOR_MATRIX_MULTIPLY = False
DOWNSAMPLE = 2                         #downsample factor for prior updates

SHOW_SIMPLE_KERNEL = False
SHOW_COMPLEX_KERNEL = False        #shows kernel at intruder's curreny xy coords

SHOW_INITIAL_PROBABILITY_MAP = False
SHOW_ISOVIST = True
SHOW_WORLD_TERRAIN = True
SHOW_POLYGONS = False
SHOW_COPTER_PATH = True


COLOR_SCALE = XDIM*YDIM*2.5
KERNEL_COLOR_SCALE = 1200

#P_HEARD_SOMETHING_IF_NO_INTRUDER = 1e-5
#P_HEARD_SOMETHING_IF_INTRUDER = 0.999
P_HEARD_SOMETHING_IF_NO_INTRUDER = 0.0                #probablity of false pos.
P_HEARD_SOMETHING_IF_INTRUDER = 1.0                #probability of hearing intr.
P_SAW_SOMETHING_IF_NO_INTRUDER = 0.0                 #probability of false pos.
P_SAW_SOMETHING_IF_INTRUDER = 1.0                #probability of seeing intr.


#SOME TENSORFLOW CALCULATIONS TO LET US ENGAGE THE GPU

#calculations for predict_intruder_location(), simple kernel
tf_priors = tf.placeholder(tf.float32, [1, None, None, 1], name='priors')
tf_simple_kernel = tf.placeholder(tf.float32,[KERNEL_SIZE,KERNEL_SIZE,1,1], name='simple_kernel')

tf_simple_kernel_convolution = tf.nn.conv2d(tf_priors, tf_simple_kernel, [1,1,1,1], "SAME")


#calculations for predict_intruder_location(), complex kernel
#(In trials, this ran much more slowly than the numpy version
#so it was cut.)
tf_complex_priors_segment = tf.placeholder(tf.float32, [None, None], name='priors')
#tf_complex_kernel = tf.placeholder(tf.float32,[None,None,KERNEL_SIZE,KERNEL_SIZE], name='complex_kernel')
tf_kernel_slice = tf.placeholder(tf.float32,[None,None], name='kernel')
tf_prob = tf.placeholder(tf.float32, shape=(), name='relative_coordinates') #adjusted i,j values

tf_new_priors = tf_complex_priors_segment + tf_prob*tf_kernel_slice       

sess = tf.Session()




#FUNCTION DEFINITIONS
def intruder_in_hearing_range(c,i):
    if (c.x - c.hearing_range/2) < i.x and i.x < (c.x + c.hearing_range/2):
        if (c.y - c.hearing_range/2) < i.y and i.y < (c.y + c.hearing_range/2):
                return True
    return False


def heardSomething(c, i):
    if intruder_in_hearing_range(c,i):            
        return rand.random() < P_HEARD_SOMETHING_IF_INTRUDER
    else:
        return rand.random() < P_HEARD_SOMETHING_IF_NO_INTRUDER


def policy_rollout(intruder_list):
    print "BEGINNING POLICY ROLLOUT..."

    for intruder in intruder_list:
         #establish a base kernel that will model the intruder behavior
        simple_kernel = np.zeros([KERNEL_SIZE,KERNEL_SIZE], dtype=np.float32) 
        
        #also create a complex kernel construct: a separate
        #movement kernel for each x,y location                
        kernel = np.zeros([intruder.xdim, intruder.ydim, KERNEL_SIZE,KERNEL_SIZE], dtype=np.float32)
        midpoint = KERNEL_SIZE/2

        if UNIFORM_PRIOR == True:
            geographic_probability_map = np.ones([intruder.xdim,intruder.ydim], dtype=np.float32)
        else:

            #TRANSITION MATRIX
            #represents the probability of transferring from any given
            #state (x,y location) to any other state
            if USE_VECTOR_MATRIX_MULTIPLY == True:
                x_offset = intruder.my_world.movement_bounds[0]
                y_offset = intruder.my_world.movement_bounds[2]
                xdim = intruder.my_world.movement_bounds[1] - intruder.my_world.movement_bounds[0] + 1
                ydim = intruder.my_world.movement_bounds[3] - intruder.my_world.movement_bounds[2] + 1
                #transition_matrix = lil_matrix(np.zeros([xdim*ydim/DOWNSAMPLE, xdim*ydim/DOWNSAMPLE], dtype=np.float32))
                transition_matrix = np.zeros([xdim*ydim/DOWNSAMPLE, xdim*ydim/DOWNSAMPLE], dtype=np.float32)

            #this represents the overall probability of the intruder 
            #being at any given x,y coordinate
            geographic_probability_map = np.zeros([intruder.xdim,intruder.ydim], dtype=np.float32)

            for i in range(int(ROLLOUT_EPOCHS)):
                #start the intruder at a random location
                intruder.select_random_location()
                intruder.select_waypoint()

                for j in range(int(ROLLOUT_TIME_STEPS)):

                    #move the intruder to a new square
                    prev_x = intruder.x
                    prev_y = intruder.y
                    intruder.step()

                    #determine the kernel location that corresponds
                    #to the intruder's new position
                    delta_x = intruder.x-prev_x
                    delta_y = intruder.y-prev_y

                    #update simple and complex kernels
                    try:
                        simple_kernel[midpoint+delta_x][midpoint+delta_y] += 1
                        kernel[prev_x][prev_y][midpoint + delta_x][midpoint+delta_y] += 1
                    except IndexError:
                        print('Intruder stepped too far.')
                    #update overall probability map
                    geographic_probability_map[intruder.x][intruder.y] += 1

                    #update transition matrix
                    if USE_VECTOR_MATRIX_MULTIPLY == True:
                        #transition_matrix[((prev_x-x_offset)*ydim+(prev_y-y_offset))/DOWNSAMPLE , ((intruder.x-x_offset)*ydim+(intruder.y-y_offset))/DOWNSAMPLE] += 1
                        transition_matrix[((prev_x-x_offset)*ydim+(prev_y-y_offset))/DOWNSAMPLE][((intruder.x-x_offset)*ydim+(intruder.y-y_offset))/DOWNSAMPLE] += 1
                    

        #normalize the simple kernel
        intruder.simple_kernel = simple_kernel/(np.sum(simple_kernel)+1e-100)

        #normalize each x,y kernel in the complex kernel
  #       for i in range(0, XDIM):
  #           for j in range(0,YDIM):
                # if np.sum(kernel[i][j]) != 0:
  #                   kernel[i][j] = kernel[i][j]/np.sum(kernel[i][j])

        kernel_sums = np.sum(kernel, axis=(2, 3)) # (1000, 1000)
        not_zero_positions = np.where(kernel_sums != 0)
        kernel[not_zero_positions] /= np.expand_dims(np.expand_dims(kernel_sums[not_zero_positions], axis=2), axis=2)

        intruder.kernel = kernel

        #normalize the overall probability map
        intruder.geographic_probability_map = geographic_probability_map/np.sum(geographic_probability_map)

        #normalize transition matrix
        if USE_VECTOR_MATRIX_MULTIPLY == True:
            sums = transition_matrix.sum(axis=1)
            for i in range(transition_matrix.shape[0]):
                my_sum = sums[i]
                if my_sum > 0:
                    transition_matrix[i] = transition_matrix[i]/my_sum
            #intruder.transition_matrix = csr_matrix(transition_matrix)
            intruder.transition_matrix = transition_matrix

def predict_intruder_location_optimized(priors, intruder, mode, w, x_offset=0, y_offset=0):
    new_priors = np.copy(priors)
    xlim, ylim = new_priors.shape[0], new_priors.shape[1]

    if mode == 0: #simple kernel - a single kernel summarizes intruder behavior
        
        new_priors = sess.run(tf_simple_kernel_convolution, feed_dict={tf_priors:np.reshape(new_priors,[1,xlim,ylim,1]),tf_simple_kernel:np.reshape(intruder.simple_movement_kernel(),[KERNEL_SIZE,KERNEL_SIZE,1,1])})
        
        return np.reshape(new_priors, [xlim,ylim])

    else: #complex kernel - a separate movement kernel for each xy location

        print [xlim, ylim]

        #convert the priors into a vector
        #priors_vector = csr_matrix(new_priors.reshape([xlim*ylim]))
        priors_vector = new_priors.reshape([xlim*ylim])

        #print "vector shapes"
        #print priors_vector.toarray().shape
        #print intruder.transition_matrix.toarray().shape

        #downsample if desired:
        if DOWNSAMPLE > 1:
           #downsampled_vector = csr_matrix(np.zeros([xlim*ylim/DOWNSAMPLE]))
           downsampled_vector = np.zeros([xlim*ylim/DOWNSAMPLE])
           for val in range(0, xlim*ylim, DOWNSAMPLE):
              downsampled_vector[val/DOWNSAMPLE] = (priors_vector[val:val+DOWNSAMPLE].sum())
        
           #multiply the vector with the transistion matrix
           new_downsampled_vector = downsampled_vector.dot(intruder.transition_matrix) 
           new_priors = np.zeros([xlim*ylim])
           for val in range(0, new_downsampled_vector.shape[0]):
                new_priors[val/DOWNSAMPLE:val/DOWNSAMPLE+DOWNSAMPLE] = downsampled_vector[val]
        else:
            #multiply the vector with the transistion matrix
            #new_priors_vector = np.dot(priors_vector,intruder.transition_matrix) 
            new_priors_vector = priors_vector.dot(intruder.transition_matrix) 
        #reshape back into 2D array
        #return np.reshape(new_priors_vector.toarray(), [xlim,ylim])
        return np.reshape(new_priors_vector, [xlim,ylim])

def predict_intruder_location(priors, intruder, mode, w, x_offset=0, y_offset=0):
    new_priors = np.copy(priors)
    xlim, ylim = new_priors.shape[0], new_priors.shape[1]

    if mode == 0: #simple kernel - a single kernel summarizes intruder behavior
        
        new_priors = sess.run(tf_simple_kernel_convolution, feed_dict={tf_priors:np.reshape(new_priors,[1,xlim,ylim,1]),tf_simple_kernel:np.reshape(intruder.simple_movement_kernel(),[KERNEL_SIZE,KERNEL_SIZE,1,1])})
        
        return np.reshape(new_priors, [xlim,ylim])

    else: #complex kernel - a separate movement kernel for each xy location

        complex_kernel = intruder.complex_movement_kernel()
        midpoint = KERNEL_SIZE/2        #all xy kernels have the same shape,
                        
        kx_min, kx_max, ky_min, ky_max = 0, KERNEL_SIZE, 0, KERNEL_SIZE

        for i in range(0+rand.randint(0, DOWNSAMPLE-1),xlim,DOWNSAMPLE):
            xmin = max(0,i-midpoint)
            xmax = min(i+1+midpoint, xlim)
            kx_min = 0
            kx_max = KERNEL_SIZE
            if i-midpoint <= 0:
                kx_min = midpoint - i
            if i+midpoint >= xlim:
                kx_max = xlim - i + midpoint

            for j in range(0+rand.randint(0,DOWNSAMPLE-1),ylim,DOWNSAMPLE):
                if w.is_valid(i+x_offset,j+y_offset):
                    ymin = max(0,j-midpoint)
                    ymax = min(j+1+midpoint, ylim)
                    ky_min = 0
                    ky_max = KERNEL_SIZE
                    if j-midpoint <= 0:
                        ky_min = midpoint - j
                    if j+midpoint >= ylim:
                        ky_max = ylim - j + midpoint

                    #get the movement kernel associated with
                    #this x,y position
                    kernel = complex_kernel[i+x_offset][j+y_offset]

                    #update the priors based on the (possibly sliced) kernel
                    kernel_slice = kernel[kx_min:kx_max,ky_min:ky_max]
                    new_priors[xmin:xmax,ymin:ymax] = new_priors[xmin:xmax,ymin:ymax] + priors[i][j]*kernel_slice       

                    #tensorflow implementation: didn't work, ran slower
                    #new_priors[xmin:xmax,ymin:ymax] = sess.run(tf_new_priors, feed_dict={tf_complex_priors_segment:PRIORS[xmin:xmax,ymin:ymax],tf_kernel_slice:kernel_slice,tf_prob:priors[i][j]})

        return new_priors/(np.sum(new_priors)+1e-100)

def update_priors(PRIORS, w, c, i):
    #If we spotted the intruder, update OBSERVATION with his location
    #(We assume determinism: if he's in our range of vision, we spotted him.)
    if heardSomething(c, i):
        #determine the location at which the "sound" was heard
        if intruder_in_hearing_range(c,i):
            #we really did hear the intruder
            targ_x = i.x
            targ_y = i.y
            print("INTRUDER HEARD at" + str([targ_x,targ_y]) + "!")
        else:
            #this was a false noise,
            #so we randomly assign it to 
            #a position in hearing_range
            boundary = c.get_hearing_boundaries()
            targ_x = rand.randint(boundary[0], boundary[1])
            targ_y = rand.randint(boundary[2], boundary[3])
            print("SOUND HEARD at" + str([targ_x,targ_y]) + "!")

        #derive: Probability that intruder is here given that I heard a sound
        #p(Intruder | heard something) = p(heard something | intruder)*p(intruder)/p(heard something)
        p_heard_something = PRIORS[targ_x][targ_y] * P_HEARD_SOMETHING_IF_INTRUDER + (1.-PRIORS[targ_x][targ_y]) * P_HEARD_SOMETHING_IF_NO_INTRUDER
        p_intruder_if_heard_something = P_HEARD_SOMETHING_IF_INTRUDER * PRIORS[targ_x][targ_y] / (p_heard_something + 1e-100)

        PRIORS = PRIORS * (1.-p_intruder_if_heard_something)
        PRIORS[targ_x][targ_y] = p_intruder_if_heard_something
        
        #WE HEARD SOMETHING
        #so assume that our current waypoint may no longer be accurate
        c.waypoint = c.select_waypoint(PRIORS)

    else: #we didn't hear anything

        #derive: Probability that intruder is here given that I heard NO sound
        #p(Intruder | no sound) = p(didn't hear anything | intruder) * p(Intruder) / p(heard something)
        boundaries = c.get_hearing_boundaries()
        p_intruder_in_hearing_range = np.mean(PRIORS[boundaries[0]:boundaries[1], boundaries[2]:boundaries[3]])
        p_heard_something = p_intruder_in_hearing_range * P_HEARD_SOMETHING_IF_INTRUDER + (1.-p_intruder_in_hearing_range) * P_HEARD_SOMETHING_IF_NO_INTRUDER
        p_intruder_if_no_sound = (1.-P_HEARD_SOMETHING_IF_INTRUDER)*p_intruder_in_hearing_range/(1. - (p_heard_something) + 1e-100)
        
        OBSERVATION = np.ones([XDIM, YDIM])
        OBSERVATION = OBSERVATION * (1.-p_intruder_if_no_sound)
        OBSERVATION[boundaries[0]:boundaries[1], boundaries[2]:boundaries[3]] = p_intruder_if_no_sound
        PRIORS = np.multiply(PRIORS, OBSERVATION)

    #if we think the intruder is nowhere,
    #then assume he could be anywhere
    if np.sum(PRIORS) == 0:
        print("PRIORS SUM TO ZERO. Reseting heat map...")
        #PRIORS = np.ones([XDIM, YDIM])
        PRIORS = i.geographic_probability_map
    PRIORS = PRIORS / np.sum(PRIORS) #renormalize
    return PRIORS

def show_priors(priors):
    screen.fill(BLACK)
    img = pygame.surfarray.make_surface((priors*COLOR_SCALE).astype(int))
    screen.blit(img, img.get_rect())
    pygame.display.flip()
    raw_input("Press enter to continue")

def paint_to_screen(PRIORS, w, c, i):
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
             drone_isovist = w.isovist.GetIsovistIntersections((c.x,c.y), c.movement_vector(), 90)
             for point in drone_isovist:
                 #print point
                 x = point[0]
                 y = point[1]
                 pygame.draw.rect(screen, DRK_GREEN, [x - ICON_SIZE/4, y - ICON_SIZE/4, ICON_SIZE/2, ICON_SIZE/2])
             if len(drone_isovist) > 2:
                 isovist_surface = pygame.Surface((XDIM,YDIM))
                 isovist_surface = pygame.surfarray.make_surface((PRIORS*COLOR_SCALE).astype(int))
                 isovist_surface.set_alpha(80)
                 pygame.draw.polygon(isovist_surface, WHITE, drone_isovist)
                 screen.blit(isovist_surface, isovist_surface.get_rect())

        #TREATS
        for k in w.cookies:
            img = pygame.image.load("imgs/cookie.png")
            img = pygame.transform.scale(img, (24,24))
            screen.blit(img, (k[0]-12, k[1]-12))

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
w = world.World(XDIM, YDIM, TREATS, OBSTACLES)

print('Initializing intruder...')
#i = intruder.Intruder(w, MESSY_WORLD, "cookies", INTRUDER_MOMENTUM)
#i = intruder.Trespasser(w, MESSY_WORLD, "cookies", INTRUDER_MOMENTUM)
i = intruder2.Intruder(w, MESSY_WORLD, "cookies", INTRUDER_MOMENTUM)

print('Rolling out policy...')
policy_rollout([i])
PRIORS = i.geographic_probability_map
#PRIORS = np.ones([XDIM,YDIM])
#PRIORS = PRIORS/np.sum(PRIORS)

c = copter.Copter(w, PRIORS)


if HEADLESS != True:
    size = (XDIM, YDIM)
    screen = pygame.display.set_mode(size)
    pygame.display.set_caption("Heat Map")
    clock = pygame.time.Clock()

#while we haven't caught the intruder...
done = False

print('Entering main loop...')
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

    c.step(PRIORS)
    i.step()

    #update probs based on observations
    PRIORS = update_priors(PRIORS, w, c, i)

    #predict intruder's coords on next time step
    #PRIORS = predict_intruder_location(PRIORS, i, MODE, w)    
    
    x1, x2, y1, y2 = w.movement_bounds[0], w.movement_bounds[1]+1, w.movement_bounds[2], w.movement_bounds[3]+1
    #x1, x2, y1, y2 = 0, XDIM, 1, YDIM
    #update only those portions of the priors that are 
    #within the movement bounds specified by the world
    if USE_VECTOR_MATRIX_MULTIPLY == True:
        print "BEGINNING VECTOR MATRIX MULTIPLY"
        print [x1, x2, y1, y2]
        PRIORS[x1:x2,y1:y2] = predict_intruder_location_optimized(PRIORS[x1:x2,y1:y2], i, MODE, w, x1, y1)
    else:
        PRIORS[x1:x2,y1:y2] = predict_intruder_location(PRIORS[x1:x2,y1:y2], i, MODE, w, x1, y1)    

    #suppress probabilities in regions
    #containing obstacles
    if MODE == 0: #simple kernel
	#for x in range(x1, x2):
	#    for y in range(y1, y2):
	#        PRIORS[x][y] *= w.is_valid(x,y)	
	PRIORS[x1:x2,y1:y2] = PRIORS[x1:x2,y1:y2]*w.validity_map.T[x1:x2,y1:y2]

    #renormalize priors after update
    PRIORS = PRIORS/(np.sum(PRIORS)+1e-100)


    paint_to_screen(PRIORS, w, c, i) #display the current world state and sprite locations
     
    if HEADLESS != True:
        clock.tick(60) #60 frames per second
        
    if PAUSE_BETWEEN_TIME_STEPS > 0:
        time.sleep(PAUSE_BETWEEN_TIME_STEPS)
    elif PAUSE_BETWEEN_TIME_STEPS == -1:
        raw_input("Press <ENTER> to continue")

    #debug printing
    #print(c.x)


#loop a bit after the intruder is caught
if HEADLESS != True:
    while True:
        pass
