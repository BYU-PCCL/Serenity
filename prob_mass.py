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
from PIL import Image, ImageDraw

import world
import copter
import intruder

##DEPRECATED###
#KERNEL_SIZE = 11 #must be an odd number

P_HEARD_SOMETHING_IF_NO_INTRUDER = 1e-3
P_HEARD_SOMETHING_IF_INTRUDER = 0.9
#P_HEARD_SOMETHING_IF_NO_INTRUDER = 0.0                #probablity of false pos.
#P_HEARD_SOMETHING_IF_INTRUDER = 1.0                #probability of hearing intr.
P_SAW_SOMETHING_IF_NO_INTRUDER = 0.0                 #probability of false pos.
P_SAW_SOMETHING_IF_INTRUDER = 1.0                #probability of seeing intr.



#SOME TENSORFLOW CALCULATIONS TO LET US ENGAGE THE GPU

#calculations for predict_intruder_location(), simple kernel
tf_priors = tf.placeholder(tf.float32, [1, None, None, 1], name='priors')
#tf_simple_kernel = tf.placeholder(tf.float32,[KERNEL_SIZE,KERNEL_SIZE,1,1], name='simple_kernel')
tf_simple_kernel = tf.placeholder(tf.float32,[None,None,1,1], name='simple_kernel')

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


class prob_mass:

    def __init__(self, w, i, i_model, c, mode=1, kernel_size=11, rollout_epochs=1000, rollout_steps=1000, downsample=2):
        #w = world instantiation
	#i = the "real" intruder
	#i_model = our model of the intruder
	#c = the quadcopter trying to "catch" the intruder

	self.w = w
	self.c = c
	self.i = i
	self.i_model = i_model

	self.MODE = mode
	self.KERNEL_SIZE=kernel_size
	self.DOWNSAMPLE = downsample

        print('Rolling out policy...')
        self.policy_rollout([i], rollout_epochs, rollout_steps)

	self.initialize_priors()
        #self.PRIORS = i.geographic_probability_map
        #PRIORS = np.ones([XDIM,YDIM])
	#PRIORS = PRIORS/np.sum(PRIORS)

	self.last_intruder_x=-1
	self.last_intruder_y=-1

	self.isovist_color = (255,255,255) #white
	print("INITIALIZATION COMPLETE!")


    def initialize_priors(self):
	if self.MODE == 0: #simple kernel
	    self.PRIORS = np.ones([self.w.xdim, self.w.ydim])
	else:
	    self.PRIORS = self.i.geographic_probability_map
  
    def intruder_in_sight_range(self):
	intersections = self.w.isovist.GetIsovistIntersections((self.c.x, self.c.y),self.c.movement_vector()) 
	return self.w.isovist.FindIntruderAtPoint((self.i.x, self.i.y), intersections)

    def saw_something(self):
        if self.intruder_in_sight_range():            
            return rand.random() < P_HEARD_SOMETHING_IF_INTRUDER
        else:
            return rand.random() < P_HEARD_SOMETHING_IF_NO_INTRUDER


    def intruder_in_hearing_range(self):
        c = self.c
	i = self.i
        if (c.x - c.hearing_range/2) < i.x and i.x < (c.x + c.hearing_range/2):
            if (c.y - c.hearing_range/2) < i.y and i.y < (c.y + c.hearing_range/2):
                return True
        return False


    def heardSomething(self):
        if self.intruder_in_hearing_range():            
            return rand.random() < P_HEARD_SOMETHING_IF_INTRUDER
        else:
            return rand.random() < P_HEARD_SOMETHING_IF_NO_INTRUDER


    def policy_rollout(self, intruder_list, num_epochs, num_steps):
        print "BEGINNING POLICY ROLLOUT..."

        for intruder in intruder_list:
            #establish a base kernel that will model the intruder behavior
            simple_kernel = np.zeros([self.KERNEL_SIZE,self.KERNEL_SIZE], dtype=np.float32) 
        
            #also create a complex kernel construct: a separate
            #movement kernel for each x,y location                
            kernel = np.zeros([intruder.xdim, intruder.ydim, self.KERNEL_SIZE,self.KERNEL_SIZE], dtype=np.float32)
            midpoint = self.KERNEL_SIZE/2

            if num_epochs == 0 or num_steps == 0:
                geographic_probability_map = np.ones([intruder.xdim,intruder.ydim], dtype=np.float32)
            else:

                #this represents the overall probability of the intruder 
                #being at any given x,y coordinate
                geographic_probability_map = np.zeros([intruder.xdim,intruder.ydim], dtype=np.float32)

                for i in range(int(num_epochs)):
                    #start the intruder at a random location
                    intruder.select_random_location()
                    intruder.select_waypoint()

                    for j in range(int(num_steps)):

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
                        
			####DEPRECATED###
#			if USE_VECTOR_MATRIX_MULTIPLY == True:
#                            #transition_matrix[((prev_x-x_offset)*ydim+(prev_y-y_offset))/DOWNSAMPLE , ((intruder.x-x_offset)*ydim+(intruder.y-y_offset))/DOWNSAMPLE] += 1
#                            transition_matrix[((prev_x-x_offset)*ydim+(prev_y-y_offset))/DOWNSAMPLE][((intruder.x-x_offset)*ydim+(intruder.y-y_offset))/DOWNSAMPLE] += 1
                    

            #normalize the simple kernel
            intruder.simple_kernel = simple_kernel/(np.sum(simple_kernel)+1e-100)

            kernel_sums = np.sum(kernel, axis=(2, 3)) # (1000, 1000)
            not_zero_positions = np.where(kernel_sums != 0)
            kernel[not_zero_positions] /= np.expand_dims(np.expand_dims(kernel_sums[not_zero_positions], axis=2), axis=2)

            intruder.kernel = kernel

            #normalize the overall probability map
            intruder.geographic_probability_map = geographic_probability_map/np.sum(geographic_probability_map)


    def predict_intruder_location(self, priors, x_offset=0, y_offset=0):
        new_priors = np.copy(priors)
        xlim, ylim = new_priors.shape[0], new_priors.shape[1]

        if self.MODE == 0: #simple kernel - a single kernel summarizes intruder behavior
        
            new_priors = sess.run(tf_simple_kernel_convolution, feed_dict={tf_priors:np.reshape(new_priors,[1,xlim,ylim,1]),tf_simple_kernel:np.reshape(self.i.simple_movement_kernel(),[self.KERNEL_SIZE,self.KERNEL_SIZE,1,1])})
        
            #suppress probabilities in regions
            #containing obstacles
	    print new_priors.shape
	    print self.w.validity_map.T.shape
            #new_priors = new_priors*self.w.validity_map.T[x_offset:x_offset+xlim,y_offset:y_offset+ylim]
	    print new_priors.shape

            return np.reshape(new_priors, [xlim,ylim])

        else: #complex kernel - a separate movement kernel for each xy location

            complex_kernel = self.i.complex_movement_kernel()
            midpoint = self.KERNEL_SIZE/2        #all xy kernels have the same shape,
                        
            kx_min, kx_max, ky_min, ky_max = 0, self.KERNEL_SIZE, 0, self.KERNEL_SIZE

            for i in range(0+rand.randint(0, self.DOWNSAMPLE-1),xlim,self.DOWNSAMPLE):
                xmin = max(0,i-midpoint)
                xmax = min(i+1+midpoint, xlim)
                kx_min = 0
                kx_max = self.KERNEL_SIZE
                if i-midpoint <= 0:
                    kx_min = midpoint - i
                if i+midpoint >= xlim:
                    kx_max = xlim - i + midpoint

                for j in range(0+rand.randint(0,self.DOWNSAMPLE-1),ylim,self.DOWNSAMPLE):
                    if self.w.is_valid(i+x_offset,j+y_offset):
                        ymin = max(0,j-midpoint)
                        ymax = min(j+1+midpoint, ylim)
                        ky_min = 0
                        ky_max = self.KERNEL_SIZE
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

            return new_priors/(np.sum(new_priors)+1e-100)

    def update_priors(self):
        #If we spotted the intruder, update OBSERVATION with his location
        #(We assume determinism: if he's in our range of vision, we spotted him.)
        if self.saw_something():
            #determine the location at which the "sound" was heard
	    self.isovist_color = (255,255,255) #white
            if self.intruder_in_sight_range():
                #we really did hear the intruder
                targ_x = self.i.x
                targ_y = self.i.y
                print("INTRUDER SIGHTED at" + str([targ_x,targ_y]) + "!")
            else:
                #this was a false noise,
                #so we randomly assign it to 
                #a position in hearing_range
                boundary = self.c.get_hearing_boundaries()
                targ_x = rand.randint(boundary[0], boundary[1])
                targ_y = rand.randint(boundary[2], boundary[3])
                print("SOMETHING SEEN at" + str([targ_x,targ_y]) + "!")
	
            #derive: Probability that intruder is here given that I heard a sound
            #p(Intruder | heard something) = p(heard something | intruder)*p(intruder)/p(heard something)
            p_heard_something = self.PRIORS[targ_x][targ_y] * P_HEARD_SOMETHING_IF_INTRUDER + (1.-self.PRIORS[targ_x][targ_y]) * P_HEARD_SOMETHING_IF_NO_INTRUDER
            p_intruder_if_heard_something = P_HEARD_SOMETHING_IF_INTRUDER * self.PRIORS[targ_x][targ_y] / (p_heard_something + 1e-100)

            self.PRIORS = self.PRIORS * (1.-p_intruder_if_heard_something)
            self.PRIORS[targ_x][targ_y] = p_intruder_if_heard_something
        
            #WE HEARD SOMETHING
            #so assume that our current waypoint may no longer be accurate
            self.c.waypoint = self.c.select_waypoint(self.PRIORS)
	
	    #TRY TO PREDICT WHERE THE INTRUDER IS HEADED
	    if self.last_intruder_x >= 0 and self.last_intruder_y >=0:
	        #This is the second consecutive sighting of the intruder
	        #So let's infer something from his trajectory
	        trajectory=(self.last_intruder_x-targ_x, self.last_intruder_y-targ_y)

	        #TO-DO!!!
	        #next, we'll calculate possible RRT paths and
	        #keep the ones that begin on a near-matching trajectory
	
	    #remember the last known sighting
	    self.last_intruder_x = targ_x
	    self.last_intruder_y = targ_y

        else: #we didn't hear anything
	    self.isovist_color = (0,0,150) #medium green
  
	    #we didn't see the intruder, so reset
	    #the last sighting flags
	    self.last_intruder_x = -1
	    self.last_intruder_y = -1


            #derive: Probability that intruder is here given that I heard NO sound
            #p(Intruder | no sound) = p(didn't hear anything | intruder) * p(Intruder) / p(heard something)
	    if False:
		#THIS IS THE OLD WAY,
		#based on a square of "hearing" rather than an isovist
                boundaries = self.c.get_hearing_boundaries()
                p_intruder_in_hearing_range = np.mean(self.PRIORS[int(boundaries[0]):int(boundaries[1]), int(boundaries[2]):int(boundaries[3])])
                p_heard_something = p_intruder_in_hearing_range * P_HEARD_SOMETHING_IF_INTRUDER + (1.-p_intruder_in_hearing_range) * P_HEARD_SOMETHING_IF_NO_INTRUDER
                p_intruder_if_no_sound = (1.-P_HEARD_SOMETHING_IF_INTRUDER)*p_intruder_in_hearing_range/(1. - (p_heard_something) + 1e-100)
        
                OBSERVATION = np.ones([self.w.xdim, self.w.ydim])
                OBSERVATION = OBSERVATION * (1.-p_intruder_if_no_sound)
                OBSERVATION[int(boundaries[0]):int(boundaries[1]), int(boundaries[2]):int(boundaries[3])] = p_intruder_if_no_sound

	    else:
		#This is the new way. Isovist-based
	  	polygon = self.w.isovist.GetIsovistIntersections((self.c.x,self.c.y), self.c.movement_vector(), self.c.isovist_angle)
		polygon.append((self.c.x,self.c.y))

		if len(polygon) > 2:
		    img = Image.new('L', (self.w.xdim, self.w.ydim), 0)
		    ImageDraw.Draw(img).polygon(polygon, outline=1, fill=1)
		    isovist_mask = np.array(img).T
                
		    #boundaries = self.c.get_hearing_boundaries()
                    #p_intruder_in_hearing_range = np.mean(self.PRIORS[int(boundaries[0]):int(boundaries[1]), int(boundaries[2]):int(boundaries[3])])
                    #p_heard_something = p_intruder_in_hearing_range * P_HEARD_SOMETHING_IF_INTRUDER + (1.-p_intruder_in_hearing_range) * P_HEARD_SOMETHING_IF_NO_INTRUDER
                    #p_intruder_if_no_sound = (1.-P_HEARD_SOMETHING_IF_INTRUDER)*p_intruder_in_hearing_range/(1. - (p_heard_something) + 1e-100)

                    p_intruder_in_hearing_range = np.sum(self.PRIORS*isovist_mask) / np.sum(self.PRIORS)
                    p_heard_something = p_intruder_in_hearing_range * P_HEARD_SOMETHING_IF_INTRUDER + (1.-p_intruder_in_hearing_range) * P_HEARD_SOMETHING_IF_NO_INTRUDER
                    p_intruder_if_no_sound = (1.-P_HEARD_SOMETHING_IF_INTRUDER)*p_intruder_in_hearing_range/(1. - (p_heard_something) + 1e-100)

		    OBSERVATION = isovist_mask
		    print "p_intruder_in_hearing_range = %f" % (p_intruder_in_hearing_range)
		    print "p_heard_something = %f" % (p_heard_something)
		    print "p_intruder_if_no_sound = %f" % (p_intruder_if_no_sound)
		    OBSERVATION = OBSERVATION * p_intruder_if_no_sound + (1-isovist_mask) * (1-p_intruder_if_no_sound)

		else:
		    #We made no observation, so priors will stay unchanged
                    OBSERVATION = np.ones([self.w.xdim, self.w.ydim])
	
	    #print "POLYGON"
	    #print polygon
	    #raw_input("test")

            self.PRIORS = np.multiply(self.PRIORS, OBSERVATION)

        #if we think the intruder is nowhere,
        #then assume he could be anywhere
        if np.sum(self.PRIORS) == 0:
            print("PRIORS SUM TO ZERO. Reseting heat map...")
	    self.initialize_priors()
            #PRIORS = np.ones([XDIM, YDIM])
            #self.PRIORS = self.i.geographic_probability_map
        self.PRIORS = self.PRIORS / np.sum(self.PRIORS) #renormalize
        return self.PRIORS


    def step(self):

        #update probs based on observations
        self.update_priors()

        #predict intruder's current/future location
	w = self.w
        x1, x2, y1, y2 = w.movement_bounds[0], w.movement_bounds[1]+1, w.movement_bounds[2], w.movement_bounds[3]+1
        self.PRIORS[x1:x2,y1:y2] = self.predict_intruder_location(self.PRIORS[x1:x2,y1:y2], x1, y1)

        if self.MODE == 0: #simple kernel
            ##suppress probabilities in regions
            ##containing obstacles
            self.PRIORS[x1:x2,y1:y2] = self.PRIORS[x1:x2,y1:y2]*self.w.validity_map.T[x1:x2,y1:y2]

        #renormalize priors after update
        self.PRIORS = self.PRIORS/(np.sum(self.PRIORS)+1e-100)
	return self.PRIORS

