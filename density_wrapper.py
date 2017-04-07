
import numpy as np
import world
import copter
import intruder
import prob_mass
import math
import isovist

from PIL import Image, ImageDraw, ImageOps


MESSY_WORLD = True
INTRUDER_MOMENTUM=4
#COLOR_SCALE = 1000 * 1000 * 2.5
COLOR_SCALE = 1000 * 1000 * 30

MODE = 0 #0 for simple kernel, 1 for complex kernel
kernel_size = 11
rollout_epochs = 1000
rollout_steps = 1000
downsample = 4


#rotation code from http://stackoverflow.com/questions/34372480/rotate-point-about-another-point-in-degrees-python
def rotate(origin, point, angle):
    """
    Rotate a point counterclockwise by a given angle around a given origin.

    The angle should be given in radians.
    """
    ox, oy = origin
    px, py = point

    qx = ox + math.cos(angle) * (px - ox) - math.sin(angle) * (py - oy)
    qy = oy + math.sin(angle) * (px - ox) + math.cos(angle) * (py - oy)
    return qx, qy


class density_wrapper:

    def __init__(self, XDIM=1000, YDIM=1000, MODE=1, NUM_TREATS=3, MAP="bremen"):

        #A world with cookies for grabbing, based on the Bremen point cloud
        self.w = world.World(XDIM, YDIM, NUM_TREATS, "bremen", True)

        #An intruder who likes cookies and walks linearly from waypoint to waypoint
        self.i = intruder.Thief(self.w, MESSY_WORLD, "cookies", INTRUDER_MOMENTUM)

        #use the actual intruder as our model, for now
        self.i_model = self.i

        self.c=copter.Copter(self.w)
        self.pm = prob_mass.prob_mass(self.w, self.i, self.i_model, self.c, MODE, kernel_size, rollout_epochs, rollout_steps, downsample)
	self.DENSITY_MAP = self.pm.PRIORS
        self.c.set_initial_path(self.DENSITY_MAP)

    def step(self):
	self.c.step(self.DENSITY_MAP)
        self.i.step()
    	self.DENSITY_MAP = COLOR_SCALE * self.pm.step()

	#THE VIEWER DISPLAY IS ROTATED 90% FROM THE PROB_MASS CALCULATIONS
	#So we rotate the density mass
	tmp = Image.fromarray(np.uint8(self.DENSITY_MAP))
	tmp = tmp.transpose(Image.ROTATE_90)
        rotated_densities = np.asarray(tmp)

	#we also rotate the copter and intruder coordinates...
	#except for weird and unknown reasons, we must also transpose them first
	
	#transpose
	c_x = self.c.y
	c_y = self.c.x
	i_x = self.i.y
        i_y = self.i.x
	movement_vector = (self.c.movement_vector()[1], self.c.movement_vector()[0])

	#rotate
	rotated_copter = rotate((self.w.xdim/2,self.w.ydim/2), (c_x,c_y), 3*math.pi/2);
	rotated_intruder = rotate((self.w.xdim/2,self.w.ydim/2), (i_x,i_y), 3*math.pi/2);
	rotated_movement_vector = rotate((0,0), movement_vector, 3*math.pi/2)

	c_x = rotated_copter[0] 
	c_y = rotated_copter[1]
	i_x = rotated_intruder[0]
        i_y = rotated_intruder[1]


	#and we have to transpose and rotate the polygon 
	#segments for the isovist, too
	transposed_segments= []
	for polygon in self.w.polygon_segments:
	    segment_list = []
	    for segment in polygon:

		#transpose
		point1 = (segment[0][1], segment[0][0])
		point2 = (segment[1][1], segment[1][0])

		#rotate
		rotated_point1 = rotate((self.w.xdim/2,self.w.ydim/2), point1, 3*math.pi/2)
		rotated_point2 = rotate((self.w.xdim/2,self.w.ydim/2), point2, 3*math.pi/2)

	        new_segment = [rotated_point1, rotated_point2]
		segment_list.append(new_segment)
	    transposed_segments.append(segment_list) 

	adjusted_isovist = isovist.Isovist(transposed_segments)

        polygon = adjusted_isovist.GetIsovistIntersections((c_x,c_y), rotated_movement_vector, self.c.isovist_angle)
        polygon.append((c_x,c_y))

        isovist_mask = np.zeros([self.w.xdim, self.w.ydim])
        if len(polygon) > 2:
           img = Image.new('L', (self.w.xdim, self.w.ydim), 255)
           ImageDraw.Draw(img).polygon(polygon, outline=0, fill=0)
	   #flip isovist top to bottom.
	   #why? I don't know why. Because it was displaying upside down
	   #img = img.transpose(Image.FLIP_TOP_BOTTOM)
           isovist_mask = np.array(img)

        bremen_img = np.stack((rotated_densities, rotated_densities, 255-isovist_mask), axis=-1)

        bremen_img[c_y-5:c_y+5,c_x-5:c_x+5] = [0, 255, 0]
        bremen_img[i_y-5:i_y+5,i_x-5:i_x+5] = [255, 0, 0]

        return bremen_img
        
