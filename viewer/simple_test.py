import numpy as np

import os,sys,inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0,parentdir) 
import q

from my_rrt import *

from numpy import atleast_2d
a2d = atleast_2d

path = [ [ 0.4, 0.0, ],
         [ 0.6, 0.0, ],
         [ 0.6, 0.8, ],
         [ 1.0, 0.8, ],
         [ 1.0, 1.0, ],
         [ 0.0, 1.0, ],
         [ 0.0, 0.8, ],
         [ 0.4, 0.8, ] ]

locs = [ [ 0.1, 0.9 ],
         [ 0.5, 0.1 ],
         [ 0.9, 0.9 ] ]

path = a2d( path )
locs = a2d( locs )

rx1,ry1,rx2,ry2 = polygons_to_segments( [path] )

Q = q.Q()
Q.always_sample = True

def run_model():

    start_loc = locs[ Q.choice( range(3) ) ]
    goal_loc = locs[ Q.choice( range(3) ) ]

    path = run_rrt( start_loc, goal_loc, rx1,ry1,rx2,ry2 )
