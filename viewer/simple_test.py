
import matplotlib.pyplot as plt

import numpy as np

import os,sys,inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0,parentdir) 

import q
import isovist as iso

from my_rrt import *

from numpy import atleast_2d
a2d = atleast_2d

def getPolygonSegments_thall():
	polygonSegments = []
        c = 1000.0
	polygonSegments.append([ 
                [ ( c*0.400, c*0.0 ),  ( c*0.600, c*0.0 ) ],
                [ ( c*0.600, c*0.0 ),  ( c*0.600, c*0.800 ) ],
                [ ( c*0.600, c*0.800 ),  ( c*1.000, c*0.800 ) ],
                [ ( c*1.000, c*0.800 ),  ( c*1.000, c*1.000 ) ],
                [ ( c*1.000, c*1.000 ),  ( c*0.0, c*1.000 ) ],
                [ ( c*0.0, c*1.000 ),  ( c*0.0, c*0.800 ) ],
                [ ( c*0.0, c*0.800 ),  ( c*0.400, c*0.800 ) ],
                [ ( c*0.400, c*0.800 ),  ( c*0.400, c*0.0 ) ]
        ])
        return polygonSegments

halls = [ [ 0.4, 0.0 ],
         [ 0.6, 0.0 ],
         [ 0.6, 0.8 ],
         [ 1.0, 0.8 ],
         [ 1.0, 1.0 ],
         [ 0.0, 1.0 ],
         [ 0.0, 0.8 ],
         [ 0.4, 0.8 ],
         [ 0.4, 0.0 ] ]

locs = [ [ 0.1, 0.9 ],
         [ 0.5, 0.1 ],
         [ 0.9, 0.9 ] ]

halls = a2d( halls )
locs = a2d( locs )

def listify_segs( rx1,ry1,rx2,ry2 ):
    result = []
    for i in range(rx1.shape[0]):
        result.append( [ (rx1[i],ry1[i]), (rx2[i],ry2[i]) ] )
    return result

# for compatibility with polygons_to_segments
tmp_halls = np.vstack(( np.mean(halls, axis=0, keepdims=True), halls ))
rx1,ry1,rx2,ry2 = polygons_to_segments( [tmp_halls] )

Q = q.Q()
Q.always_sample = True

#isovist = iso.Isovist( [listify_segs(rx1,ry1,rx2,ry2)] )
isovist = iso.Isovist( getPolygonSegments_thall() )
c = 1000.0
UAVLocation = (c*0.5, 0.11)
UAVForwardVector = (0.0,0.010)
intersections = isovist.GetIsovistIntersections( UAVLocation, UAVForwardVector )
print intersections


# def run_model():

#     start_loc = a2d( locs[ Q.choice( range(3), name="sloc" ) ] )
#     goal_loc = a2d( locs[ Q.choice( range(3), name="gloc" ) ] )

#     rrt_path = run_rrt( start_loc, goal_loc, rx1,ry1,rx2,ry2 )

# #    isIntruderFound = isovist.IsIntruderSeen( rrt_path, UAVLocation,
# #                                              UAVForwardVector, UAVFieldOfVision = 45 )

#     intersections = isovist.GetIsovistIntersections( 1000*UAVLocation, 1000*UAVForwardVector )

#     plt.figure()
#     for s in [halls]:
#         for i in range( 0, s.shape[0]-1 ):
#             plt.plot( [ s[i,0], s[i+1,0] ], [ s[i,1], s[i+1,1] ], 'k' )
#     for i in range( 0, len(rrt_path)-1 ):
#         plt.plot( [ rrt_path[i][0], rrt_path[i+1][0] ], [ rrt_path[i][1], rrt_path[i+1][1] ], 'b' )
#     plt.scatter( start_loc[0,0], start_loc[0,1] )
#     plt.scatter( goal_loc[0,0], goal_loc[0,1] )
#     plt.show()

# run_model()
