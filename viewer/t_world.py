import os,sys,inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0,parentdir) 

import q
import isovist as iso

from my_rrt import *

from numpy import atleast_2d
a2d = atleast_2d

class Tworld( object ):
    def __init__( self ):
        halls = [ [ 0.4, 0.0 ],
         [ 0.6, 0.0 ],
         [ 0.6, 0.8 ],
         [ 1.0, 0.8 ],
         [ 1.0, 1.0 ],
         [ 0.0, 1.0 ],
         [ 0.0, 0.8 ],
         [ 0.4, 0.8 ],
         [ 0.4, 0.0 ] ]

        self.locs = [ [ 0.1, 0.9 ],
         [ 0.5, 0.1 ],
         [ 0.9, 0.9 ] ]

        halls = a2d( halls )
        self.locs = a2d( self.locs )

        # for compatibility with polygons_to_segments
        tmp_halls = np.vstack(( np.mean(halls, axis=0, keepdims=True), halls )) 
        rx1,ry1,rx2,ry2 = polygons_to_segments( [tmp_halls] )

        self.my_rrt = lambda start_loc, goal_loc: run_rrt( start_loc, goal_loc, rx1,ry1,rx2,ry2 )

        self.isovist = iso.Isovist( [self.listify_segs(rx1,ry1,rx2,ry2)] )

        #self.UAVLocation = ( 0.5, 0.11 )
        #self.UAVLocation = ( 0.2, 0.82 )
        self.UAVLocation = ( 0.2, 0.801 )

        self.UAVForwardVector = ( 0.0, 0.010 )

    def listify_segs( self, rx1,ry1,rx2,ry2 ):
        result = []
        for i in range(rx1.shape[0]):
            result.append( [ (rx1[i],ry1[i]), (rx2[i],ry2[i]) ] )
        return result

    def run( self, Q ):
#        sc = Q.choice( p=1.0/3.0*np.ones((1,3)), name="sloc" )
#        gc = Q.choice( p=1.0/3.0*np.ones((1,3)), name="gloc" )

        sc = Q.choice( p=a2d([0.2,0.4,0.4]), name="sloc" )
        gc = Q.choice( p=a2d([0.2,0.4,0.4]), name="gloc" )

        start_loc = a2d( self.locs[ sc ] )
        goal_loc = a2d( self.locs[ gc ] )

        rrt_path = self.my_rrt( start_loc, goal_loc )

        isIntruderFound = self.isovist.IsIntruderSeen( rrt_path,
                                                  self.UAVLocation,
                                                  self.UAVForwardVector,
                                                  UAVFieldOfVision = 45 )

        if isIntruderFound:
            p = 0.9
        else:
            p = 0.1
            
#        Q.flip( p=p, name="data" )

        return sc,gc,isIntruderFound
