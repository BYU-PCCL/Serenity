import os,sys,inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0,parentdir) 

import q
import isovist as iso

from my_rrt import *

from numpy import atleast_2d
a2d = atleast_2d

class Timeworld( object ):
    def __init__( self ):

        self.sensors = [
            [ 0.574, 1-0.624 ],
            [ 0.539, 1-0.404 ] ]

        self.locs = [
            [ 0.100, 1-0.900 ],
            [ 0.566, 1-0.854 ],
            [ 0.761, 1-0.665 ],
            [ 0.523, 1-0.604 ],
            [ 0.241, 1-0.660 ],
            [ 0.425, 1-0.591 ],
            [ 0.303, 1-0.429 ],
            [ 0.815, 1-0.402 ],
            [ 0.675, 1-0.075 ],
            [ 0.432, 1-0.098 ] ]

        self.sensors = a2d( self.sensors )
        self.locs = a2d( self.locs )

        rx1,ry1,rx2,ry2 = polygons_to_segments( load_polygons( "./paths.txt" ) )

        self.my_rrt = lambda start_loc, goal_loc: run_rrt( start_loc, goal_loc, rx1,ry1,rx2,ry2 )
        self.isovist = iso.Isovist( [self.listify_segs(rx1,ry1,rx2,ry2)] )

        self.UAVLocation = ( 0.468, 1-0.764 )
        self.UAVForwardVector = ( 0.0, 0.010 )

    def listify_segs( self, rx1,ry1,rx2,ry2 ):
        result = []
        for i in range(rx1.shape[0]):
            result.append( [ (rx1[i],ry1[i]), (rx2[i],ry2[i]) ] )
        return result

    def run( self, Q ):
        cnt = len( self.locs )

        sc = Q.choice( p=1.0/cnt*np.ones((1,cnt)), name="sloc" )
        gc = Q.choice( p=1.0/cnt*np.ones((1,cnt)), name="gloc" )

        start_loc = np.atleast_2d( self.locs[sc] )
        goal_loc = np.atleast_2d( self.locs[gc] )

        rrt_path = self.my_rrt( start_loc, goal_loc )

        pvals = np.zeros((2,100))
        for ind, pt in enumerate( rrt_path ):
            dists = distance_to_other_points( pt, self.sensors ) < 0.01
            pvals[:,ind:ind+1] = 0.999*dists + 0.001*(1-dists)

        data = Q.flip( p=pvals, name="data" )

        return start_loc,goal_loc,data,rrt_path,pvals



# ========================================================
# ========================================================
# ========================================================

        # if rrt_path == []:
        #     isIntruderFound = False
        # else:
        #     rrt_pts = np.vstack( rrt_path )
        #     dists = distance_to_other_points( a2d(self.UAVLocation), rrt_pts )
        #     isIntruderFound = np.any( dists < 0.1 )
        # if isIntruderFound:
        #     p = 0.9999999
        # else:
        #     p = 0.0000001
            

#        isIntruderFound = self.isovist.IsIntruderSeen( rrt_path,
#                                                  self.UAVLocation,
#                                                  self.UAVForwardVector,
#                                                  UAVFieldOfVision = 45 )
#        intersections = self.isovist.GetIsovistIntersections( self.UAVLocation, self.UAVForwardVector )


# def S_mog_grid( Q, sz, name=None ):
#     w = Q.rand( sz[0], sz[1], name=name+'_w' )
#     w = w/np.sum( w )

#     inds = np.reshape( range(0,np.prod(sz)), sz )
#     mc = Q.choice( inds.ravel(), p=w.ravel(), name=name+'_mc' )

#     X,Y = np.meshgrid( range(0,sz[0]), range(0,sz[1]) )
#     Xc = float( X.ravel()[ mc ] ) / float( sz[0] )
#     Yc = float( Y.ravel()[ mc ] ) / float( sz[1] )

#     xv = 0.1*np.sqrt( 1.0/float(sz[0]) )
#     yv = 0.1*np.sqrt( 1.0/float(sz[1]) )

#     return np.atleast_2d( ( Xc + xv*Q.randn( name=name+'_gx' ),
#                             Yc + yv*Q.randn( name=name+'_gy' ) ) )

# def S_intruder_loc( Q ):
#     return S_mog_grid( Q, (20,20), name='iloc' )

# def S_goal_loc( Q ):
#     return S_mog_grid( Q, (20,20), name='gloc' )
