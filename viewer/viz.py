
import threading

import numpy as np
from glumpy import app, gl, gloo, glm, data
from glumpy.transforms import Position, PanZoom

from dwtrackball import DWTrackball

import matplotlib.cm as cm

from PIL import Image

from shaders import *
from my_rrt import *
from local import *

import os,sys,inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0,parentdir) 
import q

RRT_LEN = 100
RRT_CNT = 100

# our approximation object
Q = q.Q()
Q.always_sample = True

# ========================================================
# ========================================================
# ========================================================

def S_mog_grid( Q, sz, name=None ):
    w = Q.rand( sz[0], sz[1], name=name+'_w' )
    w = w/np.sum( w )

    inds = np.reshape( range(0,np.prod(sz)), sz )
    mc = Q.choice( inds.ravel(), p=w.ravel(), name=name+'_mc' )

    X,Y = np.meshgrid( range(0,sz[0]), range(0,sz[1]) )
    Xc = float( X.ravel()[ mc ] ) / float( sz[0] )
    Yc = float( Y.ravel()[ mc ] ) / float( sz[1] )

    xv = 0.1*np.sqrt( 1.0/float(sz[0]) )
    yv = 0.1*np.sqrt( 1.0/float(sz[1]) )

    return np.atleast_2d( ( Xc + xv*Q.randn( name=name+'_gx' ),
                            Yc + yv*Q.randn( name=name+'_gy' ) ) )

def S_intruder_loc( Q ):
    return S_mog_grid( Q, (20,20), name='iloc' )

def S_goal_loc( Q ):
    return S_mog_grid( Q, (20,20), name='gloc' )

#lst = []
#for i in range(2000):
#    lst.append( S_mog_grid( Q, (20,20), "bob" ) )
#data = np.vstack( lst )
#plt.scatter( data[:,0], data[:,1] ); plt.show()

# ========================================================
# ========================================================
# ========================================================

def stdize( x ):
    x = x - np.min( x )
    x = x / np.max( x )
    return x

def make_gloo_indices( start, end ):
    return np.asarray( range(start,end), dtype=np.uint32 ).view( gloo.IndexBuffer )

rrt_index = 0
rx1,ry1,rx2,ry2 = polygons_to_segments( load_polygons( "./paths.txt" ) )

def sample_rrt( ):
    global Q, rrt_index, rx1,ry1,rx2,ry2

    start_pt = np.atleast_2d( [0.1,0.1] )
    goal_pt = np.atleast_2d( [0.9,0.9] )

#    start_pt = S_intruder_loc( Q )
#    goal_pt = S_goal_loc( Q )

    path = run_rrt( start_pt, goal_pt, rx1,ry1,rx2,ry2 )

    if not path == []:
        tmp = np.vstack( path )
        tmp = np.hstack(( tmp, 0.001*np.ones((tmp.shape[0],1)) )) # add a z coordinate
        foo = np.zeros((100,3))
        foo[ 0:tmp.shape[0], : ] = tmp

        tmp = paths['position']
        tmp[ :, 2 ] *= 0.95 # decay z just a bit; newer are always on top
        tmp[ rrt_index*RRT_LEN:(rrt_index+1)*RRT_LEN, :]  = foo

        tmp = paths['color']
        tmp *= 0.95 # decay colors towards black
        tmp[ rrt_index*RRT_LEN:(rrt_index+1)*RRT_LEN, : ] = np.atleast_2d( [0.8,0.8,1,0.9] )
        paths['color'] = tmp
        rrt_index = (rrt_index+1) % RRT_CNT

# ==========================================================================

def rrt_thread():
    print "RRT_THREAD"
    while True:
        sample_rrt()

# ========================================================

print "Loading points..."
xyz = np.load( MY_DATA_PATH + 'final_xyz.npy' )
colors_1 = np.load( MY_DATA_PATH + 'final_ref.npy' )

# or do all of this
# normalize and clip out the interesting bits
gnd_pts = xyz[:,2] <= 0.0

xyz = xyz[ ~gnd_pts, 0:3 ]
colors_1 = colors_1[ ~gnd_pts ]

xyz -= np.min(xyz)
xyz /= np.max(xyz)

xyz[ xyz[:,0] < 0.30, 0 ] = 0.30
xyz[ xyz[:,0] > 0.65, 0 ] = 0.65

xyz[ xyz[:,1] < 0.4, 1 ] = 0.4
xyz[ xyz[:,1] > 0.8, 1 ] = 0.8

xyz -= np.min(xyz,axis=0)
xyz /= np.max(xyz)

# blend reflectance and height for our final color
colors_1 = 0.5*(2.0*stdize( colors_1 )) + 0.8*stdize(xyz[:,2])

# translate into a RGB colorspace
colors_4 = cm.viridis( colors_1 )
#colors_4 = cm.jet( colors_1 )
colors_4[:,3] = 0.4  # alpha channel

#
# ==========================================================================
#

my_transform = DWTrackball( Position() )

# the laser scan point cloud
pnt_cloud = gloo.Program( pnt_cld_vertex, pnt_cld_fragment, count=xyz.shape[0] )
pnt_cloud['position'] = xyz
pnt_cloud['bg_color'] = colors_4
pnt_cloud['transform'] = my_transform

# the ground quad
quad = gloo.Program( tex_quad_vertex, tex_quad_fragment, count=4 )
quad['position'] = [ (0,0), (0,1), (1,0), (1,1) ]
quad['a_texcoord'] =  [ (0,0), (0,1), (1,0), (1,1) ]
quad['u_texture'] = np.array( Image.open("./cnts2.png") )
quad['u_texture'].interpolation = gl.GL_LINEAR
quad['transform'] = my_transform

# buildings
bdata = []
bldg_inds = []
cnt = 0
for x in open("./paths.txt"):
    tmp = np.fromstring( x, dtype=float, sep=' ' )
    tmp = np.reshape( tmp/1000.0, (-1,2) )
    tmp = np.vstack(( np.mean(tmp,axis=0,keepdims=True), tmp, tmp[0,:] ))
    tmp = np.hstack(( tmp, 0.001*np.ones((tmp.shape[0],1)) )) # add a z coordinate
    tmp[:,1] = 1.0 - tmp[:,1]  # flip on the y axis
    bdata.append( tmp )
    bldg_inds.append( make_gloo_indices( cnt, cnt+tmp.shape[0] ) )
    cnt += tmp.shape[0]

alldata = np.vstack( bdata )

cnt = alldata.shape[0]
bldgs = gloo.Program( quad_vertex, quad_fragment, count=cnt )
bldgs['position'] = alldata
bldgs['color'] = np.zeros(( cnt, 4 ))
bldgs['color'][...,0] = 1.0
bldgs['color'][...,3] = 0.2
bldgs['transform'] = my_transform

# the paths
paths = gloo.Program( quad_vertex, quad_fragment, count=RRT_CNT*RRT_LEN )
#paths = gloo.Program( paths_vertex, paths_fragment, count=RRT_CNT*RRT_LEN )
paths['position']  = np.zeros((RRT_CNT*RRT_LEN,3))
paths['color'] = np.zeros(( RRT_CNT*RRT_LEN, 4 ))
paths['transform'] = my_transform

paths_inds = []
cnt = 0
for i in range( 0, RRT_CNT ):
    paths_inds.append( make_gloo_indices( cnt, cnt+RRT_LEN ) )
    cnt += RRT_LEN

#
# ==========================================================================
#

window = app.Window( width=800, height=800, color=(0,0,0,1) )
telapsed = 0.0

should_draw_pnts = True
should_draw_rrt = True
should_draw_bldgs = True
should_draw_heatmap = True

#texture = np.ones((1000,1000,4),np.float32).view( gloo.TextureFloat2D )
#texture = np.zeros((1000,1000,4),np.float32).view( gloo.TextureFloat2D )
#framebuffer = gloo.FrameBuffer( color=[texture], depth=gloo.DepthBuffer(1000,1000) )
#framebuffer = gloo.FrameBuffer( color=[texture] )

@window.event
def on_key_press( key, modifiers ):
    global should_draw_pnts, should_draw_rrt, should_draw_bldgs, should_draw_heatmap
    global my_transform

    if key == ord('p') or key == ord('P'):
        should_draw_pnts = not should_draw_pnts
    elif key == ord('r') or key == ord('R'):
        should_draw_rrt = not should_draw_rrt
    elif key == ord('b') or key == ord('B'):
        should_draw_bldgs = not should_draw_bldgs
    elif key == ord('h') or key == ord('H'):
        should_draw_heatmap = not should_draw_heatmap

    elif key == 49: # '1'
        my_transform.view_xyz = [ 0.46, 0.70, -0.38 ]
        my_transform.view_hpr = [ -0.75,  84.25,  0 ]
        my_transform.do_update()
    elif key == 50: # '2'
        my_transform.view_xyz = [ 0.42, 0.49, 0.07 ]
        my_transform.view_hpr = [ -0.5, 46,  0 ]
        my_transform.do_update()
    elif key == 51: # '2'
        my_transform.view_xyz = [ 0.5, 0.02, -0.30 ]
        my_transform.view_hpr = [ 31.25,  -0.25, 0 ]
        my_transform.do_update()

    elif key == ord('s') or key == ord('S'):
        print "Saving framebuffer"
        np.save( '/tmp/bob.npy', framebuffer.color[0] )

    else:
        print "Unknown key %d" % key

@window.event
def on_draw(dt):
    window.clear()

    # framebuffer.activate()
    # gl.glViewport( 0, 0, 1000, 1000 )
    # for piset in paths_inds:
    #     paths.draw( gl.GL_LINE_STRIP, indices = piset )
    # framebuffer.deactivate()
#    global cur_width, cur_height
#    gl.glViewport( 0, 0, cur_width, cur_height )
#    quad['u_texture'] = framebuffer.color[0]

    # this is the heatmap
    if should_draw_heatmap:
        quad.draw( gl.GL_TRIANGLE_STRIP )

    # these are the buildings
    if should_draw_bldgs:
        for iset in bldg_inds:
            bldgs.draw( gl.GL_TRIANGLE_FAN, indices = iset )

    # this is the point cloud representing sensor readings
    if should_draw_pnts:
        pnt_cloud.draw( gl.GL_POINTS )

    # these are the RRT paths
    if should_draw_rrt:
        for piset in paths_inds:
            paths.draw( gl.GL_LINE_STRIP, indices = piset )

cur_width = 800
cur_height = 800

@window.event
def on_resize(width,height):
    cur_width = width
    cur_height = height
    pnt_cloud['projection'] = glm.perspective( 45.0, width / float(height), 1.0, 1000.0 )
    quad['projection'] = glm.perspective( 45.0, width / float(height), 1.0, 1000.0 )
    bldgs['projection'] = glm.perspective( 45.0, width / float(height), 1.0, 1000.0 )
    paths['projection'] = glm.perspective( 45.0, width / float(height), 1.0, 1000.0 )

@window.timer(1/60.)
def timer(fps):
    global telapsed
    quad['u_texture'] += 1
#    pnt_cloud['position'][...,2] += 0.01*np.sin( telapsed )
#    bldgs['position'][...,2] += 0.01*np.sin( telapsed )
#    telapsed += 0.1
    sample_rrt()

window.attach( my_transform )

gl.glEnable( gl.GL_DEPTH_TEST )

#t = threading.Thread( target=rrt_thread )
#t.start()
#print "HERE!"

app.run()
