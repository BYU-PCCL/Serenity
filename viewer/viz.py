
import numpy as np
from glumpy import app, gl, gloo, glm, data
from glumpy.transforms import Position, PanZoom

from dwtrackball import DWTrackball

import matplotlib.cm as cm

from PIL import Image

from shaders import *

# ========================================================

def stdize( x ):
    x = x - np.min( x )
    x = x / np.max( x )
    return x

# ========================================================

print "Loading points..."
#xyz = np.load( '/opt/wingated/nancy/final_xyz.npy' )
xyz = np.load( '../point_clouds/final_xyz.npy' )
#colors_1 = np.load( '/opt/wingated/nancy/final_ref.npy' )
colors_1 = np.load( '../point_clouds/final_ref.npy' )

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

# the laser scan point cloud
pnt_cloud = gloo.Program( pnt_cld_vertex, pnt_cld_fragment, count=xyz.shape[0] )
pnt_cloud['position'] = xyz
pnt_cloud['bg_color'] = colors_4
pnt_cloud['transform'] = DWTrackball( Position() )

# the ground quad
quad = gloo.Program( tex_quad_vertex, tex_quad_fragment, count=4 )
quad['position'] = [ (0,0), (0,1), (1,0), (1,1) ]
quad['a_texcoord'] =  [ (0,0), (0,1), (1,0), (1,1) ]
quad['u_texture'] = np.array( Image.open("./cnts2.png") )
quad['u_texture'].interpolation = gl.GL_LINEAR
quad['transform'] = DWTrackball( Position() )

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
    bldg_inds.append( np.asarray( range(cnt, cnt+tmp.shape[0]), dtype=np.uint32 ).view( gloo.IndexBuffer ) )
    cnt += tmp.shape[0]

alldata = np.vstack( bdata )

cnt = alldata.shape[0]
bldgs = gloo.Program( quad_vertex, quad_fragment, count=cnt )
bldgs['position'] = alldata
bldgs['color'] = np.zeros(( cnt, 4 ))
bldgs['color'][...,2:4] = 1.0
bldgs['transform'] = DWTrackball( Position() )

#
# ==========================================================================
#

window = app.Window( width=800, height=800, color=(0,0,0,1) )
telapsed = 0.0

@window.event
def on_draw(dt):
    window.clear()
    quad.draw( gl.GL_TRIANGLE_STRIP )
    pnt_cloud.draw( gl.GL_POINTS )
    for iset in bldg_inds:
        bldgs.draw( gl.GL_TRIANGLE_FAN, indices = iset )

@window.event
def on_resize(width,height):
    pnt_cloud['projection'] = glm.perspective( 45.0, width / float(height), 1.0, 1000.0 )
    quad['projection'] = glm.perspective( 45.0, width / float(height), 1.0, 1000.0 )
    bldgs['projection'] = glm.perspective( 45.0, width / float(height), 1.0, 1000.0 )

@window.timer(1/60.)
def timer(fps):
    global telapsed
#    quad['u_texture'] += 1
#    pnt_cloud['position'][...,2] += 0.01*np.sin( telapsed )
    bldgs['position'][...,2] += 0.01*np.sin( telapsed )
    telapsed += 0.1

window.attach( pnt_cloud['transform'] )
window.attach( quad['transform'] )
window.attach( bldgs['transform'] )

gl.glEnable( gl.GL_DEPTH_TEST )

app.run()
