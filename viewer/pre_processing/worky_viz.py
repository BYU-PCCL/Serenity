
import numpy as np
from glumpy import app, gl, gloo, glm, data
from glumpy.transforms import Position, Trackball, PanZoom

import matplotlib.cm as cm

# ========================================================

def stdize( x ):
    x = x - np.min( x )
    x = x / np.max( x )
    return x

# ========================================================

print "Loading points..."
xyz = np.load( '/opt/wingated/nancy/final_xyz.npy' )
colors_1 = np.load( '/opt/wingated/nancy/final_ref.npy' )

# do this
#xyz = xyz / 20.0

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

#xyz = (xyz - 0.5)*10.0

# end or

# blend reflectance and height for our final color
colors_1 = 0.5*(2.0*stdize( colors_1 )) + 0.8*stdize(xyz[:,2])

# translate into a RGB colorspace
colors_4 = cm.viridis( colors_1 )
#colors_4 = cm.jet( colors_1 )
colors_4[:,3] = 0.2  # alpha channel

# ========================================================

vertex_pc = """
#version 120
uniform mat4 model;
uniform mat4 view;
uniform mat4 projection;
attribute vec4  bg_color;
attribute vec3  position;
varying vec4  v_bg_color;

void main (void) {
    v_bg_color = bg_color;

    gl_Position = <transform.trackball_projection> *
                  <transform.trackball_view> *
                  <transform.trackball_model> *
                  vec4(position,1.0);

    gl_PointSize = 2 * (1.0 + 1.0 + 1.5);
}
"""
#    gl_Position = projection * view * model * vec4(position,1.0);

fragment_pc = """
#version 120

varying vec4  v_bg_color;

void main() {
    float r = (1.0 + 1.0 + 1.5);
    float signed_distance = length(gl_PointCoord.xy - vec2(0.5,0.5)) * 2 * r - 1.0;

    if ( signed_distance < 0 ) {
      gl_FragColor = v_bg_color;
    } else {
      discard;
    }
}
"""

# ==========================================================================

vertex_q = """
  uniform mat4 model;
  uniform mat4 view;
  uniform mat4 projection;

  attribute vec2 position;
  attribute vec4 color;
  varying vec4 v_color;

  void main() {
    gl_Position = <transform.trackball_projection> *
                  <transform.trackball_view> *
                  <transform.trackball_model> *
                  vec4( position, 0.0, 1.0 );

    v_color = color;
  } """

fragment_q = """
  varying vec4 v_color;
  void main() {
      gl_FragColor = v_color;
  } """

# ==========================================================================

window = app.Window(width=800, height=800, color=(0,0,0,1))

# Build the program and corresponding buffers (with 4 vertices)
quad = gloo.Program( vertex_q, fragment_q, count=4 )

# Upload data into GPU
quad['color'] = [ (1,0,0,1), (0,1,0,1), (0,0,1,1), (1,1,0,1) ]
quad['position'] = [ (0,0),   (0,+1),   (+1,0),   (+1,+1)   ]

cube['u_texture'] = data.get("crate.png")

data.get(


quad['model'] = np.eye( 4, dtype=np.float32 )
quad['projection'] = np.eye( 4, dtype=np.float32 )
quad['view'] = np.eye( 4, dtype=np.float32 )
quad['transform'] = Trackball(Position())

#
# ==========================================================================
#

pnt_cloud = gloo.Program( vertex_pc, fragment_pc, count=xyz.shape[0] )

pnt_cloud['position'] = xyz
pnt_cloud['bg_color'] = colors_4
pnt_cloud['model'] = np.eye( 4, dtype=np.float32 )
pnt_cloud['projection'] = np.eye( 4, dtype=np.float32 )
pnt_cloud['view'] = np.eye( 4, dtype=np.float32 )
pnt_cloud['transform'] = Trackball(Position())

#pnt_cloud['transform'] = PanZoom()

glm.translate( np.eye( 4, dtype=np.float32 ), 0, 0, -5 )

#
# ==========================================================================
#

@window.event
def on_draw(dt):
    window.clear()
    quad.draw( gl.GL_TRIANGLE_STRIP )
#    pnt_cloud.draw( gl.GL_POINTS )

@window.event
def on_resize(width,height):
    pnt_cloud['projection'] = glm.perspective( 45.0, width / float(height), 1.0, 1000.0 )
    quad['projection'] = glm.perspective( 45.0, width / float(height), 1.0, 1000.0 )

window.attach( pnt_cloud['transform'] )
window.attach( quad['transform'] )

gl.glEnable( gl.GL_DEPTH_TEST )
app.run()
