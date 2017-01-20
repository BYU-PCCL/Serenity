#
# ========================================================
#
# shaders for the point cloud
#
# ========================================================
#

pnt_cld_vertex = """
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

pnt_cld_fragment = """
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

#
# ========================================================
#
# shaders for a textured quad
#
# ========================================================
#

tex_quad_vertex = """
  uniform mat4 model;
  uniform mat4 view;
  uniform mat4 projection;

  attribute vec2 position;

  attribute vec2 a_texcoord;      // Vertex texture coordinates
  varying vec2   v_texcoord;      // Interpolated fragment texture coordinates (out)

  void main() {
    gl_Position = <transform.trackball_projection> *
                  <transform.trackball_view> *
                  <transform.trackball_model> *
                  vec4( position, 0.0, 1.0 );

    v_texcoord  = a_texcoord;

  } """

tex_quad_fragment = """
  uniform sampler2D u_texture;  // Texture 
  varying vec2      v_texcoord; // Interpolated fragment texture coordinates (in)
  void main() {
      vec4 t_color = texture2D( u_texture, v_texcoord );
      gl_FragColor = t_color;
  } """

#
# ========================================================
#
# shaders for a NON textured triangle strip
#
# ========================================================
#

quad_vertex = """
uniform mat4   model;         // Model matrix
uniform mat4   view;          // View matrix
uniform mat4   projection;    // Projection matrix
attribute vec4 color;         // Vertex color
attribute vec3 position;      // Vertex position
varying vec4   v_color;       // Interpolated fragment color (out)

void main() {
  v_color = color;
  gl_Position = <transform.trackball_projection> *
                <transform.trackball_view> *
                <transform.trackball_model> *
                vec4( position, 1.0 );
} """

quad_fragment = """
varying vec4   v_color;         // Interpolated fragment color (in)
void main() {
  gl_FragColor = v_color;
} """

# ==========================================================================
