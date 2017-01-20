
import numpy as np
from glumpy.transforms import Transform
from glumpy import gl, glm, library

class DWTrackball(Transform):
    """
    D. Wingate's 3D transform

    """

    aliases = { "view"       : "trackball_view",
                "model"      : "trackball_model",
                "projection" : "trackball_projection" }

    def __init__(self, *args, **kwargs):
        """
        Initialize the transform.
        """

        code = library.get("transforms/trackball.glsl")
        Transform.__init__( self, code, *args, **kwargs )

        self._width = 1
        self._height = 1

#        self.view_xyz = np.asarray( [ 2, 2, 0.5 ] )  # camera xyz
#        self.view_hpr = np.asarray( [ 187, 0, 0 ] )  # camera heading, pitch, roll (degrees)
#        self.view_xyz = np.asarray( [ 0.5, 0.25, 6.0 ] )  # camera xyz
#        self.view_hpr = np.asarray( [ 270, -22.5, 0 ] )  # camera heading, pitch, roll (degrees)

        self.view_xyz = np.asarray( [ 0.0, 0.0, 1.0 ] )  # camera xyz
        self.view_hpr = np.asarray( [ 0.0, 0.0, 0.0 ] )  # camera heading, pitch, roll (degrees)

        self._model = np.eye( 4, dtype=np.float32 )
        self._projection = np.eye( 4, dtype=np.float32 )
        self._view = np.eye( 4, dtype=np.float32 )

        glm.translate( self._model, 0, 0, -1 )

    def on_attach(self, program):
        self["view"] = self._view
        self["model"] = self._model
        self["projection"] = self._projection

    def on_resize( self, width, height ):
        vnear = 0.01
        vfar = 10
        k = 0.8
        if width >= height:
            k2 = (1.0*height)/(1.0*width)
            self['projection'] = glm.frustum( -vnear*k, vnear*k, -vnear*k*k2, vnear*k*k2, vnear, vfar  )
        else:
            k2 = (1.0*width)/(1.0*height)
            self['projection'] = glm.frustum( -vnear*k*k2, vnear*k*k2, -vnear*k, vnear*k, vnear, vfar  )

        Transform.on_resize( self, width, height )

    def on_mouse_drag( self, x, y, dx, dy, button ):
        dx = dx*0.5
        dy = dy*0.5

        side = 0.01 * dx
        DEG_TO_RAD = (3.141592654 / 180.0)

        s = np.sin( self.view_hpr[0]*DEG_TO_RAD )
        c = np.cos( self.view_hpr[0]*DEG_TO_RAD )

        if button == 2: # left mouse
            self.view_hpr[0] += dx * 0.5
            self.view_hpr[1] += dy * 0.5

        elif button == 4: # middle mouse
            fwd = 0.01 * dy
            self.view_xyz[0] -= 0.25*c*side # LR
            self.view_xyz[2] -= 0.25*s*side # LR
            self.view_xyz[1] += 0.0025 * dy  # UD

        elif button == 8: # right mouse
            fwd = 0.01 * dy
            self.view_xyz[0] -= 0.25*s*fwd # LR
            self.view_xyz[2] += 0.25*c*fwd # LR

#        print "----"
#        print self.view_xyz
#        print self.view_hpr

        self.do_update()

    def on_mouse_scroll(self, x, y, dx, dy):
        self.on_mouse_drag( 0, 0, 0, -5*dy, 8 )

    def do_update( self ):
        M = np.eye( 4 )
        M = glm.rotate( M, -90, 1, 0, 0 ) # rotates CCW
        glm.translate( M, -self.view_xyz[0], -self.view_xyz[1], -self.view_xyz[2] )
        glm.rotate( M, self.view_hpr[0], 0, 1, 0 )
        glm.rotate( M, self.view_hpr[1], 1, 0, 0 )

        self._model = M
        self["model"] = self._model
