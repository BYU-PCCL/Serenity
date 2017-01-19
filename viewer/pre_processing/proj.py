
#np.max(xyz,axis=0)  # array([ 310.0701501,  424.4992927])
#np.min(xyz,axis=0)  # array([-505.35308745, -316.42401899])


import numpy as np
import matplotlib.cm as cm
import matplotlib.pyplot as plt

def sigmoid(x):
    return 1.0 / ( 1.0 + np.exp( -x ) )

print "Loading points..."
xyz = np.load( '/opt/wingated/nancy/final_xyz.npy' )
colors_1 = np.load( '/opt/wingated/nancy/final_ref.npy' )

# clip out all of the ground points.  they just confuse things
gnd_pts = xyz[:,2] <= 0.0

# normalize and clip out the interesting bits
xy = xyz[ ~gnd_pts, 0:2 ]
xy -= np.min(xy)
xy /= np.max(xy)

xy[ xy[:,0] < 0.30, 0 ] = 0.30
xy[ xy[:,0] > 0.65, 0 ] = 0.65

xy[ xy[:,1] < 0.4, 1 ] = 0.4
xy[ xy[:,1] > 0.8, 1 ] = 0.8

xy -= np.min(xy,axis=0)
xy /= np.max(xy)

# now all interesting points are in [0,1]x[0,1]

#
# generate a 1k x 1k image
#

xy *= [1000.0,1000.0]

inds = xy[:,0].astype(int) + 1000*(xy[:,1].astype(int))
inds = inds.astype( int )

# compute a density
cnts = np.bincount( inds, minlength=1000*1000 )
cnts = cnts[0:1000*1000]

img = np.reshape( cnts, (1000,1000) )
img = np.flipud( img )

# two different visualizations
plt.imshow(img>1); plt.colorbar(); plt.show()
plt.imshow(sigmoid(-5.0 + img/10.0)); plt.colorbar(); plt.show()

tmp_cnts = img
tmp_cnts[ tmp_cnts > 255 ] = 255

foo = np.zeros((1000,1000,3))
foo[:,:,0] = tmp_cnts
foo[:,:,1] = tmp_cnts
foo[:,:,2] = tmp_cnts

plt.imshow( foo ); plt.show()


