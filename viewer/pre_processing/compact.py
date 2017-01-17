
import numpy as np

# ========================================================

CNT = 13
ds_list = range( 0, CNT )
dss = []
css = []
for i in ds_list:
    print "Loading dataset %d..." % i

    A_tmp = np.load( '/opt/wingated/nancy/bremen%03d.npy' % i )

    # reduce dataset size by 90%
    A_tmp = A_tmp[0::10,:]

    # add a bias
    xyz_tmp = np.hstack(( A_tmp[:,0:3], np.ones((A_tmp.shape[0],1)) ))

    A_tmat = np.loadtxt( '/opt/wingated/nancy/finalposes/scan%03d.dat' % i )

    # transform the points
    xyz_trans = np.dot( xyz_tmp, A_tmat.T )

    dss.append( xyz_trans[:,0:3] )
    css.append( A_tmp[:,6] )

xyz = np.vstack( dss )
colors_1 = np.hstack( css )

np.save( '/opt/wingated/nancy/final_xyz.npy', xyz, allow_pickle=False )
np.save( '/opt/wingated/nancy/final_ref.npy', colors_1, allow_pickle=False )
