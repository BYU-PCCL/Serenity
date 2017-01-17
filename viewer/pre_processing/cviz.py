
import numpy as np

import matplotlib.cm as cm

for i in range(12,13):
    infn = './bremen_city/scan%03d.txt' % i
    outfn = '/opt/wingated/nancy/bremen%03d.npy' % i
    print "%s -> %s" % ( infn, outfn )

    A = np.loadtxt( infn, delimiter=' ', skiprows=1 )
    np.save( outfn, A, allow_pickle=False )
