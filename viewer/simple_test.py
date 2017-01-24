
import matplotlib.pyplot as plt
import numpy as np

import os,sys,inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0,parentdir) 

import q

from t_world import Tworld

model = Tworld()

Q = q.Q()
Q.set_model( model )
#Q.condition( name="data", value=True )

Q.analyze()

SS = 0.01

results = []

for iter in range( 0, 100 ):

    score = Q.calc_rolled_out_gradient( cnt=10 )
    print "\n%d: %.2f" % ( iter, score )

    results.append( np.copy( Q.var_db[ "gloc" ][ "p" ] ) )

    for k in Q.grad_db:
        for j in Q.grad_db[ k ]:
            print "[%s][%s]" % (k,j)
            print "  ", Q.var_db[ k ][ j ]
            print "  ", Q.grad_db[ k ][ j ]
            Q.var_db[ k ][ j ] -= SS * Q.grad_db[ k ][ j ]
