
import os,sys,inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0,parentdir) 
import q
import erps

import matplotlib.pyplot as plt
import time_world
from my_rrt import *

bdata = load_polygons( "./paths.txt" )
model = time_world.Timeworld()

Q = q.Q( model )

data = np.zeros((2,100))
data[1,10:18] = 1
Q.condition( name="data", value=data )  # condition on the fact that we DID see the intruder at certain times

Q.analyze()

start_loc,goal_loc,data,rrt_path,pvals = Q.run_model()

#data = erps.flip_erp.sample( p=pvals )

#results, scores = Q.opt_adam( itercnt=1000, rolloutcnt=10 )
#results, scores = Q.opt_sgd( alpha=0.001, itercnt=1000, rolloutcnt=10 )

#results, scores = Q.opt_adam( alpha=0.001, itercnt=300, rolloutcnt=10 )
#results, scores = Q.opt_adam( alpha=0.001, itercnt=300, rolloutcnt=3 )
#results, scores = Q.opt_adam( alpha=0.0025, itercnt=300, rolloutcnt=10 )

results, scores = Q.opt_adam( alpha=0.01, itercnt=300, rolloutcnt=10 )

parm_data = np.vstack( results )
plt.figure(); plt.plot( parm_data ); plt.show()


plt.figure()

for s in bdata:
    for i in range( 1, s.shape[0]-1 ):
        plt.plot( [ s[i,0], s[i+1,0] ],
                  [ s[i,1], s[i+1,1] ], 'k' )

for p in range(0,100):
    start_loc,goal_loc,jed_data,rrt_path,ints = Q.run_model()
    for i in range( 0, len(rrt_path)-1 ):
        if data[0,i] or data[1,i]:
            color = 'r'
        else:
            color = 'b'

        plt.plot( [ rrt_path[i][0], rrt_path[i+1][0] ],
                  [ rrt_path[i][1], rrt_path[i+1][1] ], color )

plt.scatter( model.sensors[0,0], model.sensors[0,1] )
plt.scatter( model.sensors[1,0], model.sensors[1,1] )

plt.scatter( start_loc[0,0], start_loc[0,1] )
plt.scatter( goal_loc[0,0], goal_loc[0,1] )

bob = np.copy( results[-1] )
bob = bob - np.min( bob )
bob = bob / np.max( bob )
plt.scatter( model.locs[:,0], model.locs[:,1], s=1000*bob )

plt.xlim( -0.05, 0.95 ); plt.ylim( -0.05, 1.05 )
plt.show()

#plt.scatter( model.UAVLocation[0], model.UAVLocation[1] )
#ints.append(ints[0])
#for i in range( 0, len(ints)-1 ):
#    plt.plot( [ ints[i][0], ints[i+1][0] ],
#              [ ints[i][1], ints[i+1][1] ], 'g' )

