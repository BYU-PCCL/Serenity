
import os,sys,inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0,parentdir) 
import q

import matplotlib.pyplot as plt
import full_world
from my_rrt import *

bdata = load_polygons( "./paths.txt" )
model = full_world.Fullworld()

Q = q.Q( model )
Q.condition( name="data", value=True )  # condition on the fact that we DID see the intruder
Q.analyze()

#results, scores = Q.opt_adam( itercnt=1000, rolloutcnt=10 )
#results, scores = Q.opt_sgd( alpha=0.001, itercnt=1000, rolloutcnt=10 )

#results, scores = Q.opt_adam( alpha=0.001, itercnt=300, rolloutcnt=10 )
#results, scores = Q.opt_adam( alpha=0.001, itercnt=300, rolloutcnt=3 )
results, scores = Q.opt_adam( alpha=0.0025, itercnt=100, rolloutcnt=10 )

'''
start_loc,goal_loc,isfnd,rrt_path,ints = Q.run_model()

if isfnd:
    color = 'r'
else:
    color = 'b'

plt.figure()

for s in bdata:
    for i in range( 0, s.shape[0]-1 ):
        plt.plot( [ s[i,0], s[i+1,0] ],
                  [ s[i,1], s[i+1,1] ], 'k' )

for i in range( 0, len(rrt_path)-1 ):
    plt.plot( [ rrt_path[i][0], rrt_path[i+1][0] ],
              [ rrt_path[i][1], rrt_path[i+1][1] ], color )

ints.append(ints[0])
for i in range( 0, len(ints)-1 ):
    plt.plot( [ ints[i][0], ints[i+1][0] ],
              [ ints[i][1], ints[i+1][1] ], 'g' )

plt.scatter( model.UAVLocation[0], model.UAVLocation[1] )
plt.scatter( start_loc[0,0], start_loc[0,1] )
plt.scatter( goal_loc[0,0], goal_loc[0,1] )
plt.xlim( 0, 1 ); plt.ylim( 0, 1 )
plt.show()
'''
