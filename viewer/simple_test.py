
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

results, scores = Q.opt_sgd()
results, scores = Q.opt_adam()

data = np.vstack( results )
plt.plot( data ); plt.show()

