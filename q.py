#import numpy as np
import autograd.numpy as np
import scipy.stats as ss
from dautograd import dgrad_named

#
# ==================================================================
#

class choice_erp:
    @staticmethod
    def diffparms():
        return ["p"]

    @staticmethod
    def sample( p=[1.0] ):
        p = p.ravel()
        p /= np.sum( p )  # ties the parameters together
        return np.random.choice( range(len(p)), p=p )

    @staticmethod
    def score( X, p=[1.0] ):
        p = p.ravel()
        p /= np.sum( p )  # ties the parameters together
        return np.sum( np.log( p[X] ) )

    @staticmethod        
    def new_var_params( p=[1.0] ):
        cnt = np.prod( p.shape )
        return { "p": (1.0/(float(cnt))) * np.ones( p.shape ) }

# -------------------------------------------
    
class randn_erp:
    @staticmethod
    def diffparms():
        return ["mu","sigma"]

    @staticmethod
    def sample( sz=(1,1), mu=0.0, sigma=1.0 ):
        return mu + sigma*np.random.randn( sz[0], sz[1] )
    
    @staticmethod
    def score( X, sz=(1,1), mu=0.0, sigma=1.0 ):
        return np.sum( ss.norm.logpdf( x, loc=mu, scale=sigma ) )

    @staticmethod    
    def new_var_params( sz=(1,1), mu=0.0, sigma=1.0 ):
        return { "mu": np.zeros( sz ),
                 "sigma": np.ones( sz ) }

# -------------------------------------------

class flip_erp:
    @staticmethod
    def diffparms():
        return ["p"]

    @staticmethod
    def sample( sz=(1,1), p=0.5 ):
        return np.random.rand( *sz ) > p

    @staticmethod
    def score( X, sz=(1,1), p=0.5 ):
        return np.sum( X * np.log(p) - (1.0-X)*np.log(1.0-p) )

    @staticmethod
    def new_var_params( sz=(1,1), p=0.5 ):
        return { "p": 0.5*np.ones( sz ) }

#
# ==================================================================
#
    
class Q( object ):

    def __init__( self ):

        self.model = None
        self.inject_q_objs = False

        self.var_db = {}
        self.cond_db = {}

        # this gets reset at each gradient iteration
        self.grad_db = {}

        # this gets reset at each rollout
        self.cur_grad_db = {}
        self.cur_trace_score = 0.0

        self.always_sample = False

        self.choice = self.make_erp( choice_erp )
        self.randn = self.make_erp( randn_erp )
        self.flip = self.make_erp( flip_erp )

#
# ------------------------------------------------------
#

    def condition( self, name=None, value=None ):
        self.cond_db[ name ] = value

    def make_erp( self, erp_class ):

        # we use autograd to automatically create a differentiable
        # score function for each ERP type.
        erp_class.var_grads = {}
        for p in erp_class.diffparms():
            erp_class.var_grads[p] = dgrad_named( erp_class.score, p )

        return lambda *args, **kwargs: self.do_var_erp( erp_class, *args, **kwargs )

    def set_model( self, model=None ):
        self.model = model # should be an object

#
# ------------------------------------------------------
#

    def run_model( self ):
        self.cur_trace_score = 0.0
        self.cur_grad_db = {}
        return self.model.run( self ) # passes this Q object in as second parameter

#
# ------------------------------------------------------
#

    def analyze( self ):
        if self.model == None:
            raise(Exception('Must specify a model before analysis'))

        self.inject_q_objs = True
        self.run_model()
        self.inject_q_objs = False

        #
        # collect and analyze results!
        #

        # initialize all of the gradient accumulators
        for k in self.cur_grad_db:
            self.grad_db[ k ] = {}
            for j in self.cur_grad_db[k]:
                self.grad_db[ k ][ j ] = 0.0

#
# ------------------------------------------------------
#

    def calc_rolled_out_gradient( self, cnt=100 ):

        total_score = 0.0

        # perform cnt rollouts
        for i in range(cnt):
            self.run_model()
            total_score += self.cur_trace_score
            for k in self.cur_grad_db:
                for j in self.cur_grad_db[k]:
                    self.grad_db[ k ][ j ] += self.cur_trace_score * self.cur_grad_db[ k ][ j ]

        # XXX we should have a baseline estimator here!

        # normalize
        for k in self.cur_grad_db:
            for j in self.cur_grad_db[k]:
                    self.grad_db[ k ][ j ] /= cnt

        return total_score / float(cnt)
#
# ------------------------------------------------------
#
    
    def do_var_erp( self, erp_class, *args, **kwargs ):

        if kwargs.has_key( 'name' ):
            name = kwargs['name']
            del kwargs['name']
        else:
            raise(Exception('All ERPs must have a name!'))

        if self.var_db.has_key( name ):
            var_params = self.var_db[ name ]
        else:
            var_params = erp_class.new_var_params( *args, **kwargs )
            self.var_db[ name ] = var_params

        if self.cond_db.has_key( name ):
            new_val = self.cond_db[ name ]
            trace_score = erp_class.score( new_val, *args, **kwargs )
            my_score = -trace_score

        else:
            # we always sample from the variational distribution
            new_val = erp_class.sample( **var_params )
            # score under the variational parameters and the trace parameters
            var_score = erp_class.score( new_val, **var_params )
            trace_score = erp_class.score( new_val, *args, **kwargs )

            tmp = {}
            for p in erp_class.diffparms():            
                tmp[p] = erp_class.var_grads[p]( new_val, **var_params )
            self.cur_grad_db[name] = tmp
            my_score = var_score - trace_score

        # XXX broadcast this score only to parents in the dependency graph!

        self.cur_trace_score += my_score

        if self.inject_q_objs:
            return new_val
        else:
            return new_val

#
# ------------------------------------------------------
#
    
    def opt_sgd( self, alpha=0.01, itercnt=100, rolloutcnt=10 ):
        results = []
        scores = []

        for iter in range( 0, itercnt ):
            score = self.calc_rolled_out_gradient( cnt=rolloutcnt )
            print "\n%d: %.2f" % ( iter, score )
            results.append( np.copy( self.var_db[ "gloc" ][ "p" ] ) )
            scores.append( score )

            for k in self.grad_db:
                for j in self.grad_db[ k ]:
                    self.var_db[ k ][ j ] -= alpha * self.grad_db[ k ][ j ]

        return results, scores
#
# ------------------------------------------------------
#

    def opt_adam( self, alpha=0.01, beta_1=0.9, beta_2=0.999, epsilon=10e-8, itercnt=100, rolloutcnt=10 ):
        results = []
        scores = []

        m_t = {}
        v_t = {}
        for k in self.grad_db:
            m_t[k] = {}
            v_t[k] = {}
            for j in self.grad_db[ k ]:
                m_t[k][j] = 0
                v_t[k][j] = 0

        for iter in range( 0, itercnt ):

            score = self.calc_rolled_out_gradient( cnt=rolloutcnt )
            print "\n%d: %.2f" % ( iter, score )

            results.append( np.copy( self.var_db[ "gloc" ][ "p" ] ) )
            scores.append( score )

            for k in self.grad_db:
                for j in self.grad_db[ k ]:
                    m_t[k][j] = beta_1 * m_t[k][j] + (1.0 - beta_1) * self.grad_db[ k ][ j ]
                    v_t[k][j] = beta_2 * v_t[k][j] + (1.0 - beta_2) * self.grad_db[ k ][ j ]**2.0

            for k in self.grad_db:
                for j in self.grad_db[ k ]:
                    m_hat_t = m_t[k][j] / (1.0 - beta_1**float(iter+1))
                    v_hat_t = v_t[k][j] / (1.0 - beta_2**float(iter+1))
            
                    self.var_db[ k ][ j ] -= alpha * m_hat_t / (v_hat_t**0.5 + epsilon)

        return results, scores
 
#
# ==================================================================
#
