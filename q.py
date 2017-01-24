import numpy as np
import scipy.stats as ss

#
# ==================================================================
#

class choice_erp:
    @staticmethod
    def sample( p=[1.0] ):
        p = p.ravel()
        return np.random.choice( range(len(p)), p=p )

    @staticmethod
    def score( X, p=[1.0] ):
        p = p.ravel()
        return np.sum( np.log( p[X] ) )

    @staticmethod        
    def new_var_params( p=[1.0] ):
        cnt = np.prod( p.shape )
        return { "p": (1.0/(float(cnt))) * np.ones( p.shape ) }

# -------------------------------------------
    
class randn_erp:
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
    def sample( sz=(1,1), p=0.5 ):
        return np.random.rand( *sz ) > p

    @staticmethod
    def score( X, sz=(1,1), p=0.5 ):
        return np.sum( ss.bernoulli.logpmf( X, p ) )

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

        self.always_sample = False

        self.trace_score = 0.0

        self.choice = self.make_erp( choice_erp )
        self.randn = self.make_erp( randn_erp )
        self.flip = self.make_erp( flip_erp )



    def set_model( self, model=None ):
        self.model = model # should be an object

    def run_model( self ):
        self.trace_score = 0.0
        return self.model.run( self ) # passes this Q object in as second parameter

    def analyze( self ):
        if self.model == None:
            raise(Exception('Must specify a model before analysis'))

        self.inject_q_objs = True
        self.run_model()
        self.inject_q_objs = False

        # XXX collect and analyze results!
        
    def condition( self, name=None, value=None ):
        self.cond_db[ name ] = value

    def make_erp( self, erp_class ):
       return lambda *args, **kwargs: self.do_var_erp( erp_class, *args, **kwargs )
    
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
            self.trace_score += -trace_score

        else:
            # we always sample from the variational distribution
            new_val = erp_class.sample( **var_params )
            # score under the variational parameters and the trace parameters
            var_score = erp_class.score( new_val, **var_params )
            trace_score = erp_class.score( new_val, *args, **kwargs )
            self.trace_score += var_score - trace_score

        if self.inject_q_objs:
            return new_val
        else:
            return new_val
    
#
# ==================================================================
#

    # def mk_name( self, name ):
    #     if name == None:
    #         # someday, might replace this with auto-naming?
    #         # check out the inspect module...
    #         raise( Exception('All ERPs must have a name!') )
    #     return name
#            name = None
#        name = self.mk_name( name )


    # def make_erp( self, samp_func, score_func=None ):
    #    return lambda *args, **kwargs: self.do_erp( samp_func, *args, **kwargs )

    # def do_erp( self, samp_func, *args, **kwargs ):
    #     if kwargs.has_key( 'name' ):
    #         name = kwargs['name']
    #     else:
    #         name = None

    #     if not self.always_sample:
    #         name = self.mk_name( name )
    #         if self.db.has_key( name ):
    #             return self.db[ name ]

    #     new_val = samp_func( *args )
    #     self.db[ name ] = new_val

    #     return new_val
    
