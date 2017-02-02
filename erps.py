import autograd.numpy as np
import scipy.stats as ss

a2d = np.atleast_2d

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

    @staticmethod
    def project_param( name, val ):
        if name == 'p':
            val = a2d( val )
            val[val<0.0] = 0.0
            # val[val>1.0] = 1.0 # XXX optional, since we renormalize
            return val
        else:
            return val

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

    @staticmethod
    def project_params( name, val ):
        return val

# -------------------------------------------

class flip_erp:
    @staticmethod
    def diffparms():
        return ["p"]

    @staticmethod
    def sample( sz=None, p=0.5 ):
        p = a2d( p )
        if sz == None:
            sz = p.shape
        return np.random.rand( *sz ) < p

    @staticmethod
    def score( X, sz=None, p=0.5 ):
        epsilon = 1e-20
        return np.sum( X * np.log(p+epsilon) - (1.0-X)*np.log(1.0-p+epsilon) )

    @staticmethod
    def new_var_params( sz=None, p=0.5 ):
        if sz == None:
            sz = p.shape
        return { "p": 0.5*np.ones( sz ) }

    @staticmethod
    def project_param( name, val ):
        if name == 'p':
            val = a2d( val )
            val[val<0.0] = 0.0
            val[val>1.0] = 1.0
            return val
        else:
            return val
