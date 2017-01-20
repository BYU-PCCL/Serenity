import numpy as np
import scipy.stats as ss

class Q( object ):

    def __init__( self ):
        self.db = {}
        self.always_sample = False

        self.choice = self.make_erp( np.random.choice )
        self.randint = self.make_erp( np.random.randint )
        self.rand = self.make_erp( np.random.rand )
        self.randn = self.make_erp( np.random.randn )

        self.dirichlet = self.make_erp( ss.dirichlet.rvs, ss.dirichlet.logpdf )

    def make_erp( self, samp_func, score_func=None ):
        return lambda *args, **kwargs: self.do_erp( samp_func, *args, **kwargs )

    def do_erp( self, samp_func, *args, **kwargs ):
        if kwargs.has_key( 'name' ):
            name = kwargs['name']
        else:
            name = None

        if not self.always_sample:
            name = self.mk_name( name )
            if self.db.has_key( name ):
                return self.db[ name ]

        new_val = samp_func( *args )
        self.db[ name ] = new_val

        return new_val

    def mk_name( self, name ):
        if name == None:
            # someday, might replace this with auto-naming?
            raise( Exception('All ERPs must have a name!') )
        return name

