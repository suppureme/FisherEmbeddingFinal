import sys

import sklearn
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score
from functools import partial

from sklearn.metrics import silhouette_score
from sklearn.model_selection import train_test_split
import vix_utils
#import pandas_datareader as pdr
from datetime import datetime
from datetime import timedelta
from MCMC.DataframeManipulator import DataframeManipulator
from MCMC.Misc import Misc
from MCMC.Indicator import *
from scipy.stats import percentileofscore
from sklearn.metrics import normalized_mutual_info_score, adjusted_mutual_info_score, mutual_info_score
from sklearn.feature_selection import mutual_info_classif
from MCMC.Metrics import *

# define an example,
# define a method to define alpha functions bound with dataframe


class MCMC():

    def __init__(self, alpha_fn, alpha_fn_params_0, target, num_iters, prior, burn = 0.00, optimize_fn = None, lower_limit = -10000,
                 upper_limit = 10000):
        self.alpha_fn = alpha_fn
        self.lower_limit = lower_limit
        self.upper_limit = upper_limit
        self.initial_params = alpha_fn_params_0
        self.target = target
        if optimize_fn is not None:
            self.optimize_fn = optimize_fn
        else:
            self.optimize_fn = MCMC.nmi
        self.num_iters = num_iters
        self.burn = burn
        self.prior = prior

    def transition_fn(self, cur, iter ):

        #print("Inside transition_fn")


        std = self.std_guess( iter, self.num_iters )
        new_guesses = []
        for c, s in zip( cur, std):

            #print("Inside for loop")

            loop = True
            while loop:

                #print("Inside while loop")

                new_guess = np.random.normal( c, s, (1,))

                #print(f"New guess {new_guess}")
                #print(f"c: {c}")
                #print(f"s: {s}")

                if new_guess[0] <= self.upper_limit and new_guess[0] >= self.lower_limit:
                    new_guesses.append( new_guess[0] )
                    loop = False
        return list( new_guesses )

    @staticmethod
    def __to_percentile( arr ):
        pct_arr = []
        for idx in range( 0, len(arr)):
            pct_arr.append( round( percentileofscore( np.array( arr ), arr[ idx ]  ) ) )
        return pct_arr

    @staticmethod
    def __shan_entropy(c):
        c_normalized = c / float(np.sum(c))
        c_normalized = c_normalized[np.nonzero(c_normalized)]
        H = -sum(c_normalized * np.log2(c_normalized))
        return H

    @staticmethod
    def nmi( X, Y, bins ):

        c_XY = np.histogram2d(X, Y, bins)[0]
        c_X = np.histogram(X, bins)[0]
        c_Y = np.histogram(Y, bins)[0]

        H_X = MCMC.__shan_entropy(c_X)
        H_Y = MCMC.__shan_entropy(c_Y)
        H_XY = MCMC.__shan_entropy(c_XY)

        NMI = 2*(H_X + H_Y - H_XY)/(H_X+H_Y)
        return NMI


    def do_step(self, iter, prev_params, prev_nmi ):

        #print("Inside do_step")

        next_params = self.transition_fn( prev_params, iter )

        if self.prior( next_params ) != 0:

            #y_pred = MCMC.__to_percentile( self.alpha_fn( *next_params ) )
            #print( y_pred )
            #y_true = MCMC.__to_percentile( self.target )
            #print( y_true )

            X = self.alpha_fn( *next_params )
            Y = self.target
            next_nmi = self.optimize_fn( X , Y, round( len( X )/5 ) )

            #print("Iter:", iter)
            #print( "Next MI:" + str( next_nmi ))

            if next_nmi > prev_nmi:
                # print( "Exploit:")
                # print( next_nmi )
                # print( next_params )
                #print( self.std_guess(iter, self.num_iters))
                #print( self.explore_factor(iter, self.num_iters))
                return [ next_params, next_nmi ]
            else:
                ratio = next_nmi/prev_nmi

                uniform = np.random.uniform(0,1 )
                if ratio > uniform * self.explore_factor( iter, self.num_iters ):
                    # print("Explore:")
                    # print(next_nmi)
                    # print(next_params)
                    #print(self.std_guess(iter, self.num_iters))
                    #print(self.explore_factor(iter, self.num_iters))
                    return [ next_params, next_nmi ]
                else:
                    return [ prev_params, prev_nmi ]
        else:
            return [ prev_params, prev_nmi ]

    def optimize(self):

        prev_params = self.initial_params
        [ prev_params, prev_nmi] = self.do_step( 0, prev_params, -1 )
        all_results = []

        for i in range( 0, self.num_iters):
            # print( i )
            # if round( i / 100 ) == i/100:
            #     print( "Current: "  + str( i ) + " of " + str( self.num_iters ))
            [next_params, next_nmi] = self.do_step( i, prev_params, prev_nmi)
            all_results.append( [next_params, next_nmi, i ])
            prev_params = next_params
            prev_nmi = next_nmi

        return all_results

    def explore_factor( self, iter, num_iters ):
        if iter < 0.1 * num_iters:
            return 0.5
        if iter < 0.3 * num_iters:
            return 0.8
        if iter < 0.5 * num_iters:
            return 1
        if iter < 0.75 * num_iters:
            return 1.5
        if iter < 0.8 * num_iters:
            return 2
        if iter < 0.9 * num_iters:
            return 3
        if iter < 1 * num_iters:
            return 4
        return 5
        #return 0.1

    def std_guess( self, iter, num_iters ):
        stds = []
        guesses = self.initial_params
        for guess in guesses:
            num_digits = len( str( round(guess) ))
            std = (10 ** ( num_digits-2 ))
            if iter < 0.5 * num_iters:
                std_factor = 2
            elif iter < 0.65 * num_iters:
                std_factor = 1
            elif iter < 0.85 * num_iters:
                std_factor = 0.75
            elif iter < 0.95 * num_iters:
                std_factor = 0.5
            elif iter < 0.99 * num_iters:
                std_factor = 0.1
            elif iter < num_iters:
                std_factor = 0.01
            #std_factor = 0.1
            stds.append( std * std_factor )
        return stds


    def analyse_results(self, all_results, top_n = 5 ):
        params = [ x[0] for x in all_results[round(self.burn*len(all_results)):]]
        nmis = [ x[1] for x in all_results[round(self.burn*len(all_results)):]]
        iteration = [x[2] for x in all_results[round(self.burn * len(all_results)):]]
        best_nmis = sorted( nmis, reverse=True)
        best_nmis = best_nmis[:top_n]

        best_params = []
        best_nmi = []
        best_iteration = []

        for p, n, it in zip( params, nmis, iteration ):
            if n >= best_nmis[-1]:
                best_params.append( p )
                best_nmi.append( n )
                best_iteration.append(it)
            if len( best_nmi ) == top_n:
                break

        return best_params, best_nmi, best_iteration


class MCMC_Indicator( MCMC ):


    def __init__(self, indicator, initial_args, feature, target_col, df, num_iters, prior, fltr ):
        self.target_col = target_col
        self.filter = fltr
        self.indicator = indicator
        self.feature = feature
        self.df = df
        MCMC.__init__( self, alpha_fn=self.create_alpha_fn(), alpha_fn_params_0=self.create_alpha_args( initial_args ),
                       target = self.create_target(),
                       num_iters = num_iters, prior=prior )


    def transition_fn(self, cur, iter ):
        std = self.std_guess( iter, self.num_iters )
        return [ round( x ) for x in np.random.normal( cur, std, ( len(cur),) ) ]


    def create_alpha_fn(self):
        indicator = self.indicator
        def alpha_fn( *args_to_optimize ):
            feature = self.feature
            df = self.df
            ind_args = list( args_to_optimize )
            print( ind_args)
            ind_args.append( feature )
            print( "Indicator initialization args")
            print( ind_args )
            id = indicator(*ind_args)
            print( "Indicator application args" )
            modified_df = id.apply( df )

            modified_df = modified_df.drop([ self.target_col ], axis = 1)
            modified_df = pd.concat( [ modified_df, self.df[ self.target_col ] ], axis = 1, join="inner")
            modified_df = self.filter( modified_df, id.describe(), self.target_col )
            modified_df = modified_df.dropna()

            self.target = modified_df[ self.target_col ].values
            return modified_df[ id.describe() ].values

        return alpha_fn

    def create_target(self):
        target = self.df[ self.target_col ]
        print( target.tail(10))
        return target

    def create_alpha_args(self, args ):

        all_args = args

        print( "Alpha args")
        print( all_args )
        return all_args





simple_test = True
if __name__ == "__main__":

    if simple_test:

        m = np.random.normal( 0, 1, 5000 )
        obs = [ 3 ** x for x in m ]

        def optim_fn( x, y, bins ):
            diff = [ (a-b)*(a-b) for a, b in zip( x,y)]
            return -sum( diff )

        def alpha_fn( a, b ):
    #        u = np.random.normal( 0, 1, 5000 )
            y = [( a * b ) ** x for x in m ]
            return y


        def prior( params ):
            if params[ 0 ] < 0 or params[1] < 0:
                return 0
            return 1



    #    from functools import partial
        guess = [ 3, 2 ]
    #    b_std = partial( std_guess, guess )

        mc = MCMC( alpha_fn, guess, obs, 10000, prior =  prior, optimize_fn=optim_fn )
        rs = mc.optimize()
        print( mc.analyse_results( rs, top_n=2 ))

    else:

        def prior( params ):
            if params[ 0 ] < 5:
                return 0
            return 1

        def fltr( df, indicator_col, target_col ):
            p = Pivot( 1, indicator_col )
            new_df = p.apply( df )
            new_df = new_df[new_df[p.describe()] == 1]
            #mask = (new_df[target_col] >= 1) | (new_df[target_col] <= -1)
            #new_df = new_df[ mask ]

            return new_df

        from Backtester import Backtester

        b = Backtester(["FCEL", "MSTR"], {"FCEL": "united states", "MSTR": "united states"}, {}, 0.001, 0.002, None)
        b.insert_rebal_set("1/1/2021", "FCEL", 0.5)
        b.insert_rebal_set("1/1/2021", "MSTR", 0.5)
        b.insert_rebal_set("21/2/2021", "FCEL", 0)
        b.insert_rebal_set("21/2/2021", "MSTR", 0)
        b.insert_rebal_set("1/5/2021", "FCEL", 1)
        b.insert_rebal_set("1/5/2021", "MSTR", 0)
        b.do_work("1/1/2013", "6/5/2021")
        df = b.full_data
        rr = RollingFSharpe( 30, "FCEL_Close" )
        df = rr.apply( df )
        dfm = DataframeManipulator( df )
        dfm.add_lookback_func( rr.describe(), "percentile", 260, new_column_name=rr.describe() + "_P" )
        df = dfm.df
        df = df.dropna()
        mcmc = MCMC_Indicator( Fisher, [ 100 ], "FCEL_Close", rr.describe(), df, 1000, prior, fltr )
        rs = mcmc.optimize()
        print( mcmc.analyse_results( rs, top_n=10 ))





