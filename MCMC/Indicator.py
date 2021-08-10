import sklearn
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import scipy
from hurst import compute_Hc, random_walk
import pandas as pd
import numpy as np
import datetime
import math
import mplfinance as mpf
import investpy
import sklearn
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score
from sklearn.metrics import silhouette_score
from sklearn.model_selection import train_test_split
import vix_utils
#import pandas_datareader as pdr
from datetime import datetime
from datetime import timedelta
from MCMC.DataframeManipulator import DataframeManipulator
from MCMC.Misc import Misc
class Indicator:


    def __init__(self, feature ):
        self.feature = feature

    def apply(self, df ):
        pass

    def describe(self):
        return ( "SHELL_INDICATOR" )


class EMA( Indicator ):

    def __init__(self, period, feature ):
        self.period = period
        Indicator.__init__( self, feature )

    def describe(self):
        return ( "EMA_" + self.feature + "_" + str( self.period ))

    def apply(self, df ):
        dfm = DataframeManipulator( df )
        dfm.add_lookback_func( self.feature, "ema", self.period, self.describe() )
        return dfm.df



class CHG(Indicator):

    def __init__(self, period, feature):
        self.period = period
        Indicator.__init__(self, feature)

    def describe(self):
        return ("CHG_" + self.feature + "_" + str(self.period))


    def apply(self, df):
        dfm = DataframeManipulator(df)
        dfm.look_back(self.feature, self.period )
        dfm.extend_with_func( Misc.change, self.describe(), ( self.period, self.feature, ) )
        return dfm.df



class ROC(Indicator):

    def __init__(self, period, feature):
        self.period = period
        Indicator.__init__(self, feature)

    def describe(self):
        return ("ROC_" + self.feature + "_" + str(self.period))


    def apply(self, df):
        dfm = DataframeManipulator(df)
        dfm.look_back(self.feature, self.period )
        dfm.extend_with_func( Misc.roc_pct, self.describe(), ( self.period, self.feature, ) )
        return dfm.df


class RSI(Indicator):

    def __init__(self, period, feature):
        self.period = period
        Indicator.__init__(self, feature)

    def describe(self):
        return ("RSI_" + self.feature + "_" + str(self.period))

    @staticmethod
    def __is_gain(row, roc ):
        if row[ roc.describe() ] > 0:
            return 1
        else:
            return 0

    @staticmethod
    def __is_loss( row, roc ):
        if row[ roc.describe() ] < 0:
            return 1
        else:
            return 0

    @staticmethod
    def __zero_out_gains(row, roc ):
        if row[ roc.describe() ] > 0:
            return 0
        else:
            return row[ roc.describe() ]

    @staticmethod
    def __zero_out_losses(row, roc ):
        if row[ roc.describe() ] < 0:
            return 0
        else:
            return row[ roc.describe() ]

    def apply(self, df):
        self.chg = CHG( 1, self.feature )
        df = self.chg.apply( df )
        dfm = DataframeManipulator( df )
        adv = "__adv_" + self.feature + "_" + str( self.period )
        dec = "__dec_" + self.feature + "_" + str( self.period )
        n_adv = "__n_adv_" + self.feature + "_" + str( self.period )
        n_dec = "__n_dec_" + self.feature + "_" + str( self.period )


        dfm.extend_with_func( RSI.__zero_out_losses, adv, ( self.chg, ) )
        dfm.extend_with_func( RSI.__zero_out_gains, dec, ( self.chg, ) )
        dfm.extend_with_func( RSI.__is_loss, n_dec, ( self.chg, ) )
        dfm.extend_with_func( RSI.__is_gain, n_adv, ( self.chg, ) )

        dfm.add_lookback_func( adv, "rma", self.period, adv + "_mean" )
        dfm.add_lookback_func( dec, "rma", self.period, dec + "_mean" )
        dfm.add_lookback_func( n_adv, "sum", self.period, n_adv + "_sum" )
        dfm.add_lookback_func( n_dec, "sum", self.period, n_dec + "_sum" )

        dfm.extend_with_func( Misc.rsi, self.describe(), ( adv + "_mean", dec + "_mean", n_adv + "_sum", n_dec + "_sum" ) )
        dfm.delete_cols( [ adv + "_mean", dec + "_mean", n_adv + "_sum", n_dec + "_sum", n_adv, n_dec, adv, dec ])
        return dfm.df

class ConstanceRSI(Indicator):

    def __init__(self, rsi_period, feature, rsi_ma_period = 3, rsi_delta_len = 9, rsi_ma_len = 3, sma_fast_len = 13, sma_slow_len = 33 ):
        self.period = rsi_period
        self.rsi_ma_period = rsi_ma_period
        self.rsi_ma_len = rsi_ma_len
        self.rsi_delta_len = rsi_delta_len
        self.sma_fast_len = sma_fast_len
        self.sma_slow_len = sma_slow_len
        Indicator.__init__(self, feature)

    def describe(self):
        return ("CRSI_" + self.feature + "_" + str(self.period))


    @staticmethod
    def __rsi_composite( row, rsi_delta, rsi_ma ):
        return row[ rsi_delta ] + row[rsi_ma]

    def apply(self, df):
        self.rsi = RSI( self.period, self.feature )
        self.rsi_ma = RSI( self.rsi_ma_period, self.feature )
        df = self.rsi.apply( df )
        df = self.rsi_ma.apply( df )
        dfm = DataframeManipulator( df )
        dfm.add_lookback_func( self.rsi_ma.describe() , "sma", self.rsi_ma_len, self.rsi_ma.describe() + "_mean")
        dfm.look_back( self.rsi.describe(), self.rsi_delta_len )
        dfm.extend_with_func( Misc.change, "__RSI_Delta", ( self.rsi_delta_len, self.rsi.describe(),  ) )
        dfm.extend_with_func( ConstanceRSI.__rsi_composite, self.describe(), ( "__RSI_Delta", self.rsi_ma.describe() + "_mean"))
        dfm.add_lookback_func( self.describe(), "sma", self.sma_fast_len )
        dfm.add_lookback_func( self.describe(), "sma", self.sma_slow_len )

        return dfm.df

class Pivot(Indicator):
    def __init__(self, period, feature ):
        self.period = period
        Indicator.__init__(self, feature)

    def describe(self):
        return ("Pivot_" + self.feature + "_" + str(self.period) )

    @staticmethod
    def __is_l_pivot(row, lb, lf, feature):
        cur = row[feature]
        nxt = row[lf]
        prv = row[lb]

        if cur < nxt and cur < prv:
            return 1
        return 0

    @staticmethod
    def __is_pivot(row, l_pivot, h_pivot):
        if row[ l_pivot ] == 1 or row[ h_pivot ] == 1:
            return 1
        return 0

    @staticmethod
    def __is_h_pivot( row, lb, lf, feature ):
        cur = row[ feature ]
        nxt = row[ lf ]
        prv = row[ lb ]

        if cur > nxt and cur > prv:
            return 1
        return 0


    def apply(self, df):
        dfm = DataframeManipulator(df)
        dfm.look_forward( self.feature, self.period )
        dfm.look_back( self.feature, self.period )

        dfm.extend_with_func( Pivot.__is_h_pivot, self.describe() + "_HI", ( self.feature + "_T+" + str( self.period ),
                                                                             self.feature + "_T-" + str( self.period ),
                                                                             self.feature,) )
        dfm.extend_with_func( Pivot.__is_l_pivot, self.describe() + "_LO", ( self.feature + "_T+" + str( self.period ),
                                                                             self.feature + "_T-" + str( self.period ),
                                                                             self.feature ,) )
        dfm.extend_with_func( Pivot.__is_pivot, self.describe(), ( self.describe() + "_LO", self.describe() + "_HI",) )
        return dfm.df

class Fisher(Indicator):

    def __init__(self, period, feature, high = None, low = None ):
        self.period = period
        if high is None:
            high = feature.replace( "_Close","_High")
        if low is None:
            low = feature.replace( "_Close","_Low")
        self.high = high
        self.low = low
        Indicator.__init__(self, feature)

    def describe(self):
        return ("Fisher_" + self.feature + "_" + str(self.period))


    @staticmethod
    def __rsi_composite( row, rsi_delta, rsi_ma ):
        return row[ rsi_delta ] + row[rsi_ma]

    @staticmethod
    def __v_raw( row, period ):
        hl2 = row[ "HL2" ]
        hl2_max = row["HL2_max_"+ str(period)]
        hl2_min = row["HL2_min_"+ str(period)]
        v = 2 * ( hl2 - hl2_min)/(hl2_max-hl2_min) - 1
        return v

    @staticmethod
    def __fisher_raw( row ):
        v = row[ "V" ]
        if v > 0.99:
            v = 0.999
        elif v < -0.99:
            v = -0.999

        fish = math.log( (1 + v )/(1 - v ))
        return fish

    def apply(self, df):

        dfm = DataframeManipulator(df)
        dfm.make_hl2( self.high, self.low )
        dfm.add_lookback_func( "HL2", "max", self.period )
        dfm.add_lookback_func( "HL2", "min", self.period )
        dfm.df = dfm.df.dropna()
        dfm.extend_with_func( Fisher.__v_raw, "V_Raw", ( self.period, )  )
        df = dfm.df
        #print( dfm.df.tail(10) )
        df[ "V" ] = math.nan
        df.iloc[0, df.columns.get_loc("V")] = 0
        count_row = df.shape[0]
        for i in range(1, count_row):
            df.iloc[ i, df.columns.get_loc( "V" ) ] = 0.33 * df.iloc[ i, df.columns.get_loc( "V_Raw" )] + \
                                                      0.67 * df.iloc[ i-1, df.columns.get_loc( "V" ) ]

        dfm = DataframeManipulator( df )
        dfm.extend_with_func( Fisher.__fisher_raw, "FISH_Raw", ( ) )

        df = dfm.df
        df[  self.describe()  ]= math.nan
        df.iloc[ 0, df.columns.get_loc( self.describe()) ] = 0
        for i in range( 1, count_row ):
            df.iloc[ i, df.columns.get_loc(  self.describe() ) ] = \
                0.5 * df.iloc[ i, df.columns.get_loc( "FISH_Raw" )] +\
                0.5 * df.iloc[ i - 1, df.columns.get_loc(  self.describe() ) ]

        return df

# Next do future metrics
if __name__ == "__main__":
    from Backtester import Backtester
    b = Backtester(["FCEL", "MSTR"], {"FCEL":"united states","MSTR": "united states"}, {}, 0.001, 0.002,None)
    b.insert_rebal_set( "1/1/2021", "FCEL", 0.5 )
    b.insert_rebal_set( "1/1/2021", "MSTR", 0.5 )
    b.insert_rebal_set( "21/2/2021", "FCEL", 0 )
    b.insert_rebal_set( "21/2/2021", "MSTR", 0 )
    b.insert_rebal_set( "1/5/2021", "FCEL", 1 )
    b.insert_rebal_set( "1/5/2021", "MSTR", 0 )
    b.do_work( "1/1/2019", "6/5/2021" )
    df = b.full_data
    ema = EMA( 21, "FCEL_Close" )
    df = ema.apply( df )
    print( df[ ema.describe() ].tail(10 ) )

    roc = ROC( 1, "FCEL_Close" )
    df = roc.apply( df )
    print( df[ roc.describe() ].tail(10 ) )

    rsi = RSI( 14, "FCEL_Close" )
    df = rsi.apply( df )
    print( df[ rsi.describe() ].tail(10 ) )

    crsi = ConstanceRSI( 14, "FCEL_Close" )
    df = crsi.apply( df )
    print( df[ [ crsi.describe(), crsi.describe() + "_sma_13", crsi.describe()+"_sma_33" ] ].tail(10 ) )

    f = Fisher( 260, "FCEL_Close", "FCEL_High", "FCEL_Low" )
    df = f.apply( df )
    print( df[ [ f.describe()] ].tail(10 ) )

    # Changed lookforward to shift(-lookfwd) from -lookfwd +1 is this okay?
    p = Pivot( 1, crsi.describe() )
    df = p.apply( df )
    d = df[ df[p.describe()] == 1]
    print( d[ [ p.describe(), crsi.describe() ] ].tail(10 ) )
