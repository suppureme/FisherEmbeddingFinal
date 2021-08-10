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
from MCMC.Indicator import CHG, ROC
class Metrics():

    def __init__(self,feature ):
        self.feature = feature
        pass

    def describe(self):
        return ( "Empty" )

    def apply(self, df ):
        return None

class RollingSharpe( Metrics ):
    def __init__(self, lookback, feature ):
        self.lookback = lookback
        Metrics.__init__( self, feature )

    def describe(self):
        return ( "RSharpe_" + str( self.lookback ) + "_" + self.feature)

    @staticmethod
    def __make_sharpe( row, roc_col, std_col ):
        roc = row[ roc_col ]
        std = row[ std_col ]
        return roc/std * math.sqrt( 252 )

    def apply(self, df):
        roc = ROC( self.lookback, self.feature )
        df = roc.apply( df )
        df[ self.describe() ] = df[ roc.describe() ]
        dfm = DataframeManipulator( df )
        dfm.add_lookback_func( self.feature, "std", self.lookback )
        dfm.extend_with_func( RollingSharpe.__make_sharpe, self.describe(), (roc.describe(), self.feature + "_std_" + str( self.lookback ),) )
        dfm.delete_cols( [ roc.describe(), self.feature + "_std_" + str( self.lookback ) ])
        return dfm.df


class RollingFSharpe(Metrics):
    def __init__(self, lookfwd, feature):
        self.lookfwd = lookfwd
        Metrics.__init__(self, feature)

    def describe(self):
        return ("RFSharpe_" + str(self.lookfwd) + "_" + self.feature)


    def apply(self, df):
        delete_cols = []
        sharpe = RollingSharpe( self.lookfwd, self.feature )
        try:
            idx = df.columns.get_loc( sharpe.describe())
        except:
            df = sharpe.apply( df )
            delete_cols.append( sharpe.describe() )
        dfm = DataframeManipulator( df )
        dfm.look_forward( sharpe.describe(), self.lookfwd - 1, self.describe())
        if delete_cols != []:
            dfm.delete_cols( delete_cols )
        return dfm.df


class RollingMax( Metrics ):
    def __init__(self, lookback, feature ):
        self.lookback = lookback
        Metrics.__init__( self, feature )


    def describe(self):
        return ( "RMX_" + str( self.lookback ) + "_" + self.feature)

    @staticmethod
    def __Max( row, du_col, feature ):
        return ( max( row[ feature ], row[ du_col ] ) )

    def apply(self, df):
        dfm = DataframeManipulator( df )
        dfm.add_lookback_func( self.feature, "max", self.lookback )
        dfm.extend_with_func( RollingMax.__Max, self.describe(), ( self.feature + "_max_" + str( self.lookback ), self.feature) )
        dfm.delete_cols( [ self.feature + "_max_" + str( self.lookback ) ])
        return dfm.df



class RollingFMax( Metrics ):
    def __init__(self, lookfwd, feature ):
        self.lookfwd = lookfwd
        Metrics.__init__( self, feature )


    def describe(self):
        return ( "RFMX_" + str( self.lookfwd ) + "_" + self.feature)

    def apply(self, df):
        to_delete = []
        rmx = RollingMax( self.lookfwd, self.feature )
        [ df, to_delete ] = Misc.apply_if_not_present( df, rmx, to_delete )
        dfm = DataframeManipulator( df )
        dfm.look_forward( rmx.describe(), self.lookfwd - 1, self.describe() )
        dfm.delete_cols( to_delete )
        return dfm.df


class RollingMin( Metrics ):
    def __init__(self, lookback, feature ):
        self.lookback = lookback
        Metrics.__init__( self, feature )


    def describe(self):
        return ( "RMN_" + str( self.lookback ) + "_" + self.feature)

    @staticmethod
    def __Min( row, dd_col, feature ):
        return ( min( row[ feature ], row[ dd_col ] ) )

    def apply(self, df):
        dfm = DataframeManipulator( df )
        dfm.add_lookback_func( self.feature, "min", self.lookback )
        dfm.extend_with_func( RollingMin.__Min, self.describe(), ( self.feature + "_min_" + str( self.lookback ), self.feature) )
        dfm.delete_cols( [ self.feature + "_min_" + str( self.lookback ) ])
        return dfm.df



class RollingFMin( Metrics ):
    def __init__(self, lookfwd, feature ):
        self.lookfwd = lookfwd
        Metrics.__init__( self, feature )


    def describe(self):
        return ( "RFMN_" + str( self.lookfwd ) + "_" + self.feature)

    def apply(self, df):
        to_delete = []
        rmn = RollingMin( self.lookfwd, self.feature )
        [ df, to_delete ] = Misc.apply_if_not_present( df, rmn, to_delete )

        dfm = DataframeManipulator( df )
        dfm.look_forward( rmn.describe(), self.lookfwd - 1, self.describe() )
        dfm.delete_cols( to_delete )
        return dfm.df

class RollingFDD( Metrics ):
    def __init__(self, lookback, feature ):
        self.lookback = lookback
        Metrics.__init__( self, feature )


    def describe(self):
        return ( "RFDD_" + str( self.lookback ) + "_" + self.feature)

    @staticmethod
    def __DD( row, dd_col, feature ):

        return ( row[ dd_col ]  - row[ feature ] )/row[ feature ]

    def apply(self, df):
        to_delete = []
        fmn = RollingFMin(self.lookback, self.feature)
        [df, to_delete] = Misc.apply_if_not_present(df, fmn, to_delete)
        dfm = DataframeManipulator( df )
        dfm.extend_with_func( RollingFDD.__DD, self.describe(), ( fmn.describe(), self.feature  ))

        dfm.delete_cols( to_delete )
        return dfm.df

class RollingFDU( Metrics ):
    def __init__(self, lookback, feature ):
        self.lookback = lookback
        Metrics.__init__( self, feature )


    def describe(self):
        return ( "RFDU_" + str( self.lookback ) + "_" + self.feature)

    @staticmethod
    def __DU( row, du_col, feature ):
        return ( row[ du_col ] - row[ feature ] )/row[ feature ]

    def apply(self, df):
        to_delete = []
        fmx = RollingFMax( self.lookback, self.feature )
        [df, to_delete] = Misc.apply_if_not_present(df, fmx, to_delete)

        dfm = DataframeManipulator( df )
        dfm.extend_with_func( RollingFDU.__DU, self.describe(), ( fmx.describe(), self.feature  ))
        dfm.delete_cols( to_delete )
        return dfm.df



class RollingReturn( Metrics ):
    def __init__(self, lookback, feature ):
        self.lookback = lookback
        Metrics.__init__( self, feature )


    def describe(self):
        return ( "RRet_" + str( self.lookback ) + "_" + self.feature)

    def apply(self, df):
        roc = ROC( self.lookback, self.feature )
        df = roc.apply( df )
        df[ self.describe() ] = df[ roc.describe() ].copy()
        dfm = DataframeManipulator( df )
        dfm.delete_cols( [ roc.describe() ])
        return dfm.df


class RollingFReturn(Metrics):
    def __init__(self, lookfwd, feature):
        self.lookfwd = lookfwd
        Metrics.__init__( self, feature)

    def describe(self):
        return ("RFRet_" + str(self.lookfwd) + "_" + self.feature)

    def apply(self, df):
        to_delete = []
        rr = RollingReturn( self.lookfwd, self.feature )
        [ df, to_delete ] = Misc.apply_if_not_present( df, rr, to_delete )

        dfm = DataframeManipulator( df )
        dfm.look_forward( rr.describe(), self.lookfwd - 1, self.describe() )
        dfm.delete_cols( to_delete )
        df = dfm.df
        return df



class RollingFRR(Metrics):
    def __init__(self, lookfwd, feature):
        self.lookfwd = lookfwd
        Metrics.__init__( self, feature)

    @staticmethod
    def __do_rr( row, du_col, dd_col ):
        if row[ dd_col ] == 0:
            return 100000
        else:
            return abs( row[ du_col ]/row[ dd_col ] )

    def describe(self):
        return ("RFRR_" + str(self.lookfwd) + "_" + self.feature)

    def apply(self, df):
        to_delete = []
        du = RollingFDU( self.lookfwd, self.feature )
        dd = RollingFDD( self.lookfwd, self.feature )

        [df, to_delete] = Misc.apply_if_not_present(df, du, to_delete)
        [df, to_delete] = Misc.apply_if_not_present(df, dd, to_delete)

        dfm = DataframeManipulator( df )
        dfm.extend_with_func( RollingFRR.__do_rr, self.describe(), ( du.describe(), dd.describe(), ) )

        dfm.delete_cols( to_delete )
        df = dfm.df

        return df

if __name__ == "__main__":
    from Backtester import Backtester

    b = Backtester(["FCEL", "MSTR"], {"FCEL": "united states", "MSTR": "united states"}, {}, 0.001, 0.002, None)
    b.do_work("1/1/2019", "6/5/2021")
    df = b.full_data

    RR = RollingReturn( 10, "FCEL_Close" )
    d1 = RR.apply( df )
    print( d1[ RR.describe()].tail(10 ))

    RF = RollingFReturn(10, "FCEL_Close")
    d2 = RF.apply(df)
    print( d2[ RF.describe()].tail(20 ))

    SR = RollingSharpe( 10, "FCEL_Close" )
    d1 = SR.apply( df )
    print( d1[ SR.describe()].tail(10 ))

    mx = RollingMax( 10, "FCEL_Close" )
    d1 = mx.apply( df )
    print( d1[ mx.describe()].tail(20 ))

    mx = RollingFMax( 10, "FCEL_Close" )
    d1 = mx.apply( df )
    print( d1[ [ "FCEL_Close", mx.describe()]].tail(20 ))

    mn = RollingMin( 10, "FCEL_Close" )
    d1 = mn.apply( df )
    print( d1[ mn.describe()].tail(10 ))

    mn = RollingFMin( 10, "FCEL_Close" )
    d1 = mn.apply( df )
    print( d1[ mn.describe()].tail(20 ))

    SF = RollingFSharpe(10, "FCEL_Close")
    d2 = SF.apply(df)
    print( d2[ SF.describe()].tail(20 ))

    DD = RollingFDD( 10, "FCEL_Close" )
    d2 = DD.apply(d2)
    print( d2[ DD.describe()].tail(20 ))

    DU = RollingFDU( 10, "FCEL_Close" )
    d2 = DU.apply(d2)
    print( d2[ DU.describe()].tail(20 ))

    RR = RollingFRR( 10, "FCEL_Close" )
    d2 = RR.apply(d2)
    print( d2[ RR.describe()].tail(20 ))
