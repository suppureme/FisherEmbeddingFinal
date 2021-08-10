import sklearn
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import scipy
from scipy import stats
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

class DataframeManipulator:

    def __init__(self, df ):
        self.df = df

    def look_back( self, column_name, num_rows, new_column_name = None ):
        if new_column_name is None:
            new_column_name = column_name + "_T-" + str( num_rows )
        self.df[ new_column_name ] = self.df[ column_name ].shift( num_rows )

    def look_forward( self, column_name, num_rows, new_column_name = None ):
        if new_column_name is None:
            new_column_name = column_name + "_T+" + str( num_rows )
        self.df[ new_column_name ] = self.df[ column_name ].shift( -num_rows )

    def extend_explicit(self, values, new_column_name ):
        self.df[ new_column_name ] = values

    def delete_cols(self, column_names ):
        if column_names != []:
            self.df = self.df.drop( column_names, axis = 1)

    def make_hl2(self, high, low ):
        self.df[ "HL2" ] = (self.df[ high ] + self.df[low])/2

    def extend_with_func(self, func, new_column_name, args = () ):
        self.df[ new_column_name ] = self.df.apply( func, axis = 1, args = args )

    def drop_na(self):
        self.df =  self.df.dropna().copy()

    def add_lookback_func(self, column_name, lookback_fn, lookback_dur, new_column_name = None, adjust = False ):
        df_temp = self.df[column_name]
        if new_column_name is None:
            new_column_name = column_name + "_" + lookback_fn + "_" + str( lookback_dur )
        if lookback_fn == "max":
            r = df_temp.rolling(lookback_dur, min_periods=lookback_dur)
            self.df[ new_column_name ] = r.max()
        elif lookback_fn == "rma":
            r = df_temp.ewm( min_periods=lookback_dur, adjust=adjust, alpha = 1/lookback_dur)
            self.df[new_column_name] = r.mean()
        elif lookback_fn == "ema":
            r = df_temp.ewm( com = lookback_dur - 1, min_periods=lookback_dur, adjust = adjust)
            self.df[new_column_name] = r.mean()
        elif lookback_fn == "sma":
            r = df_temp.rolling(lookback_dur, min_periods=lookback_dur)
            self.df[new_column_name] = r.mean()
        elif lookback_fn == "max":
            r = df_temp.rolling(lookback_dur, min_periods=lookback_dur)
            self.df[new_column_name] = r.max()
        elif lookback_fn == "min":
            r = df_temp.rolling(lookback_dur, min_periods=lookback_dur)
            self.df[new_column_name] = r.min()
        elif lookback_fn == "percentile":
            r = df_temp.rolling(lookback_dur, min_periods=lookback_dur)
            self.df[new_column_name] = r.apply( lambda x: scipy.stats.percentileofscore( x, x[-1]))
        elif lookback_fn == "std":
            r = df_temp.rolling(lookback_dur, min_periods=lookback_dur)
            self.df[new_column_name] = r.std()
        elif lookback_fn == "sum":
            r = df_temp.rolling(lookback_dur, min_periods=lookback_dur)
            self.df[new_column_name] = r.sum()


    def reverse_column(self, column_name, new_column_name ):
        df_temp = self.df[ column_name ]
        df_temp = df_temp.iloc[::-1].values
        if new_column_name is None:
            self.df[ column_name] = df_temp
        else:
            self.df[ new_column_name] = df_temp


    def find_filter(self, column_name, filter_mask ):
        df_temp = self.df[ filter_mask ]
        return df_temp[ column_name ]


