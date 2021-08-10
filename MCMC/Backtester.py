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


class Backtester:

    def __init__(self, ric_list, symbol_countries, weights, commisions, slippage, inflows ):
        """Need the rics and their weights by time along with commisions/slippage plus inflows"""
        self.ric_list = ric_list
        self.weights = weights
        self.symbol_countries = symbol_countries
        self.commisions = commisions
        self.slippage = slippage
        self.inflows = inflows
        self.symbol_data = {}
        self.full_data = None


    def __get_data(self, symbol, country, from_date, to_date, resolution = "daily"):

        find = investpy.search.search_quotes(text=symbol, products=["stocks", "etfs"])
        for f in find:
            print(f)

            if f.symbol.lower() == symbol.lower() and f.country.lower() == country.lower():
                break
        if f.symbol.lower() != symbol.lower():
            return None
        ret = f.retrieve_historical_data(from_date=from_date, to_date=to_date)
        if ret is None:
            try:
                ret = investpy.get_stock_historical_data(stock=symbol,
                                                         country=country,
                                                         from_date=from_date,
                                                         to_date=to_date)
            except:
                ret = None
        if ret is None:
            try:
                ret = investpy.get_etf_historical_data(etf=symbol,
                                                       country=country,
                                                       from_date=from_date,
                                                       to_date=to_date)
            except:
                ret = None

        if ret is None:
            try:
                ret = investpy.get_index_historical_data(index=symbol,
                                                         country=country,
                                                         from_date=from_date,
                                                         to_date=to_date)
            except:
                ret = None

        # print(ret)

        return ret


    def data_worker(self, from_date, to_date ):
        full_df = None
        for symbol in self.ric_list:
            print(symbol)
            d = self.__get_data( symbol, self.symbol_countries[ symbol ], from_date, to_date, "daily"  )
            self.symbol_data[symbol] = d.copy()
            d[symbol + "_Low"] = d["Low"]
            d[symbol + "_High"] = d["High"]
            d[symbol + "_Close"] = d["Close"]
            d[ symbol + "_Open" ] = d["Open"]
            d = d[[  symbol + "_Open",  symbol + "_Close",  symbol + "_High",  symbol + "_Low" ]]
            if full_df is None:
                full_df = d
            else:
                full_df = full_df.join(d, on='Date')
        self.full_data = full_df

    def do_one_bar(self, bar, date, pre_rebalance_units, prev_cash ):
        total_costs = 0
        new_cash = prev_cash
        cur_index = 0
        post_rebalance_units = {}
        for symbol, unit in pre_rebalance_units.items():
            close = bar[symbol + "_Close"]
            holding = unit * close
            cur_index += holding
        cur_index += prev_cash
        if date in self.weights:
            print("Rebalance date")
            print( "Index pre rebalance " + str( cur_index ) )
            print( "Cash pre rebalance " + str( prev_cash ) )
            new_cash = prev_cash
            weights = self.weights[date]
            for symbol, w in weights.items():
                close = bar[symbol+"_Close"]
                allocation = w * ( cur_index * ( 1 - self.commisions ) )
                units = allocation/close

                print(symbol+" has number of units post rebalance " + str(units) + " from " + str( pre_rebalance_units[ symbol ]) )
                total_costs += self.slippage * abs( pre_rebalance_units[ symbol ] - units ) * close
                new_cash -= self.slippage * abs( pre_rebalance_units[ symbol ] - units ) * close
                new_cash -= (units - pre_rebalance_units[ symbol ]) * close
                post_rebalance_units[ symbol ] = units
            fee = self.commisions * cur_index
            new_cash -= fee
            total_costs += fee
            cur_index_post_rebalance = 0
            for symbol, unit in post_rebalance_units.items():
                close = bar[symbol + "_Close"]
                holding = unit * close
                cur_index_post_rebalance += holding
            cur_index_post_rebalance += new_cash
            print( "Cost " + str( total_costs ))
            print( "Index post rebalance " + str( cur_index_post_rebalance ) )
            print( "Cash post rebalance " + str( new_cash ) )

        else:
            post_rebalance_units = pre_rebalance_units
            cur_index_post_rebalance = cur_index
        return [ total_costs, post_rebalance_units, new_cash, cur_index_post_rebalance ]


    def insert_rebal_set(self, dt, symbol, weight ):
        dt_date = datetime.strptime( dt, "%d/%m/%Y").date()
        if dt_date not in self.weights:
            self.weights[ dt_date ] = {}
        self.weights[dt_date][ symbol ] = weight
        total_weight = 0
        for item, weight in self.weights[ dt_date ].items():
            total_weight += weight
        if total_weight > 1:
            print( "error!")
            return None

    def do_work(self, frm, to, index = 100 ):

        self.data_worker( from_date=frm, to_date=to)
        units = {}
        for symbol in self.ric_list:
            units[ symbol ] = 0
        cash = index
        index_list = {}
        units_list = {}

        total_costs = 0
        for index, row in self.full_data.iterrows():
            dt = index.to_pydatetime().date()
            [ costs, units, cash, idx ] = self.do_one_bar( row, dt, pre_rebalance_units=units, prev_cash=cash )
            print( dt, " ", idx )
            total_costs += costs
            index_list[ dt ] = idx
            units_list[ dt ] = units
        return [units_list, index_list, total_costs]

    def plot_backtest(self, index):
        plt.plot(index.keys(), index.values())
        plt.show()

    def get_stats(self, index, rf_rate ):
        returns = []
        highest = -1
        prev = None
        max_dd = -1
        reqd_dd = []
        for dt, id in index.items():
            if prev is None:
                prev = id
                continue
            if highest < id:
                highest = id
            dd = abs( id - highest )/highest
            if max_dd < dd:
                max_dd = dd
                reqd_dd = [ highest, id, dt ]

            ret = ( id - prev )/prev
            returns.append( ret )
            prev = id
        avg = sum( returns )/len( returns )
        std = np.std( np.array( returns ) )
        sharpe = np.sqrt( 252 ) * (( avg - rf_rate/365)/std)
        all_dates = list( index.keys() )
        num_days = all_dates[-1] - all_dates[0]
        all_indices = list( index.values() )
        print( num_days.days )
        years =  num_days.days/365
        ret = ( all_indices[-1] )/all_indices[0]
        irr = pow( ret, 1/years ) - 1
        return { "ret": ret, "IRR": irr, "Sharpe": sharpe, "Max DD": max_dd, "DD Desc": reqd_dd }



if __name__ == "__main__":
    b = Backtester(["INFY", "SBI"], {"SBI":"india","INFY": "india"}, {}, 0.001, 0.002,None)
    b.insert_rebal_set( "1/1/2021", "SBI", 0 )
    b.insert_rebal_set( "1/1/2021", "INFY", 1 )
    b.insert_rebal_set( "1/2/2021", "SBI", 1 )
    b.insert_rebal_set( "1/2/2021", "INFY", 0 )
    b.insert_rebal_set( "1/3/2021", "SBI", 1 )
    b.insert_rebal_set( "1/3/2021", "INFY", 0 )
    b.insert_rebal_set( "1/4/2021", "SBI", 0 )
    b.insert_rebal_set( "1/4/2021", "INFY", 1 )
    units, index, costs = b.do_work( "1/1/2021", "5/5/2021")
    print( index )
    print( costs )

    print( b.get_stats( index, 0.05 ))