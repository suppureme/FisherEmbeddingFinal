import multiprocessing
import warnings
warnings.filterwarnings('ignore')
from helper_functions import *
import pickle
import matplotlib
matplotlib.use('Agg')
from datetime import timedelta
import os

def select_all_strategies(train_monthsf, datesf, temp_ogf, ticker, save=True):
    inputs =[]
    for date_i in range(len(datesf)-(int(train_monthsf/3)+1)):
        inputs.append([date_i, datesf, temp_ogf, train_monthsf])
    try:
        pool = multiprocessing.Pool(processes=7, maxtasksperchild=1)
        results = pool.map(get_strategies_brute_force, inputs)
    finally: # To make sure processes are closed in the end, even if errors happen
        pool.close()
        pool.join()

    res_test = [pd.DataFrame(columns=["Lookback", "Low Bound", "High Bound", "AvgWinLoss",\
                                       "Optimization_Years"])] * (len(datesf)-(int(24/3)+1))
    for i in range(len(results)):
        res_test[results[i][0]+int((train_monthsf-24)/3)] = pd.concat([res_test[results[i][0]],results[i][1].reset_index().drop(['index'], axis=1)], axis=0)

    if save==True:
        with open(f'TickerResults/{ticker}/SelectedStrategies/{ticker}_TrainYrs_{int(train_monthsf/12)}_All_Strategies.pkl', 'wb') as file:
            pickle.dump(res_test, file)

    return res_test

def select_strategies_from_corr_filter(res_testf2,res_testf4,res_testf8, datesf, temp_ogf, num_opt_periodsf,num_strategiesf, ticker, save=True):
    train_monthsf = 24  #minimum optimization lookback
    res_total = [None]*(len(datesf)-(int(train_monthsf/3)+1))
    for i in range(len(datesf)-(int(train_monthsf/3)+1)):
        if num_opt_periodsf==1:
            res_total[i] = pd.concat([res_testf2[i]], axis = 0)
        if num_opt_periodsf==2:
            res_total[i] = pd.concat([res_testf2[i],res_testf4[i]], axis=0)
        if num_opt_periodsf==3:
            res_total[i] = pd.concat([res_testf2[i],res_testf4[i],res_testf8[i]], axis=0)
        res_total[i] = res_total[i].reset_index().drop(['index'], axis=1)

    ss_test = [None]*(len(datesf)-(int(train_monthsf/3)+1))
    res_test = [None]*(len(datesf)-(int(train_monthsf/3)+1))
    inputs = []
    for date_i in range(len(datesf)-(int(train_monthsf/3)+1)):
        inputs.append([date_i, datesf, temp_ogf,res_total, num_strategiesf,train_monthsf])
    try:
        pool = multiprocessing.Pool(processes=7, maxtasksperchild=1)
        results_filtered = pool.map(corr_sortino_filter, inputs)
    finally: # To make sure processes are closed in the end, even if errors happen
        pool.close()
        pool.join()

    for i in range(len(datesf)-(int(train_monthsf/3)+1)):
        ss_test[results_filtered[i][0]] = results_filtered[i][1]
        res_test[results_filtered[i][0]] = results_filtered[i][2]

    if save==True:
        with open(f'TickerResults/{ticker}/SelectedStrategies/{ticker}_OptPeriods_{num_opt_periodsf}_Selected_Strategies_ss.pkl', 'wb') as file:
            pickle.dump(ss_test, file)
        with open(f'TickerResults/{ticker}/SelectedStrategies/{ticker}_OptPeriods_{num_opt_periodsf}_Selected_Strategies_res.pkl', 'wb') as file:
            pickle.dump(res_test, file)

    return ss_test, res_test

if __name__ == '__main__':

    #tickers = niftydata()

    tickers = ["USO", "CPER", "GLD", "000001.SS"]  #, ##, , "^N225"

    for ticker in tickers:

        if not os.path.exists(f'TickerResults/{ticker}/SelectedStrategies'):
            os.makedirs(f'TickerResults/{ticker}/SelectedStrategies')

        print(f"Processing {ticker}")
        temp_og = get_data(ticker, "yfinance")
        dates = valid_dates(pd.date_range(start=str(temp_og.iloc[1]['Date'] + timedelta(days=365))[:10], end="2024-06-15", freq=f'3M'))
        res_test2 = select_all_strategies(24,dates, temp_og, ticker,save=True)
        res_test4 = select_all_strategies(48,dates, temp_og, ticker,save=True)
        res_test8 = select_all_strategies(96,dates, temp_og, ticker,save=True)
        ss_test, res_test = select_strategies_from_corr_filter(res_test2,res_test4,res_test8, dates, temp_og, 1,10, ticker, save=True)
        ss_test, res_test = select_strategies_from_corr_filter(res_test2,res_test4,res_test8, dates, temp_og, 2,10, ticker, save=True)
        ss_test, res_test = select_strategies_from_corr_filter(res_test2,res_test4,res_test8, dates, temp_og, 3,10, ticker, save=True)
