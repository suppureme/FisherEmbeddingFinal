import investpy
from datetime import date, datetime
import math
import numpy as np
import pandas as pd
from MCMC.MCMC import MCMC
import datetime as dt
import dateutil.relativedelta
import multiprocessing
import pickle
import eikon as ek
import yfinance as yf
from datetime import timedelta

def niftydata():
    # /****************************************************************************************************************/
    # Load nifty50 list

    data_nifty = pd.read_csv("Nifty_50_Components_10Mar2021_csv.csv")
    data_nifty["Identifier (RIC)"] = data_nifty["Identifier (RIC)"].str.replace('.NS', '')
    Ticker = data_nifty["Identifier (RIC)"]

    # /****************************************************************************************************************/
    # Load Leavers and Joiners list

    data_Leavers_Joiners = pd.read_csv("Leavers_Jnrs_10_Mar_2021_csv_new.csv")
    data_Leavers_Joiners = data_Leavers_Joiners.drop(['Company'], axis=1).dropna()
    Date_req = "01-Jan-2021"  # input("Enter the date in format XX-Jan-20YY : ")
    x = data_Leavers_Joiners.apply(lambda x: datetime.strptime(x['Date'], "%d-%b-%Y"), axis=1)
    data_Leavers_Joiners['Date'] = data_Leavers_Joiners.apply(lambda x: datetime.strptime(x['Date'], "%d-%b-%Y"),
                                                              axis=1)
    row_req = data_Leavers_Joiners[data_Leavers_Joiners['Date'] > datetime.strptime(Date_req, "%d-%b-%Y")]

    rows_add = row_req[row_req["Unnamed: 0"] == '-']
    rows_sub = row_req[row_req["Unnamed: 0"] == '+']

    for i in range(len(row_req)):
        if row_req["Unnamed: 0"][i] == '-':
            Ticker = Ticker[Ticker != row_req['Code'][i]]
        else:
            A = pd.Series({'Totals (50)': [row_req["Code"][i]]})
            Ticker.append(pd.Series(A))

    return Ticker[1:].to_list()

def get_data_investpy( symbol, country, from_date, to_date ):
  find = investpy.search.search_quotes(text=symbol, products =["stocks", "etfs", "indices"] )
  for f in find:
    #print( f )

    if f.symbol.lower() == symbol.lower() and f.country.lower() == country.lower():
      break
  if f.symbol.lower() != symbol.lower():
    return None
  ret = f.retrieve_historical_data(from_date=from_date, to_date=to_date )
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
  ret.drop(["Change Pct"], axis=1, inplace=True)
  return ret

def get_data(ticker, api):

    if api == "yfinance":
        temp_og = yf.download(ticker, start = '2007-01-01', end= str(date.today()+timedelta(1)))
        temp_og.reset_index(inplace=True)
        temp_og.drop(['Adj Close'], axis=1, inplace=True)
        temp_og = temp_og.loc[temp_og["Close"]>1]
        temp_og = add_fisher(temp_og)

    if api =="investpy":
        temp_og = get_data_investpy(symbol=ticker, country='india', from_date="01/07/2007",to_date=(date.today()+timedelta(1)).strftime("%d/%m/%Y"))
        temp_og.reset_index(inplace=True)
        temp_og = add_fisher(temp_og)

    if api == "reuters":
        temp_og = ek.get_timeseries(ticker, start_date='2007-01-01', end_date=str(date.today() + timedelta(1)))
        temp_og.reset_index(inplace=True)
        temp_og.rename(columns={"HIGH": "High", "CLOSE": "Close", "LOW": "Low", "OPEN": "Open", "VOLUME": "Volume"},
                       inplace=True)
        temp_og.drop(['COUNT'], axis=1, inplace=True)

    return temp_og

def add_fisher(temp):
    for f_look in range(50, 400, 20):
        temp[f'Fisher{f_look}'] = fisher(temp, f_look)
    return temp

def fisher(ohlc, period):
    def __round(val):
        if (val > .99):
            return .999
        elif val < -.99:
            return -.999
        return val

    from numpy import log, seterr
    seterr(divide="ignore")
    med = (ohlc["High"] + ohlc["Low"]) / 2
    ndaylow = med.rolling(window=period).min()
    ndayhigh = med.rolling(window=period).max()
    med = [0 if math.isnan(x) else x for x in med]
    ndaylow = [0 if math.isnan(x) else x for x in ndaylow]
    ndayhigh = [0 if math.isnan(x) else x for x in ndayhigh]
    raw = [0] * len(med)
    for i in range(0, len(med)):
        try:
            raw[i] = 2 * ((med[i] - ndaylow[i]) / (ndayhigh[i] - ndaylow[i]) - 0.5)
        except:
            ZeroDivisionError
    value = [0] * len(med)
    value[0] = __round(raw[0] * 0.33)
    for i in range(1, len(med)):
        try:
            value[i] = __round(0.33 * raw[i] + 0.67 * value[i - 1])
        except:
            ZeroDivisionError
    _smooth = [0 if math.isnan(x) else x for x in value]
    fish1 = [0] * len(_smooth)
    for i in range(1, len(_smooth)):
        fish1[i] = ((0.5 * (np.log((1 + _smooth[i]) / (1 - _smooth[i]))))) + (0.5 * fish1[i - 1])
    fish2 = fish1[1:len(fish1)]
    # plt.figure(figsize=(18, 8))
    # plt.plot(ohlc.index, fish1, linewidth=1, label="Fisher_val")
    # plt.legend(loc="upper left")
    # plt.show()
    return fish1

def valid_dates(dates_all):
    dates = []
    i = 0
    while True:
        dates.append(dates_all[i])
        if dates_all[i] > pd.to_datetime(date.today()):
            break
        i = i + 1
    return dates

def create_final_signal_weights(signal, params, weights, nos):
    params = params[:nos]
    for i in range(len(params)):
        if i==0:
            signals =  signal[params.iloc[i]["Name"]].to_frame().rename(columns={'params.iloc[i]["Name"]': f'Signal{i + 1}'})
        else:
            signals = pd.merge(signals, signal[params.iloc[i]["Name"]].to_frame().rename(columns={'params.iloc[i]["Name"]': f'Signal{i + 1}'}), left_index=True, right_index=True)
            #signalsg = pd.concat([signalsg, signalg[paramsg.iloc[i]["Name"]].to_frame().rename(columns={'paramsg.iloc[i]["Name"]': f'Signal{i + 1}'})],axis=1)

    sf = pd.DataFrame(np.dot(np.where(np.isnan(signals),0,signals), weights))
    #return sf.set_index(signals.index).rename(columns={0: 'signal'})

    return pd.DataFrame(np.where(sf > 0.5, 1, 0)).set_index(signals.index).rename(columns={0: 'signal'})

    #return pd.DataFrame(np.where(signalsg.mean(axis=1, skipna=True) > 0.5, 1, 0)).set_index(signalsg.index).rename(columns={0:'signal'}), \
    #        pd.DataFrame(np.where(signalsn.mean(axis=1, skipna=True) > 0.5, 1, 0)).set_index(signalsn.index).rename(columns={0: 'signal'})

    #portfolio scaling
    #return pd.DataFrame(signalsg.mean(axis=1, skipna=True)).rename(columns={0:'signal'}), \
    #       pd.DataFrame(signalsn.mean(axis=1, skipna=True)).rename(columns={0:'signal'})

def backtest_live(input):

    date_i = input[0]
    dates = input[1]
    temp_og = input[2]
    ss_test = input[3]
    res_test = input[4]
    num_strategies = input[5]
    weights = input[6]
    recalib_months = input[7]
    dates_all = input[8]

    if (date_i - int(24 / recalib_months)) < 0:
        temp = temp_og.loc[
            (temp_og["Date"] >= str(dates_all[dates_all.index(dates[date_i]) - int(24 / 3)])) & (
                    temp_og["Date"] < str(dates[date_i + 1]))].reset_index().drop(
            ['index'], axis=1)
    else:
        temp = temp_og.loc[
            (temp_og["Date"] >= str(dates[date_i - int(24 / recalib_months)])) & (
                    temp_og["Date"] < str(dates[date_i + 1]))].reset_index().drop(
            ['index'], axis=1)

    test = temp.loc[(temp["Date"] >= str(dates[date_i]))].reset_index().drop(['index'], axis=1)

    if len(ss_test[date_i]) > 0:
        if len(ss_test[date_i]) > num_strategies:
            selected_strategies = ss_test[date_i][:num_strategies]
        else:
            selected_strategies = ss_test[date_i]

        if len(res_test[date_i]) > num_strategies:
            res = res_test[date_i][:num_strategies]
        else:
            res = res_test[date_i]

        # print("Optimizing weights")
        strategies = ["Strategy" + str(i) for i in range(1, len(res) + 1)]
        params = selected_params(strategies, res)

        _, signal = top_n_strat_params_rolling_live(temp, res, to_train=False, num_of_strat=len(selected_strategies),
                                               split_date=str(dates[date_i]))

        # Weights
        signal_final = create_final_signal_weights(signal, params, weights, len(selected_strategies))

        print_strategies = pd.concat([params.drop(["Name"], axis=1), weights.rename(columns={0: 'Weights'})], axis=1)
        print_strategies["Signal"] = signal.iloc[-1].reset_index().drop(['index'],axis=1)

        # equi-weighted
        # signal_final = create_final_signal(signal, params, len(selected_strategies))

        inp = pd.merge(test.set_index(test["Date"]), signal_final, left_index=True, right_index=True)

    else:

        inp = test.set_index(test["Date"])
        inp['signal'] = 0

    test_strategy = FISHER_MCMC_live(inp.drop(['signal', 'Date'], axis=1), inp['signal'])
    _ = test_strategy.generate_signals()
    tt = test_strategy.signal_performance(1, 6)
    return tt.set_index(pd.to_datetime(tt["Date"])).drop(columns="Date"), print_strategies

def top_n_strat_params_rolling_live(temp, res, to_train, num_of_strat, split_date):
    if len(res)>0:
        for i in range(num_of_strat):
            f_look = res.iloc[i, 0]
            bf = res.iloc[i, 1]
            sf = res.iloc[i, 2]
            temp["fisher"] = temp[f'Fisher{int(f_look)}']
            if to_train:
                train = temp.loc[(temp["Date"] <= split_date)].reset_index().drop(['index'], axis=1)
            else:
                train = temp.loc[temp["Date"] > split_date].reset_index().drop(['index'], axis=1)
            test_strategy = FISHER_bounds_strategy_opt_live(train, zone_low=bf, zone_high=sf)
            dummy_signal = test_strategy.generate_signals()
            dummy = test_strategy.signal_performance(10000, 6)
            if i==0:
                strat_sig_returns = dummy['S_Return'].to_frame().rename(columns={'S_Return': f'Strategy{i + 1}'}).set_index(dummy["Date"])
                strat_sig = dummy_signal['signal'].to_frame().rename(columns={'signal': f'Strategy{i + 1}'}).set_index(dummy_signal["Date"])
                #fisher_test = temp["fisher"].to_frame().rename(columns={"fisher": f'Fisher{asset}{i + 1}'}).set_index(temp["Date"])
            else:
                strat_sig_returns = pd.merge(strat_sig_returns, (dummy['S_Return'].to_frame().rename(columns={'S_Return': f'Strategy{i + 1}'}).set_index(dummy["Date"])), left_index=True, right_index=True)
                strat_sig = pd.concat([strat_sig, (dummy_signal['signal'].to_frame().rename(columns={'signal': f'Strategy{i + 1}'}).set_index(dummy_signal["Date"]))], axis = 1)
                #fisher_test = pd.concat([fisher_test, (temp["fisher"].to_frame().rename(columns={'fisher': f'Fisher{asset}{i + 1}'}).set_index(temp["Date"]))], axis = 1)
            #strat_sig_returns = pd.merge(strat_sig_returns,dummy['S_Return'].to_frame().rename(columns = {'S_Return':f'Strategy{i + 1}'}).set_index(dummy["Date"]), left_index=True, right_index=True)
        #return dummy
        return strat_sig_returns, strat_sig#, fisher_test
    else:
        return pd.DataFrame(), pd.DataFrame()

def optimize_weights_and_backtest(input):

    def get_equity_curve(*args):
        weights = []
        for weight in args:
            weights.append(weight)
        # weights.append((1-sum(weights)))
        weights = pd.DataFrame(weights)
        weights = weights / weights.sum()
        signal_final = create_final_signal_weights(signal, params, weights, num_strategies)
        inp = pd.merge(train.set_index(train["Date"]), signal_final, left_index=True, right_index=True)
        test_strategy = FISHER_MCMC(inp.drop(['signal', 'Date'], axis=1), inp['signal'])
        _ = test_strategy.generate_signals()
        ec = test_strategy.signal_performance(10000, 6)

        return ec

    def sortino(x, y, bins):
        ecdf = x[["S_Return"]]
        stdev_down = ecdf.loc[ecdf["S_Return"] < 0, "S_Return"].std() * (252 ** .5)
        if math.isnan(stdev_down):
            stdev_down = 0.0
        if stdev_down != 0.0:
            sortino = (ecdf["S_Return"].mean() - 0.06 / 252) * 252 / stdev_down
        else:
            sortino = 0
        return np.float64(sortino)

    def sharpe(x, y, bins):
        ecdf = x[["S_Return"]]

        stdev = ecdf["S_Return"].std() * (252 ** .5)
        if math.isnan(stdev):
            stdev = 0.0
        if stdev != 0.0:
            sharpe = (ecdf["S_Return"].mean() - 0.06 / 252) * 252 / stdev
        else:
            sharpe = 0
        return np.float64(sharpe)

    def rolling_sharpe(x, y, bins):
        r_window = 252
        ecdf = x[["Date", "S_Return"]]
        ecdf['RSharpeRatio_Series'] = (ecdf["S_Return"].rolling(window=r_window, min_periods=1).mean() - 0.06 / 252) / ecdf[
            "S_Return"].rolling(window=r_window, min_periods=1).std() * (252 ** .5)
        ecdf['RSharpeRatio_Series'] = ecdf['RSharpeRatio_Series'].replace(to_replace=math.inf, value=0)
        ecdf['RSharpeRatio_Series'] = ecdf['RSharpeRatio_Series'].replace(to_replace=math.nan, value=0)
        RSharpeRatio = ecdf['RSharpeRatio_Series'].mean()
        return np.float64(RSharpeRatio)

    def rolling_sortino(x, y, bins):
        r_window = 252
        ecdf = x[["Date", "S_Return"]]
        ecdf["S_Return_stdev"] = ecdf["S_Return"].copy()
        ecdf["S_Return_stdev"] = np.where(ecdf["S_Return_stdev"] >= 0, ecdf["S_Return_stdev"], np.nan)
        ecdf['RStDev Annualized Downside Return_Series'] = ecdf["S_Return_stdev"].rolling(window=r_window,
                                                                                          min_periods=1).std() * (
                                                                       252 ** .5)
        ecdf['RStDev Annualized Downside Return_Series'] = ecdf['RStDev Annualized Downside Return_Series'].replace(
            to_replace=math.inf, value=0)
        ecdf['RStDev Annualized Downside Return_Series'] = ecdf['RStDev Annualized Downside Return_Series'].replace(
            to_replace=math.nan, value=0)
        ecdf['RSortinoRatio_Series'] = np.where(ecdf['RStDev Annualized Downside Return_Series'] != 0.0,
                                                (ecdf["S_Return"].rolling(window=r_window, min_periods=1).mean() - 0.06 / 252) * 252 /
                                                ecdf['RStDev Annualized Downside Return_Series'], 0)
        RSortinoRatio = ecdf['RSortinoRatio_Series'].mean()
        return np.float64(RSortinoRatio)

    def rolling_cagr(x, y, bins):
        ecdf = x[["Date", "S_Return"]]
        ecdf["Date"] = pd.to_datetime(ecdf["Date"])
        ecdf = ecdf['S_Return'].to_frame().set_index(ecdf["Date"])
        ecdf['Portfolio'] = 1 + ecdf['S_Return'].expanding().sum()
        ecdf['Portfolio'] = ecdf['Portfolio']
        ecdf['Portfolio_1yr'] = ecdf['Portfolio'].to_frame().shift(365, freq='D')
        ecdf['Portfolio_1yr'] = ecdf['Portfolio_1yr'].fillna(method="ffill")
        ecdf['RCAGR_Strategy_Series'] = ecdf['Portfolio'] / ecdf['Portfolio_1yr'] - 1
        RCAGR_Strategy = ecdf['RCAGR_Strategy_Series'].mean()
        return np.float64(RCAGR_Strategy)

    def maxdrawup_by_maxdrawdown(x, y, bins):
        r_window = 252
        ecdf = x[["Date", "S_Return"]]
        ecdf['Strategy_Return'] = ecdf['S_Return'].expanding().sum()
        ecdf['Portfolio Value'] = ((ecdf['Strategy_Return'] + 1) * 1)
        ecdf['Portfolio Value'][0] = 1
        ecdf['Drawdown_Series'] = 1 - ecdf['Portfolio Value'] / np.maximum.accumulate(ecdf['Portfolio Value'])
        ecdf['Max_Drawdown_Series'] = ecdf['Drawdown_Series'].rolling(window=r_window, min_periods=1).max()
        ecdf['Drawup_Series'] = ecdf['Portfolio Value'] / np.minimum.accumulate(ecdf['Portfolio Value']) - 1
        ecdf['Max_Drawup_Series'] = ecdf['Drawup_Series'].rolling(window=r_window, min_periods=1).max()
        ecdf['Drawup/Drawdown_Series'] = ecdf['Max_Drawup_Series'] / ecdf['Max_Drawdown_Series']
        ecdf['Drawup/Drawdown_Series'] = ecdf['Drawup/Drawdown_Series'].replace(math.inf, 100)
        RDrawupDrawdown = ecdf['Drawup/Drawdown_Series'].mean()
        # ecdf['Drawdown_Series'] = 1 - ecdf['Portfolio Value'] / np.maximum.accumulate(ecdf['Portfolio Value'])
        # ecdf['Drawup_Series'] = ecdf['Portfolio Value'] / np.minimum.accumulate(ecdf['Portfolio Value']) - 1
        # RDrawupDrawdown = ecdf['Drawup_Series'].max() / ecdf['Drawdown_Series'].max()
        return np.float64(RDrawupDrawdown)

    def outperformance(x, y, bins):
        r_window = 252
        x["Date"] = pd.to_datetime(x["Date"])
        ecdf1 = x['S_Return'].to_frame().set_index(x["Date"])
        ecdf2 = x['Return'].to_frame().set_index(x["Date"])
        ecdf1['Portfolio'] = 1 + ecdf1['S_Return'].expanding().sum()
        ecdf1['Portfolio_1yr'] = ecdf1['Portfolio'].to_frame().shift(365, freq='D')
        ecdf2['Portfolio'] = 1 + ecdf2['Return'].expanding().sum()
        ecdf2['Portfolio_1yr'] = ecdf2['Portfolio'].to_frame().shift(365, freq='D')
        ecdf1['Portfolio_1yr'] = ecdf1['Portfolio_1yr'].fillna(method="ffill")
        ecdf1['RCAGR_Strategy_Series'] = ecdf1['Portfolio'] / ecdf1['Portfolio_1yr'] - 1
        ecdf2['Portfolio_1yr'] = ecdf2['Portfolio_1yr'].fillna(method="ffill")
        ecdf2['RCAGR_Market_Series'] = ecdf2['Portfolio'] / ecdf2['Portfolio_1yr'] - 1
        RCAGR_Strategy = ecdf1['RCAGR_Strategy_Series'].mean()
        RCAGR_Market = ecdf2['RCAGR_Market_Series'].mean()
        ROutperformance = RCAGR_Strategy - RCAGR_Market
        return np.float64(ROutperformance)

    def prior(params):
        return 1

    date_i = input[0]
    dates = input[1]
    temp_og = input[2]
    ss_test = input[3]
    res_test = input[4]
    num_strategies = input[5]
    metric = input[6]
    recalib_months = input[7]
    dates_all = input[8]

    #print(f"Training period begins: {str(dates[date_i])}")
    if (date_i - int(24 / recalib_months)) < 0:
        temp = temp_og.loc[
            (temp_og["Date"] > str(dates_all[dates_all.index(dates[date_i]) - int(24 / 3)])) & (
                        temp_og["Date"] < str(dates[date_i + 1]))].reset_index().drop(
            ['index'], axis=1)
    else:
        temp = temp_og.loc[
            (temp_og["Date"] > str(dates[date_i - int(24 / recalib_months)])) & (
                        temp_og["Date"] < str(dates[date_i + 1]))].reset_index().drop(
            ['index'], axis=1)


    train = temp.loc[(temp["Date"] < str(dates[date_i]))].reset_index().drop(['index'], axis=1)
    test = temp.loc[(temp["Date"] >= str(dates[date_i]))].reset_index().drop(['index'], axis=1)

    if len(ss_test[date_i]) > 0:
        if len(ss_test[date_i]) > num_strategies:
            selected_strategies = ss_test[date_i][:num_strategies]
        else:
            selected_strategies = ss_test[date_i]

        if len(res_test[date_i]) > num_strategies:
            res = res_test[date_i][:num_strategies]
        else:
            res = res_test[date_i]

        # print("Optimizing weights")
        strategies = ["Strategy" + str(i) for i in range(1, len(res) + 1)]
        params = selected_params(strategies, res)

        _, signal = top_n_strat_params_rolling(temp, res, to_train=True, num_of_strat=len(selected_strategies),
                                               split_date=str(dates[date_i]))
        #print("Running MCMC")
        guess = (0.5 * np.ones([1, len(selected_strategies)])).tolist()[0]

        if len(guess)>1:
            if metric == 'rolling_sharpe':
                mc = MCMC(alpha_fn=get_equity_curve, alpha_fn_params_0=guess, target=_, num_iters=2000, prior=prior,
                          optimize_fn=rolling_sharpe, lower_limit=0, upper_limit=1)
                rs = mc.optimize()

            if metric == 'rolling_sortino':
                mc = MCMC(alpha_fn=get_equity_curve, alpha_fn_params_0=guess, target=_, num_iters=2000, prior=prior,
                          optimize_fn=rolling_sortino, lower_limit=0, upper_limit=1)
                rs = mc.optimize()

            if metric == 'rolling_cagr':
                mc = MCMC(alpha_fn=get_equity_curve, alpha_fn_params_0=guess, target=_, num_iters=2000, prior=prior,
                          optimize_fn=rolling_cagr, lower_limit=0, upper_limit=1)
                rs = mc.optimize()

            if metric == 'maxdrawup_by_maxdrawdown':
                mc = MCMC(alpha_fn=get_equity_curve, alpha_fn_params_0=guess, target=_, num_iters=2000, prior=prior,
                          optimize_fn=maxdrawup_by_maxdrawdown, lower_limit=0, upper_limit=1)
                rs = mc.optimize()

            if metric == 'outperformance':
                mc = MCMC(alpha_fn=get_equity_curve, alpha_fn_params_0=guess, target=_, num_iters=2000, prior=prior,
                          optimize_fn=outperformance, lower_limit=0, upper_limit=1)
                rs = mc.optimize()

            # Printing results:
            weights = []
            for weight in mc.analyse_results(rs, top_n=1)[0][0]:
                weights.append(weight)
            weights = pd.DataFrame(weights)
            weights = weights / weights.sum(axis=0)

        else:
            weights = pd.DataFrame([1])

        #print(pd.concat([params.drop(["Name"], axis=1), weights.rename(columns={0: 'Weights'})], axis=1))

        # signal_final = create_final_signal_weights(signal, params, weights, len(selected_strategies))
        # inp = pd.merge(train.set_index(train["Date"]), signal_final, left_index=True, right_index=True)
        #
        # test_strategy = FISHER_MCMC(inp.drop(['signal', 'Date'], axis=1), inp['signal'])
        # _ = test_strategy.generate_signals()
        # _ = test_strategy.signal_performance(1, 6)

        # print(f"Training period ends: {str(dates[date_i + 2])}")
        # print(f"Sortino for training period is: {test_strategy.daywise_performance['SortinoRatio']}")
        # print(f"Testing period begins: {str(dates[date_i + 2])}")

        _, signal = top_n_strat_params_rolling(temp, res, to_train=False, num_of_strat=len(selected_strategies),
                                               split_date=str(dates[date_i]))

        # Weights
        signal_final = create_final_signal_weights(signal, params, weights, len(selected_strategies))

        # equi-weighted
        # signal_final = create_final_signal(signal, params, len(selected_strategies))

        inp = pd.merge(test.set_index(test["Date"]), signal_final, left_index=True, right_index=True)

    else:
        inp = test.set_index(test["Date"])
        inp['signal'] = 0
        weights = pd.DataFrame()

    test_strategy = FISHER_MCMC(inp.drop(['signal', 'Date'], axis=1), inp['signal'])
    _ = test_strategy.generate_signals()
    tt = test_strategy.signal_performance(1, 6)
    return date_i, tt.set_index(pd.to_datetime(tt["Date"])).drop(columns="Date"), weights
    # print(f"Testing period ends: {str(dates[date_i + 3])}")
    # print(f"Sortino for testing period is: {test_strategy.daywise_performance['SortinoRatio']}")

def get_strategies_brute_force(inp):
    def get_equity_curve_embeddings(*args):
        f_look = args[0]
        f_look = 1 * round(f_look / 1)
        lb = round(10 * args[1]) / 10
        ub = round(10 * args[2]) / 10

        temp["fisher"] = temp[f'Fisher{f_look}']

        test_strategy = FISHER_bounds_strategy_opt(temp, lb, ub)
        _ = test_strategy.generate_signals()
        ec = test_strategy.signal_performance(10000, 6)
        return ec

    def AvgWinLoss(x, y, bins):
        ecdf = x[["S_Return", "Close", "signal", "trade_num"]]
        ecdf = ecdf[ecdf["signal"] == 1]
        trade_wise_results = []
        for i in range(max(ecdf['trade_num'])):
            trade_num = i + 1
            entry = ecdf[ecdf["trade_num"] == trade_num].iloc[0]["Close"]
            exit = ecdf[ecdf["trade_num"] == trade_num].iloc[-1]["Close"]
            trade_wise_results.append({'Trade Number': trade_num, 'Entry': entry, 'Exit': exit})
        trade_wise_results = pd.DataFrame(trade_wise_results)
        d_tp = {}
        if len(trade_wise_results) > 0:
            trade_wise_results["Win/Loss"] = np.where(trade_wise_results["Exit"] > trade_wise_results["Entry"], "Win",
                                                      "Loss")
            trade_wise_results["Return on Trade"] = trade_wise_results["Exit"] / trade_wise_results["Entry"] - 1
            d_tp["TotalWins"] = len(trade_wise_results[trade_wise_results["Win/Loss"] == "Win"])
            d_tp["TotalLosses"] = len(trade_wise_results[trade_wise_results["Win/Loss"] == "Loss"])
            d_tp['TotalTrades'] = d_tp["TotalWins"] + d_tp["TotalLosses"]
            if d_tp['TotalTrades'] == 0:
                d_tp['HitRatio'] = 0
            else:
                d_tp['HitRatio'] = round(d_tp['TotalWins'] / d_tp['TotalTrades'], 4)
            d_tp['AvgWinRet'] = np.round(
                trade_wise_results[trade_wise_results["Win/Loss"] == "Win"]["Return on Trade"].mean(), 4)
            if math.isnan(d_tp['AvgWinRet']):
                d_tp['AvgWinRet'] = 0.0
            d_tp['AvgLossRet'] = np.round(
                trade_wise_results[trade_wise_results["Win/Loss"] == "Loss"]["Return on Trade"].mean(), 4)
            if math.isnan(d_tp['AvgLossRet']):
                d_tp['AvgLossRet'] = 0.0
            if d_tp['AvgLossRet'] != 0:
                d_tp['WinByLossRet'] = np.round(abs(d_tp['AvgWinRet'] / d_tp['AvgLossRet']), 2)
            else:
                d_tp['WinByLossRet'] = 0
            if math.isnan(d_tp['WinByLossRet']):
                d_tp['WinByLossRet'] = 0.0
            if math.isinf(d_tp['WinByLossRet']):
                d_tp['WinByLossRet'] = 0.0
        else:
            d_tp["TotalWins"] = 0
            d_tp["TotalLosses"] = 0
            d_tp['TotalTrades'] = 0
            d_tp['HitRatio'] = 0
            d_tp['AvgWinRet'] = 0
            d_tp['AvgLossRet'] = 0
            d_tp['WinByLossRet'] = 0

        return np.float64(d_tp['WinByLossRet'])

    date_i = inp[0]
    dates = inp[1]
    temp_og = inp[2]
    train_months = inp[3]

    temp = temp_og.loc[(temp_og["Date"] > str(dates[date_i])) & (temp_og["Date"] < str(
        dates[date_i + int(train_months / 3)]))].reset_index().drop(['index'], axis=1)
    res = pd.DataFrame(columns=["Lookback", "Low Bound", "High Bound", "AvgWinLoss"])

    for f_look in range(50, 410, 20):
        max_metric = 0
        for lb in np.round(np.arange(-7, 7, 0.25), decimals=1):
            for ub in np.round(np.arange(-7, 7, 0.25), decimals=1):
                metric = AvgWinLoss(get_equity_curve_embeddings(f_look, lb, ub), 0, 0)
                if metric > max_metric:
                    max_metric = metric
                    res_iter = pd.DataFrame(
                        [{"Lookback": f_look, "Low Bound": lb, "High Bound": ub, "AvgWinLoss": metric}])
                    res = pd.concat([res, res_iter], axis=0)

    res.drop_duplicates(keep='first', inplace=True, ignore_index=True)
    res.sort_values("AvgWinLoss", axis=0, ascending=False, inplace=True)
    res["Optimization_Years"] = train_months / 12
    res = res.reset_index().drop(['index'], axis=1)
    return (date_i, res)

def backtest_sortino(x, y, bins):
    ecdf = x[["S_Return"]]
    stdev_down = ecdf.loc[ecdf["S_Return"] < 0, "S_Return"].std() * (252 ** .5)
    if math.isnan(stdev_down):
        stdev_down = 0.0
    if stdev_down != 0.0:
        sortino = (ecdf["S_Return"].mean() - 0.06 / 252) * 252 / stdev_down
    else:
        sortino = 0
    return np.float64(sortino)
def backtest_sharpe(x, y, bins):
    ecdf = x[["S_Return"]]

    stdev = ecdf["S_Return"].std() * (252 ** .5)
    if math.isnan(stdev):
        stdev = 0.0
    if stdev != 0.0:
        sharpe = (ecdf["S_Return"].mean() - 0.06 / 252) * 252 / stdev
    else:
        sharpe = 0
    return np.float64(sharpe)
def backtest_rolling_sharpe(x, y, bins):
    r_window = 252
    ecdf = x[["Date", "S_Return"]]
    ecdf['RSharpeRatio_Series'] = (ecdf["S_Return"].rolling(window=r_window, min_periods=1).mean() - 0.06 / 252) / ecdf[
        "S_Return"].rolling(window=r_window, min_periods=1).std() * (252 ** .5)
    ecdf['RSharpeRatio_Series'] = ecdf['RSharpeRatio_Series'].replace(to_replace=math.inf, value=0)
    ecdf['RSharpeRatio_Series'] = ecdf['RSharpeRatio_Series'].replace(to_replace=math.nan, value=0)
    RSharpeRatio = ecdf['RSharpeRatio_Series'].mean()
    return np.float64(RSharpeRatio)
def backtest_rolling_sortino(x, y, bins):
    r_window = 252
    ecdf = x[["Date", "S_Return"]]
    ecdf["S_Return_stdev"] = ecdf["S_Return"].copy()
    ecdf["S_Return_stdev"] = np.where(ecdf["S_Return_stdev"] >= 0, ecdf["S_Return_stdev"], np.nan)
    ecdf['RStDev Annualized Downside Return_Series'] = ecdf["S_Return_stdev"].rolling(window=r_window,
                                                                                      min_periods=1).std() * (
                                                               252 ** .5)
    ecdf['RStDev Annualized Downside Return_Series'] = ecdf['RStDev Annualized Downside Return_Series'].replace(
        to_replace=math.inf, value=0)
    ecdf['RStDev Annualized Downside Return_Series'] = ecdf['RStDev Annualized Downside Return_Series'].replace(
        to_replace=math.nan, value=0)
    ecdf['RSortinoRatio_Series'] = np.where(ecdf['RStDev Annualized Downside Return_Series'] != 0.0,
                                            (ecdf["S_Return"].rolling(window=r_window,
                                                                      min_periods=1).mean() - 0.06 / 252) * 252 /
                                            ecdf['RStDev Annualized Downside Return_Series'], 0)
    RSortinoRatio = ecdf['RSortinoRatio_Series'].mean()
    return np.float64(RSortinoRatio)
def backtest_rolling_cagr(x, y, bins):
    ecdf = x[["Date", "S_Return"]]
    ecdf = ecdf['S_Return'].to_frame().set_index(ecdf["Date"])
    ecdf['Portfolio'] = 1 + ecdf['S_Return'].expanding().sum()
    ecdf['Portfolio'] = ecdf['Portfolio']
    ecdf['Portfolio_1yr'] = ecdf['Portfolio'].to_frame().shift(365, freq='D')
    ecdf['Portfolio_1yr'] = ecdf['Portfolio_1yr'].fillna(method="ffill")
    ecdf['RCAGR_Strategy_Series'] = ecdf['Portfolio'] / ecdf['Portfolio_1yr'] - 1
    RCAGR_Strategy = ecdf['RCAGR_Strategy_Series'].mean()
    return np.float64(RCAGR_Strategy)
def backtest_rolling_maxdrawup_by_maxdrawdown(x, y, bins):
    r_window = 252
    ecdf = x[["Date", "S_Return"]]
    ecdf['Strategy_Return'] = ecdf['S_Return'].expanding().sum()
    ecdf['Portfolio Value'] = ((ecdf['Strategy_Return'] + 1) * 1)
    ecdf['Portfolio Value'][0] = 1
    ecdf['Drawdown_Series'] = 1 - ecdf['Portfolio Value'] / np.maximum.accumulate(ecdf['Portfolio Value'])
    ecdf['Max_Drawdown_Series'] = ecdf['Drawdown_Series'].rolling(window=r_window, min_periods=1).max()
    ecdf['Drawup_Series'] = ecdf['Portfolio Value'] / np.minimum.accumulate(ecdf['Portfolio Value']) - 1
    ecdf['Max_Drawup_Series'] = ecdf['Drawup_Series'].rolling(window=r_window, min_periods=1).max()
    ecdf['Drawup/Drawdown_Series'] = ecdf['Max_Drawup_Series'] / ecdf['Max_Drawdown_Series']
    ecdf['Drawup/Drawdown_Series'] = ecdf['Drawup/Drawdown_Series'].replace(math.inf, 100)
    RDrawupDrawdown = ecdf['Drawup/Drawdown_Series'].mean()
    # ecdf['Drawdown_Series'] = 1 - ecdf['Portfolio Value'] / np.maximum.accumulate(ecdf['Portfolio Value'])
    # ecdf['Drawup_Series'] = ecdf['Portfolio Value'] / np.minimum.accumulate(ecdf['Portfolio Value']) - 1
    # RDrawupDrawdown = ecdf['Drawup_Series'].max() / ecdf['Drawdown_Series'].max()
    return np.float64(RDrawupDrawdown)
def backtest_maxdrawup_by_maxdrawdown(x, y, bins):
    r_window = 252
    ecdf = x[["Date", "S_Return"]]
    ecdf['Strategy_Return'] = ecdf['S_Return'].expanding().sum()
    ecdf['Portfolio Value'] = ((ecdf['Strategy_Return'] + 1) * 1)
    ecdf['Portfolio Value'][0] = 1
    # ecdf['Drawdown_Series'] = 1 - ecdf['Portfolio Value'] / np.maximum.accumulate(ecdf['Portfolio Value'])
    # ecdf['Max_Drawdown_Series'] = ecdf['Drawdown_Series'].rolling(window=r_window, min_periods=1).max()
    # ecdf['Drawup_Series'] = ecdf['Portfolio Value'] / np.minimum.accumulate(ecdf['Portfolio Value']) - 1
    # ecdf['Max_Drawup_Series'] = ecdf['Drawup_Series'].rolling(window=r_window, min_periods=1).max()
    # ecdf['Drawup/Drawdown_Series'] = ecdf['Max_Drawup_Series'] / ecdf['Max_Drawdown_Series']
    # ecdf['Drawup/Drawdown_Series'] = ecdf['Drawup/Drawdown_Series'].replace(math.inf, 100)
    # RDrawupDrawdown = ecdf['Drawup/Drawdown_Series'].mean()
    ecdf['Drawdown_Series'] = 1 - ecdf['Portfolio Value'] / np.maximum.accumulate(ecdf['Portfolio Value'])
    ecdf['Drawup_Series'] = ecdf['Portfolio Value'] / np.minimum.accumulate(ecdf['Portfolio Value']) - 1
    DrawupDrawdown = ecdf['Drawup_Series'].max() / ecdf['Drawdown_Series'].max()
    return np.float64(DrawupDrawdown)
def backtest_rolling_outperformance(x, y, bins):
    r_window = 252
    ecdf1 = x['S_Return'].to_frame().set_index(x["Date"])
    ecdf2 = x['Return'].to_frame().set_index(x["Date"])
    ecdf1['Portfolio'] = 1 + ecdf1['S_Return'].expanding().sum()
    ecdf1['Portfolio_1yr'] = ecdf1['Portfolio'].to_frame().shift(365, freq='D')
    ecdf2['Portfolio'] = 1 + ecdf2['Return'].expanding().sum()
    ecdf2['Portfolio_1yr'] = ecdf2['Portfolio'].to_frame().shift(365, freq='D')
    ecdf1['Portfolio_1yr'] = ecdf1['Portfolio_1yr'].fillna(method="ffill")
    ecdf1['RCAGR_Strategy_Series'] = ecdf1['Portfolio'] / ecdf1['Portfolio_1yr'] - 1
    ecdf2['Portfolio_1yr'] = ecdf2['Portfolio_1yr'].fillna(method="ffill")
    ecdf2['RCAGR_Market_Series'] = ecdf2['Portfolio'] / ecdf2['Portfolio_1yr'] - 1
    RCAGR_Strategy = ecdf1['RCAGR_Strategy_Series'].mean()
    RCAGR_Market = ecdf2['RCAGR_Market_Series'].mean()
    ROutperformance = RCAGR_Strategy - RCAGR_Market
    return np.float64(ROutperformance)
def backtest_outperformance(x, y, bins):
    r_window = 252
    ecdf1 = x['S_Return'].to_frame().set_index(x["Date"])
    ecdf2 = x['Return'].to_frame().set_index(x["Date"])
    ecdf1['Portfolio'] = 1 + ecdf1['S_Return'].expanding().sum()
    ecdf2['Portfolio'] = 1 + ecdf2['Return'].expanding().sum()
    RCAGR_Strategy = ecdf1['Portfolio'][-1]/ecdf1['Portfolio'][1]-1
    RCAGR_Market = ecdf2['Portfolio'][-1]/ecdf2['Portfolio'][1]-1
    Outperformance = RCAGR_Strategy - RCAGR_Market
    return np.float64(Outperformance)

def get_constituents(index):
    ic, err = ek.get_data(index, ['TR.IndexConstituentRIC'])
    lj, err = ek.get_data(index,
                          ['TR.IndexJLConstituentChangeDate',
                           'TR.IndexJLConstituentRIC.change',
                           'TR.IndexJLConstituentRIC'],
                          {'SDate': '0D', 'EDate': '-55M', 'IC': 'B'})
    lj['Date'] = pd.to_datetime(lj['Date']).dt.date
    lj.sort_values(['Date', 'Change'], ascending=False, inplace=True)
    dates = [dt.date(2007, 1, 30)]
    i = 0
    while (dates[0] + dateutil.relativedelta.relativedelta(months=+i + 1)) < dt.date.today():
        dates.append(dates[0] + dateutil.relativedelta.relativedelta(months=+i + 1))
        i = i + 1
    dates.append(dt.date.today())
    df = pd.DataFrame(index=dates, columns=['Index Constituents'])
    ic_list = ic['Constituent RIC'].tolist()
    for i in range(len(dates)):
        #print(str(dates[len(dates) - i - 1]))
        df.at[dates[len(dates) - i - 1], 'Index Constituents'] = ic_list[:]
        for j in lj.index:
            if lj['Date'].loc[j] <= dates[len(dates) - i - 1]:
                if lj['Date'].loc[j] > dates[len(dates) - i - 2]:
                    if lj['Change'].loc[j] == 'Joiner':
                        #print('Removing ' + lj['Constituent RIC'].loc[j])
                        ic_list.remove(lj['Constituent RIC'].loc[j])
                    elif lj['Change'].loc[j] == 'Leaver':
                        #print('Adding ' + lj['Constituent RIC'].loc[j])
                        ic_list.append(lj['Constituent RIC'].loc[j])
                else:
                    break
    df.index = pd.date_range(start=str(df.index[0])[:10], end=str(df.index[-1].replace(month=df.index[-1].month+1))[:10], freq=f'1M')
    return df

def prepare_portfolio_data(tickers, recalibrating_months,api):
    data = pd.DataFrame()
    for ticker in tickers:
        try:
            temp_og = get_data(ticker, api)
            data = pd.concat([data, temp_og["Close"].to_frame().astype(float).rename(columns={"Close":ticker}).set_index(temp_og["Date"])], axis=1)
            data[f"{ticker}Return"] = np.log(data[ticker]/data[ticker].shift(1))
            data[f"{ticker}ROC0.5"] = data[ticker].pct_change(10)
            data[f"{ticker}ROC1"] = data[ticker].pct_change(21)
            data[f"{ticker}ROC3"] = data[ticker].pct_change(63)
            data[f"{ticker}ROC6"] = data[ticker].pct_change(126)
            data[f"{ticker}ROC9"] = data[ticker].pct_change(189)
            data[f"{ticker}ROC12"] = data[ticker].pct_change(252)
            data[f"{ticker}SD12"] = data[ticker].pct_change().rolling(252).std()
            data[f"{ticker}FReturn"] = data[ticker].pct_change(-recalibrating_months*21)
        except:
            print(f"{ticker} not processed")
    data.reset_index(inplace=True)
    return data

def get_weights_stocks(constituents, topn, test_monthsf, train_monthsf, datesf, temp_ogf, save=True):
    inputs = []
    for date_i in range(len(datesf) - (int(train_monthsf/test_monthsf) + 1)):
        inputs.append([temp_ogf, topn,datesf,date_i,train_monthsf,test_monthsf, constituents ])
    try:
        pool = multiprocessing.Pool(processes=7, maxtasksperchild=1)
        results = pool.map(recalibrate_weights_stocks, inputs)
    finally:  # To make sure processes are closed in the end, even if errors happen
        pool.close()
        pool.join()

    # res_test2 = [pd.DataFrame(columns=["Lookback", "Low Bound", "High Bound", "AvgWinLoss",  "Sortino", "Optimization_Years"])] * len(results2)
    res_test = [pd.DataFrame(columns=["Ticker", "WtROC0.5", "WtROC1", "WtROC3", "WtROC6", "WtROC9", "WtROC12", "Accelerating Momentum"])] * (len(datesf) - (int(24/test_monthsf) + 1))
    for i in range(len(results)):
        res_test[results[i][0] + int((train_monthsf - 24) / test_monthsf)] = pd.concat(
            [res_test[results[i][0]], results[i][1].reset_index().drop(['index'], axis=1)], axis=0)

    if save == True:
        with open(f'TrainYrs_{int(train_monthsf / 12)}_Weights.pkl', 'wb') as file:
            pickle.dump(res_test, file)
    return res_test

def recalibrate_weights_stocks(inp):

    def alpha(*args):
        weights = pd.DataFrame(args)
        weights = weights / weights.sum()
        for ticker in tickers:
            df = data[[f"{ticker}ROC0.5", f"{ticker}ROC1", f"{ticker}ROC3", f"{ticker}ROC6", f"{ticker}ROC9",
                       f"{ticker}ROC12", f"{ticker}SD12", f"{ticker}FReturn"]]
            df[f"{ticker}AM"] = np.dot(df[[f"{ticker}ROC0.5", f"{ticker}ROC1", f"{ticker}ROC3", f"{ticker}ROC6",
                                           f"{ticker}ROC9", f"{ticker}ROC12"]], weights)
            data[f"{ticker}AM"] = df[f"{ticker}AM"] / df[f"{ticker}SD12"]
        return data[[f"{ticker}AM" for ticker in tickers]].to_numpy().ravel()

    def prior(params):
        if (params[0] < 0) | (params[0] > 1):
            return 0
        if (params[1] < 0) | (params[1] > 1):
            return 0
        if (params[2] < 0) | (params[2] > 1):
            return 0
        if (params[3] < 0) | (params[3] > 1):
            return 0
        if (params[4] < 0) | (params[4] > 1):
            return 0
        if (params[5] < 0) | (params[5] > 1):
            return 0
        return 1

    # Optimizing weights for entire portfolio
    temp_ogf = inp[0]
    top_n = inp[1]
    datesf = inp[2]
    date_i = inp[3]
    train_monthsf = inp[4]
    test_monthsf = inp[5]
    constituents = inp[6]

    # data_og = temp_ogf.loc[(temp_ogf["Date"] > str(datesf[date_i])) & (temp_ogf["Date"] < str(datesf[date_i + int(train_monthsf/test_monthsf) + int(test_monthsf/test_monthsf)]))].reset_index().drop(['index'], axis=1)

    # Adjustment made for forward returns
    data_og = temp_ogf.loc[(temp_ogf["Date"] > str(datesf[date_i])) & (
                temp_ogf["Date"] < str(datesf[date_i + int(train_monthsf / test_monthsf) - 1]))].reset_index().drop(
        ['index'], axis=1)

    tickers_in_index = constituents.loc[datesf[date_i]][0]

    data = data_og.dropna(axis=1, how='all')
    data = data.dropna()
    tickers1 = []
    for column in data.columns[1:]:
        if column.endswith("Return") | column.endswith("FReturn") | column.endswith("ROC0.5") | column.endswith(
                "ROC1") | column.endswith("ROC3") | \
                column.endswith("ROC6") | column.endswith("ROC9") | column.endswith("ROC12") | column.endswith("SD12"):
            continue
        else:
            tickers1.append(column)
    tickers = []
    for ticker in tickers1:
        if ((f"{ticker}" in data.columns[1:]) & (f"{ticker}ROC0.5" in data.columns[1:]) & (
                f"{ticker}ROC1" in data.columns[1:]) & (
                f"{ticker}ROC3" in data.columns[1:]) & (f"{ticker}ROC6" in data.columns[1:]) &
                (f"{ticker}ROC9" in data.columns[1:]) & (f"{ticker}ROC12" in data.columns[1:]) & (
                        f"{ticker}SD12" in data.columns[1:]) & (f"{ticker}FReturn" in data.columns[1:])):
            print(ticker)
            tickers.append(ticker)

    for ticker in tickers:
        if not ticker in tickers_in_index:
            tickers.remove(ticker)

    random_starts = 10
    iterations = 100
    guess_list = [np.random.dirichlet(np.ones(6), size=1).tolist()[0] for i in range(random_starts)]
    res = pd.DataFrame(columns=["WtROC0.5", "WtROC1", "WtROC3", "WtROC6", "WtROC9", "WtROC12", "NMIS"])

    for guess in guess_list:
        mc = MCMC(alpha_fn=alpha, alpha_fn_params_0=guess,
                  target=data[[f"{ticker}FReturn" for ticker in tickers]].to_numpy().ravel(), num_iters=iterations,
                  prior=prior, optimize_fn=None, lower_limit=0, upper_limit=1)
        rs = mc.optimize()
        res_iter = [{"WtROC0.5": mc.analyse_results(rs, top_n=iterations)[0][i][0],
                     "WtROC1": mc.analyse_results(rs, top_n=iterations)[0][i][1],
                     "WtROC3": mc.analyse_results(rs, top_n=iterations)[0][i][2],
                     "WtROC6": mc.analyse_results(rs, top_n=iterations)[0][i][3],
                     "WtROC9": mc.analyse_results(rs, top_n=iterations)[0][i][4],
                     "WtROC12": mc.analyse_results(rs, top_n=iterations)[0][i][5],
                     "NMIS": mc.analyse_results(rs, top_n=iterations)[1][i]} \
                    for i in range(iterations)]
        res_iter = pd.DataFrame(res_iter)
        res = pd.concat([res, res_iter], axis=0)
    res = res.sort_values("NMIS", axis=0, ascending=False).reset_index(drop=True)
    chosen_weights = pd.DataFrame(
        [res.iloc[0]["WtROC0.5"], res.iloc[0]["WtROC1"], res.iloc[0]["WtROC3"], res.iloc[0]["WtROC6"],
         res.iloc[0]["WtROC9"], res.iloc[0]["WtROC12"]])
    chosen_weights = chosen_weights / chosen_weights.sum()
    am = []
    for ticker in tickers:
        am.append({"Ticker": ticker, "WtROC0.5": float(chosen_weights.iloc[0]), "WtROC1": float(chosen_weights.iloc[1]),
                   "WtROC3": float(chosen_weights.iloc[2]), "WtROC6": float(chosen_weights.iloc[3]),
                   "WtROC9": float(chosen_weights.iloc[4]), "WtROC12": float(chosen_weights.iloc[5]),
                   "Accelerating Momentum": np.dot(
                       data_og.iloc[-1][[f"{ticker}ROC0.5", f"{ticker}ROC1", f"{ticker}ROC3",
                                         f"{ticker}ROC6", f"{ticker}ROC9", f"{ticker}ROC12"]],
                       chosen_weights)[0] / data_og.iloc[-1][f"{ticker}SD12"]})
    am = pd.DataFrame(am)
    am = am.sort_values("Accelerating Momentum", axis=0, ascending=False).reset_index(drop=True)
    return date_i, am.iloc[:top_n]

def backtest_AM_daily_rebalance(input):

    date_i = input[0]
    dates_rebalancing = input[1]
    data_inp = input[2]
    assetsb = input[3]

    try:
        test = data_inp.loc[
            (data_inp["Date"] > str(dates_rebalancing[date_i])) & (
                        data_inp["Date"] < str(dates_rebalancing[date_i + 1]))].reset_index().drop(['index'], axis=1)
        test.set_index(test["Date"], inplace=True)
    except:
        test = data_inp.loc[(data_inp["Date"] > str(dates_rebalancing[date_i]))]
        test.set_index(test["Date"], inplace=True)
    tickers = assetsb[date_i]["Ticker"].to_list()
    returns = pd.DataFrame()
    returns["Return"] = test[[f"{ticker}Return" for ticker in tickers]].mean(axis=1)

    return returns["Return"]

def backtest_Alpha_AM_daily_rebalance(input):

    date_i = input[0]
    dates_rebalancing = input[1]
    data_inp = input[2]
    assetsb = input[3]

    try:
        test = data_inp.loc[
            (data_inp["Date"] > str(dates_rebalancing[date_i])) & (
                        data_inp["Date"] < str(dates_rebalancing[date_i + 1]))].reset_index().drop(['index'], axis=1)
        test.set_index(test["Date"], inplace=True)
    except:
        test = data_inp.loc[(data_inp["Date"] > str(dates_rebalancing[date_i]))]
        test.set_index(test["Date"], inplace=True)
    tickers = assetsb[date_i]["Ticker"].to_list()
    returns = pd.DataFrame()
    returns["Return_nifty"] = test[[f"{ticker}Return" for ticker in tickers]].mean(axis=1)*test["signal_nifty"].shift(1)#+(6/ 25200) * (1 - test["signal"].shift(1))
    returns["signal_nifty"] = test["signal_nifty"].shift(1)
    returns["Return_gold"] = np.log(test["Close_gold"]/test["Close_gold"].shift(1))*test["signal_gold"].shift(1)
    returns["signal_gold"] = test["signal_gold"].shift(1)
    returns["Return"] = 0

    for i in range(len(returns)):

        if returns["signal_nifty"].iloc[i]==1:
            returns["Return"].iloc[i] = returns["Return_nifty"].iloc[i]
            continue
        if returns["signal_gold"].iloc[i]==1:
            returns["Return"].iloc[i] = 0.5*returns["Return_gold"].iloc[i]
            continue
        returns["Return"].iloc[i] = (6/ 25200)


    return returns["Return"]

def backtest_Alpha_AM(dates_rebalancing,data_inp,assetsb):
    current_balance = 1494000
    gold_allocation = 0
    nifty_allocation = 0
    cash_allocation = 0
    portfolio_value = pd.DataFrame()

    for date_i in range(len(dates_rebalancing) - 1):
        try:
            test = data_inp.loc[
                (data_inp["Date"] >= str(dates_rebalancing[date_i])) & (
                        data_inp["Date"] < str(dates_rebalancing[date_i + 1]))].reset_index().drop(['index'], axis=1)
            test.set_index(test["Date"], inplace=True)
        except:
            test = data_inp.loc[(data_inp["Date"] > str(dates_rebalancing[date_i]))]
            test.set_index(test["Date"], inplace=True)
        tickers = assetsb[date_i]["Ticker"].to_list()

        percent_tracker_current_balance_ticker = {}
        percent_tracker_units_ticker = {}
        percent_ticker = {}
        for ticker in tickers:
            percent_tracker_current_balance_ticker[ticker] = current_balance / len(tickers)
            percent_tracker_units_ticker[ticker] = percent_tracker_current_balance_ticker[ticker] / test.iloc[0][ticker]

        current_balance_ticker = {}
        units_ticker = {}
        for ticker in tickers:
            current_balance_ticker[ticker] = current_balance / len(tickers)
            units_ticker[ticker] = current_balance_ticker[ticker] / test.iloc[0][ticker]

        units_gold = 0

        for i in range(len(test)):

            for ticker in tickers:
                percent_tracker_current_balance_ticker[ticker] = percent_tracker_units_ticker[ticker] *  test.iloc[i][ticker]
            for ticker in tickers:
                percent_ticker[ticker] = percent_tracker_current_balance_ticker[ticker] / sum(percent_tracker_current_balance_ticker.values())

            signal_nifty = test["signal_nifty"].iloc[i]
            signal_gold = test["signal_gold"].iloc[i]

            if signal_nifty == 1:
                nifty_allocation = current_balance
                gold_allocation = 0
                cash_allocation = 0
            if (signal_nifty == 0) & (signal_gold == 1):
                nifty_allocation = 0
                gold_allocation = current_balance / 2
                cash_allocation = current_balance / 2
            if (signal_nifty == 0) & (signal_gold == 0):
                nifty_allocation = 0
                gold_allocation = 0
                cash_allocation = current_balance

            if ((test["signal_nifty"].iloc[i] == 1) & (test["signal_nifty"].shift(1).fillna(0).iloc[i] == 0)):
                for ticker in tickers:
                    current_balance_ticker[ticker] = nifty_allocation * percent_ticker[ticker]
                    units_ticker[ticker] = current_balance_ticker[ticker] / test.iloc[i][ticker]

            if ((test["signal_nifty"].iloc[i] == 0) & (test["signal_nifty"].shift(1).fillna(1).iloc[i] == 1)):
                for ticker in tickers:
                    current_balance_ticker[ticker] = 0
                    units_ticker[ticker] = 0
                if signal_gold == 1:
                    units_gold = gold_allocation / test.iloc[i]["Close_gold"]

            if ((test["signal_gold"].iloc[i] == 1) & (test["signal_gold"].shift(1).fillna(0).iloc[i] == 0)):
                units_gold = gold_allocation / test.iloc[i]["Close_gold"]

            if ((test["signal_gold"].iloc[i] == 0) & (test["signal_gold"].shift(1).fillna(1).iloc[i] == 1)):
                units_gold = 0

            if signal_nifty == 1:
                units_gold = 0
                nifty_allocation = 0
                for ticker in tickers:
                    current_balance_ticker[ticker] = units_ticker[ticker] * test.iloc[i][ticker]
                    nifty_allocation = nifty_allocation + current_balance_ticker[ticker]
            if (signal_nifty == 0) & (signal_gold == 1):
                gold_allocation = units_gold * test.iloc[i]["Close_gold"]
            cash_allocation = cash_allocation * (1 + 6 / 25200)
            current_balance = nifty_allocation + gold_allocation + cash_allocation
            portfolio_day = {'Date': test.iloc[i]["Date"], 'signal_nifty': signal_nifty, 'signal_gold':signal_gold,'units_gold':units_gold,'nifty_allocation':nifty_allocation,'gold_allocation':gold_allocation,'cash_allocation':cash_allocation,'Pvalue': current_balance}
            portfolio_day[f"Gold_close"] = test.iloc[i]["Close_gold"]
            for ticker in tickers:
                portfolio_day[f"{ticker}_close"] = test.iloc[i][ticker]
                portfolio_day[f"{ticker}_percent"] = percent_ticker[ticker]
                portfolio_day[f"{ticker}_units"] = units_ticker[ticker]
                portfolio_day[f"{ticker}_current_balance"] = current_balance_ticker[ticker]
            portfolio_day = pd.DataFrame([portfolio_day])
            portfolio_day = portfolio_day.set_index("Date")
            portfolio_value = pd.concat([portfolio_value, portfolio_day], axis=0, join="outer")

    #print(portfolio_value)
    #returns = pd.DataFrame(np.log(portfolio_value / portfolio_value.shift(1))).rename(columns={"Pvalue": "Return"})
    return portfolio_value,units_ticker,units_gold

def corr_sortino_filter(inp):
    date_i = inp[0]
    dates = inp[1]
    temp_og = inp[2]
    res_total = inp[3]
    num_strategies = inp[4]
    train_monthsf = inp[5]

    temp = temp_og.loc[(temp_og["Date"] > str(dates[date_i])) & (temp_og["Date"] < str(dates[date_i + (int(train_monthsf/3)+1)]))].reset_index().drop(['index'], axis=1)
    res = res_total[date_i]
    x, y = corr_filter(temp, res, dates, date_i, num_strategies, train_monthsf)
    return date_i,x,y

def corr_filter(temp, res, dates, date_i, num_strategies, train_monthsf):
    res.sort_values("AvgWinLoss", axis=0, ascending=False, inplace=True)
    res.reset_index().drop(['index'], axis=1)
    returns, _ = top_n_strat_params_rolling(temp, res, to_train=True, num_of_strat=len(res), split_date =str(dates[date_i+int(train_monthsf/3)]))
    if returns.empty:
        return [], pd.DataFrame( columns=["Lookback", "Low Bound", "High Bound", "AvgWinLoss", "Optimization_Years"])
    corr_mat = returns.corr()
    first_selected_strategy = 'Strategy1'
    selected_strategies = strategy_selection(returns, corr_mat, num_strategies, first_selected_strategy)
    params = selected_params(selected_strategies, res)
    res = params.drop(["Name"], axis=1)
    return (selected_strategies, res)

def top_n_strat_params_rolling(temp, res, to_train, num_of_strat, split_date):
    if len(res)>0:
        for i in range(num_of_strat):
            f_look = res.iloc[i, 0]
            bf = res.iloc[i, 1]
            sf = res.iloc[i, 2]
            temp["fisher"] = temp[f'Fisher{int(f_look)}']
            if to_train:
                train = temp.loc[(temp["Date"] <= split_date)].reset_index().drop(['index'], axis=1)
            else:
                train = temp.loc[temp["Date"] > split_date].reset_index().drop(['index'], axis=1)
            test_strategy = FISHER_bounds_strategy_opt(train, zone_low=bf, zone_high=sf)
            dummy_signal = test_strategy.generate_signals()
            dummy = test_strategy.signal_performance(10000, 6)
            if i==0:
                strat_sig_returns = dummy['S_Return'].to_frame().rename(columns={'S_Return': f'Strategy{i + 1}'}).set_index(dummy["Date"])
                strat_sig = dummy_signal['signal'].to_frame().rename(columns={'signal': f'Strategy{i + 1}'}).set_index(dummy_signal["Date"])
                #fisher_test = temp["fisher"].to_frame().rename(columns={"fisher": f'Fisher{asset}{i + 1}'}).set_index(temp["Date"])
            else:
                strat_sig_returns = pd.merge(strat_sig_returns, (dummy['S_Return'].to_frame().rename(columns={'S_Return': f'Strategy{i + 1}'}).set_index(dummy["Date"])), left_index=True, right_index=True)
                strat_sig = pd.concat([strat_sig, (dummy_signal['signal'].to_frame().rename(columns={'signal': f'Strategy{i + 1}'}).set_index(dummy_signal["Date"]))], axis = 1)
                #fisher_test = pd.concat([fisher_test, (temp["fisher"].to_frame().rename(columns={'fisher': f'Fisher{asset}{i + 1}'}).set_index(temp["Date"]))], axis = 1)
            #strat_sig_returns = pd.merge(strat_sig_returns,dummy['S_Return'].to_frame().rename(columns = {'S_Return':f'Strategy{i + 1}'}).set_index(dummy["Date"]), left_index=True, right_index=True)
        #return dummy
        return strat_sig_returns, strat_sig#, fisher_test
    else:
        return pd.DataFrame(), pd.DataFrame()

def strategy_selection(returns, corr_mat, num_strat, first_selected_strategy):
    strategies = [column for column in returns]
    selected_strategies = [first_selected_strategy]
    strategies.remove(first_selected_strategy)
    last_selected_strategy = first_selected_strategy

    while len(selected_strategies) < num_strat:
        corrs = corr_mat.loc[strategies][last_selected_strategy]
        corrs = corrs.loc[corrs>0.9]
        strategies = [st for st in strategies if st not in corrs.index.to_list()]

        if len(strategies)==0:
            break

        strat = strategies[0]

        selected_strategies.append(strat)
        strategies.remove(strat)
        last_selected_strategy = strat

    return selected_strategies

def selected_params(selected_strategies, res):
    selected_params = []
    for strategy in selected_strategies:
        selected_params.append(
            {"Name": strategy, "Lookback": res.iloc[int(strategy[8:])-1]["Lookback"],
             "Low Bound": res.iloc[int(strategy[8:])-1]["Low Bound"],
             "High Bound": res.iloc[int(strategy[8:])-1]["High Bound"],
             #"Sortino": res.iloc[int(strategy[8:])-1]["Sortino"],
             "AvgWinLoss": res.iloc[int(strategy[8:])-1]["AvgWinLoss"],
             "Optimization_Years": res.iloc[int(strategy[8:])-1]["Optimization_Years"]})
    selected_params = pd.DataFrame(selected_params)
    return selected_params

class FISHER_bounds_strategy_opt:

    def __init__(self, data, zone_low, zone_high, start=None, end=None):
        self.zl = zone_low
        self.zh = zone_high
        self.data = data  # the dataframe
        #self.data['yr'] = self.data['Date'].dt.year
        #self.n_days = (self.data.Date.iloc[-1] - self.data.Date.iloc[0])
        self.start = start  # the beginning date of the sample period
        self.end = end  # the ending date of the sample period

    def generate_signals(self, charts=True):
        self.data = self.data.loc[(self.data.fisher != 0)]
        self.data["fisher_lag"] = self.data.fisher.shift(1)
        self.data["lb"] = self.zl
        self.data["ub"] = self.zh
        self.data.dropna()
        self.data.reset_index(inplace=True)
        self.data = self.data.drop(['index'], axis=1)

        ## creating signal
        ##Upper and lower bounds
        buy_mask = (self.data["fisher"] > self.data["lb"]) & (self.data["fisher_lag"] < self.data["lb"])
        sell_mask = ((self.data["fisher"] < self.data["ub"]) & (self.data["fisher_lag"] > self.data["ub"])) | (
                    self.data["fisher"] < np.minimum(self.data["lb"], self.data["ub"]))

        bval = +1
        sval = 0  # -1 if short selling is allowed, otherwise 0

        self.data['signal_bounds'] = np.nan
        self.data.loc[buy_mask, 'signal_bounds'] = bval
        self.data.loc[sell_mask, 'signal_bounds'] = sval
        # initialize with long
        self.data["signal_bounds"][0] = 1

        self.data.signal_bounds = self.data.signal_bounds.fillna(method="ffill")

        self.data["signal"] = self.data.signal_bounds

        # Closing positions at end of time period
        self.data["signal"][-1:] = 0

        mask = ((self.data.signal == 1) & (self.data.signal.shift(1) == 0)) & (self.data.signal.notnull())
        self.data['trade_num'] = np.where(mask, 1, 0).cumsum()

        return self.data[["Date", "signal"]]

    def signal_performance(self, allocation, interest_rate):
        """
        Another instance method
        """
        self.allocation = allocation
        self.int = interest_rate

        # creating returns and portfolio value series
        self.data['Return'] = np.log(self.data['Close'] / self.data['Close'].shift(1))
        self.data['S_Return'] = self.data['signal'].shift(1) * self.data['Return']
        self.data['S_Return'] = self.data['S_Return'].fillna(0)
        self.data["S_Return"] = self.data["S_Return"] + (self.int / 25200) * (1 - self.data['signal'].shift(1))
        self.data['Market_Return'] = self.data['Return'].expanding().sum()
        self.data['Strategy_Return'] = self.data['S_Return'].expanding().sum()
        self.data['Portfolio Value'] = ((self.data['Strategy_Return'] + 1) * self.allocation)
        self.data['Wins'] = np.where(self.data['S_Return'] > 0, 1, 0)
        self.data['Losses'] = np.where(self.data['S_Return'] < 0, 1, 0)

        return self.data[['Date', 'Close', 'signal', 'S_Return', 'trade_num', 'Return']]

class FISHER_MCMC:

    def __init__(self, data, signals, start=None, end=None):

        self.signals = signals
        self.data = data  # the dataframe
        self.start = start  # the beginning date of the sample period
        self.end = end  # the ending date of the sample period

    def generate_signals(self, charts=True):

        self.data["signal"] = self.signals

        # Closing positions at end of time period
        self.data["signal"][-1:] = 0

        mask = ((self.data.signal == 1) & (self.data.signal.shift(1)==0)) & (self.data.signal.notnull())
        self.data['trade_num'] = np.where(mask, 1, 0).cumsum()

    def signal_performance(self, allocation, interest_rate):

        self.allocation = allocation
        self.int = interest_rate
        self.data = self.data.reset_index().rename(columns={'index':'Date'})
        # self.data['yr'] = self.data['Date'].dt.year
        # self.n_days = (self.data.Date.iloc[-1] - self.data.Date.iloc[0])

        # creating returns and portfolio value series
        self.data['Return'] = np.log(self.data['Close'] / self.data['Close'].shift(1))
        self.data['S_Return'] = self.data['signal'].shift(1) * self.data['Return']
        self.data['S_Return'] = self.data['S_Return'].fillna(0)
        self.data['Return'] = self.data['Return'].fillna(0)
        self.data["S_Return"] = self.data["S_Return"] + (self.int/25200)*(1-self.data['signal'].shift(1))
        self.data['Market_Return'] = self.data['Return'].expanding().sum()
        self.data['Strategy_Return'] = self.data['S_Return'].expanding().sum()
        self.data['Portfolio Value'] = ((self.data['Strategy_Return'] + 1) * self.allocation)
        self.data['Wins'] = np.where(self.data['S_Return'] > 0, 1, 0)
        self.data['Losses'] = np.where(self.data['S_Return'] < 0, 1, 0)

        return self.data[['Date', 'Close', 'signal', 'S_Return', 'trade_num', 'Return']]

class FISHER_bounds_strategy_opt_live:
    metrics = {'Sharpe': 'Sharpe Ratio',
               'CAGR': 'Compound Average Growth Rate',
               'MDD': 'Maximum Drawdown',
               'NHR': 'Normalized Hit Ratio',
               'OTS': 'Optimal trade size'}

    def __init__(self, data, zone_low, zone_high, start=None, end=None):

        self.zl = zone_low
        self.zh = zone_high
        self.data = data  # the dataframe
        self.data['yr'] = self.data['Date'].dt.year
        self.n_days = (self.data.Date.iloc[-1] - self.data.Date.iloc[0])
        self.start = start  # the beginning date of the sample period
        self.end = end  # the ending date of the sample period

    def generate_signals(self, charts=True):

        self.data = self.data.loc[(self.data.fisher != 0)]
        self.data["fisher_lag"] = self.data.fisher.shift(1)
        self.data["lb"] = self.zl
        self.data["ub"] = self.zh
        self.data.dropna()
        self.data.reset_index(inplace=True)
        self.data = self.data.drop(['index'], axis=1)

        ## creating signal
        ##Upper and lower bounds
        buy_mask = (self.data["fisher"] > self.data["lb"]) & (self.data["fisher_lag"] < self.data["lb"])

        sell_mask = ((self.data["fisher"] < self.data["ub"]) & (self.data["fisher_lag"] > self.data["ub"]))|(self.data["fisher"] < np.minimum(self.data["lb"], self.data["ub"]))

        bval = +1
        sval = 0 # -1 if short selling is allowed, otherwise 0

        self.data['signal_bounds'] = np.nan
        self.data.loc[buy_mask, 'signal_bounds'] = bval
        self.data.loc[sell_mask, 'signal_bounds'] = sval
        # initialize with long
        self.data["signal_bounds"][0] = 1

        self.data.signal_bounds = self.data.signal_bounds.fillna(method="ffill")
        #self.data.signal_bounds = self.data.signal_bounds.fillna(0)

        self.data["signal"] = self.data.signal_bounds

        mask = ((self.data.signal == 1) & (self.data.signal.shift(1)==0)) & (self.data.signal.notnull())
        self.data['trade_num'] = np.where(mask, 1, 0).cumsum()

        return self.data[["Date", "signal"]]

    def signal_performance(self, allocation, interest_rate):

        self.allocation = allocation
        self.int = interest_rate

        # creating returns and portfolio value series
        self.data['Return'] = np.log(self.data['Close'] / self.data['Close'].shift(1))
        self.data['S_Return'] = self.data['signal'].shift(1) * self.data['Return']
        self.data['S_Return'] = self.data['S_Return'].fillna(0)
        self.data["S_Return"] = self.data["S_Return"] + (self.int/25200)*(1-self.data['signal'].shift(1))
        self.data['Market_Return'] = self.data['Return'].expanding().sum()
        self.data['Strategy_Return'] = self.data['S_Return'].expanding().sum()
        self.data['Portfolio Value'] = ((self.data['Strategy_Return'] + 1) * self.allocation)
        self.data['Wins'] = np.where(self.data['S_Return'] > 0, 1, 0)
        self.data['Losses'] = np.where(self.data['S_Return'] < 0, 1, 0)

        return self.data[['Date', 'Close', 'signal', 'S_Return', 'trade_num', 'Return']]

class FISHER_MCMC_live:

    def __init__(self, data, signals, start=None, end=None):

        self.signals = signals
        self.data = data  # the dataframe
        self.start = start  # the beginning date of the sample period
        self.end = end  # the ending date of the sample period

    def generate_signals(self, charts=True):

        self.data["signal"] = self.signals

        mask = ((self.data.signal == 1) & (self.data.signal.shift(1)==0)) & (self.data.signal.notnull())
        self.data['trade_num'] = np.where(mask, 1, 0).cumsum()

    def signal_performance(self, allocation, interest_rate):

        self.allocation = allocation
        self.int = interest_rate
        self.data = self.data.reset_index().rename(columns={'index':'Date'})
        self.data['yr'] = self.data['Date'].dt.year
        self.n_days = (self.data.Date.iloc[-1] - self.data.Date.iloc[0])

        # creating returns and portfolio value series
        self.data['Return'] = np.log(self.data['Close'] / self.data['Close'].shift(1))
        self.data['S_Return'] = self.data['signal'].shift(1) * self.data['Return']
        self.data['S_Return'] = self.data['S_Return'].fillna(0)
        self.data['Return'] = self.data['Return'].fillna(0)
        self.data["S_Return"] = self.data["S_Return"] + (self.int/25200)*(1-self.data['signal'].shift(1))
        self.data['Market_Return'] = self.data['Return'].expanding().sum()
        self.data['Strategy_Return'] = self.data['S_Return'].expanding().sum()
        self.data['Portfolio Value'] = ((self.data['Strategy_Return'] + 1) * self.allocation)
        self.data['Wins'] = np.where(self.data['S_Return'] > 0, 1, 0)
        self.data['Losses'] = np.where(self.data['S_Return'] < 0, 1, 0)

        return self.data[['Date', 'Close', 'signal', 'S_Return', 'trade_num', 'Return']]

def optimize_weights_live(input):

    def get_equity_curve(*args):
        weights = []
        for weight in args:
            weights.append(weight)
        # weights.append((1-sum(weights)))
        weights = pd.DataFrame(weights)
        weights = weights / weights.sum()
        signal_final = create_final_signal_weights(signal, params, weights, num_strategies)
        inp = pd.merge(train.set_index(train["Date"]), signal_final, left_index=True, right_index=True)
        test_strategy = FISHER_MCMC(inp.drop(['signal', 'Date'], axis=1), inp['signal'])
        _ = test_strategy.generate_signals()
        ec = test_strategy.signal_performance(10000, 6)

        return ec

    def sortino(x, y, bins):
        ecdf = x[["S_Return"]]
        stdev_down = ecdf.loc[ecdf["S_Return"] < 0, "S_Return"].std() * (252 ** .5)
        if math.isnan(stdev_down):
            stdev_down = 0.0
        if stdev_down != 0.0:
            sortino = (ecdf["S_Return"].mean() - 0.06 / 252) * 252 / stdev_down
        else:
            sortino = 0
        return np.float64(sortino)

    def sharpe(x, y, bins):
        ecdf = x[["S_Return"]]

        stdev = ecdf["S_Return"].std() * (252 ** .5)
        if math.isnan(stdev):
            stdev = 0.0
        if stdev != 0.0:
            sharpe = (ecdf["S_Return"].mean() - 0.06 / 252) * 252 / stdev
        else:
            sharpe = 0
        return np.float64(sharpe)

    def rolling_sharpe(x, y, bins):
        r_window = 252
        ecdf = x[["Date", "S_Return"]]
        ecdf['RSharpeRatio_Series'] = (ecdf["S_Return"].rolling(window=r_window, min_periods=1).mean() - 0.06 / 252) / ecdf[
            "S_Return"].rolling(window=r_window, min_periods=1).std() * (252 ** .5)
        ecdf['RSharpeRatio_Series'] = ecdf['RSharpeRatio_Series'].replace(to_replace=math.inf, value=0)
        ecdf['RSharpeRatio_Series'] = ecdf['RSharpeRatio_Series'].replace(to_replace=math.nan, value=0)
        RSharpeRatio = ecdf['RSharpeRatio_Series'].mean()
        return np.float64(RSharpeRatio)

    def rolling_sortino(x, y, bins):
        r_window = 252
        ecdf = x[["Date", "S_Return"]]
        ecdf["S_Return_stdev"] = ecdf["S_Return"].copy()
        ecdf["S_Return_stdev"] = np.where(ecdf["S_Return_stdev"] >= 0, ecdf["S_Return_stdev"], np.nan)
        ecdf['RStDev Annualized Downside Return_Series'] = ecdf["S_Return_stdev"].rolling(window=r_window,
                                                                                          min_periods=1).std() * (
                                                                       252 ** .5)
        ecdf['RStDev Annualized Downside Return_Series'] = ecdf['RStDev Annualized Downside Return_Series'].replace(
            to_replace=math.inf, value=0)
        ecdf['RStDev Annualized Downside Return_Series'] = ecdf['RStDev Annualized Downside Return_Series'].replace(
            to_replace=math.nan, value=0)
        ecdf['RSortinoRatio_Series'] = np.where(ecdf['RStDev Annualized Downside Return_Series'] != 0.0,
                                                (ecdf["S_Return"].rolling(window=r_window, min_periods=1).mean() - 0.06 / 252) * 252 /
                                                ecdf['RStDev Annualized Downside Return_Series'], 0)
        RSortinoRatio = ecdf['RSortinoRatio_Series'].mean()
        return np.float64(RSortinoRatio)

    def rolling_cagr(x, y, bins):
        ecdf = x[["Date", "S_Return"]]
        ecdf = ecdf['S_Return'].to_frame().set_index(ecdf["Date"])
        ecdf['Portfolio'] = 1 + ecdf['S_Return'].expanding().sum()
        ecdf['Portfolio'] = ecdf['Portfolio']
        ecdf['Portfolio_1yr'] = ecdf['Portfolio'].to_frame().shift(365, freq='D')
        ecdf['Portfolio_1yr'] = ecdf['Portfolio_1yr'].fillna(method="ffill")
        ecdf['RCAGR_Strategy_Series'] = ecdf['Portfolio'] / ecdf['Portfolio_1yr'] - 1
        RCAGR_Strategy = ecdf['RCAGR_Strategy_Series'].mean()
        return np.float64(RCAGR_Strategy)

    def maxdrawup_by_maxdrawdown(x, y, bins):
        r_window = 252
        ecdf = x[["Date", "S_Return"]]
        ecdf['Strategy_Return'] = ecdf['S_Return'].expanding().sum()
        ecdf['Portfolio Value'] = ((ecdf['Strategy_Return'] + 1) * 1)
        ecdf['Portfolio Value'][0] = 1
        ecdf['Drawdown_Series'] = 1 - ecdf['Portfolio Value'] / np.maximum.accumulate(ecdf['Portfolio Value'])
        ecdf['Max_Drawdown_Series'] = ecdf['Drawdown_Series'].rolling(window=r_window, min_periods=1).max()
        ecdf['Drawup_Series'] = ecdf['Portfolio Value'] / np.minimum.accumulate(ecdf['Portfolio Value']) - 1
        ecdf['Max_Drawup_Series'] = ecdf['Drawup_Series'].rolling(window=r_window, min_periods=1).max()
        ecdf['Drawup/Drawdown_Series'] = ecdf['Max_Drawup_Series'] / ecdf['Max_Drawdown_Series']
        ecdf['Drawup/Drawdown_Series'] = ecdf['Drawup/Drawdown_Series'].replace(math.inf, 100)
        RDrawupDrawdown = ecdf['Drawup/Drawdown_Series'].mean()
        # ecdf['Drawdown_Series'] = 1 - ecdf['Portfolio Value'] / np.maximum.accumulate(ecdf['Portfolio Value'])
        # ecdf['Drawup_Series'] = ecdf['Portfolio Value'] / np.minimum.accumulate(ecdf['Portfolio Value']) - 1
        # RDrawupDrawdown = ecdf['Drawup_Series'].max() / ecdf['Drawdown_Series'].max()
        return np.float64(RDrawupDrawdown)

    def outperformance(x, y, bins):
        r_window = 252
        ecdf1 = x['S_Return'].to_frame().set_index(x["Date"])
        ecdf2 = x['Return'].to_frame().set_index(x["Date"])
        ecdf1['Portfolio'] = 1 + ecdf1['S_Return'].expanding().sum()
        ecdf1['Portfolio_1yr'] = ecdf1['Portfolio'].to_frame().shift(365, freq='D')
        ecdf2['Portfolio'] = 1 + ecdf2['Return'].expanding().sum()
        ecdf2['Portfolio_1yr'] = ecdf2['Portfolio'].to_frame().shift(365, freq='D')
        ecdf1['Portfolio_1yr'] = ecdf1['Portfolio_1yr'].fillna(method="ffill")
        ecdf1['RCAGR_Strategy_Series'] = ecdf1['Portfolio'] / ecdf1['Portfolio_1yr'] - 1
        ecdf2['Portfolio_1yr'] = ecdf2['Portfolio_1yr'].fillna(method="ffill")
        ecdf2['RCAGR_Market_Series'] = ecdf2['Portfolio'] / ecdf2['Portfolio_1yr'] - 1
        RCAGR_Strategy = ecdf1['RCAGR_Strategy_Series'].mean()
        RCAGR_Market = ecdf2['RCAGR_Market_Series'].mean()
        ROutperformance = RCAGR_Strategy - RCAGR_Market
        return np.float64(ROutperformance)

    def prior(params):
        return 1

    date_i = input[0]
    dates = input[1]
    temp_og = input[2]
    ss_test = input[3]
    res_test = input[4]
    num_strategies = input[5]
    metric = input[6]
    recalib_months = input[7]
    dates_all = input[8]

    if (date_i - int(24 / recalib_months)) < 0:
        temp = temp_og.loc[
            (temp_og["Date"] > str(dates_all[dates_all.index(dates[date_i]) - int(24 / 3)])) & (
                    temp_og["Date"] < str(dates[date_i + 1]))].reset_index().drop(
            ['index'], axis=1)
    else:
        temp = temp_og.loc[
            (temp_og["Date"] > str(dates[date_i - int(24 / recalib_months)])) & (
                    temp_og["Date"] < str(dates[date_i + 1]))].reset_index().drop(
            ['index'], axis=1)

    train = temp.loc[(temp["Date"] < str(dates[date_i]))].reset_index().drop(['index'], axis=1)
    test = temp.loc[(temp["Date"] >= str(dates[date_i]))].reset_index().drop(['index'], axis=1)

    if len(ss_test[date_i]) > 0:
        if len(ss_test[date_i]) > num_strategies:
            selected_strategies = ss_test[date_i][:num_strategies]
        else:
            selected_strategies = ss_test[date_i]

        if len(res_test[date_i]) > num_strategies:
            res = res_test[date_i][:num_strategies]
        else:
            res = res_test[date_i]

        # print("Optimizing weights")
        strategies = ["Strategy" + str(i) for i in range(1, len(res) + 1)]
        params = selected_params(strategies, res)

        _, signal = top_n_strat_params_rolling(temp, res, to_train=True, num_of_strat=len(selected_strategies),
                                               split_date=str(dates[date_i]))
        #print("Running MCMC")
        guess = (0.5 * np.ones([1, len(selected_strategies)])).tolist()[0]

        if len(guess)>1:
            if metric == 'rolling_sharpe':
                mc = MCMC(alpha_fn=get_equity_curve, alpha_fn_params_0=guess, target=_, num_iters=10000, prior=prior,
                          optimize_fn=rolling_sharpe, lower_limit=0, upper_limit=1)
                rs = mc.optimize()

            if metric == 'rolling_sortino':
                mc = MCMC(alpha_fn=get_equity_curve, alpha_fn_params_0=guess, target=_, num_iters=10000, prior=prior,
                          optimize_fn=rolling_sortino, lower_limit=0, upper_limit=1)
                rs = mc.optimize()

            if metric == 'rolling_cagr':
                mc = MCMC(alpha_fn=get_equity_curve, alpha_fn_params_0=guess, target=_, num_iters=10000, prior=prior,
                          optimize_fn=rolling_cagr, lower_limit=0, upper_limit=1)
                rs = mc.optimize()

            if metric == 'maxdrawup_by_maxdrawdown':
                mc = MCMC(alpha_fn=get_equity_curve, alpha_fn_params_0=guess, target=_, num_iters=10000, prior=prior,
                          optimize_fn=maxdrawup_by_maxdrawdown, lower_limit=0, upper_limit=1)
                rs = mc.optimize()

            if metric == 'outperformance':
                mc = MCMC(alpha_fn=get_equity_curve, alpha_fn_params_0=guess, target=_, num_iters=10000, prior=prior,
                          optimize_fn=outperformance, lower_limit=0, upper_limit=1)
                rs = mc.optimize()

            # Printing results:
            weights = []
            for weight in mc.analyse_results(rs, top_n=1)[0][0]:
                weights.append(weight)
            weights = pd.DataFrame(weights)
            weights = weights / weights.sum(axis=0)

        else:
            weights = pd.DataFrame([1])

    else:
        weights = pd.DataFrame()

    return date_i, weights