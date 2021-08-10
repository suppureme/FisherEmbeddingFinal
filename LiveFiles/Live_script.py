import random
import multiprocessing
import pandas as pd
import pickle
from datetime import date, datetime
import time

import warnings
warnings.filterwarnings('ignore')
from helper_functions import *
import pickle
from datetime import timedelta
import os
import smtplib
from email.mime.text import MIMEText
from email.mime.image import MIMEImage
from email.mime.multipart import MIMEMultipart

import matplotlib.pyplot as plt
plt.rcParams["figure.figsize"] = (15,5)
plt.rcParams['axes.grid'] = False
import seaborn as sns
sns.set_style("whitegrid", {'axes.grid' : False})

from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()

def get_data(ticker):
    import yfinance as yf
    temp_og = yf.download(ticker, start = '2007-01-01', end= str(date.today()))
    temp_og.reset_index(inplace=True)
    temp_og.drop(['Adj Close'], axis=1, inplace=True)
    temp_og = temp_og.loc[temp_og["Close"]>1]

    today_data = yf.download(ticker, start=str(date.today() - timedelta(days=1)), interval="1m")
    today_data.reset_index(inplace=True)
    today_data.drop(['Adj Close'], axis=1, inplace=True)
    today_data = today_data.loc[today_data["Close"] > 1]

    today_time_close = today_data.iloc[-1]["Datetime"]

    temp_og = pd.concat([temp_og, pd.DataFrame([{"Date": pd.Timestamp(year=today_data.iloc[-1]["Datetime"].year,
                                                                      month=today_data.iloc[-1]["Datetime"].month,
                                                                      day=today_data.iloc[-1]["Datetime"].day),\
                                                 "Open": today_data.iloc[-1]["Open"],
                                                 "Close": today_data.iloc[-1]["Close"],
                                                 "High": today_data.iloc[-1]["High"],
                                                 "Low": today_data.iloc[-1]["Low"],
                                                 "Volume": today_data.iloc[-1]["Volume"]}])],axis=0).reset_index().drop(['index'], axis=1)
    temp_og.drop_duplicates(subset="Date",
                            keep='first', inplace=True)
    temp_og = add_fisher(temp_og)
    return temp_og, today_time_close

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
        with open(f'{ticker}/SelectedStrategies/{ticker}_TrainYrs_{int(train_monthsf/12)}_All_Strategies.pkl', 'wb') as file:
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
        with open(f'{ticker}/SelectedStrategies/{ticker}_OptPeriods_{num_opt_periodsf}_Selected_Strategies_ss.pkl', 'wb') as file:
            pickle.dump(ss_test, file)
        with open(f'{ticker}/SelectedStrategies/{ticker}_OptPeriods_{num_opt_periodsf}_Selected_Strategies_res.pkl', 'wb') as file:
            pickle.dump(res_test, file)

    return ss_test, res_test

def SendMail(ticker, text, printdf, ImgFileName):
    msg = MIMEMultipart()
    msg['Subject'] = f'Strategy Update on {ticker}'
    msg['From'] = 'suprabhashsahu@acsysindia.com'
    msg['To'] = 'suprabhashsahu@acsysindia.com, aditya@shankar.biz'

    text = MIMEText(text)
    msg.attach(text)

    with open(ImgFileName, 'rb') as f:
        img_data = f.read()
    image = MIMEImage(img_data, name=os.path.basename(ImgFileName))
    msg.attach(image)

    strategies = """\
    <html>
      <head></head>
      <body>
        {0}
      </body>
    </html>
    """.format(printdf.to_html())

    part1 = MIMEText(strategies, 'html')
    msg.attach(part1)

    s = smtplib.SMTP('smtp.gmail.com', 587)
    s.ehlo()
    s.starttls()
    s.ehlo()
    s.login('suprabhashsahu@acsysindia.com', 'esahYah8')
    s.sendmail('suprabhashsahu@acsysindia.com', ['suprabhashsahu@acsysindia.com',  'aditya@shankar.biz'], msg.as_string())
    s.quit()

def signal_print(inp):
    if inp == 0:
        signal = "Neutral on Equity, Long on Fixed Income"
    else:
        signal = "Long on Equity, Neutral on Fixed Income"
    return signal

if __name__ == '__main__':

    run_hour = 23
    run_minute = 59
    time_now = datetime.now()
    first_run = datetime.now()
    #first_run = first_run.replace(day=first_run.day + 1).replace(hour=13).replace(minute=52).replace(second=00)
    first_run = first_run.replace(day=first_run.day).replace(hour=run_hour).replace(minute=run_minute).replace(second=00)

    print("Sleeping")

    time.sleep((first_run - time_now).seconds-100)

    print("Woken up")

    while True:
        if (datetime.today().isoweekday() < 6) & (datetime.now().hour==run_hour) & (datetime.now().minute==run_minute) & (datetime.now().second==00):

            print("Executing")
            # Define all run parameters
            ticker = "GOLD"
            number_of_optimization_periods = 1
            recalib_months = 3
            num_strategies = 3
            metric = 'rolling_sortino'
            recalibrate_today = False
            text = ""

            #Download data
            temp_og, today_time_close = get_data(ticker)

            dates_all_ss = pd.date_range(start="2011-01-02", end="2024-06-15", freq=f'3M')
            dates_ss = valid_dates(dates_all_ss)

            text = text + f"Recalibrating on {str(dates_ss[-1])[:11]}" + "\n" + "\n"
            print("Importing selected strategies")
            with open(f'LiveFiles/{ticker}_OptPeriods_{number_of_optimization_periods}_Selected_Strategies_ss.pkl','rb') as file:
                ss_test_imp = pickle.load(file)
            with open(f'LiveFiles/{ticker}_OptPeriods_{number_of_optimization_periods}_Selected_Strategies_res.pkl','rb') as file:
                res_test_imp = pickle.load(file)

            res_test = []
            ss_test = []
            dates = []
            for date_i in range(len(dates_ss) - (int(24 / 3) + 1)):
                if (3 * date_i) % recalib_months == 0:
                    dates.append(dates_ss[date_i + int(24 / 3)])
                    ss_test.append(ss_test_imp[date_i])
                    res_test.append(res_test_imp[date_i])

            dates.append(date.today())
            date_p = [date_i for date_i in range(len(dates)-1)][-1]
            print(f"Selected Strategies for Testing period beginning: {str(dates[date_p])} and ending: {str(dates[date_p+1])}")
            print(res_test[date_p])

            print("Importing Weights")
            with open(f'LiveFiles/{ticker}_OptPeriods_{number_of_optimization_periods}_Weights.pkl','rb') as file:
                weights = pickle.load(file)

            inputs = []
            for date_i in range(len(dates) - 1):
                inputs.append([date_i, dates, temp_og, ss_test, res_test, num_strategies, weights[date_i], recalib_months,dates_ss])
            try:
                pool = multiprocessing.Pool(processes=7, maxtasksperchild=1)
                results_backtest = pool.map(backtest_live, inputs)
            finally:  # To make sure processes are closed in the end, even if errors happen
                pool.close()
                pool.join()

            results_final = pd.DataFrame()
            for tt in results_backtest:
                results_final = pd.concat([results_final, tt[0]], axis=0)
            temp_res = results_final

            temp_res['Return'] = np.log(temp_res['Close'] / temp_res['Close'].shift(1))
            temp_res['Market_Return'] = temp_res['Return'].expanding().sum()
            temp_res['Strategy_Return'] = temp_res['S_Return'].expanding().sum()
            temp_res['Portfolio Value'] = ((temp_res['Strategy_Return'] + 1) * 10000)
            temp_res.reset_index(inplace=True)

            temp_res.to_csv(f"LiveFiles/Performance{ticker}.csv")

            plt.plot(temp_res['Date'], temp_res['Market_Return'], color='black', label='Market Returns')
            plt.plot(temp_res['Date'], temp_res['Strategy_Return'], color='blue', label='Strategy Returns')
            plt.title('Strategy Backtest')
            plt.legend(loc=0)
            plt.tight_layout()
            plt.savefig(f"LiveFiles/Performance{ticker}.jpg")
            plt.clf()

            text = text + "Stats for last 252 trading days:" + "\n"
            text = text +f"Sortino: {np.round(backtest_sortino(temp_res[-252:],0,0), 2)}"+ "\n"
            text = text +f"Sharpe: {np.round(backtest_sharpe(temp_res[-252:],0,0), 2)}"+ "\n"
            text = text +f"Rolling Sortino: {np.round(backtest_rolling_sortino(temp_res[-252:],0,0), 2)}"+ "\n"
            text = text +f"Rolling CAGR: {np.round(backtest_rolling_cagr(temp_res[-252:],0,0), 2)}"+ "\n"
            text = text +f"MaxDrawup/MaxDrawdown: {np.round(backtest_maxdrawup_by_maxdrawdown(temp_res[-252:],0,0), 2)}"+ "\n"+ "\n"
            text = text + "Selected Strategies: " + "\n" + "\n"
            #text = text +f"Outperformance: {backtest_outperformance(temp_res[-252:],0,0)}"+ "\n"
            text = f"Signal at : {str(today_time_close)[:19]} (timezone=America/New_York): {signal_print(temp_res.iloc[-1]['signal'])}" + "\n" f"Signal at Yesterday Close: {signal_print(temp_res.iloc[-2]['signal'])}" + "\n" f"Signal at Day before Yesterday Close: {signal_print(temp_res.iloc[-3]['signal'])}" + "\n" + "\n" + "Overall Performance:" + "\n" + f"Portfolio Value: {np.round(temp_res.iloc[-1]['Portfolio Value'], 2)}" + "\n" + "\n" + text

            SendMail(ticker, text, results_backtest[-1][1], f"LiveFiles/Performance{ticker}.jpg")

            if (pd.to_datetime(date.today()) in dates_all_ss) or recalibrate_today==True:

                print("Recalibrating")
                res_test2 = select_all_strategies(24,dates_ss, temp_og, ticker,save=True)
                # res_test4 = select_all_strategies(48, dates_ss, temp_og, ticker, save=True)
                # res_test8 = select_all_strategies(96, dates_ss, temp_og, ticker, save=True)
                ss_test_imp, res_test_imp = select_strategies_from_corr_filter(res_test2,0,0, dates_ss, temp_og, number_of_optimization_periods,10, ticker, save=True)

                res_test = []
                ss_test = []
                dates = []
                for date_i in range(len(dates_ss) - (int(24 / 3) + 1)):
                    if (3 * date_i) % recalib_months == 0:
                        dates.append(dates_ss[date_i + int(24 / 3)])
                        ss_test.append(ss_test_imp[date_i])
                        res_test.append(res_test_imp[date_i])

                print("Recalibrating Weights")
                inputs = []
                for date_i in range(len(dates)-1):
                    inputs.append([date_i, dates, temp_og, ss_test, res_test, num_strategies, metric, recalib_months,dates_ss])
                try:
                    pool = multiprocessing.Pool(processes=7, maxtasksperchild=1)
                    weights_all = pool.map(optimize_weights_live, inputs)
                finally: # To make sure processes are closed in the end, even if errors happen
                    pool.close()
                    pool.join()

                weights = [None]*(len(dates)-1)
                for date_i in range(len(dates)-1):
                    weights[weights_all[date_i][0]] = weights_all[date_i][1]

                with open(f'LiveFiles/{ticker}_OptPeriods_{number_of_optimization_periods}_Weights.pkl', 'wb') as file:
                    pickle.dump(weights, file)

        if datetime.now()<datetime.now().replace(hour=run_hour).replace(minute=run_minute).replace(second=30):
            continue

        print("Sleeping")

        time_now = datetime.now()
        next_run = datetime.now()
        next_run = next_run.replace(day=next_run.day + 1).replace(hour=run_hour).replace(minute=run_minute).replace(second=00)

        time.sleep((next_run - time_now).seconds-100)

        print("Woken Up")





