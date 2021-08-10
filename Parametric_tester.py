import multiprocessing
import os.path
from os import path

import warnings

import pandas as pd

warnings.filterwarnings('ignore')

import os, sys
currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
sys.path.append(parentdir)
from helper_functions import *
import pickle

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
from datetime import timedelta

def SendMail(ticker, sortby, ImgFileNameList):
    msg = MIMEMultipart()
    msg['Subject'] = f'{ticker} Top 3 strategies sorted by {sortby}'
    msg['From'] = 'suprabhashsahu@acsysindia.com'
    msg['To'] = 'suprabhashsahu@acsysindia.com, aditya@shankar.biz'   #

    text = MIMEText(f'{ticker} Top 3 strategies sorted by {sortby}')
    msg.attach(text)
    for ImgFileName in ImgFileNameList:
        with open(ImgFileName, 'rb') as f:
            img_data = f.read()
        image = MIMEImage(img_data, name=os.path.basename(ImgFileName))
        msg.attach(image)

    s = smtplib.SMTP('smtp.gmail.com', 587)
    s.ehlo()
    s.starttls()
    s.ehlo()
    s.login('suprabhashsahu@acsysindia.com', 'esahYah8')
    s.sendmail('suprabhashsahu@acsysindia.com', ['suprabhashsahu@acsysindia.com', 'aditya@shankar.biz'], msg.as_string())  #
    s.quit()

if __name__ == '__main__':
    #tickers = niftydata()
    tickers = ["ETH-USD", "BTC-USD","^GSPC"] #

    for ticker in tickers:
        for number_of_optimization_periods in [1,2,3]: #1,2,3
            for recalib_months in [3,6,12]:  #,6,12
                for num_strategies in [1,3,5,7]: #
                    for metric in ['rolling_sharpe', 'rolling_sortino', 'rolling_cagr', 'maxdrawup_by_maxdrawdown','outperformance']: #

                        if not os.path.exists(f'TickerResults/{ticker}/equity_curves'):
                            os.makedirs(f'TickerResults/{ticker}/equity_curves')

                        if not os.path.exists(f'TickerResults/{ticker}/csv_files'):
                            os.makedirs(f'TickerResults/{ticker}/csv_files')

                        if not os.path.exists(f'TickerResults/{ticker}/weights'):
                            os.makedirs(f'TickerResults/{ticker}/weights')

                        if path.exists(f"TickerResults/{ticker}\csv_files\Results_Ticker{ticker}_LP{number_of_optimization_periods}_Recal{recalib_months}_NS{num_strategies}_M{metric}.csv"):
                            print("Already processed")
                            continue

                        #try:
                        temp_og = get_data(ticker, "yfinance")
                        dates = valid_dates(pd.date_range(start=str(temp_og.iloc[1]['Date'] + timedelta(days=365))[:10],
                                                          end="2024-06-15", freq=f'3M'))

                        temp_og["Date"] = pd.to_datetime(temp_og["Date"])

                        with open(f'TickerResults/{ticker}/SelectedStrategies/{ticker}_OptPeriods_{number_of_optimization_periods}_Selected_Strategies_ss.pkl', 'rb') as file:
                            ss_test_imp = pickle.load(file)
                        with open(f'TickerResults/{ticker}/SelectedStrategies/{ticker}_OptPeriods_{number_of_optimization_periods}_Selected_Strategies_res.pkl', 'rb') as file:
                            res_test_imp = pickle.load(file)

                        res_test = []
                        ss_test = []
                        datesb = []
                        for date_i in range(len(dates) - (int(24 / 3) + 1)):
                            if (3 * date_i) % recalib_months == 0:
                                datesb.append(dates[date_i + int(24 / 3)])
                                ss_test.append(ss_test_imp[date_i])
                                res_test.append(res_test_imp[date_i])

                        datesb.append(date.today())
                        inputs = []
                        for date_i in range(len(datesb)-1):
                            inputs.append([date_i, datesb, temp_og, ss_test, res_test, num_strategies, metric, recalib_months,dates])
                        try:
                            pool = multiprocessing.Pool(processes=7, maxtasksperchild=1)
                            results_multi = pool.map(optimize_weights_and_backtest, inputs)
                        finally: # To make sure processes are closed in the end, even if errors happen
                            pool.close()
                            pool.join()

                        weights = [None] * (len(datesb)-1)
                        for date_i in range(len(datesb)-1):
                            weights[results_multi[date_i][0]] = results_multi[date_i][2]

                        with open(f"TickerResults/{ticker}/weights/Results_Ticker{ticker}_LP{number_of_optimization_periods}_Recal{recalib_months}_NS{num_strategies}_M{metric}.pkl",'wb') as file:
                            pickle.dump(weights, file)

                        results_final = pd.DataFrame()
                        for tt in results_multi:
                            results_final = pd.concat([results_final, tt[1]], axis=0)

                        temp_res = results_final
                        temp_res['Return'] = np.log(temp_res['Close'] / temp_res['Close'].shift(1))
                        temp_res['Market_Return'] = temp_res['Return'].expanding().sum()
                        temp_res['Strategy_Return'] = temp_res['S_Return'].expanding().sum()
                        temp_res['Portfolio Value'] = ((temp_res['Strategy_Return'] + 1) * 10000)
                        temp_res.reset_index(inplace=True)

                        temp_res.to_csv(f"TickerResults/{ticker}/csv_files/Results_Ticker{ticker}_LP{number_of_optimization_periods}_Recal{recalib_months}_NS{num_strategies}_M{metric}.csv")
                        # except:
                        #     print(f"Could not process for Ticker: {ticker} Number of Optimization Periods: {number_of_optimization_periods}, recalib_months: {recalib_months}, num_strategies: {num_strategies}, metric: {metric}")
        res = []
        for number_of_optimization_periods in [1, 2, 3]:
            for recalib_months in [3, 6, 12]:
                for num_strategies in [1, 3, 5, 7]:
                    for metric in ['rolling_sharpe', 'rolling_sortino', 'rolling_cagr', 'maxdrawup_by_maxdrawdown',
                                   'outperformance']:
                        try:
                            temp_res = pd.read_csv(
                                f"TickerResults/{ticker}/csv_files/Results_Ticker{ticker}_LP{number_of_optimization_periods}_Recal{recalib_months}_NS{num_strategies}_M{metric}.csv",
                                parse_dates=True)
                            temp_res['Date'] = pd.to_datetime(temp_res['Date'])
                            plt.plot(temp_res['Date'], temp_res['Market_Return'], color='black', label='Market Returns')
                            plt.plot(temp_res['Date'], temp_res['Strategy_Return'], color='blue', label='Strategy Returns')
                            plt.title('Strategy Backtest')
                            plt.legend(loc=0)
                            plt.tight_layout()
                            plt.savefig(
                                f"TickerResults/{ticker}/equity_curves/Results_Ticker{ticker}_LP{number_of_optimization_periods}_Recal{recalib_months}_NS{num_strategies}_M{metric}.jpg")
                            plt.clf()
                            res.append({'Ticker': ticker, "Optimization Periods": number_of_optimization_periods,
                                        "Recalibration Months": recalib_months, "Number of Strategies": num_strategies,
                                        "Metric": metric, "Sortino": backtest_sortino(temp_res, 0, 0),
                                        "Sharpe": backtest_sharpe(temp_res, 0, 0),
                                        "Rolling Sortino": backtest_rolling_sortino(temp_res, 0, 0),
                                        "Rolling Sharpe": backtest_rolling_sharpe(temp_res, 0, 0),
                                        "Rollling CAGR": backtest_rolling_cagr(temp_res, 0, 0),
                                        "MaxDrawupByMaxDrawdown": backtest_maxdrawup_by_maxdrawdown(temp_res, 0, 0),
                                        "Outperformance": backtest_outperformance(temp_res, 0, 0)})
                        except:
                            print("Not processed")

        pd.DataFrame(res).to_csv(f"TickerResults/{ticker}/Results_Parametric.csv")

        #Emailer for top3 strategies
        for sortby in ["Outperformance", "Sharpe", "MaxDrawupByMaxDrawdown"]:
            res_sorted = pd.DataFrame(res).sort_values(sortby, ascending=False)
            for i in range(3):
                number_of_optimization_periods = res_sorted.iloc[i]["Optimization Periods"]
                recalib_months = res_sorted.iloc[i]["Recalibration Months"]
                num_strategies = res_sorted.iloc[i]["Number of Strategies"]
                metric = res_sorted.iloc[i]["Metric"]
                temp_res = pd.read_csv(
                    f"TickerResults/{ticker}/csv_files/Results_Ticker{ticker}_LP{number_of_optimization_periods}_Recal{recalib_months}_NS{num_strategies}_M{metric}.csv",
                    parse_dates=True)
                temp_res['Date'] = pd.to_datetime(temp_res['Date'])
                plt.plot(temp_res['Date'], temp_res['Market_Return'], color='black', label='Market Returns')
                plt.plot(temp_res['Date'], temp_res['Strategy_Return'], color='blue', label='Strategy Returns')
                plt.title('Strategy Backtest')
                plt.legend(loc=0)
                plt.tight_layout()
                plt.savefig(
                    f"TickerResults/{ticker}/SortedBy_{sortby}_{(i+1)}_Results_Ticker{ticker}_LP{number_of_optimization_periods}_Recal{recalib_months}_NS{num_strategies}_M{metric}.jpg")
                plt.clf()
            path_mail = f"TickerResults/{ticker}"
            files = os.listdir(path_mail)
            images = []
            for file in files:
                if file.startswith(f"SortedBy_{sortby}") & file.endswith('.jpg'):
                    img_path = path_mail + '/' + file
                    images.append(img_path)
            SendMail(ticker, sortby, images)
