import sys
import outputReport
import warnings
import itertools
import matplotlib.pyplot as plt
import numpy as np
warnings.filterwarnings("ignore")
plt.style.use('fivethirtyeight')
import pandas as pd
import statsmodels.api as sm
import matplotlib
import math
matplotlib.rcParams['axes.labelsize'] = 14
matplotlib.rcParams['xtick.labelsize'] = 12
matplotlib.rcParams['ytick.labelsize'] = 12
matplotlib.rcParams['text.color'] = 'G'

def sarima(name,target,date,forecast_type,train_percentage):
    df = pd.read_csv(name, ",")
    df.iloc[:, target] = df.iloc[:, target].astype(float)
    y = df.set_index(df.iloc[:, date])
    y.index = pd.DatetimeIndex(y.index).to_period(forecast_type)
    train_date_index=math.floor(train_percentage*len(df)/100)


    p = d = q = range(0, 2)
    pdq = list(itertools.product(p, d, q))
    seasonal_pdq = [(x[0], x[1], x[2], 12) for x in list(itertools.product(p, d, q))]
    result = '<br>Examples of parameter for SARIMA...'
    print('Examples of parameter for SARIMA...')
    print('SARIMAX: {} x {}'.format(pdq[1], seasonal_pdq[1]))
    result += '<br>SARIMAX: {} x {}'.format(pdq[1], seasonal_pdq[1])
    print('SARIMAX: {} x {}'.format(pdq[1], seasonal_pdq[2]))
    result += '<br>SARIMAX: {} x {}'.format(pdq[1], seasonal_pdq[2])
    print('SARIMAX: {} x {}'.format(pdq[2], seasonal_pdq[3]))
    result += '<br>SARIMAX: {} x {}'.format(pdq[2], seasonal_pdq[3])
    print('SARIMAX: {} x {}'.format(pdq[2], seasonal_pdq[4]))
    result += '<br>SARIMAX: {} x {}'.format(pdq[2], seasonal_pdq[4])
    aic = 100000000000
    best_order = [-5, -5, -5]
    best_sorder = [-5, -5, -5, -5]

    result += "<br>AIC RESULTS FOR DIFFERENT SARIMAX PARAMETERS"
    for param in pdq:
        for param_seasonal in seasonal_pdq:
            try:
                mod = sm.tsa.statespace.SARIMAX(y.iloc[:train_date_index, target], order=param,
                                                seasonal_order=param_seasonal)
                results = mod.fit(disp=False)
                print('ARIMA{}x{}12 - AIC:{}'.format(param, param_seasonal, results.aic))
                result += '<br>ARIMA{}x{}12 - AIC:{}'.format(param, param_seasonal, results.aic)
                if (results.aic <= aic):
                    aic = results.aic
                    best_order = param
                    best_sorder = param_seasonal
            except:
                print(sys.exc_info()[0])
                continue
    print(aic, " ", best_order, " ", best_sorder)
    # bestorder= " ".join(str(v) for v in best_order)
    # bestsorder= " ".join(str(v) for v in best_sorder)
    #
    # result+="<BR>BEST CHOOSEN PARAMETERS<BR>",str(aic),bestorder,bestsorder

    mod = sm.tsa.statespace.SARIMAX(y.iloc[train_date_index:, target],
                                    order=best_order,
                                    seasonal_order=best_sorder,
                                    enforce_stationarity=False,
                                    enforce_invertibility=False)
    results = mod.fit(disp=False)
    pred = results.get_prediction(start=pd.to_datetime(y.iloc[train_date_index, date]),end=pd.to_datetime(y.iloc[-1, date]), dynamic=False)
    prediction = pd.DataFrame(pred.predicted_mean)
    result += str(prediction.to_html(formatters={'Name': lambda x: '<b>' + x + '</b>'}))

    pathName,result_output=outputReport.regression_extanded_results(y.iloc[train_date_index:,date],y.iloc[train_date_index:,target],prediction,"sarimax")
    result+=result_output
    MSE,MAD,MAPE=outputReport.regression_basic_results(y.iloc[train_date_index:,target], pred.predicted_mean)
    score=find_score(y.iloc[train_date_index:,target],pred.predicted_mean)
    return(score,pathName,MSE,MAD,MAPE,result)

def find_score(y_test,y_pred):
    # correct=0
    # for i in range(len(y_test)):
    #     if(abs(y_test.iloc[i]-y_pred.iloc[i])<=0.1*y_pred.iloc[i] ):
    #         correct+=1
    # return (correct/len(y_test))
    correlation_matrix = np.corrcoef(y_test, y_pred)
    correlation_xy = correlation_matrix[0, 1]
    r_squared = correlation_xy ** 2
    return r_squared



def sarima_forecast(name,target,date,forecast_type,train_date_index):
    df = pd.read_csv(name,",")
    df.iloc[:,target] = df.iloc[:,target].astype(float)
    y = df.set_index(df.iloc[:,date])
    y.index = pd.DatetimeIndex(y.index).to_period(forecast_type)


    p = d = q = range(0, 2)
    pdq = list(itertools.product(p, d, q))
    seasonal_pdq = [(x[0], x[1], x[2], 12) for x in list(itertools.product(p, d, q))]
    result='<br>Examples of parameter for SARIMA...'
    print('Examples of parameter for SARIMA...')
    print('SARIMAX: {} x {}'.format(pdq[1], seasonal_pdq[1]))
    result+='<br>SARIMAX: {} x {}'.format(pdq[1], seasonal_pdq[1])
    print('SARIMAX: {} x {}'.format(pdq[1], seasonal_pdq[2]))
    result+='<br>SARIMAX: {} x {}'.format(pdq[1], seasonal_pdq[2])
    print('SARIMAX: {} x {}'.format(pdq[2], seasonal_pdq[3]))
    result+='<br>SARIMAX: {} x {}'.format(pdq[2], seasonal_pdq[3])
    print('SARIMAX: {} x {}'.format(pdq[2], seasonal_pdq[4]))
    result+='<br>SARIMAX: {} x {}'.format(pdq[2], seasonal_pdq[4])
    aic=100000000000
    best_order=[-5,-5,-5]
    best_sorder=[-5,-5,-5,-5]

    result+="<br>AIC RESULTS FOR DIFFERENT SARIMAX PARAMETERS"
    for param in pdq:
        for param_seasonal in seasonal_pdq:
            try:
                mod = sm.tsa.statespace.SARIMAX(y.iloc[:train_date_index,target],order=param,seasonal_order=param_seasonal)
                results = mod.fit(disp=False)
                print('ARIMA{}x{}12 - AIC:{}'.format(param,param_seasonal,results.aic))
                result+='<br>ARIMA{}x{}12 - AIC:{}'.format(param,param_seasonal,results.aic)
                if(results.aic<=aic):
                    aic=results.aic
                    best_order=param
                    best_sorder=param_seasonal
            except:
                print(sys.exc_info()[0])
                continue
    print(aic," ",best_order," ",best_sorder)
    # bestorder= " ".join(str(v) for v in best_order)
    # bestsorder= " ".join(str(v) for v in best_sorder)
    #
    # result+="<BR>BEST CHOOSEN PARAMETERS<BR>",str(aic),bestorder,bestsorder

    mod = sm.tsa.statespace.SARIMAX(y.iloc[train_date_index:,target],
                                    order=best_order,
                                    seasonal_order=best_sorder,
                                    enforce_stationarity=False,
                                    enforce_invertibility=False)
    results = mod.fit(disp=False)
    pred = results.get_prediction(start=pd.to_datetime(y.iloc[train_date_index,date]), dynamic=False)
    prediction=pd.DataFrame(pred.predicted_mean)
    result+=str(prediction.to_html(formatters={'Name': lambda x: '<b>' + x + '</b>'}))

    pathName=outputReport.regression_extanded_results_forecast(y.iloc[train_date_index:,date], pred.predicted_mean, "sarimax")

    return(pathName,result)

