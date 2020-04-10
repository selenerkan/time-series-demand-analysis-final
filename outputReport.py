import numpy as np
import sklearn.metrics as metrics
from statsmodels import robust
import time
import matplotlib.pyplot as plt
import math
from sklearn.metrics import confusion_matrix,auc,roc_curve,roc_auc_score
from sklearn.preprocessing import LabelBinarizer
import os
import pandas as pd
def mean_absolute_percentage_error(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

def regression_basic_results(y_true, y_pred):
    # Regression metrics
    MSE = metrics.mean_squared_error(y_true, y_pred)
    MAD=robust.mad(y_pred)
    if(len(np.where( y_true == 0))>0):
        MAPE="zero term occured in target"
    else:
        MAPE=round(mean_absolute_percentage_error(y_true, y_pred),4)
    return round(MSE, 4),round(MAD,4),MAPE


def regression_extanded_results(timestamp,y_true, y_pred,model):
    fgr = plt.figure()
    ax = fgr.add_subplot(111)
    plt.xlabel("Time")
    plt.ylabel("Forecast")
    ax.plot(timestamp, y_true, "r",label="Actual Data")
    ax.plot(timestamp, y_pred, "b",label="Forecast")
    if(len(timestamp)>10):
        xlabelindexes=np.arange(0,len(timestamp.index),math.floor(len(timestamp)/10))
        plt.xticks(timestamp.iloc[xlabelindexes],rotation=30)
    else:
        plt.xticks(timestamp,rotation=30)

    ax.legend()

    fileName=model+str(time.time())+".png"
    pathforHTML="pictures/"+fileName
    directory="src/pictures/"
    pathName = directory+fileName

    if not os.path.exists(directory):
        os.makedirs(directory)

    plt.savefig(pathName)

    MAE = metrics.mean_absolute_error(y_true, y_pred)
    MSE = metrics.mean_squared_error(y_true, y_pred)
    RMSE = round(np.sqrt(MSE), 4)

    explained_variance = metrics.explained_variance_score(y_true, y_pred)

    if(len(np.where( y_true == 0))>0):
        mean_squared_log_error="zero term occured in target"
    else:
        mean_squared_log_error = round(metrics.mean_squared_log_error(y_true, y_pred),4)


    result = "Explained Variance: "+ str(round(explained_variance, 4))+"<br>Mean Squared Log Error: "+ str(mean_squared_log_error)+ "<br>MAE: "+str(round(MAE, 4),)+"<br>RMSE: "+str(RMSE)

    values=pd.DataFrame({"TRUE VALUES":y_true}).reset_index()
    values["FORECASTED VALUES"]=y_pred

    result+="<br>OUTPUTS<br>"+str(values.head(10).to_html(formatters={'Name': lambda x: '<b>' + x + '</b>'}))+"<br>"
    return pathforHTML, result

def regression_extanded_results_forecast(timestamp, y_pred,model):
    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    plt.xlabel("Time")
    plt.ylabel("Forecast")
    ax1.plot(timestamp, y_pred, "b",label="Forecast")
    if(len(timestamp)>10):
        xlabelindexes=np.arange(0,len(timestamp.index),math.floor(len(timestamp)/10))
        plt.xticks(timestamp.iloc[xlabelindexes],rotation=30)
    else:
        plt.xticks(timestamp,rotation=30)
    ax1.legend()

    fileName=model+str(time.time())+".png"
    pathforHTML="pictures/"+fileName
    directory="src/pictures/"
    pathName = directory+fileName

    if not os.path.exists(directory):
        os.makedirs(directory)

    plt.savefig(pathName)
    return pathforHTML


def classification_results(y_true, y_pred,y,model):

    cnf_matrix = confusion_matrix(y_true, y_pred)

    expanded="<br>CONFUSION MATRIX<br>"+str(cnf_matrix).replace("\n","<br>")+"<br>"

    FP = cnf_matrix.sum(axis=0) - np.diag(cnf_matrix)
    FN = cnf_matrix.sum(axis=1) - np.diag(cnf_matrix)
    TP = np.diag(cnf_matrix)
    TN = cnf_matrix.sum() - (FP + FN + TP)
    n_class = len(TP)

    FP = sum(FP.astype(float))/n_class
    FN = sum(FN.astype(float))/n_class
    TP = sum(TP.astype(float))/n_class
    TN = sum(TN.astype(float))/n_class


    # Sensitivity, hit rate, recall, or true positive rate
    AVG_TPR = TP / (TP + FN)
    # Specificity or true negative rate
    AVG_TNR = TN / (TN + FP)
    # Precision or positive predictive value
    AVG_PPV = TP / (TP + FP)
    # Negative predictive value
    AVG_NPV = TN / (TN + FN)
    # Fall out or false positive rate
    AVG_FPR = FP / (FP + TN)
    # False negative rate
    AVG_FNR = FN / (TP + FN)
    # False discovery rate
    AVG_FDR = FP / (TP + FP)
    # Overall accuracy
    AVG_ACC = (TP + TN) / (TP + FP + FN + TN)

    expanded += "<br>AVERAGE RATES"+"<br>AVERAGE TRUE POSITIVE RATE "+str(AVG_TPR)+"<br>AVERAGE TRUE NEGATIVE RATE "\
             + str(AVG_TNR)+"<br>AVERAGE POSITIVE PREDICTED VALUE "+str(AVG_PPV)+"<br>AVERAGE NEGATIVE PREDICTED VALUE "\
             + str(AVG_NPV)+"<br>AVERAGE FALSE POSITIVE RATE "+str(AVG_FPR)+"<br>AVERAGE FALSE NEGATIVE RATE "+\
              str(AVG_FNR)+"<br>AVERAGE FALSE DISCOVERY RATE "+str(AVG_FDR)+"<br>"

    lb=LabelBinarizer()
    lb.fit(y)

    y_true_bin=lb.transform(y_true)
    y_pred_bin=lb.transform(y_pred)

    classification_fgr=plt.figure()
    class_ax=classification_fgr.add_subplot(111)
    class_ax.plot([0, 1], [0, 1], color='navy',  linestyle='--')

    fpr=dict()
    tpr=dict()
    thresholds=dict()
    auc_score=dict()
    for i in range(y_true_bin.shape[1]):
        fpr[i], tpr[i], thresholds[i] = roc_curve(y_true_bin[:,i], y_pred_bin[:,i])
        auc_score[i]=auc(fpr[i], tpr[i])
        class_ax.plot(fpr[i], tpr[i], label='ROC curve (area = %0.2f)' % auc_score[i]+" for class: "+str(i))

    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic (ROC)')
    class_ax.legend(loc="lower right")

    fileName=model+str(time.time())+"ROC.png"
    pathforHTML="pictures/"+fileName
    directory="src/pictures/"
    pathName = directory+fileName

    if not os.path.exists(directory):
        os.makedirs(directory)

    plt.savefig(pathName)


    values=pd.DataFrame({"TRUE VALUES":y_true}).reset_index()
    values["FORECASTED VALUES"]=y_pred
    expanded += "<br>OUTPUTS<br>" + str(values.head(10).to_html(formatters={'Name': lambda x: '<b>' + x + '</b>'})) + "<br>"

    return AVG_ACC,FP,FN,TP,TN,expanded,pathforHTML



def classification_results_forecast(timestamp,y_pred,model):
    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    plt.xlabel("Time")
    plt.ylabel("Forecast")
    plt.scatter(timestamp, y_pred, c="b",label="Forecast")
    if(len(timestamp)>10):
        xlabelindexes=np.arange(0,len(timestamp.index),math.floor(len(timestamp)/10))
        plt.xticks(timestamp.iloc[xlabelindexes],rotation=30)
    else:
        plt.xticks(timestamp, rotation=30)

    ax1.legend()

    fileName=model+str(time.time())+".png"
    pathforHTML="pictures/"+fileName
    directory="src/pictures/"
    pathName = directory+fileName

    if not os.path.exists(directory):
        os.makedirs(directory)

    plt.savefig(pathName)

    return pathforHTML














