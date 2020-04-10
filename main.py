import outputReport
import featureGeneration
import dataPreperation
from sklearn.linear_model import Lasso
from sklearn.linear_model import LogisticRegression
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
import json
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
import matplotlib as mpl
import webbrowser
from pyearth import Earth
import pandas as pd
import tkinter as tk
from tkinter import filedialog
from tkinter import *
import os
import sarimax

mpl.rcParams['agg.path.chunksize'] = 10000

def pca(feature_arr, feature_num):
    pipeline = Pipeline([('pca', PCA(n_components=feature_num))])
    feature_arr = pipeline.fit_transform(feature_arr)
    return (feature_arr)

def prepare_data(csvfile,target,timestampindex,ispca,feature_num,istest,index):
    # read data
    df = dataPreperation.open_file(csvfile)

    # split timestamp into day, month, year
    df=pd.DataFrame(featureGeneration.split_timestamp(df,timestampindex))


    # split timestamp from dataframe
    timestamp=df.iloc[:,timestampindex]
    df=dataPreperation.delete_col(timestampindex,df)

    # convert dataset into float
    df=pd.DataFrame(dataPreperation.convert_data_float(df))

    # split data into feature and target, store timestamp for graphs
    #target-1 because we delete timestamp
    target=target-1
    feature,target_column=dataPreperation.split_target_feature(df,target)

    proffesional_report="Dataset after encoding:<br>"+str(feature.head(10).to_html(formatters={'Name': lambda x: '<b>' + x + '</b>'}))+"<br>"
    # split data
    # target,feature,timestamp=split_labels_features(target_column,df,timestamp)

    # generate new features
    feature,new_feature_names=featureGeneration.create_substract_features(feature,target_column)
    feature=pd.DataFrame(feature)

    feature,new_feature_names=featureGeneration.create_features(feature)
    feature=pd.DataFrame(feature)
    proffesional_report+="Dataset after new added features:<br>"+str(feature.head(10).to_html(formatters={'Name': lambda x: '<b>' + x + '</b>'}))+"<br>"

    # clean nan,infinity values and columns with only one type of variable
    feature,labels=dataPreperation.clean_dataframe(feature)


    #normalize data
    feature=pd.DataFrame(dataPreperation.normalize_data(feature),columns=labels)
    proffesional_report+="Dataset after normalization:<br>"+str(feature.head(10).to_html(formatters={'Name': lambda x: '<b>' + x + '</b>'}))+"<br>"
    # apply PCA or feature selection (user choice)
    if (ispca == 1):
        selected_features = pd.DataFrame(pca(feature, feature_num))
        # selected_features = feature[feature_names.keys()]

        if (selected_features.shape[1] == 0):
            proffesional_report+="PCA Features:<br>"+"NO FEATURES CREATED, TEST USED THE DATASET WITHOUT APPLYING PCA"+"<br>"
            selected_features=feature

        else:
            proffesional_report+="PCA Features:<br>"+str(selected_features.head(10).to_html(formatters={'Name': lambda x: '<b>' + x + '</b>'}))+"<br>"
    else:
        feature_names, heatMapName = featureGeneration.feature_selection(target_column, feature)
        selected_features = feature[feature_names.keys()]
        if(selected_features.shape[1]==0):
            proffesional_report+="Selected Features:<br>"+"NO FEATURES SELECTED (THRESHOLD VALUE FOR CORRELATION IS 0.5, YOU CAN DECREASE THE VALUE)"+"<br>"
            selected_features=feature

        else:
            proffesional_report+="Selected Features:<br>"+str(selected_features.head(10).to_html(formatters={'Name': lambda x: '<b>' + x + '</b>'}))+"<br>"



    # create training and testing datasets
    x_train,x_test,y_train,y_test,timestamp = dataPreperation.split_test_train(target_column,selected_features,timestamp,index,istest)
    return x_train, x_test, y_train, y_test, timestamp,proffesional_report



def logistic_regression(x_train,x_test,y_train,y_test):

    model = LogisticRegression()

    # predict
    model = model.fit(x_train, y_train)

    y_pred = model.predict(x_test)

    # score
    score = model.score(x_test, y_test)

    y=pd.DataFrame(np.append(np.unique(y_train),np.unique(y_test)))

    AVG_ACC,FP,FN,TP,TN,expanded,pathforHTML=outputReport.classification_results(y_test, y_pred, y, "logistic")

    return AVG_ACC,FP,FN,TP,TN,expanded,pathforHTML


def logistic_forecast(x_train,x_test,y_train,timestamp):
    # set model
    model = LogisticRegression()

    print("2 ", timestamp)
    # predict
    model = model.fit(x_train, y_train)

    y_pred = pd.DataFrame(model.predict(x_test),columns=["Forecasted Values"])

    result=str(y_pred.to_html(formatters={'Name': lambda x: '<b>' + x + '</b>'}))

    filename=outputReport.classification_results_forecast(timestamp, y_pred, "logistic forecast")

    return filename,result

def random_forest(x_train,x_test,y_train,y_test):
    model = RandomForestClassifier(n_estimators=100,
                                   bootstrap=True,
                                   max_features='sqrt')

    model.fit(x_train, y_train)

    y_pred = model.predict(x_test)

    y=pd.DataFrame(np.append(np.unique(y_train),np.unique(y_test)))

    AVG_ACC,FP,FN,TP,TN,expanded,pathforHTML=outputReport.classification_results(y_test, y_pred, y, "randomforest")

    return AVG_ACC,FP,FN,TP,TN,expanded,pathforHTML


def random_forest_forecast(x_train,x_test,y_train,timestamp):
    # set model
    model = RandomForestClassifier(n_estimators=100,
                                   bootstrap=True,
                                   max_features='sqrt')

    # predict
    model.fit(x_train, y_train)

    y_pred = pd.DataFrame(model.predict(x_test),columns=["Forecasted Values"])

    result=str(y_pred.to_html(formatters={'Name': lambda x: '<b>' + x + '</b>'}))

    filename=outputReport.classification_results_forecast(timestamp, y_pred, "random forest forecast")

    return filename,result

def decision_tree(x_train,x_test,y_train,y_test):

    model = DecisionTreeClassifier()

    # predict
    model = model.fit(x_train, y_train)

    y_pred = model.predict(x_test)

    # score
    score = model.score(x_test, y_test)

    y=pd.DataFrame(np.append(np.unique(y_train),np.unique(y_test)))

    AVG_ACC,FP,FN,TP,TN,expanded,pathforHTML=outputReport.classification_results(y_test, y_pred, y, "deicisiontree")

    return AVG_ACC,FP,FN,TP,TN,expanded,pathforHTML


def decision_tree_forecast(x_train,x_test,y_train,timestamp):
    # set model
    model = DecisionTreeClassifier()

    print("2 ", timestamp)
    # predict
    model = model.fit(x_train, y_train)

    y_pred = pd.DataFrame(model.predict(x_test),columns=["Forecasted Values"])

    result=str(y_pred.to_html(formatters={'Name': lambda x: '<b>' + x + '</b>'}))

    filename=outputReport.classification_results_forecast(timestamp, y_pred, "decision tree forecast")

    return filename,result


def linear_regression(x_train,x_test,y_train,y_test,timestamp):
    # set model
    model=LinearRegression()

    # predict
    model = model.fit(x_train, y_train)

    y_pred = model.predict(x_test)

    # score
    # score = model.score(x_test, y_test)
    MSE,MAD,MAPE=outputReport.regression_basic_results(y_test, y_pred)
    fileName,result=outputReport.regression_extanded_results(timestamp, y_test, y_pred, "linear")

    correlation_matrix = np.corrcoef(y_test, y_pred)
    correlation_xy = correlation_matrix[0, 1]
    score = correlation_xy ** 2

    return score,fileName,MSE,MAD,MAPE,result

def linear_regression_forecast(x_train,x_test,y_train,timestamp):
    # set model
    model = LinearRegression()

    # predict
    model = model.fit(x_train, y_train)

    y_pred = pd.DataFrame(model.predict(x_test),columns=["Forecasted Values"])

    result=str(y_pred.to_html(formatters={'Name': lambda x: '<b>' + x + '</b>'}))

    filename=outputReport.regression_extanded_results_forecast(timestamp, y_pred, "linear forecast")

    return filename,result

def mars(x_train,x_test,y_train,y_test,timestamp):
    # set model
    model=Earth(max_degree=1, penalty=1.0, endspan=5)

    # predict
    model = model.fit(x_train, y_train)

    y_pred = model.predict(x_test)

    # score
    # score=model.score(x_test,y_test)

    correlation_matrix = np.corrcoef(y_test, y_pred)
    correlation_xy = correlation_matrix[0, 1]
    score = correlation_xy ** 2

    MSE,MAD,MAPE=outputReport.regression_basic_results(y_test, y_pred)
    fileName,result=outputReport.regression_extanded_results(timestamp, y_test, y_pred, "mars")
    try:
        model_summary=str(model.summary())
        model_summary_final=model_summary.replace("\n", "<br>")
        result += "<br>Model Parameters:<br>"+str(model.get_params()) + "<br>Model Summary:<br>" + model_summary_final
    except:
        result+="<br>Model Summary is not available for MARS"
    return score,fileName,MSE,MAD,MAPE,result


def mars_forecast(x_train, x_test, y_train, timestamp):
    # set model
    model = Earth(max_degree=1, penalty=1.0, endspan=5)

    # predict
    model = model.fit(x_train, y_train)

    y_pred = pd.DataFrame(model.predict(x_test),columns=["Forecasted Values"])

    filename = outputReport.regression_extanded_results_forecast(timestamp, y_pred, "mars forecast")

    try:
        model_summary = str(model.summary())
        model_summary_final = model_summary.replace("\n", "<br>")
        result = "<br>Model Parameters:<br>" + str(model.get_params()) + "<br>Model Summary:<br>" + model_summary_final
    except:
        result = "Model Summary is not available for MARS"

    result += str(y_pred.to_html(formatters={'Name': lambda x: '<b>' + x + '</b>'}))

    return filename,result

def lasso_reg(x_train,x_test,y_train,y_test,timestamp):
    # set model
    model = Lasso()

    # predict
    model = model.fit(x_train, y_train)

    y_pred = model.predict(x_test)

    # score
    # score = model.score(x_test, y_test)
    correlation_matrix = np.corrcoef(y_test, y_pred)
    correlation_xy = correlation_matrix[0, 1]
    score = correlation_xy ** 2

    MSE, MAD, MAPE = outputReport.regression_basic_results(y_test, y_pred)
    fileName, result = outputReport.regression_extanded_results(timestamp, y_test, y_pred, "lasso")

    return score, fileName, MSE, MAD, MAPE, result


def lasso_regression_forecast(x_train,x_test,y_train,timestamp):
    # set model
    model = Lasso()

    # predict
    model = model.fit(x_train, y_train)

    y_pred = pd.DataFrame(model.predict(x_test),columns=["Forecasted Values"])

    result=str(y_pred.to_html(formatters={'Name': lambda x: '<b>' + x + '</b>'}))

    filename=outputReport.regression_extanded_results_forecast(timestamp, y_pred, "lasso forecast")

    return filename,result

def create_data_json(scores,MSE,MAD,MAPE,fileName,expanded,professional_report):
    data = 'data = ' + json.dumps({
        "linear": {
            "accuracy": scores["linear"],
            "MSE": MSE["linear"],
            "MAD": MAD["linear"],
            "MAPE": MAPE["linear"],
            "fileName": fileName["linear"],
            "expanded":expanded["linear"],
            "professional":professional_report
        },
        "lasso": {
            "accuracy": scores["lasso"],
            "MSE": MSE["lasso"],
            "MAD": MAD["lasso"],
            "MAPE": MAPE["lasso"],
            "fileName": fileName["lasso"],
            "expanded": expanded["lasso"],
            "professional":professional_report
        },
        "mars": {
            "accuracy": scores["mars"],
            "MSE": MSE["mars"],
            "MAD": MAD["mars"],
            "MAPE": MAPE["mars"],
            "fileName": fileName["mars"],
            "expanded":expanded["mars"],
            "professional": professional_report
        },
        "sarimax": {
            "accuracy": scores["sarimax"],
            "MSE": MSE["sarimax"],
            "MAD": MAD["sarimax"],
            "MAPE": MAPE["sarimax"],
            "fileName": fileName["sarimax"],
            "expanded":expanded["sarimax"],
            "professional": professional_report
        }
    })
    return data


def create_classification_data_json(scores,FP,FN,TP,TN,fileName,expanded,professional_report):
    data = 'data_classification = ' + json.dumps({
        "decisiontree": {
            "accuracy": scores["decisiontree"],
            "FP": FP["decisiontree"],
            "FN": FN["decisiontree"],
            "TP": TP["decisiontree"],
            "TN": TN["decisiontree"],
            "fileName": fileName["decisiontree"],
            "expanded":expanded["decisiontree"],
            "professional":professional_report
        },
        "logistic": {
            "accuracy": scores["logistic"],
            "FP": FP["logistic"],
            "FN": FN["logistic"],
            "TP": TP["logistic"],
            "TN": TN["logistic"],
            "fileName": fileName["logistic"],
            "expanded":expanded["logistic"],
            "professional":professional_report
        },
        "randomforest": {
            "accuracy": scores["randomforest"],
            "FP": FP["randomforest"],
            "FN": FN["randomforest"],
            "TP": TP["randomforest"],
            "TN": TN["randomforest"],
            "fileName": fileName["randomforest"],
            "expanded":expanded["randomforest"],
            "professional":professional_report
        }
    })
    return data

def main_func(algorithms_reg,algorithms_class,csvfile,targetCol,timestamp,ispca,feature_num,istest,index,isregression):

    x_train,x_test,y_train,y_test,timestamp_test,professional_report=prepare_data(csvfile, targetCol,timestamp,ispca,feature_num,istest,index)

    scores = {"lasso": None, "linear": None, "mars": None, "sarimax": None}
    fileName = {"lasso": None, "linear": None, "mars": None, "sarimax": None}
    MSE = {"lasso": None, "linear": None, "mars": None, "sarimax": None}
    MAD = {"lasso": None, "linear": None, "mars": None, "sarimax": None}
    MAPE = {"lasso": None, "linear": None, "mars": None, "sarimax": None}
    expanded = {"lasso": None, "linear": None, "mars": None, "sarimax": None}

    scores_class = {"decisiontree": None, "randomforest": None, "logistic": None }
    fileName_class = {"decisiontree": None, "randomforest": None, "logistic": None}
    FP = {"decisiontree": None, "randomforest": None, "logistic": None}
    FN= {"decisiontree": None, "randomforest": None, "logistic": None}
    TN= {"decisiontree": None, "randomforest": None, "logistic": None}
    TP= {"decisiontree": None, "randomforest": None, "logistic": None}
    expanded_class = {"decisiontree": None, "randomforest": None, "logistic": None}

    if(isregression):

        if len(algorithms_reg) == 0:
            algorithms_reg = {"Mars Regression", "Linear Regression", "Lasso Regression", "sarimax"}

        for each in algorithms_reg:
            if (istest == 1):
                if each == "Mars Regression":
                    scores["mars"], fileName["mars"], MSE["mars"], MAD["mars"], \
                    MAPE["mars"], expanded["mars"] = mars(x_train, x_test, y_train, y_test, timestamp_test)
                elif each == "Linear Regression":
                    scores["linear"], fileName["linear"], MSE["linear"], MAD["linear"], \
                    MAPE["linear"], expanded["linear"] = linear_regression(x_train, x_test, y_train, y_test,
                                                                           timestamp_test)
                elif each == "Lasso Regression":
                    scores["lasso"], fileName["lasso"], MSE["lasso"], MAD["lasso"], \
                    MAPE["lasso"], expanded["lasso"] = lasso_reg(x_train, x_test, y_train, y_test, timestamp_test)
                else:
                    scores["sarimax"], fileName["sarimax"], MSE["sarimax"], MAD["sarimax"], \
                    MAPE["sarimax"], expanded["sarimax"] = sarimax.sarima(csvfile, targetCol, timestamp, "D", index)

            else:
                if each == "Mars Regression":
                    fileName["mars"], expanded["mars"] = mars_forecast(x_train, x_test, y_train, timestamp_test)
                elif each == "Linear Regression":
                    fileName["linear"], expanded["linear"] = linear_regression_forecast(x_train, x_test, y_train,
                                                                                        timestamp_test)
                elif each == "Lasso Regression":
                    fileName["lasso"], expanded["lasso"] = lasso_regression_forecast(x_train, x_test, y_train,
                                                                                     timestamp_test)
                else:
                    fileName["sarimax"], expanded["sarimax"] = sarimax.sarima_forecast(csvfile, targetCol, timestamp, "D",
                                                                                       index)

    else:

        if len(algorithms_class) == 0:
            algorithms_class = {"Decision Tree", "Random Forest", "Logistic Regression"}

        print("1 ",timestamp)

        for each in algorithms_class:
            if (istest == 1):
                if each == "Decision Tree":
                    scores_class["decisiontree"],FP["decisiontree"],FN["decisiontree"],TP["decisiontree"],TN["decisiontree"]\
                        ,expanded_class["decisiontree"],fileName_class["decisiontree"]\
                        = decision_tree(x_train, x_test, y_train, y_test)
                elif each == "Logistic Regression":
                    scores_class["logistic"],FP["logistic"],FN["logistic"],TP["logistic"],TN["logistic"]\
                        ,expanded_class["logistic"],fileName_class["logistic"]\
                        = logistic_regression(x_train, x_test, y_train, y_test)
                else:
                    scores_class["randomforest"],FP["randomforest"],FN["randomforest"],TP["randomforest"],TN["randomforest"]\
                        ,expanded_class["randomforest"],fileName_class["randomforest"]\
                        = random_forest(x_train, x_test, y_train, y_test)
            else:
                if each == "Decision Tree":
                    fileName_class["decisiontree"], expanded_class["decisiontree"] = decision_tree_forecast(x_train,x_test,y_train,timestamp_test)
                elif each=="Logistic Regression":
                    fileName_class["logistic"], expanded_class["logistic"] = logistic_forecast(x_train,x_test,y_train,timestamp_test)
                else:
                    fileName_class["randomforest"], expanded_class["randomforest"] = random_forest_forecast(x_train,x_test,y_train,timestamp_test)



    data_class = create_classification_data_json(scores_class,FP,FN,TP,TN,fileName_class,expanded_class,professional_report)
    data = create_data_json(scores, MSE, MAD, MAPE, fileName, expanded, professional_report)

    f = open("src/data_classification.js", "w")
    f.write(data_class)
    f.close()

    f = open("src/data.js", "w")
    f.write(data)
    f.close()




    webbrowser.open("file://"+os.path.realpath('src/output_report.html'))


class Interface():
    def __init__(self):

        self.csv=""
        self.root = tk.Tk()
        self.root.geometry('600x450')
        self.istest = tk.IntVar()
        self.ispcavar = tk.IntVar()
        self.isregressionvar = tk.IntVar()


        self.filelabel = tk.Label(self.root, text="No file chosen!!!")
        self.filebutton = tk.Button(self.root,text="CHOOSE CSV FILE",command=self.csv_file)
        self.targetlabel=tk.Label(self.root,text="Enter the target column:")
        self.target=tk.Entry(self.root)
        self.timestamplabel=tk.Label(self.root,text="Enter the timestamp column:")
        self.timestamp=tk.Entry(self.root)
        self.pca=tk.Radiobutton(self.root,text="Apply PCA",variable=self.ispcavar,value=1)
        self.featureselection=tk.Radiobutton(self.root,text="Apply Feature Selection",variable=self.ispcavar,value=0)
        self.featurenumberlabel=tk.Label(self.root,text="Enter total feature numbers after pca(0 for no pca):")
        self.featurenumber=tk.Entry(self.root)
        self.indexlabel=tk.Label(self.root,text="Enter the index to start forecasting (ex:17)(for test algorithms enter percentage of training set (ex:80))")
        self.index=tk.Entry(self.root)
        self.run=tk.Button(self.root,text="Run Algorithms",command=self.main)
        self.testing = tk.Radiobutton(self.root, text='Test Algorithms', variable=self.istest, value=1)
        self.forecast = tk.Radiobutton(self.root, text='Forecast', variable=self.istest, value=0)
        self.labelclassificationlistbox=tk.Label(self.root,text="Choose classification algorithms:")
        self.listboxclassification = tk.Listbox(self.root, height=5, selectmode='multiple')
        self.labelregressionlistbox=tk.Label(self.root,text="Choose regression algorithms:")
        self.listboxregression = tk.Listbox(self.root, height=5, selectmode='multiple')
        self.isregressionbox=tk.Checkbutton(self.root,text="Conduct Regression",variable=self.isregressionvar)

        dataReg = ("Linear Regression", "Mars Regression", "Lasso Regression","Sarimax")

        for val in dataReg:
            self.listboxregression.insert(END, val)

        dataClas = ("Decision Tree", "Logistic Regression", "Random Forest")

        for val in dataClas:
            self.listboxclassification.insert(END, val)

        self.root.title('Time Series Demand Analysis')

        self.filelabel.place(x=20, y=20)
        self.filebutton.place(x=20, y=40)
        self.targetlabel.place(x=20,y=80)
        self.target.place(x=20,y=100)
        self.timestamplabel.place(x=20,y=140)
        self.timestamp.place(x=20,y=160)
        self.pca.place(x=16,y=200)
        self.featureselection.place(x=100,y=200)
        self.featurenumberlabel.place(x=20,y=250)
        self.featurenumber.place(x=20,y=270)
        self.testing.place(x=20,y=300)
        self.forecast.place(x=170,y=300)
        self.indexlabel.place(x=20,y=340)
        self.index.place(x=20,y=360)
        self.run.place(x=20,y=410)
        self.labelclassificationlistbox.place(x=400, y=20)
        self.listboxclassification.place(x=400, y=50)
        self.labelregressionlistbox.place(x=400, y=180)
        self.listboxregression.place(x=400, y=210)
        self.isregressionbox.place(x=400, y=310)
        self.root.mainloop()

    def csv_file(self):
        csvfile = filedialog.askopenfilename()
        if csvfile == "":
            self.filelabel['text'] = "No file chosen!!!"
            self.csv=""
        else:
            txt = "csv file: ", csvfile

            self.filelabel['text'] = txt
            self.csv=csvfile

    def main(self):
        values_regression = [self.listboxregression.get(idx) for idx in self.listboxregression.curselection()]
        values_class = [self.listboxclassification.get(idx) for idx in self.listboxclassification.curselection()]
        targetCol=self.target.get()
        timestampCol=self.timestamp.get()
        featureNumValue=self.featurenumber.get()
        if self.csv!="" and targetCol!="":
            main_func(values_regression,values_class,self.csv,int(targetCol),int(timestampCol),int(self.ispcavar.get()),int(featureNumValue),int(self.istest.get()),
                      int(self.index.get()),int(self.isregressionvar.get()))


app=Interface()
