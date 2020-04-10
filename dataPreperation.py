import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn import preprocessing
import pandas as pd

def open_file(csv_file):
    df=pd.read_csv(csv_file, delimiter=",")
    return (df)

def delete_col(col_idx, df):
    return df.drop(df.columns[[col_idx]], axis=1)


def delete_row(row_idx, df):
    return df.drop(df.rows[[row_idx]], axis=0)

# ENCODE
def encode_columns(df, col_indexes):
    lab_enc = preprocessing.LabelEncoder()
    for i in col_indexes:
        df.iloc[:, i] = lab_enc.fit_transform(df.iloc[:, i])
    return (df)


def split_test_train(target,feature,timestamp,index,istest):
    if(istest==1):
        size=int((index/100)*len(feature))
        y_train=target.iloc[:size]
        x_train=feature[:size]
        y_test=target.iloc[size:]
        x_test=feature[size:]
        timestamp=timestamp[size:]

    else:
        y_train=target.iloc[:index]
        x_train=feature[:index]
        y_test=target.iloc[index:]
        x_test=feature[index:]
        timestamp=timestamp[index:]


    return x_train,x_test,y_train,y_test,timestamp

# NORMALIZATION
def normalize_data(df):
    pipeline = Pipeline([('scaling', StandardScaler())])
    df = pipeline.fit_transform(df)
    return (df)

def convert_data_float(df):
    idx = []  # indexes for columns with string variables
    for i in range(df.shape[1]):
        try:
            df.iloc[:, i] = df.iloc[:, i].astype(float)
        except:
            idx.append(i)

    # encode string variables
    df = encode_columns(df, idx)

    # convert full df into float
    df = df.astype(float)
    return df

def clean_dataframe(df):
    # np where first element=row , second element = column
    # find infinity variables
    inf_place = np.where(np.isinf(df))

    # delete columns with infinity variables
    df.drop(df.columns[inf_place[1]], axis="columns", inplace=True)

    # find nan variables
    nan_place = np.where(np.isnan(df))

    # delete features with nan variables
    df.drop(df.columns[nan_place[1]], axis="columns", inplace=True)

    # find features with only 1 variable
    indices = []
    for i in range(df.shape[1]):
        if np.unique(df.iloc[:, i]).size <= 1:
            indices.append(i)

    # delete features with only 1 variable
    df = df.drop(df.columns[[indices]], axis=1)
    return df,df.columns


def split_target_feature(df,target):
    target_col = df.iloc[:, target]
    feature=delete_col(target,df)
    return feature,target_col