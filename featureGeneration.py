import numpy as np
import time
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import os

def create_substract_features(df,target):
    df["past"]=target.shift(fill_value=0)
    col=df.shape[1]
    created_feature_names=[]
    for c in range(col-1):
        for j in range(c+1,col):
            name="substract"+str(c)
            created_feature_names.append(name)
            df[name]=df.iloc[:,c]-df.iloc[:,j]
    return df,created_feature_names


def create_features(df):
    np.seterr(all="raise")
    row = df.shape[0]
    created_feature_names=[]
    for c in range(df.shape[1]):
        name1 = "mean_minus_row", c
        created_feature_names.append(name1)
        df[name1] = df.iloc[:, c].mean() - df.iloc[:, c]
        name2 = "median_minus_row", c
        created_feature_names.append(name2)
        df[name2] = df.iloc[:, c].median() - df.iloc[:, c]
        name5 = "max_minus_row", c
        created_feature_names.append(name5)
        df[name5] = df.iloc[:, c].max() - df.iloc[:, c]
        name6 = "min_minus_row", c
        created_feature_names.append(name6)
        df[name6] = df.iloc[:, c].min() - df.iloc[:, c]
        name8= "95percentile_minus_instance",c
        created_feature_names.append(name8)
        df[name8]=df.iloc[:,c].quantile(0.95)-df.iloc[:,c]
        name9 = "75percentile_minus_instance", c
        created_feature_names.append(name9)
        df[name9] = df.iloc[:, c].quantile(0.75) - df.iloc[:, c]
        name10 = "25percentile_minus_instance", c
        created_feature_names.append(name10)
        df[name10] = df.iloc[:, c].quantile(0.25) - df.iloc[:, c]
        name7 = "if_95_percentile_1", c
        #
        # if  df.iloc[:, c].quantile(0.95) < df.iloc[:, c]:
        #     df[name7] = 1
        # else:
        #     df[name7] = 0
        try:
            name3 = "logn", c
            created_feature_names.append(name3)
            df[name3] = np.log(df.iloc[:, c])
            name4 = "log10", c
            created_feature_names.append(name4)
            df[name4] = np.log10(df.iloc[:, c])
        except:
            print("features log and logn could not be created for the column number ",c,"\n")


        for r in range (row-1):

            name11= "row_minus_other_row",c
            created_feature_names.append(name11)
            df[name11]=df.iloc[r,c]-df.iloc[r+1,c]

    return df,created_feature_names

def split_timestamp(df,timestamp):
    df[["day", "month", "year"]] = df.iloc[:,timestamp].str.split("-", expand=True)

    created_feature_names = []
    name12 = "first_day_ofmonth"
    created_feature_names.append(name12)
    # if df['day'] == 1.0:
    #     df.loc[name12] = 1
    # else:
    #     df.loc[name12] = 0
    name12 = "first_day_ofmonth"
    df.loc[df['day'].astype(int) == 1, name12] = 1
    df.loc[df['day'].astype(int) > 1, name12] = 0

    name13 = "first_seven_day"
    df.loc[df['day'].astype(int) < 8, name13] = 1
    df.loc[df['day'].astype(int) > 7, name13] = 0

    name14 = "day15"
    df.loc[df['day'].astype(int) == 15, name14] = 1
    df.loc[df['day'].astype(int) != 15, name14] = 0

    return df

def feature_selection(target,feature):
    df=pd.concat([feature, target], axis=1, sort=False)
    #graph corrolations
    fg_hm=plt.figure(figsize=(20, 20))
    hm=fg_hm.add_subplot(111)
    cor = df.corr()
    sns.heatmap(cor, annot=True, cmap=plt.cm.Reds)
    figureName="heatMap"+str(time.time())+".png"
    directory="src/heatmaps/"
    pathName = directory+figureName

    if not os.path.exists(directory):
        os.makedirs(directory)

    plt.savefig(pathName)

    # Correlation with output variable
    cor_target = abs(cor.iloc[:,-1])

    # Selecting highly correlated features
    relevant_features = cor_target[abs(cor_target) >0.2]

    return relevant_features[:-1],figureName