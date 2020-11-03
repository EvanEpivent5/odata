import sklearn.preprocessing as prepro
import pandas as pd


def preprocessingfunc(data):
    pays=data.iloc[1:,0]
    data=data.drop(['country'],axis=1)
    scaler=prepro.StandardScaler()
    scaler.fit(data)
    data=scaler.transform(data)
    return(data,pays)