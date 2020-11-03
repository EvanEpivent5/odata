import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn.preprocessing as prepro
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
import sklearn.metrics
import sklearn.cluster

data_init = pd.read_csv('hotels.csv', sep=',')
nompaysetoile=data_init.iloc[:,0:3]
data = data_init.drop(['NOM','PAYS'], axis=1) 
nomcolonnesrestantes=data.iloc[0,:]
#print(nomcolonnesrestantes)

corr=data.corr()
#print(corr)
#pd.plotting.scatter_matrix(data)

datanp=data.to_numpy()
#print(datanp)

scaler=prepro.StandardScaler()
scaler.fit(datanp)
datanp=scaler.transform(datanp)

#print(datanp)
#print(np.mean(datanp))

l=linkage(datanp,method='complete',metric='euclidean')
print(l)
dendrogram(l)

clusters=fcluster(l,4,criterion='distance')
print(clusters)

#sklearn.metrics.silhouette_score(l,clusters)

#--------------------------------------------------------------
kmeans1=sklearn.cluster.KMeans(n_clusters=8,init="k-means++",n_init=10)
kmeans1.fit(datanp)
kmeans1.predict(datanp)
resultkmeans1=kmeans1.transform(datanp)

kmeans2=sklearn.cluster.KMeans(n_clusters=8,init="k-means++",n_init=10)
kmeans2.fit(datanp)
kmeans2.predict(datanp)
resultkmeans2=kmeans2.transform(datanp)

#print(sklearn.metrics.adjusted_rand_score(kmeans1.labels_,kmeans2.labels_))
#print(resultkmeans)
