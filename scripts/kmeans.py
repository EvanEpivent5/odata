from sklearn.cluster import KMeans
import pandas as pd
import matplotlib.pyplot as plt

data=pd.read_csv('src/data.csv',sep=',')


corr=data.corr()
#print(corr)

a=pd.read_csv('src/g2-2-20.txt',sep='     ')

print(a)



plt.savefig('img.png')