from sklearn.cluster import KMeans
import pandas as pd
import matplotlib.pyplot as plt

data=pd.read_csv('src/data.csv',sep=',')


corr=data.corr()
#print(corr)

a=pd.DataFrame(pd.read_csv('src/g2-2-20.txt'))
print(type(a))

pd.DataFrame.plot(a)

plt.plot([[1,1],[2,3]])
plt.savefig('img.png')