# -*- coding: utf-8 -*-
"""
Created on Fri Oct 17 08:27:16 2023

@author: Dnyaneshwari...
"""
#Clustering
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

#K-Means for 2D data
X=np.random.uniform(0,1,50)
Y=np.random.uniform(0,1,50)

#create a empty dataframe with 0 column 0 rows

df_xy=pd.DataFrame(columns=['X','Y'])
df_xy
df_xy.X=X
df_xy.Y=Y

df_xy.plot(x='X',y='Y',kind='scatter')
model1=KMeans(n_clusters=3).fit(df_xy)

'''
with this data x,y apply k means model and genrate scatter plot
with scale/font=10
cmap=plt.cm.coolwarm:cool colour combination

'''

model1.labels_
df_xy.plot(x='X',y='Y',c=model1.labels_,kind='scatter',
           s=10,cmap=plt.cm.coolwarm)

####################################################################

univ1=pd.read_excel("C:/Datasets/University_Clustering1.xlsx")

univ1.describe()
univ1.columns
univ=univ1.drop({'State'},axis=1)


def norm_fun(i):
    x=(i-i.min()) / (i.max()-i.min())
    return x

df_norm=norm_fun(univ.iloc[:,1:])
#what will be the cluster number,will it be 1,2,3,4....

TWSS=[]
k=list(range(2,8))
for i in k:
    kmeans=KMeans(n_clusters=1)
    kmeans.fit(df_norm)
    
    TWSS.append(kmeans.inertia_)
    
    
'''
kmeans inertia is also known as sum of squares errors or SSE,
calculates the sum of distances of all points within  a cluster
from the centroid of the point. it is  the diffrence between 
the observed value and the predicted value.

'''

TWSS

plt.plot(k,TWSS,'ro-');
plt.xlabel("No of Clusters");
plt.ylabel("Total_within_SS")

'''
How to select value of k from elbow curve
when k changes from 2,3 then decrease
in TWSS is higher
when k changes from 3,4
then TWSS is decreasing slowly and from
4,5 is conseiderablely less hence take k=3
'''

model=KMeans(n_clusters=3)
model.fit(df_norm)
model.labels_
mb=pd.Series(model.labels_)
univ['clust']=mb
univ.head()
univ=univ.iloc[:,[7,0,1,2,3,4,5,6]]
univ
univ.iloc[:,2:8].groupby(univ.clust).mean()
univ.to_csv("C:/Datasets/University_Clustering1.xlsx",encoding='utf-8')
import os
os.getcwd()


