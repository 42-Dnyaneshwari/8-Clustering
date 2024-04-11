# -*- coding: utf-8 -*-
"""
Created on Fri Oct 13 08:26:37 2023

@author: Dnyaneshwari
"""


#Agglomerative Clustering
import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt
#Import file and create the dataframe
Univ1 = pd.read_excel('University_Clustering.xlsx')
a = Univ1.describe()

#We have one column "State" which  really and create a dataframe 
Univ = Univ1.drop(["State"],axis=1)

#we know that there is scale difference among the columns, which we have 
#either by using normalization or standardization 
#whenever there is mixed data apply normalization 
def norm_func(i):
    x=(i-i.min())/(i.max()-i.min())
    return x 
#Now apply this normalization function to Univ dataframe for all the row and columns from 1 till end
#since 0th column has university name hence skipped 
df_norm = norm_func(Univ.iloc[:,1:])
#you can check the df_norm dataframe which is scaled between values from 0 to 1 
#You can apply describe function to new dataframe 

b = df_norm.describe()

#Before you apply clustering, you  need to plot  dendrogram furst 
#now to create dendrogram, we need to measure distance, 
#We have to import linkage 
from scipy.cluster.hierarchy import linkage 
import scipy.cluster.hierarchy as sch 
#linkage function gives us hierarchical or agglomerative clustering 
#ref the help for linkage 
z=linkage(df_norm, method='complete',metric='euclidean') 
plt.figure(figsize=(15,8))
plt.title('Hierarchical Clustering Dendrogram')
plt.xlabel('Index')
plt.ylabel('Distance')

#ref help of dendrogram 
#sch.dendrogram(z)
sch.dendrogram(z,leaf_rotation=0,leaf_font_size=10)
plt.show()

#dendrogram()
#applying agglomerative clustering choosing 3 as clusters 
#from dendrogram 
#whatever has been displayed in dendrogram is not clustering 
#It is just showing number of possible clusters 
from sklearn.cluster import AgglomerativeClustering 
h_complete = AgglomerativeClustering(n_clusters=3,linkage='complete',affinity='euclidean').fit(df_norm)

#apply labels to the clusters 
h_complete.labels_
cluster_labels = pd.Series(h_complete.labels_)
#This series to Univ Datafram as column and name the column as 'cluster'
#Assign this series to Univ Dataframe as column and na;me the  
Univ['Clust'] = cluster_labels 
#we want to relocate the column 7 to 0th position 
Univ1 = Univ.iloc[:,[7,1,2,3,4,5,6]]
#now check the Univ1 datafraame 
Univ1.iloc[:,2:].groupby(Univ1.Clust).mean()
#from the output cluster 2 has got highest Top10 
#lowest accept ratio, best faculty ratio and highest expenses 
#highest graduate ration 
Univ1.to_csv('D:/ANACONDA/Data Science/6-Machine Learning/University.csv',encoding='UTF-8')

import os 
os.getcwd()


#====================================Assignment=====================================================

import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt
#Import file and create the dataframe
Auto1 = pd.read_csv('AutoInsurance.csv')
a = Auto1.describe()
Auto1.columns
#We have one column "State" which  really and create a dataframe 
Auto = Auto1.loc[:,['Customer Lifetime Value','Income','Monthly Premium Auto','Months Since Last Claim', 'Months Since Policy Inception',
'Number of Open Complaints', 'Number of Policies','Total Claim Amount']]

#we know that there is scale difference among the columns, which we have 
#either by using normalization or standardization 
#whenever there is mixed data apply normalization 
def norm_func(i):
    x=(i-i.min())/(i.max()-i.min())
    return x 
#Now apply this normalization function to Univ dataframe for all the row and columns from 1 till end
#since 0th column has university name hence skipped 
df_norm = norm_func(Auto.iloc[:,:])
#you can check the df_norm dataframe which is scaled between values from 0 to 1 
#You can apply describe function to new dataframe 

b = df_norm.describe()

#Before you apply clustering, you  need to plot  dendrogram furst 
#now to create dendrogram, we need to measure distance, 
#We have to import linkage 
from scipy.cluster.hierarchy import linkage 
import scipy.cluster.hierarchy as sch 
#linkage function gives us hierarchical or agglomerative clustering 
#ref the help for linkage 
z=linkage(df_norm, method='complete',metric='euclidean') 
plt.figure(figsize=(15,8))
plt.title('Hierarchical Clustering Dendrogram')
plt.xlabel('Index')
plt.ylabel('Distance')

#ref help of dendrogram 
#sch.dendrogram(z)
sch.dendrogram(z,leaf_rotation=0,leaf_font_size=10)
plt.show()

#dendrogram()
#applying agglomerative clustering choosing 3 as clusters 
#from dendrogram 
#whatever has been displayed in dendrogram is not clustering 
#It is just showing number of possible clusters 
from sklearn.cluster import AgglomerativeClustering 
h_complete = AgglomerativeClustering(n_clusters=3,linkage='complete',affinity='euclidean').fit(df_norm)

#apply labels to the clusters 
h_complete.labels_
cluster_labels = pd.Series(h_complete.labels_)
#This series to Univ Datafram as column and name the column as 'cluster'
#Assign this series to Univ Dataframe as column and na;me the  
Univ['Clust'] = cluster_labels 
#we want to relocate the column 7 to 0th position 
Univ1 = Univ.iloc[:,[7,1,2,3,4,5,6]]
#now check the Univ1 datafraame 
Univ1.iloc[:,2:].groupby(Univ1.Clust).mean()
#from the output cluster 2 has got highest Top10 
#lowest accept ratio, best faculty ratio and highest expenses 
#highest graduate ration 
Univ1.to_csv('D:/ANACONDA/Data Science/6-Machine Learning/University.csv',encoding='UTF-8')

import os 
os.getcwd()












