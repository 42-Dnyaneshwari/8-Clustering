# -*- coding: utf-8 -*-
"""
Created on Fri Oct 13 22:42:22 2023

@author: 
"""

'''
4.	Perform clustering on mixed data. 
Convert the categorical variables to numeric by using dummies or 
label encoding and perform normalization techniques. The data set 
consists of details of customers related to their auto insurance. 
Refer to Autoinsurance.csv dataset.
'''
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


df=pd.read_csv("c:\Datasets\AutoInsurance.csv")
df
df.describe()
df.columns

#################################################################

#1.Business objectives
#To perform clustering for Airlines.
#To draw the inference.

#2.Business Constraints


##################################################################

#2. Work on each feature of the dataset to create a data dictionary 
#as displayed in the below image:

df.columns
dic={   'Feature_Name':['ID#', 'Education', 'Qual_miles', 'cc1_miles', 'cc2_miles', 'cc3_miles',
       'Bonus_miles', 'Bonus_trans', 'Flight_miles_12mo', 'Flight_trans_12',
       'Assault', 'Award?'],
         'Description':'Columns',
         'Type':['Quantitative','Nominal'],
         'Relevence':'Irrelevent'
     }
new_df=pd.DataFrame(dic)
#all array must be of same length
new_df
###################################################################

#3. Data Pre-processing 
#Data Cleaning, Feature Engineering, etc.

df.head()
#2 FINDING DUPLICATES
#drop 
duplicates=df.duplicated()
duplicates
#output is a single column it present true otherwise false.
sum(duplicates)#0
# so no duplicates are present

#3. OUTLIERS ANALYSIS
sns.boxplot(df.Education)
sns.boxplot(df.Flight_trans_12)
IQR=df.Education.quantile(0.75)-df.Education.quantile(0.25)
IQR
#73876.5

lower_limit=df.Education.quantile(0.75) - 1.5*IQR
lower_limit
#-18410.75
upper_limit=df.Education.quantile(0.75) + 1.5*IQR
upper_limit
#203218.75

######################################################


#OUTLIER TREATMENT
#TRIMMING

outliers_df=np.where(df.Education > upper_limit,True, np.where(df.Education<lower_limit,True,False)) 
outliers_df
df_trimmed=df.loc[~ outliers_df]
df_trimmed
df.shape
#(3999, 12)
df_trimmed.shape
#(3733, 12)
#therefore there are 266 outliers that is trimmed

#REPLACEMENT TECHQUIES

df_replaced=pd.DataFrame(np.where(df.Education > upper_limit , upper_limit,np.where(df.Education < lower_limit , lower_limit,df.Education)))
#if values are greter than upper limit mapped to the upper limit
#if values are lower than lower limit mapped to the lower limit

sns.boxplot(df_replaced[0])

#Winsorizer
from feature_engine.outliers import Winsorizer

winsor=Winsorizer(capping_method='iqr',
                  tail='both',
                  fold=1.5,
                  variables=['Education'])

df_t=winsor.fit_transform(df[{'Education'}])
sns.boxplot(df['Education'])
sns.boxplot(df_t['Education'])


###################################################################


#4. Exploratory Data Analysis (EDA):
#4.1. Summary.
#4.2. Univariate analysis.
#4.3. Bivariate analysis.


df.columns
df.shape
#(3999, 12)

df["Education"].value_Educations()
df["Assault"].value_Educations()

# the given dataset is a imEducationd dataset

###################################################################
#scatter plot
df.plot(kind='scatter', x='Bonus_miles', y='Education') ;
plt.show()
#2D scatter plot
sns.set_style("whitegrid");
sns.FacetGrid(df, hue="ID#").map(plt.scatter, "Assault", "Education").add_legend();
plt.show();
#pair plot
sns.pairplot(df, hue="Education");

#########################################################

#Mean, Variance, Std-deviation,  
print("Means:")
print(np.mean(df["Education"]))
#Mean with an outlier.
print(np.mean(np.append(df["Education"],50)));
print(np.mean(df["Education"]))
print(np.mean(df["Education"]))

print("\nStd-dev:")
print(np.std(df["Education"]))
print(np.std(df["Education"]))
print(np.std(df["Education"]))

print("\nMedians:")
print(np.median(df["Education"]))

#####################################################################
'''
5. Model Building 
5.1 Build the model on the scaled data (try multiple options).
5.2 Perform the hierarchical clustering and visualize the 
clusters using dendrogram.
5.3 Validate the clusters (try with different number of 
clusters) – label the clusters and derive insights 
(compare the results from multiple approaches).

'''

from scipy.cluster.hierarchy import linkage 
import scipy.cluster.hierarchy as sch 
from sklearn.cluster import AgglomerativeClustering 


z=linkage(df, method='complete',metric='euclidean') 
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
h_complete = AgglomerativeClustering(n_clusters=3,linkage='complete',affinity='euclidean').fit(df)

#apply labels to the clusters 
h_complete.labels_
cluster_labels = pd.Series(h_complete.labels_)
df['Education'] = cluster_labels 
#we want to relocate the column 7 to 0th position 
df = df.iloc[:,[7,1,2,3,4,5,6]]
#now check the Univ1 datafraame 
df.iloc[:,2:].groupby(df.Education).mean()

#########################################################################################

#6. Write about the benefits/impact of the solution - 
#in what way does the business (client) benefit from the solution provided?