# -- coding: utf-8 --
"""

@author: Jindal, Karan e078732
"""
# import dependencies

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

# reading relevant datasets - demo, cibil, financials and gst

cust_demo = pd.read_csv("C:/Users/e078732/Documents/ccaf/demographs.csv")
cust_cibil = pd.read_csv("C:/Users/e078732/Documents/ccaf/cibil_data.csv")
cust_finan = pd.read_csv("C:/Users/e078732/Documents/ccaf/financials.csv")
cust_gst = pd.read_csv("C:/Users/e078732/Documents/ccaf/gst.csv")

# create master dataset with data from all sources

master_df_v1 = pd.merge(cust_demo,cust_cibil,how='inner',on='customer_id')
master_df_v2 = pd.merge(master_df_v1,cust_finan,how='inner',on='customer_id')


# missing value imputations - filling in for missing values with mean/mode

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import Imputer

import scipy.sparse as sp
master_df_v3 = master_df_v2.replace('.',0)
imp = Imputer(missing_values='.', strategy='mean')
master_df_imputed = imp.fit(master_df_v3)
imp = Imputer(missing_values='.', strategy='most_frequent')
master_df_imputed = np.round(imp.transform(master_df_v2))

# variable transformations and Weight of evidence for categoricals

enc = preprocessing.OriginalEncoder()
enc.fit(master_df_imputed)

categorical = master_df_v2.select_dtypes(include=['object']).copy()
categorical = master_df_v3[['Region','Bank_Relationship']]

categorical['Risk_Flag'] = master_df_v2['Risk_Flag']
for column in categorical.columns:
    if column!='Risk_Flag':
        categorical[column] = categorical[column].map(categorical.groupby(column)['Risk_Flag'].mean())


# data reduction using variance inflation factor VIF and Principal component analysis PCA

from statsmodels.stats.outliers_influence import variance_inflation_factor
vif_data["VIF"] = [variance_inflation_factor(master_df_v2.values, i)
						for i in range(len(master_df_v2.columns))]
cor_columns=vif_data[[variance_inflation_factor>2]]
master_df_v3 = master_df_v3.drop(cor_columns,axis=1)

from sklearn.decomposition import PCA
pca =PCA()
x_pca=pca.fit_transform(master_df_v3)
x_pca=pd.DataFrame(x_pca)
explained_variance=pca.explained_variance_ratio_

# test train validation split - 70/30 spilt for test n train dataset
x = x_pca
y=x['Risk_Flag']
x = x.drop('Risk_Flag',axis=1)

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=42)

# Risk classification model training - logistic / DTree

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFECV

# Decision Tree Classifier

clf_gini = DecisionTreeClassifier(criterion = "gini",
            random_state = 100,max_depth=6, min_samples_leaf=5)

train_probs=clf_gini.predict_proba(x_train)[:,1]
probs=clf_gini.predict_proba(x_test)[:,1]

# Logistic Regression

log = LogisticRegression(random_state=0).fit(x_train,y_train)

train_probs=clf_gini.predict_proba(x_train)[:,1]
probs=clf_gini.predict_proba(x_test)[:,1]

from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.metrics import confusion_matrix

roc_auc_score(y_train,train_probs)
roc_auc_score(y_test,probs)

#score for unseen customer

# combine datasets from all sources
test_cust = pd.merge(demo,cibil,finan,gst,how='inner',on='customer_id')        
# transformation and WOEs
for column in test_cust.columns:
    if column in categorical:
        test_cust[column] = test_cust[column].map(woe[test_cust[column]])
test_cust = test_cust[final_columns_after_PCA]        
# scoring

print('output from Risk Classification Engine: '+log.predict(test_cust)+'\n')
