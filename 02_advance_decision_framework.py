
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

test_cust = pd.merge(demo,cibil,finan,gst,how='inner',on='customer_id')        
if log.predict(test_cust)==1:


	# Advance decision framework model training - DNN n RF

	import tensorflow as tf

	target = x['loan_scheme','value','tenure','profit_rate']
	x = x.drop(['loan_scheme','value','tenure','profit_rate'],axis=1)
	x_train,x_test,y_train,y_test=train_test_split(x,target,test_size=0.3,random_state=42)

	# deep neural network with 4 hidden layers

	model = tf.keras.models.Sequential([
	  tf.keras.layers.Flatten(input_shape=(28, 28)),
	  tf.keras.layers.Dense(128, activation='relu'),
	  tf.keras.layers.Dropout(0.2),
	  tf.keras.layers.Dense(10, activation='softmax')
	])

	model.compile(optimizer='adam',
	              loss='sparse_categorical_crossentropy',
	              metrics=['accuracy'])

	model.fit(x_train, y_train, epochs=5)
	model.evaluate(x_test, y_test)
	prediction = model.predict(x_test)

	# Step-wise feature selection using Random Forest classifier

	rfe = RFE(estimator=RandomForestClassifier(random_state=0,max_depth=9,n_estimators=20),n_features_to_select=9)
	rfe.fit(x_train,y_train)

	train_probs=rfe.predict_proba(x_train)[:,1]
	probs=rfe.predict_proba(x_test)[:,1]


	# lookalike modelling - kNN for next best action

	from sklearn.neighbors import KNeighborsClassifier
	knn = KNeighborsClassifier(n_neighbors=2)

	knn.fit(x_train,y_train)
	next_best_action = knn.predict(x_test)

	# Rules overiders from Rule book

	rules_book = pd.read_csv("C:/Users/e078732/Documents/ccaf/rules.csv")
	def check(rule,df,prediction):
	    if master_df_v3[rule['variable']]>rule['value']: return True else False 
	for rule in rules_book.shape[0]:
	    if !check(rule,master_df_v3):
	        master_df_v3[rule['variable']]=rule['value']


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

	if log.predict(test_cust)==1:
	    print('output from CCAF: '+model.predict(test_cust)+'\n')

