## TRANSFORM DATA

import pandas as pd
import math
from sklearn import preprocessing
import os
os.chdir('C:/Users/Jackie/Documents/1A-CS 412/Project')
number = preprocessing.LabelEncoder()

# Input data and select data column to predict
train = pd.read_csv('./data/train_users_2.csv')
test = pd.read_csv('./data/test_users.csv')
del train['date_first_booking']
del test['date_first_booking']
del train['timestamp_first_active']
del test['timestamp_first_active']


## Transform the data, fill in Nan, relable levels of categorical variable as numeric
def convert(data):
	data.loc[data['age'] > 110] =None
	FillingList = [37,36,25,35,34,26,33,27,32,28,29,31,30,30,31,29,28] 
	# training set has count over 3500- 6000, with duplicates on the top frequent observations
	for i in data.index:
		if math.isnan(data.iloc[i]['age']):
			data.loc[i,'age']= FillingList[i % 17]

	attr = ['gender', 'signup_method', 'language', 'affiliate_channel' ,'affiliate_provider', 'first_affiliate_tracked', 'signup_app', 'first_device_type','first_browser' ]

	for a in attr:
		data[a]= data[a].fillna('No_Record')
		data[a] = number.fit_transform(data[a])
	data = data.dropna()
	return data

train = convert(train)
test = convert(test)

train.to_csv("./Train_Transformed.csv", index=False, replace = True)
test.to_csv("./Test_Transformed.csv", index=False, replace = True)

