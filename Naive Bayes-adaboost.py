## CS 412 HW5

import pandas as pd
from sklearn.datasets import load_svmlight_files
import sys
from numpy.random import choice
import math
# draw = choice(list_of_candidates, number_of_items_to_pick, p=probability_distribution)
import os
os.chdir('C:/Users/Jackie/Documents/1A-CS 412/HW5/dataset')

#### Question 1
## STEP 1: Load data

X_train, y_train, X_test, y_test = load_svmlight_files( (sys.argv[1], sys.argv[2]))
# X_train, y_train, X_test, y_test = load_svmlight_files( ("./poker.train", "./poker.test") )

xtrain= X_train.toarray()
d = pd.DataFrame(xtrain)
d.to_csv("array.csv",  index= False, replace = True)
train = pd.read_csv("array.csv")
train["category"] = pd.Series(y_train)

xtest= X_test.toarray()
d = pd.DataFrame(xtest)
d.to_csv("array1.csv",  index= False, replace = True)
test = pd.read_csv("array1.csv")
test["category"] = pd.Series(y_test)
#print(test)

attributes= list(train.columns.values)
del attributes[-1]

for attr in attributes:
	temp =train[attr].value_counts()
	percent = max(temp)/sum(temp)
	if(percent>= 0.75):
		del train[attr]

attributes= list(train.columns.values)
del attributes[-1]
#print(attributes)
cat = train.category.unique()


n_train = train.shape[0]
n_test = test.shape[0]
weight_train = [float(1/n_train)] * n_train
# weight_test = [float(1/n_test)] * n_test


def freqTable(level,train):
	dict_level = {}
	subgroup = train[train.category == level]
	count = subgroup.count()
	for attr in attributes:
		prob_attr = pd.DataFrame(subgroup[attr].value_counts()/count[attr])
		prob_attr.columns = ['Prob']
		prob_attr['Value'] = prob_attr.index
		for i in prob_attr.index:
			dict_level.update( { (attr, prob_attr.ix[i]['Value']) : prob_attr.ix[i]['Prob'] } )
	# print(dict_level)
	return(dict_level)


def newfreqTable_all(train):
	freqTable_all = dict()
	for c in cat:
		freqTable_all.update( {c: freqTable(c, train)})
	return(freqTable_all)



def computeFreq(dictionary, observation):
	post_prob = 1
	for attr in attributes:
		prob_attr = dictionary.get( (attr,observation[attr] ), 0.01)
		post_prob = post_prob * prob_attr
	return post_prob


def cat_max_prob(train,test):
	freqTable_all = newfreqTable_all(train)
	result = []

	for i in test.index:
		prob = 0
		ret = ''
		for key in freqTable_all.keys():
			dictionary = freqTable_all[key]
			prob_temp = computeFreq(dictionary, test.ix[i])
			if prob_temp > prob:
				prob = prob_temp
				ret = key
		result += [ret]
	return(result)

def Adaboost(k):
	global weight_train
	ret_pred_train = [0]* n_train
	ret_pred_test =[0] * n_test

	for i in range(0,k):
		train_idx = choice( list(range(0, n_train)), n_train, p= weight_train, replace =True)
		train_i = pd.DataFrame(train.loc[train_idx,])
		train_i.index = list(range(0, n_train))
		
		pred_train_i = cat_max_prob(train_i, train_i)
		pred_train_o =cat_max_prob(train_i, train) # prediction of the original order
		pred_test_i = cat_max_prob(train_i, test)
		
		# find error rate, weight for classifier
		error_i = 0
		for j in range(0, len(train_i)):
			if(train_i.category[j] != pred_train_i[j]):
				error_i += weight_train[train_idx[j]]
		a_i = math.log( (1-error_i)/error_i )
		
		# modify weight
		for p in range(0, n_train):
			ret_pred_train[p] += a_i* pred_train_o[p]
			if(train_i.category[p] == pred_train_i[p]):
				weight_train[train_idx[p]] = weight_train[train_idx[p]] *error_i/(1-error_i)
		
		# normalize weight
		weight_train = [x/sum(weight_train) for x in weight_train]

		for t in range(0, n_test):
			ret_pred_test[t] += a_i* pred_test_i[t]
	
	for q in range(0, n_train):
		if(ret_pred_train[q] > 0):
			ret_pred_train[q] = 1
		else:
			ret_pred_train[q] = -1

	for r in range(0, n_test):
		if(ret_pred_test[r] > 0):
			ret_pred_test[r] = 1
		else:
			ret_pred_test[r] = -1

	return( [ret_pred_train, ret_pred_test] )


pred = Adaboost(5)
pred_train = pred[0]
pred_test = pred[1]


def output(real, pred):
	ret =[]
	for i in range(0,len(real)):
		if real[i] ==1 and pred[i] ==1:
			ret += ['TP']
		elif real[i] ==1 and pred[i] == -1:
			ret += ['FN']
		elif real[i] == -1 and pred[i] ==1:
			ret += ['FP']
		else:
			ret += ['TN']
	return(ret)


eval_train = pd.DataFrame(output(y_train, pred_train), columns=['tr'])
eval_test = pd.DataFrame(output(y_test, pred_test), columns = ['te'])

print("Using Adaboost")
cm_train = eval_train['tr'].value_counts()
cm_test = eval_test['te'].value_counts()

ret1 =[ cm_train['TP'], cm_train['FN'], cm_train['FP'], cm_train['TN'] ]
ret2 =[ cm_test['TP'], cm_test['FN'], cm_test['FP'], cm_test['TN'] ]
print(ret1)
print(ret2)

# Accuracy, Error Rate, Sensitivity, Specificity, Precision, F-1 Score, Fβ score

def model_eval(ret):
	tp=ret[0]
	fn=ret[1]
	fp=ret[2]
	tn=ret[3]
	a = sum(ret)
	accuracy = (tp+tn)/a
	error =1-accuracy
	sensitivity = tp/(tp+fn)
	specificity = tn/(tn+fp)
	precision = tp/(tp+fp)
	recall = tp/(tp+fn) # not outputed
	f_1 =(2*precision*recall)/(precision+recall)
	f_beta_1=(1.25*precision*recall)/(0.25*precision+recall)
	f_beta_2 =(5*precision*recall)/(4*precision+recall)
	result = [accuracy, error, sensitivity, specificity, precision, f_1, f_beta_1, f_beta_2]
	retList = [ '%.2f' % elem for elem in result ]
	print(retList)
	return(retList)

print("model evaluation")
model_eval(ret1)
model_eval(ret2)
