### CS 412 Project
### Jiaqi Yao
### Xuan Wang

################# Random Forest Algorithm ########################
import pandas as pd
import math
from sklearn import preprocessing
import random
import os
os.chdir('C:/Users/Jackie/Documents/1A-CS 412/Project')
number = preprocessing.LabelEncoder()

train = pd.read_csv('./train_Transformed.csv')
test = pd.read_csv('./test_Transformed.csv')
data = {'id' : list(test['id'])}
result_RandomForest = pd.DataFrame(data)


# column =['date_account_created','gender','age','signup_method','signup_flow','language','affiliate_channel','affiliate_provider','first_affiliate_tracked','signup_app','first_device_type','first_browser']
del_col = ['date_account_created','signup_flow','affiliate_channel','affiliate_provider','first_affiliate_tracked','signup_app','first_device_type','first_browser']
for d_c in del_col:
	del train[d_c]
	del test[d_c]


# Specify attribute needed to build decision tree
attributes = list(train.columns.values)
del attributes[0]
del attributes[-1]


# This function compute the Infomation I(a,b,...,e,f), take list as argument
def I(num_list):
	ret = 0
	num_sum = sum(num_list)
	for i in num_list:
		ret += (-i/num_sum)*(math.log(i/num_sum,2))
	return ret


# This function determines the min information loss for pne attribute and its corresponding split point given dataset.
# (so that we can have maximum information gain after subtraction)
def min_Info_Loss(attribute, curr_dataset):
	data = { 'attr': curr_dataset[attribute], 'dest' : curr_dataset['country_destination'] }
	dataset = pd.DataFrame(data, columns = ['attr', 'dest'])
	candidate = dataset.attr.unique()
	count = dataset.attr.count()
	ret_info =1000 # large enough
	ret_split = 0  # numeric split point
	#print("count is " % count)
	for c in candidate:
		info = 0
		subset1 = dataset[dataset.attr <= c]
		subset2 = dataset [dataset.attr > c] 
		sub_count1 = subset1.attr.count()
		sub_count2 = subset2.attr.count()
		list1 = list(subset1.dest.value_counts())
		list2 = list(subset2.dest.value_counts())
		info = sub_count1/count* I(list1) + sub_count2/count*I(list2)
		if info <= ret_info:
			ret_info = info
			ret_split = c
	return( (ret_info, ret_split) )


# this function finds the attribute and its splitpoint to split on, in a given dataset
def findsplit(curr_dataset):
	ret_info = 100
	ret_attr = ''
	ret_split = -1
	attr_Random = random.sample(attributes,3)
	for attr in attr_Random:
		temp_info = min_Info_Loss(attr, curr_dataset)[0]
		temp_split = min_Info_Loss(attr, curr_dataset)[1]
		if temp_info <= ret_info:
			ret_info = temp_info
			ret_split = temp_split
			ret_attr = attr
	return(ret_attr, ret_split)


## class for decision tree
class DecisionTree(object):
	class TreeNode():
		def __init__(self, attr = None, split =None):
			self.attr = attr
			self.split = split
			self.left = None
			self.right = None

	def Leaf_Node(self, curr_dataset):
		dest_list = curr_dataset.country_destination
		b1 = True # all falls in same group
		for attr in attributes:
			if len(curr_dataset[attr].unique())>1:
				b1 = False
		b2 = len(dest_list.unique()) == 1 # belong to same class
		b3 = dest_list.count() <= 1 # has nothing to split
		b4 = False
		# Tree Prepruning
		if dest_list.count() in range(2, 20):
			dominant = dest_list.describe()['top']
			b4 = dest_list[dest_list == dominant].count() / dest_list.count() > 0.75 #Threshold choose to be 0.75
		return b1 or b2 or b3 or b4

	def __init__(self):
		self.root = self.TreeNode()

	def DCTree_init(self, dataset):
		self.root = self.ConstructTree(dataset)

	def ConstructTree(self, dataset):
		if self.Leaf_Node(dataset):
			if dataset.country_destination.count() == 0:
				return self.TreeNode('NDF', -100)
			result = dataset.country_destination.describe()['top']
			return self.TreeNode(str(result), -100)
		else:
			attr_split = findsplit(dataset)
			a = attr_split[0]
			s = attr_split[1]
			curr_tree = self.TreeNode(a, s)
			subset_left = dataset[dataset[a] <= s]
			subset_right = dataset[dataset[a] > s]

			print([a,"split on", s, subset_left.age.count(), subset_right.age.count()])

			if subset_left.country_destination.count() == dataset.country_destination.count():
				result = dataset.country_destination.describe()['top']
				return self.TreeNode(str(result), -100)
			
			elif subset_right.country_destination.count() == dataset.country_destination.count():
				result = dataset.country_destination.describe()['top']
				return self.TreeNode(str(result), -100)
			
			else:
				curr_tree.left = self.ConstructTree(subset_left)
				curr_tree.right = self.ConstructTree(subset_right)

			return curr_tree


# This function traverse the decision tree built until reach leaf node,
# prediction for each line is returned here
def get_pred(subroot, observation):
	if subroot.left == None or subroot.right == None:
		return (str(subroot.attr))
	else:
		attr = subroot.attr
		split = subroot.split
		if observation[attr] <= split:
			return(get_pred(subroot.left, observation))
		else:
			return(get_pred(subroot.right, observation))


def get_prediction(DCTree, observation):
	return (get_pred( DCTree.root, observation))


# This function takes in a decisiontree and test dataset as input
# and it returns a list of prediction
def get_destination(DCTree, test):
	result = []
	for i in test.index:
		result += [get_prediction(DCTree, test.ix[i])]
	return result


##### RandomForest

def RandomForest(sample_size,num_Tree):
	for i in range(1, num_Tree+1):
		# Subset the train data
		subtrain = train.sample( n = sample_size)
		# Build the tree with the subset of train data
		RandomTree = DecisionTree()
		RandomTree.DCTree_init(subtrain)
		Pred_RT = get_destination(RandomTree, test)
		# write result to dataFrame
		result_RandomForest[str(i)] = Pred_RT

	result_RandomForest['Vote'] = None
	for i in result_RandomForest.index:
		ret_list = list(result_RandomForest.loc[i])
		del ret_list[0]
		result_RandomForest.loc[i,'Vote'] = pd.DataFrame(ret_list).ix[:,0].describe()['top']
		# print (result_RandomForest.loc[i])

	ret_data = {'id' : list(test['id']) , 'country': result_RandomForest['Vote'] }
	submission_RandomForest = pd.DataFrame( ret_data, columns = ['id', 'country'] )
	submission_RandomForest.to_csv("./submission_RandomForest_100000_70_3pred.csv", index=False, replace = True)


# RandomForest(10000, 7)
# 0.69674

# RandomForest(10000, 20)
# 0.69693

#RandomForest(10000, 100)
# 0.70514/0.70466

# RandomForest(10000, 100)
# 0.70636

# RandomForest(100000, 7)
# 0.68386

RandomForest(100000, 25)
# 0.70640

