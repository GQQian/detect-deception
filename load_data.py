import jsonpickle as jp
import _pickle as cPickle
import json
import os
from nltk.tokenize import word_tokenize
import numpy as np

data_dir = 'data/op_spam_v1.4'

#negative reviews that are true/false
negative_dir_true = data_dir+'/negative_polarity/truthful_from_Web'
negative_dir_false = data_dir+'/negative_polarity/deceptive_from_MTurk'

#positive reviews that are true/false
positive_dir_true = data_dir+'/positive_polarity/truthful_from_TripAdvisor'
positive_dir_false = data_dir+'/positive_polarity/deceptive_from_MTurk'






trueX = []
falseX = []

for i in range(1,6):
	
	curdir = negative_dir_true + '/fold{}'.format(i)
	trueX.append([])
	falseX.append([])
	for file in os.listdir(curdir):
		with open(curdir + '/' + file) as f:
			trueX[i-1].append(word_tokenize(f.read()))

	curdir = negative_dir_false + '/fold{}'.format(i)
	for file in os.listdir(curdir):
		with open(curdir + '/' + file) as f:
			falseX[i-1].append(word_tokenize(f.read()))

	curdir = positive_dir_true + '/fold{}'.format(i)
	for file in os.listdir(curdir):
		with open(curdir + '/' + file) as f:
			trueX[i-1].append(word_tokenize(f.read()))

	curdir = positive_dir_false + '/fold{}'.format(i)
	for file in os.listdir(curdir):
		with open(curdir + '/' + file) as f:
			falseX[i-1].append(word_tokenize(f.read()))
#trueX, falseX dims: (5*160*n_words)
# trueX = np.array(trueX)
# falseX = np.array(falseX)



all_X = []
all_Y = []
for i in range(5):
	all_X.append([])
	all_Y.append([])
	for x in trueX[i]:
		all_X[i].append(x)
		all_Y[i].append(True)
		if(falseX[i]):
			all_X[i].append(falseX[i].pop())
			all_Y[i].append(False)

	if(falseX[i]):
		all_X[i].append(falseX[i])
		all_Y[i].append([False]*len(falseX[i]))

with open('data/5_fold_input.pkl', 'w+b') as output_file:
	cPickle.dump((all_X, all_Y), output_file)



"""
positive_true = []
positive_false = []

negative_true = []
negative_false = []


for i in range(1,6):
	
	curdir = negative_dir_true + '/fold{}'.format(i)
	for file in os.listdir(curdir):
		with open(curdir + '/' + file) as f:
			negative_true.append(word_tokenize(f.read()))
	
	curdir = negative_dir_false + '/fold{}'.format(i)
	for file in os.listdir(curdir):
		with open(curdir + '/' + file) as f:
			negative_false.append(word_tokenize(f.read()))

	curdir = positive_dir_true + '/fold{}'.format(i)
	for file in os.listdir(curdir):
		with open(curdir + '/' + file) as f:
			positive_true.append(word_tokenize(f.read()))

	curdir = positive_dir_false + '/fold{}'.format(i)
	for file in os.listdir(curdir):
		with open(curdir + '/' + file) as f:
			positive_false.append(word_tokenize(f.read()))


positive_X = []
positive_Y = []
negative_X = []
negative_Y = []


for x in positive_true:
	positive_X.append(x)
	positive_Y.append(True)
	if(positive_false):
		positive_X.append(positive_false.pop())
		positive_Y.append(False)

if(positive_false):
	positive_X.append(positive_false)
	positive_Y.append([False]*len(positive_false))


for x in negative_true:
	negative_X.append(x)
	negative_Y.append(True)
	if(negative_false):
		negative_X.append(negative_false.pop())
		negative_Y.append(False)

if(negative_false):
	negative_X.append(negative_false)
	negative_Y.append([False]*len(negative_false))


with open('data/polar_data.pkl', 'w+b') as output_file:
	cPickle.dump((positive_X, positive_Y, negative_X, negative_Y), output_file)"""










