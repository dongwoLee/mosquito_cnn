import tensorflow as tf
import numpy as np
from numpy import array
import csv
import os,glob
from tensorflow.examples.tutorials.mnist import input_data

def getFileName(dir):
	os.chdir(dir)
	dateList = []
	for file in glob.glob("*.csv"):
		dateList.append(file)

	return dateList

def readCsv(csvlist):# label data should be made by this function.

	dataSet = []
	for i in range(len(csvlist)):
		with open(csvlist[i],'r') as f:
			reader = csv.reader(f,delimiter=',')
			for row in reader:
				dataSet.append(''.join(row))

	return dataSet

def makeTrainingSet():#training and test mosq
	
	location = []
	trainFolder = []
	trainCsv = []
	wholeCsv = []
	for folder in os.listdir("/Users/leedongwoo/Desktop/mosquito_cnn/Location_allDate"):
		if folder == '.DS_Store':
			continue
		else:
			location.append(folder)#all location

	for i in range(len(location)):
		trainFolder.append(getFileName("/Users/leedongwoo/Desktop/mosquito_cnn/Location_allDate/"+str(location[i])))

	for j in range(len(trainFolder)):
		for k in range(len(trainFolder[j])):
			trainFolder[j][k] = "/Users/leedongwoo/Desktop/mosquito_cnn/Location_allDate/"+str(location[j])+"/"+str(trainFolder[j][k])

	for p in range(len(trainFolder)):
		for q in range(len(trainFolder[p])):
			with open(trainFolder[p][q],'r') as f:
				reader = csv.reader(f,delimiter=',')
				 # this reader convert convert list to string
				trainCsv.append(list(reader))

	for index in range(len(trainCsv)):
		wholeCsv.append([[float(y) for y in x] for x in trainCsv[index]])

	trainingCsv = trainCsv[:int(len(trainCsv)*0.8)]
	testCsv  = trainCsv[int(len(trainCsv)*0.8):len(trainCsv)]

	return array(trainingCsv),array(testCsv)

def makeLabel_level():#training level and test level 

	train = []
	test = []
	wholeLabelCsv = getFileName("/Users/leedongwoo/Desktop/mosquito_cnn/Label_Data/Level/noDateMosq")
	wholeLabel = readCsv(wholeLabelCsv)

	train = wholeLabel[:int(len(wholeLabel)*0.8)]
	test = wholeLabel[int(len(wholeLabel)*0.8):len(wholeLabel)]

	train = list(map(int,train))
	test = list(map(int,test))

	train = np.array(train)
	test = np.array(test)

	train_ = np.zeros((len(train),9))
	test_ = np.zeros((len(test),9))

	train_[np.arange(len(train)),train] = 1
	test_[np.arange(len(test)),test] = 1

	return (train_),(test_)

if __name__ == '__main__':
	
	trainingCsv , testCsv = makeTrainingSet() 
	trainLabel,testLabel = makeLabel_level()#Do i have to change Label data using one-hot encoding?
	print (type(trainLabel))
	X = tf.placeholder(tf.float32, shape=[None,5400])
	Y = tf.placeholder(tf.float32, shape=[None,9])



	




