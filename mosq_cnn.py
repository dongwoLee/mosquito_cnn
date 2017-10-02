import tensorflow as tf
import numpy as np
from numpy import array
import csv
import os,glob

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

def readCsvTwoDimension(csvList):#training input should be made by two dimensional list.
	
	Matrix = []

	for i in range(len(csvList)):
		with open(csvList[i],'r') as f:
			reader = csv.reader(f,delimiter=',')
			Matrix.append(list(reader))

	return Matrix

def makeTrainingSet():#training and test mosq
	
	location = []
	trainFolder = []
	trainCsv = []
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

	return len(trainFolder)
	# trainFolder = getFileName("/Users/leedongwoo/Desktop/mosquito_cnn/Location_allDate/Congress/")
	
	# train_data = readCsvTwoDimension(trainCsv)
	

def makeLabel_level():#training level and test level 

	train = []
	test = []
	wholeLabelCsv = getFileName("/Users/leedongwoo/Desktop/mosquito_cnn/Label_Data/Level/noDateMosq")
	wholeLabel = readCsv(wholeLabelCsv)

	train = wholeLabel[:int(len(wholeLabel)*0.8)]
	test = wholeLabel[int(len(wholeLabel)*0.8):len(wholeLabel)]

	train = list(map(float,train))
	test = list(map(float,test))

	return array(train),array(test)


if __name__ == '__main__':
	
	#trainLabel,testLabel = makeLabel_level()
	# print(readCsvTwoDimension())
	#print ((trainLabel),(testLabel))
	print((makeTrainingSet()))

