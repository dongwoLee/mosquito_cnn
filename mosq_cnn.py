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
	
	twoDimensionData = [] 

	return 

def makeTrainingSet():#training and test mosq
	
	location = []
	trainCsv=[]
	for folder in os.listdir("/Users/leedongwoo/Desktop/mosquito_cnn/Location_allDate"):
		if folder == '.DS_Store':
			continue
		else:
			location.append(folder)

	for i in range(len(location)):
		trainCsv.extend(getFileName("/Users/leedongwoo/Desktop/mosquito_cnn/Location_allDate/"+str(location[i])))

	
	

def makeLabel_level():#training level and test level 

	train = []
	test = []
	wholeLabelCsv = getFileName("/Users/leedongwoo/Desktop/mosquito_cnn/Label_Data/Level/noDateMosq")
	
	wholeLabel = readCsv(wholeLabelCsv)

	train = wholeLabel[:int(len(wholeLabel)*0.8)]
	test = wholeLabel[int(len(wholeLabel)*0.8):len(wholeLabel)]

	return array(train),array(test)


if __name__ == '__main__':
	
	trainLabel,testLabel = makeLabel_level()
	# print ((trainLabel),(testLabel))
	print(len(makeTrainingSet()))

