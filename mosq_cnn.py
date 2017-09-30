import tensorflow as tf
import numpy as np
import csv
import os,glob

def getFileName(dir):
	os.chdir(dir)
	dateList = []
	for file in glob.glob("*.csv"):
		dateList.append(file)

	return dateList

def readCsv(csvlist):

	dataSet = []
	for i in range(len(csvlist)):
		with open(csvlist[i],'r') as f:
			reader = csv.reader(f,delimiter=',')
			for row in reader:
				dataSet.append(''.join(row))

	return dataSet

def makeTrainingSet():#training and test mosq
	
	for folder in os.listdir("/Users/leedongwoo/Desktop/mosquito_cnn/Location_allDate"):
		if(folder=='.DS_Store'):
			continue
		else:
			print (folder)


def makeLabel_level():#training level and test level 

	train = []
	test = []
	wholeLabelCsv = getFileName("/Users/leedongwoo/Desktop/mosquito_cnn/Label_Data/Level/noDateMosq")
	
	wholeLabel = readCsv(wholeLabelCsv)

	train = wholeLabel[:int(len(wholeLabel)*0.8)]
	test = wholeLabel[int(len(wholeLabel)*0.8):len(wholeLabel)]

	return train,test

if __name__ == '__main__':
	
	trainLabel,testLabel = makeLabel_level()
	print (len(trainLabel),len(testLabel))
	(makeTrainingSet())
