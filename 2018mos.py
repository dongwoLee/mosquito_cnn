import csv
import os,glob
import random
import shutil
from random import shuffle

def getInputCsv(list):

    path = in_Dir+list
    extension = 'csv'
    os.chdir(path)
    result = [i for i in glob.glob('*.{}'.format(extension))]

    return (result)

def getLabelCsv(list):

    path = le_Dir + list
    extension = 'csv'
    os.chdir(path)
    result = [i for i in glob.glob('*.{}'.format(extension))]

    return (result)

def makeRawData(inputDir,levelDir):

    resultInput = []
    resultLabel = []

    inputDirectory = os.listdir(inputDir)
    labelDirectory = os.listdir(levelDir)

    for dir in inputDirectory:
        resultInput.extend(getInputCsv(dir))

    for Ldir in labelDirectory:
        resultLabel.extend(getLabelCsv(Ldir))

    return (resultInput),(resultLabel)

def makeTrainingTest(rawList):

    splitLen = int(len(rawList)*0.7)

    trainingList = rawList[0:splitLen]
    testList = rawList[splitLen:len(rawList)]

    return trainingList,testList

def MakeInputSet(InputList):

    InputFactorCsv = []

    for i in range(len(InputList)):
        with open("C:/Users/dw/Desktop/mosquito_cnn/WholeInput/"+InputList[i][0],'r') as csvfile:
            reader = csv.reader(csvfile,delimiter=',')
            for row in reader:
               for i in range(len(row)):
                   InputFactorCsv.append(row[i])

    return InputFactorCsv

def MakeLabelSet(LabelList):

    with open("C:/Users/dw/Desktop/mosquito_cnn/WholeLevel/"+LabelList[0][1],'r') as csvfile:
        print(LabelList[0][1])
        reader = csv.reader(csvfile,delimiter=',')
        for row in reader:
            print (row)

if __name__ == '__main__':
    resultDataSet = []
    in_Dir = "C:/Users/dw/Desktop/mosquito_cnn/Location_allDate/"
    le_Dir = "C:/Users/dw/Desktop/mosquito_cnn/Label_Data/Level/DateLevel/"

    InputData, LabelData = makeRawData(in_Dir,le_Dir)

    for i in range(len(InputData)):
        temp = []
        temp.extend([InputData[i],LabelData[i]])
        resultDataSet.append(temp)


    shuffledData =[resultDataSet[i] for i in range(len(resultDataSet))]
    shuffle(shuffledData)

    trainingDataSet, testDataSet = makeTrainingTest(shuffledData)
    print(MakeInputSet(trainingDataSet))
    print(len(MakeInputSet(trainingDataSet)))