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
    # Both training and test will be adopted this function
    InputFactorCsv = []

    for i in range(len(InputList)):
        with open("C:/Users/dw/Desktop/mosquito_cnn/WholeInput/"+InputList[i][0],'r') as csvfile:
            reader = csv.reader(csvfile,delimiter=',')
            for row in reader:
               for i in range(len(row)):
                   InputFactorCsv.append(float(row[i]))

    ResultInput = [InputFactorCsv[i:i+5400] for i in range(0,len(InputFactorCsv),5400)]

    return InputList[0][0]

def MakeLabelSet(LabelList):

    LabelFactorCsv = []
    print(LabelList[0][0])
    for i in range(len(LabelList)):
        with open("C:/Users/dw/Desktop/mosquito_cnn/WholeLevel/"+LabelList[i][1],'r') as csvfile:
            reader = csv.reader(csvfile,delimiter=',')
            for row in reader:
               for i in range(len(row)):
                   LabelFactorCsv.append(int(row[i]))

    return (LabelFactorCsv[0])

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
    # TrainInput = (MakeInputSet(trainingDataSet))
    print(MakeInputSet(trainingDataSet))
    print(MakeLabelSet(trainingDataSet))
