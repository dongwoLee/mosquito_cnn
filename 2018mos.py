import csv
import os,glob
from random import shuffle
import tensorflow as tf
from numpy import array

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
    print(InputList[0][0])
    for i in range(len(InputList)):
        with open("C:/Users/dw/Desktop/mosquito_cnn/WholeInput/"+InputList[i][0],'r') as csvfile:
            reader = csv.reader(csvfile,delimiter=',')
            for row in reader:
               for i in range(len(row)):
                   InputFactorCsv.append(float(row[i]))

    ResultInput = [InputFactorCsv[i:i+5400] for i in range(0,len(InputFactorCsv),5400)]

    return array(ResultInput)

def MakeLabelSet(LabelList):

    LabelFactorCsv = []
    depth = 9
    print(LabelList[0][1])
    for i in range(len(LabelList)):
        with open("C:/Users/dw/Desktop/mosquito_cnn/WholeLevel/"+LabelList[i][1],'r') as csvfile:
            reader = csv.reader(csvfile,delimiter=',')
            for row in reader:
               for i in range(len(row)):
                   LabelFactorCsv.append(int(row[i]))

    tmp_one_hot = tf.one_hot(LabelFactorCsv,depth)

    with tf.Session() as sess:
        one_hot_label = sess.run(tmp_one_hot)

    return (one_hot_label)

def makeBatch(DataList,ListLength,batchSize):

    Output = []
    for i in range(0,ListLength,batchSize):
        Output.append(DataList[i:i+batchSize])
    Output = array(Output)

    return Output

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
    TrainInput = (MakeInputSet(trainingDataSet))
    TrainLabel = (MakeLabelSet(trainingDataSet))

    TestInput = (MakeInputSet(testDataSet))
    TestLabel = (MakeLabelSet(testDataSet))

    TrainInputBatch = makeBatch(TrainInput,6600,100)
    TrainLabelBatch = makeBatch(TrainLabel,2800,100)

    TestInputBatch = makeBatch(TestInput,6600,100)
    TestLabelBatch = makeBatch(TestLabel,2800,100)

    