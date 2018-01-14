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

    X = tf.placeholder(tf.float32, shape=[None, 180, 30, 1])
    Y = tf.placeholder(tf.float32, shape=[None, 9])
    keep_prob = tf.placeholder(tf.float32)

    W1 = tf.Variable(tf.random_normal([5, 5, 1, 32], stddev=0.01))
    L1 = tf.nn.conv2d(X, W1, strides=[1, 1, 1, 1], padding='SAME')
    L1 = tf.nn.relu(L1)
    L1 = tf.nn.max_pool(L1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

    W2 = tf.Variable(tf.random_normal([5, 5, 32, 64], stddev=0.01))
    L2 = tf.nn.conv2d(L1, W2, strides=[1, 1, 1, 1], padding='SAME')
    L2 = tf.nn.relu(L2)
    L2 = tf.nn.max_pool(L2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

    W3 = tf.Variable(tf.random_normal([5, 5, 64, 128], stddev=0.01))
    L3 = tf.nn.conv2d(L2, W3, strides=[1, 1, 1, 1], padding='SAME')
    L3 = tf.nn.relu(L3)
    L3 = tf.nn.max_pool(L3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

    W4 = tf.Variable(tf.random_normal([5, 5, 128, 256], stddev=0.01))
    L4 = tf.nn.conv2d(L3, W4, strides=[1, 1, 1, 1], padding='SAME')
    L4 = tf.nn.relu(L4)
    L4 = tf.nn.max_pool(L4, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

    W5 = tf.Variable(tf.random_normal([5, 5, 256, 512], stddev=0.01))
    L5 = tf.nn.conv2d(L4, W5, strides=[1, 1, 1, 1], padding='SAME')
    L5 = tf.nn.relu(L5)
    L5 = tf.nn.max_pool(L5, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

    W6 = tf.Variable(tf.random_normal([5, 5, 512, 1024], stddev=0.01))
    L6 = tf.nn.conv2d(L5, W6, strides=[1, 1, 1, 1], padding='SAME')
    L6 = tf.nn.relu(L6)
    L6 = tf.nn.max_pool(L6, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

    # W7 = tf.Variable(tf.random_normal([5, 5, 1024, 2048], stddev=0.01))
    # L7 = tf.nn.conv2d(L6, W7, strides=[1, 1, 1, 1], padding='SAME')
    # L7 = tf.nn.relu(L7)
    # L7 = tf.nn.max_pool(L7, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

    # W8 = tf.Variable(tf.random_normal([5,5,2048,4096],stddev=0.01))
    # L8 = tf.nn.conv2d(L7,W8,strides=[1,1,1,1],padding='SAME')
    # L8 = tf.nn.relu(L8)
    # L8 = tf.nn.max_pool(L8,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')

    W11 = tf.Variable(tf.random_normal([3 * 1 * 1024, 4096], stddev=0.01))
    L11 = tf.reshape(L6, [-1, 3 * 1 * 1024])
    L11 = tf.matmul(L11, W11)
    L11 = tf.nn.relu(L11)
    L11 = tf.nn.dropout(L11, keep_prob)

    W12 = tf.Variable(tf.random_normal([4096, 9], stddev=0.01))
    model = tf.matmul(L11, W12)

    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=model, labels=Y))
    optimizer = tf.train.AdamOptimizer(0.0001).minimize(cost)

    init = tf.global_variables_initializer()
    sess = tf.Session()
    sess.run(init)

    batch_size = 50
    total_batch = 132

    TrainInputBatch = makeBatch(TrainInput,6600,50)
    TrainLabelBatch = makeBatch(TrainLabel,6600,50)

    # TestInputBatch = makeBatch(TestInput,2800,100)
    # TestLabelBatch = makeBatch(TestLabel,2800,100)

    for epoch in range(15):

        total_cost = 0

        for i in range(total_batch):
            batch_xs = TrainInputBatch[i]
            batch_ys = TrainLabelBatch[i]
            batch_xs = batch_xs.reshape(-1, 180, 30, 1)

            # batch_ys = batch_ys.reshape(-1,9)
            _, cost_val = sess.run([optimizer, cost], feed_dict={X: batch_xs, Y: batch_ys, keep_prob: 0.5})

            total_cost += cost_val

        print('Epoch:', '%04d' % (epoch + 1), 'Avg. cost =', '{:.9f}'.format(total_cost / total_batch))

print('최적화 완료!')

is_correct = tf.equal(tf.argmax(model,1),tf.argmax(Y,1))
accuracy = tf.reduce_mean(tf.cast(is_correct,tf.float32))
print ('정확도: ',sess.run(accuracy, feed_dict={X:TestInput.reshape(-1,180,30,1),Y:TestLabel,keep_prob:1}))