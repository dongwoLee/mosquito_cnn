import tensorflow as tf
import numpy as np
from numpy import array
import csv
import os, glob

def getFileName(dir):
    os.chdir(dir)
    dateList = []
    for file in glob.glob("*.csv"):
        dateList.append(file)

    return dateList


def readCsv(csvlist):  # label data should be made by this function.

    dataSet = []
    for i in range(len(csvlist)):
        with open(csvlist[i], 'r') as f:
            reader = csv.reader(f, delimiter=',')
            for row in reader:
                dataSet.append(''.join(row))

    return dataSet

def makeTrainingSet():  # training and test mosq

    location = []
    trainFolder = []
    trainCsv = []
    wholeCsv = []
    for folder in os.listdir("C:/Users/dw/Desktop/mosquito_cnn/Location_allDate"):
        if folder == '.DS_Store':
            continue
        else:
            location.append(folder)  # all location

    for i in range(len(location)):
        trainFolder.append(getFileName("C:/Users/dw/Desktop/mosquito_cnn/Location_allDate/" + str(location[i])))

    for j in range(len(trainFolder)):
        for k in range(len(trainFolder[j])):
            trainFolder[j][k] = "C:/Users/dw/Desktop/mosquito_cnn/Location_allDate/" + str(location[j]) + "/" + str(trainFolder[j][k])

    for p in range(len(trainFolder)):
        for q in range(len(trainFolder[p])):
            with open(trainFolder[p][q], 'r') as f:
                reader = csv.reader(f, delimiter=',')
                for row in reader:
                    for element in row:
                        trainCsv.append(float(element))

    trainingCsv = trainCsv[:46472400]
    testingCsv  = trainCsv[46472400:58087800]

    trainCsv = [trainingCsv[i:i+5400] for i in range(0,len(trainingCsv),5400)]
    testCsv = [testingCsv[i:i+5400] for i in range(0,len(testingCsv),5400)]

    return (array(trainCsv)),(array(testCsv))

def makeLabel_level():  # training level and test level

    wholeLabelCsv = getFileName("C:/Users/dw/Desktop/mosquito_cnn/Label_Data/Level/noDateMosq")
    wholeLabel = readCsv(wholeLabelCsv)

    nb_classes = 9
    train = wholeLabel[:8606]
    test = wholeLabel[8606:10757]

    train_ = list(map(int, train))
    test_ = list(map(int, test))

    tmp_train_ = tf.one_hot(train_,nb_classes)
    tmp_test_ = tf.one_hot(test_,nb_classes)

    with tf.Session() as sess:
        one_hot_train_label = sess.run(tmp_train_)
        one_hot_test_label = sess.run(tmp_test_)

    return (one_hot_train_label), (one_hot_test_label)

def makeBatchInput(rawList,arrayLength,batchSize):

    inputList = []
    for i in range(0,arrayLength,batchSize):
        inputList.append(rawList[i:i+batchSize])
    inputList=array(inputList)

    return inputList

def test_new_file(file):
    testNew = []

    with open(file, "r") as f:
        reader = csv.reader(f, delimiter=',')
        for row in reader:
            for element in row:
                testNew.append(float(element))

    testNew = array(testNew)
    testNew =testNew.reshape(-1,180,30,1)

    return testNew

if __name__ == '__main__':
    trainingCsv_Data,testingCsv_Data = makeTrainingSet()
    trainLabel_Data, testLabel_Data = makeLabel_level()

    X = tf.placeholder(tf.float32, shape=[None,180,30,1])
    Y = tf.placeholder(tf.float32, shape=[None,9])
    keep_prob = tf.placeholder(tf.float32)

    W1 = tf.Variable(tf.random_normal([5,5,1,32],stddev=0.01))
    L1 = tf.nn.conv2d(X,W1,strides=[1,1,1,1], padding='SAME')
    L1 = tf.nn.relu(L1)
    L1 = tf.nn.max_pool(L1,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')

    W2 = tf.Variable(tf.random_normal([5,5,32,64],stddev=0.01))
    L2 = tf.nn.conv2d(L1,W2,strides=[1,1,1,1],padding='SAME')
    L2 = tf.nn.relu(L2)
    L2 = tf.nn.max_pool(L2,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')

    W3 = tf.Variable(tf.random_normal([5,5,64,128],stddev=0.01))
    L3 = tf.nn.conv2d(L2,W3,strides=[1,1,1,1],padding='SAME')
    L3 = tf.nn.relu(L3)
    L3 = tf.nn.max_pool(L3,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')

    W4 = tf.Variable(tf.random_normal([5,5,128,256],stddev=0.01))
    L4 = tf.nn.conv2d(L3,W4,strides=[1,1,1,1],padding='SAME')
    L4 = tf.nn.relu(L4)
    L4 = tf.nn.max_pool(L4,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')

    W5 = tf.Variable(tf.random_normal([5,5,256,512],stddev=0.01))
    L5 = tf.nn.conv2d(L4,W5,strides=[1,1,1,1],padding='SAME')
    L5 = tf.nn.relu(L5)
    L5 = tf.nn.max_pool(L5,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')

    W6 = tf.Variable(tf.random_normal([5,5,512,1024],stddev=0.01))
    L6 = tf.nn.conv2d(L5,W6,strides=[1,1,1,1],padding='SAME')
    L6 = tf.nn.relu(L6)
    L6 = tf.nn.max_pool(L6,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')

    W7 = tf.Variable(tf.random_normal([5,5,1024,2048],stddev=0.01))
    L7 = tf.nn.conv2d(L6,W7,strides=[1,1,1,1],padding='SAME')
    L7 = tf.nn.relu(L7)
    L7 = tf.nn.max_pool(L7,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')

    # W8 = tf.Variable(tf.random_normal([5,5,2048,4096],stddev=0.01))
    # L8 = tf.nn.conv2d(L7,W8,strides=[1,1,1,1],padding='SAME')
    # L8 = tf.nn.relu(L8)
    # L8 = tf.nn.max_pool(L8,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')

    W11 = tf.Variable(tf.random_normal([2*1*2048,8192],stddev=0.01))
    L11 = tf.reshape(L7,[-1,2*1*2048])
    L11 = tf.matmul(L11,W11)
    L11 = tf.nn.relu(L11)
    L11 = tf.nn.dropout(L11,keep_prob)

    W12 = tf.Variable(tf.random_normal([8192,9],stddev=0.01))
    model = tf.matmul(L11,W12)

    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=model, labels=Y))
    optimizer = tf.train.AdamOptimizer(0.0001).minimize(cost)

    init = tf.global_variables_initializer()
    sess = tf.Session()
    sess.run(init)

    batch_size = 50
    total_batch = 172

    training_img_batch = makeBatchInput(trainingCsv_Data,8600,50)
    training_label_batch = makeBatchInput(trainLabel_Data,8600,50)
    # testing_img_batch = makeBatchInput(testingCsv_Data,2100,100)
    # testing_label_batch = makeBatchInput(testLabel_Data,2100,100)

    # print (training_img_batch[0].shape,training_label_batch[0].shape)

    for epoch in range(10):

        total_cost = 0

        for i in range(total_batch):
            batch_xs = training_img_batch[i]
            batch_ys = training_label_batch[i]
            batch_xs = batch_xs.reshape(-1,180,30,1)

            # batch_ys = batch_ys.reshape(-1,9)
            _, cost_val = sess.run([optimizer,cost],feed_dict={X:batch_xs, Y: batch_ys, keep_prob:0.5})

            total_cost += cost_val

        print('Epoch:', '%04d' % (epoch + 1),'Avg. cost =', '{:.9f}'.format(total_cost / total_batch))

print('최적화 완료!')

is_correct = tf.equal(tf.argmax(model,1),tf.argmax(Y,1))
accuracy = tf.reduce_mean(tf.cast(is_correct,tf.float32))
print ('정확도: ',sess.run(accuracy, feed_dict={X:testingCsv_Data.reshape(-1,180,30,1),Y:testLabel_Data,keep_prob:1}))

a = test_new_file("C:/Users/dw/Desktop/mosquito_cnn/Location_allDate/Yoonjung_Elementary/2015-08-28.csv")
b = test_new_file("C:/Users/dw/Desktop/mosquito_cnn/Location_allDate/Yoonjung_Elementary/2015-08-29.csv")
c = test_new_file("C:/Users/dw/Desktop/mosquito_cnn/Location_allDate/Yoonjung_Elementary/2015-08-30.csv")
d = test_new_file("C:/Users/dw/Desktop/mosquito_cnn/Location_allDate/Yoonjung_Elementary/2015-08-31.csv")
e = test_new_file("C:/Users/dw/Desktop/mosquito_cnn/Location_allDate/Yoonjung_Elementary/2015-09-01.csv")
f = test_new_file("C:/Users/dw/Desktop/mosquito_cnn/Location_allDate/Yoonjung_Elementary/2015-09-02.csv")
g = test_new_file("C:/Users/dw/Desktop/mosquito_cnn/Location_allDate/Yoonjung_Elementary/2015-09-03.csv")
h = test_new_file("C:/Users/dw/Desktop/mosquito_cnn/Location_allDate/Yoonjung_Elementary/2015-09-04.csv")
i = test_new_file("C:/Users/dw/Desktop/mosquito_cnn/Location_allDate/Yoonjung_Elementary/2015-09-05.csv")
j = test_new_file("C:/Users/dw/Desktop/mosquito_cnn/Location_allDate/Yoonjung_Elementary/2015-09-06.csv")
k = test_new_file("C:/Users/dw/Desktop/mosquito_cnn/Location_allDate/Yoonjung_Elementary/2015-09-07.csv")
l = test_new_file("C:/Users/dw/Desktop/mosquito_cnn/Location_allDate/Yoonjung_Elementary/2015-09-08.csv")
m = test_new_file("C:/Users/dw/Desktop/mosquito_cnn/Location_allDate/Yoonjung_Elementary/2015-09-09.csv")
n = test_new_file("C:/Users/dw/Desktop/mosquito_cnn/Location_allDate/Yoonjung_Elementary/2015-09-10.csv")

prediction = tf.argmax(model,1)

print (sess.run(prediction,feed_dict={X:a,keep_prob:1.0}))
print (sess.run(prediction,feed_dict={X:b,keep_prob:1.0}))
print (sess.run(prediction,feed_dict={X:c,keep_prob:1.0}))
print (sess.run(prediction,feed_dict={X:d,keep_prob:1.0}))
print (sess.run(prediction,feed_dict={X:e,keep_prob:1.0}))
print (sess.run(prediction,feed_dict={X:f,keep_prob:1.0}))
print (sess.run(prediction,feed_dict={X:g,keep_prob:1.0}))
print (sess.run(prediction,feed_dict={X:h,keep_prob:1.0}))
print (sess.run(prediction,feed_dict={X:i,keep_prob:1.0}))
print (sess.run(prediction,feed_dict={X:j,keep_prob:1.0}))
print (sess.run(prediction,feed_dict={X:k,keep_prob:1.0}))
print (sess.run(prediction,feed_dict={X:l,keep_prob:1.0}))
print (sess.run(prediction,feed_dict={X:m,keep_prob:1.0}))
print (sess.run(prediction,feed_dict={X:n,keep_prob:1.0}))








