import tensorflow as tf
import numpy as np
from numpy import array
import csv
import os, glob
from random import *
from pprint import pprint
from tensorflow.examples.tutorials.mnist import input_data


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
    testCsv  = trainCsv[46472400:58087800]

    return array(trainingCsv),array(testCsv)

def makeLabel_level():  # training level and test level

    wholeLabelCsv = getFileName("C:/Users/dw/Desktop/mosquito_cnn/Label_Data/Level/noDateMosq")
    wholeLabel = readCsv(wholeLabelCsv)

    train_label = []
    test_label = []

    train = wholeLabel[:8606]
    test = wholeLabel[8606:10757]

    train = list(map(int, train))
    test = list(map(int, test))

    train = np.array(train)
    test = np.array(test)

    train_ = np.zeros((len(train), 9))
    test_ = np.zeros((len(test), 9))

    train_[np.arange(len(train)), train] = 1
    test_[np.arange(len(test)), test] = 1

    for i in range(len(train_)):
        for j in range(len(train_[i])):
            train_label.append(train_[i][j])

    for p in range(len(test_)):
        for q in range(len(test_[p])):
            test_label.append(test_[p][q])

    return (array(train_label)), (array(test_label))

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
    trainingCsv , testCsv = makeTrainingSet() #len(trainingCsv)=8605
    trainLabel,testLabel = makeLabel_level()#Do i have to change Label data using one-hot encoding?

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

    W3 = tf.Variable(tf.random_normal([45*8*64,256],stddev=0.01))
    L3 = tf.reshape(L2,[-1,45*8*64])
    L3 = tf.matmul(L3,W3)
    L3 = tf.nn.relu(L3)
    L3 = tf.nn.dropout(L3,keep_prob)

    W4 = tf.Variable(tf.random_normal([256,9],stddev=0.01))
    model = tf.matmul(L3,W4)

    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=model, labels=Y))
    #cost = tf.reduce_mean(tf.squared_difference(Y, model))
    optimizer = tf.train.AdamOptimizer(0.001).minimize(cost)

    init = tf.global_variables_initializer()
    sess = tf.Session()
    sess.run(init)

    batch_size = 100
    total_data_len = 10757
    total_batch = int(total_data_len / batch_size)

    img = [trainingCsv[i:i+5400] for i in range(0,len(trainingCsv),5400)]
    label = [trainLabel[j:j+9] for j in range(0,len(trainLabel),9)]

    img = array(img)
    training_img = []
    for i in range(len(img)):
        training_img.append(np.reshape(img[i],(1,5400)))
    training_img = array(training_img)

    label = array(label)
    training_label = []
    for i in range(len(label)):
        training_label.append(np.reshape(label[i],(1,9)))
    training_label = array(training_label)


    test_img = [testCsv[i:i + 5400] for i in range(0, len(testCsv), 5400)]
    test_label = [testLabel[j:j + 9] for j in range(0, len(testLabel), 9)]

    test_img=array(test_img)
    test_label=array(test_label)
    test_label=np.reshape(test_label,(-1,9))

    for epoch in range(10):
        total_cost = 0

        for i in range(8606):
            batch_xs = img[i]
            batch_ys = training_label[i]
            batch_xs = batch_xs.reshape(-1,180,30,1)
            #batch_ys = batch_ys.reshape(-1,9)
            _, cost_val = sess.run([optimizer,cost],feed_dict={X:batch_xs, Y: batch_ys, keep_prob:0.7})

            total_cost += cost_val

        print('Epoch:', '%04d' % (epoch + 1),'Avg. cost =', '{:.9f}'.format(total_cost / total_batch))

print('최적화 완료!')

is_correct = tf.equal(tf.argmax(model,1),tf.argmax(Y,1))
accuracy = tf.reduce_mean(tf.cast(is_correct,tf.float32))
print ('정확도: ',sess.run(accuracy, feed_dict={X:test_img.reshape(-1,180,30,1),Y:test_label,keep_prob:1}))

a = test_new_file("C:/Users/dw/Desktop/mosquito_cnn/Location_allDate/Congress/2012-07-28.csv")
b = test_new_file("C:/Users/dw/Desktop/mosquito_cnn/Location_allDate/Congress/2012-07-29.csv")
c = test_new_file("C:/Users/dw/Desktop/mosquito_cnn/Location_allDate/Congress/2012-07-30.csv")
d = test_new_file("C:/Users/dw/Desktop/mosquito_cnn/Location_allDate/Congress/2012-07-31.csv")
e = test_new_file("C:/Users/dw/Desktop/mosquito_cnn/Location_allDate/Congress/2012-08-01.csv")
f = test_new_file("C:/Users/dw/Desktop/mosquito_cnn/Location_allDate/Congress/2012-08-02.csv")
g = test_new_file("C:/Users/dw/Desktop/mosquito_cnn/Location_allDate/Congress/2012-08-03.csv")
h = test_new_file("C:/Users/dw/Desktop/mosquito_cnn/Location_allDate/Congress/2012-08-04.csv")
i = test_new_file("C:/Users/dw/Desktop/mosquito_cnn/Location_allDate/Congress/2012-08-05.csv")
j = test_new_file("C:/Users/dw/Desktop/mosquito_cnn/Location_allDate/Congress/2012-08-06.csv")
k = test_new_file("C:/Users/dw/Desktop/mosquito_cnn/Location_allDate/Congress/2012-08-07.csv")
l = test_new_file("C:/Users/dw/Desktop/mosquito_cnn/Location_allDate/Congress/2012-08-08.csv")
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
print (sess.run(prediction,feed_dict={X:l ,keep_prob:1.0}))








