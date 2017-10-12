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
	for folder in os.listdir("C:/Users/dw/Desktop/mosquito_cnn/Location_allDate"):
		if folder == '.DS_Store':
			continue
		else:
			location.append(folder)#all location

	for i in range(len(location)):
		trainFolder.append(getFileName("C:/Users/dw/Desktop/mosquito_cnn/Location_allDate/"+str(location[i])))

	for j in range(len(trainFolder)):
		for k in range(len(trainFolder[j])):
			trainFolder[j][k] = "C:/Users/dw/Desktop/mosquito_cnn/Location_allDate/"+str(location[j])+"/"+str(trainFolder[j][k])

	for p in range(len(trainFolder)):
		for q in range(len(trainFolder[p])):
			with open(trainFolder[p][q],'r') as f:
				reader = csv.reader(f,delimiter=',')
				for row in reader:
					trainCsv.append(row)

	# for index in range(len(trainCsv)):
	# 	wholeCsv.append([[float(y) for y in x] for x in trainCsv[index]])
    #
	# trainingCsv = trainCsv[:int(len(trainCsv)*0.8)]
	# testCsv  = trainCsv[int(len(trainCsv)*0.8):len(trainCsv)]

	return 

def makeLabel_level():#training level and test level 

	train = []
	test = []
	wholeLabelCsv = getFileName("C:/Users/dw/Desktop/mosquito_cnn/Label_Data/Level/noDateMosq")
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
	
	# trainingCsv , testCsv = makeTrainingSet() #len(trainingCsv)=8605
	# trainLabel,testLabel = makeLabel_level()#Do i have to change Label data using one-hot encoding?

	print(makeTrainingSet())

# 	X = tf.placeholder(tf.float32, shape=[None,180,30,1])
# 	Y = tf.placeholder(tf.float32, shape=[None,9])
# 	keep_prob = tf.placeholder(tf.float32)
#
# 	W1 = tf.Variable(tf.random_normal([3,3,1,32],stddev=0.01))
# 	L1 = tf.nn.conv2d(X,W1,strides=[1,1,1,1], padding='SAME')
# 	L1 = tf.nn.relu(L1)
# 	L1 = tf.nn.max_pool(L1,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')
#
# 	W2 = tf.Variable(tf.random_normal([3,3,32,64],stddev=0.01))
# 	L2 = tf.nn.conv2d(L1,W2,strides=[1,1,1,1],padding='SAME')
# 	L2 = tf.nn.relu(L2)
# 	L2 = tf.nn.max_pool(L2,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')
#
# 	W3 = tf.Variable(tf.random_normal([45*8*64,256],stddev=0.01))
# 	L3 = tf.reshape(L2,[-1,45*8*64])
# 	L3 = tf.matmul(L3,W3)
# 	L3 = tf.nn.relu(L3)
# 	L3 = tf.nn.dropout(L3,keep_prob)
#
# 	W4 = tf.Variable(tf.random_normal([256,9],stddev=0.01))
# 	model = tf.matmul(L3,W4)
#
# 	cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=model, labels=Y))
# 	optimizer = tf.train.AdamOptimizer(0.001).minimize(cost)
#
# 	init = tf.global_variables_initializer()
# 	sess = tf.Session()
# 	sess.run(init)
#
# 	batch_size = 100
# 	total_data_len = 10757
# 	total_batch = int(total_data_len / batch_size)
#
# 	for epoch in range(15):
# 		total_cost = 0
#
# 		for i in range(total_batch):
# 			batch_xs = trainingCsv[i]
# 			batch_ys = trainLabel[i]
#
# 			batch_xs = batch_xs.reshape(1,180,30,1)
# 			_, cost_val = sess.run([optimizer,cost],feed_dict={X:batch_xs, Y: batch_ys, keep_prob:0.7})
#
# 			total_cost += cost_val
#
# 		print('Epoch:', '%04d' % (epoch + 1),
# 			  'Avg. cost =', '{:.3f}'.format(total_cost / total_batch))
#
# print('최적화 완료!')

	