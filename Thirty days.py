import tensorflow as tf

a = [1,2,3,4]

depth = 9

b = tf.one_hot(a,depth)

with tf.Session() as sess:
	print (sess.run(b))