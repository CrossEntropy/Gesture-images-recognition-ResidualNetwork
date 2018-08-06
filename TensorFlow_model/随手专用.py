import tensorflow as tf
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

a = tf.Variable(1, dtype=tf.int32)
sess = tf.Session()
sess.run(a.assign(20))
print(sess.run(a))
