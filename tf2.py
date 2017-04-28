import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)


for _ in range(1000):
    batch_xs, batch_ys = mnist.train.next_batch(100)
    sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})


'''
--- Tensorflow variable --- 
'''
weights = tf.Variable(tf.random_normal([300, 200], stddev=0.5), name="weights")

'''
# Common tensors from the TensorFlow API docs

tf.zeros(shape, dtype=tf.float32, name=None)
tf.ones(shape, dtype=tf.float32, name=None)
tf.random_normal(shape, mean=0.0, stddev=1.0, dtype=tf.float32,
                 seed=None, name=None)
tf.truncated_normal(shape, mean=0.0, stddev=1.0, dtype=tf.float32,
                    seed=None, name=None)
tf.random_uniform(shape, minval=0, maxval=None, dtype=tf.float32,
                  seed=None, name=None)
'''
print weights


'''
--- Tensorflow Placeholder --- 
'''




'''
--- Tensorflow Session --- 
'''