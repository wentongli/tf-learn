import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)


def inference(x1):
    tf.constant_initializer(value=0)
    W = tf.get_variable("W", [784, 10],
                        initializer=tf.constant_initializer())
    b = tf.get_variable("b", [10],
                        initializer=tf.constant_initializer())
    return tf.nn.softmax(tf.matmul(x1, W) + b)


def loss(result1, y):
    dot_product = y * tf.log(result1)
    # Reduction along axis 0 collapses each column into a single
    #  value, whereas reduction along axis 1 collapses each row
    #  into a single value. In general, reduction along axis i
    #  collapses the ith dimension of a tensor to size 1.
    entropy = -tf.reduce_sum(dot_product, reduction_indices=1)
    return tf.reduce_mean(entropy)


def training(cost1, global_step1):
    tf.summary.scalar("cost", cost1)
    optimizer = tf.train.GradientDescentOptimizer(learning_rate)
    return optimizer.minimize(cost, global_step=global_step1)


def evaluate(output1, y):
    correct_prediction = tf.equal(tf.argmax(output1, 1),
                                  tf.argmax(y, 1))
    return tf.reduce_mean(tf.cast(correct_prediction, tf.float32))


# Parameters
learning_rate = 0.01
training_epochs = 1000
batch_size = 1000
display_step = 1

with tf.Graph().as_default():
    # mnist data image of shape 28*28=784
    x = tf.placeholder("float", [None, 784])
    # 0-9 digits recognition => 10 classes
    y = tf.placeholder("float", [None, 10])

    output = inference(x)

    cost = loss(output, y)

    global_step = tf.Variable(0, name='global_step', trainable=False)

    train_op = training(cost, global_step)

    eval_op = evaluate(output, y)

    summary_op = tf.summary.merge_all()

    saver = tf.train.Saver()

    sess = tf.Session()

    summary_writer = tf.summary.FileWriter("logistic_logs/", graph_def=sess.graph_def)

    init_op = tf.initialize_all_variables()
    sess.run(init_op)

    # Training cycle
    for epoch in range(training_epochs):
        avg_cost = 0.
        total_batch = int(mnist.train.num_examples/batch_size)

        # Loop over all batches
        for i in range(1):
            minibatch_x, minibatch_y = mnist.train.next_batch(batch_size)

            # Fit training using batch data
            feed_dict = {x: minibatch_x, y: minibatch_y}
            sess.run(train_op, feed_dict=feed_dict)

            # Compute average loss
            minibatch_cost = sess.run(cost, feed_dict=feed_dict)
            avg_cost += minibatch_cost/total_batch

            # Display logs per epoch step
            if epoch % display_step == 0:
                val_feed_dict = {x: mnist.validation.images, y: mnist.validation.labels}
                accuracy = sess.run(eval_op, feed_dict=val_feed_dict)
                print "Validation Error:", (1 - accuracy)
                summary_str = sess.run(summary_op, feed_dict=feed_dict)
                summary_writer.add_summary(summary_str, sess.run(global_step))
                saver.save(sess, "logistic_logs/model-checkpoint", global_step=global_step)

    print "Optimization Finished!"
    test_feed_dict = {x: mnist.test.images, y: mnist.test.labels}
    accuracy = sess.run(eval_op, feed_dict=test_feed_dict)
    print "Test Accuracy:", accuracy