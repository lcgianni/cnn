from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf


LOGDIR = './tensorflow_logs/mnist_deep'

def weight_variables(shape):
    """ generate a weight variable of a given shape """
    initial = tf.truncated_normal(shape, stddev=0.1) # will initialize randmoly based on normal distribution
    return tf.Variable(initial, name='weight')

def bias_variable(shape):
    """ generate a bias variable of a given shape """
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial, name='bias')

def main():
    mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

    # placeholder for image data
    x = tf.placeholder(tf.float32, [None, 784], name='x')
    # placeholder for image labels
    y_ = tf.placeholder(tf.float32, [None, 10], name='labels')

    with tf.name_scope('reshape'):
        x_image = tf.reshape(x, [-1, 28, 28, 1])
        tf.summary.image('input', x_image, 4)

    # convolutional layer - maps one grayscale image to 32 features
    with tf.name_scope('conv1'):
        W_conv1 = weight_variables([5, 5, 1, 32])
        b_conv1 = bias_variable([32])
        x_conv1 = tf.nn.conv2d(x_image, W_conv1, strides=[1, 1, 1, 1], padding='SAME')
        h_conv1 = tf.nn.relu(x_conv1 + b_conv1)

    # pooling layer - downsampling by 2x
    with tf.name_scope('pool1'):
        h_pool1 = tf.nn.max_pool(h_conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

    # second convolutional layer - maps 32 feature maps to 64
    with tf.name_scope('conv2'):
        W_conv2 = weight_variables([5, 5, 32, 64])
        b_conv2 = bias_variable([64])
        x_conv2 = tf.nn.conv2d(h_pool1, W_conv2, strides=[1, 1, 1, 1], padding='SAME')
        h_conv2 = tf.nn.relu(x_conv2 + b_conv2)

    # second pooling layer
    with tf.name_scope('pool2'):
        h_pool2 = tf.nn.max_pool(h_conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

    # after downsampling twice, the 28x28 image is 7x7 with 64 feature maps
    # fully connected layer
    with tf.name_scope('flatten'):
        h_pool_flat = tf.reshape(h_pool2, [-1, 7*7*64])
        W_fc1 = weight_variables([7*7*64, 1024])
        b_fc1 = bias_variable([1024])
        h_fc1 = tf.nn.relu(tf.matmul(h_pool_flat, W_fc1) + b_fc1)

    # dropout - control complexity of model to avoid overfitting
    with tf.name_scope('dropout'):
        keep_prob = tf.placeholder(tf.float32)
        h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

    # map the features to 10 classes (one for each label)
    with tf.name_scope('fc-classify'):
        W_fc2 = weight_variables([1024, 10])
        b_fc2 = bias_variable([10])
        y = tf.matmul(h_fc1_drop, W_fc2) + b_fc2

    # define loss
    with tf.name_scope('loss'):
        cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y), name='cross_entropy')
        tf.summary.histogram('loss', cross_entropy)

    # define optimizer
    with tf.name_scope('optimizer'):
        train_step = tf.train.AdamOptimizer(0.0001).minimize(cross_entropy, name='train_step')

    # define accuracy
    with tf.name_scope('accuracy'):
        correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1)) # highest probability is label predicted
        correct_prediction = tf.cast(correct_prediction, tf.float32, name='correct_prediction') # from array of boolean to array of 0s and 1s
        accuracy = tf.reduce_mean(correct_prediction, name='accuracy')
        tf.summary.scalar('accuracy', accuracy)

    # launch session
    sess = tf.InteractiveSession()

    # initialize variables (common error!!)
    tf.global_variables_initializer().run()

    # merge all summary data to avoid having to do it manually for each summary
    merged = tf.summary.merge_all()

    # create summary writer
    writer = tf.summary.FileWriter(LOGDIR, sess.graph)

    # train model
    for i in range(2000):
        batch = mnist.train.next_batch(100) # size of batch is 100. batch of 1 generates instability.
        sess.run(train_step, feed_dict={x: batch[0], y_: batch[1], keep_prob: 1.0})
        if i % 5 == 0:
            summary = sess.run(merged, feed_dict={x: batch[0], y_: batch[1], keep_prob: 1.0})
            writer.add_summary(summary, i)
        if i % 100 == 0:
            train_accuracy = sess.run(accuracy, feed_dict={x: batch[0], y_: batch[1], keep_prob: 1.0})
            print("Step %d, Training Accuracy %g" % (i, float(train_accuracy)))


    # print results on test_set
    print("Tests Accuracy %g" % sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0}))

    # close summary writer
    writer.close()

if __name__ == '__main__':
    main()
