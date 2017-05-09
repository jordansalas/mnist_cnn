# Import the converted model's class
from  kaffe.tensorflow import Network

import numpy as np
import random
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
from sklearn.metrics import confusion_matrix
import time
from datetime import timedelta
import math

class MyNet(Network):
    def setup(self):
        (self.feed('data')
             .conv(5, 5, 20, 1, 1, padding='VALID', relu=False, name='conv1')
             .max_pool(2, 2, 2, 2, name='pool1')
             .conv(5, 5, 50, 1, 1, padding='VALID', relu=False, name='conv2')
             .max_pool(2, 2, 2, 2, name='pool2')
             .fc(500, name='ip1')
             .fc(10, relu=False, name='ip2')
             .softmax(name='prob'))

# Genera data para el train, con los indices aleatorios
def gen_data(source):
    while True:
        indices = range(len(source.images))
        random.shuffle(indices)
        for i in indices:
            image = np.reshape(source.images[i], (28, 28, 1))
            label = source.labels[i]
            yield image, label

def gen_data_batch(source):
    data_gen = gen_data(source)

    while True:
        image_batch = []
        label_batch = []
        for _ in range(batch_size):
            image, label = next(data_gen)
            image_batch.append(image)
            label_batch.append(label)
        yield np.array(image_batch), np.array(label_batch)

def train_model(num_iterations,data,session):
    print('\n# PHASE: Training model')
    # Ensure we update the global variable rather than a local copy.
    global total_iterations

    # Start-time used for printing time-usage below.
    start_time = time.time()
    data_gen = gen_data_batch(data)
    for i in range(total_iterations,total_iterations + num_iterations):
        np_images, np_labels = next(data_gen)
        feed = {images: np_images, labels: np_labels}
        session.run(train_op, feed_dict=feed)

        # Print status every 10 iterations.
        if i % 10 == 0:
            # Calculate the accuracy on the training-set.
            acc = session.run(accuracy, feed_dict=feed)

            # Message for printing.
            msg = "Optimization Iteration: {0:>6}, Training Accuracy: {1:>6.1%}"

            # Print it.
            print(msg.format(i + 1, acc))
    # Update the total number of iterations performed.
    total_iterations += num_iterations
    # Ending time.
    end_time = time.time()
    # Difference between start and end-times.
    time_dif = end_time - start_time
    # Print the time-usage.
    print("Time usage: " + str(timedelta(seconds=int(round(time_dif)))))


def test_model(data,session,show_confusion_matrix=False):
    print('\n# PHASE: TEST model')
    # Number of images in the test-set.
    num_test = len(data.images)
    # Allocate an array for the predicted classes which
    # will be calculated in batches and filled into this array.
    cls_pred = np.zeros(shape=num_test, dtype=np.int)

    # The starting index for the next batch is denoted i.
    i = 0
    # k = 0
    # data_gen = gen_data_batch(data)
    while i < num_test:
        # The ending index for the next batch is denoted j.
        j = min(i + batch_size, num_test)
        # Get the images from the test-set between index i and j.
        images_ = data.images[i:j, :]
        images_ = images_.reshape(j-i,28,28,1)
        #print(images_)


        # Get the associated labels.
        labels_ = data.labels[i:j, :]
        #labels_ = labels_.reshape(batch_size, 28, 28, 1)
        # Get the images,labels from the test-set between index i and j.
        #np_images, np_labels = next(data_gen)
        #print(len(np_images[0]))
        feed = {images: images_, labels: labels_}

        # Calculate the predicted class using TensorFlow.
        if(j-i)==batch_size:
            pred_ = session.run(y_pred_cls, feed_dict=feed)
            cls_pred[i:j] = pred_
        else:
            print("Sobras del batch no medidos:",str(j-i))

        # Set the start-index for the next batch to the
        # end-index of the current batch.
        i = j
    # Convenience variable for the true class-numbers of the test-set.
    cls_true = data.cls

    # Create a boolean array whether each image is correctly classified.
    correct = (cls_true == cls_pred)
    # Calculate the number of correctly classified images.
    # When summing a boolean array, False means 0 and True means 1.
    correct_sum = correct.sum()

    # Classification accuracy is the number of correctly classified
    # images divided by the total number of images in the test-set.
    acc = float(correct_sum) / num_test

    # Print the accuracy.
    msg = "Accuracy on Test-Set: {0:.1%} ({1} / {2})"
    print(msg.format(acc, correct_sum, num_test))

    plot_confusion_matrix(data, cls_pred)

# Helper-function to plot confusion matrix
def plot_confusion_matrix(data,cls_pred):
    # This is called from print_test_accuracy() below.

    # cls_pred is an array of the predicted class-number for
    # all images in the test-set.

    # Get the true classifications for the test-set.
    cls_true = data.cls
    num_classes = 10

    # Get the confusion matrix using sklearn.
    cm = confusion_matrix(y_true=cls_true,
                          y_pred=cls_pred)

    # Print the confusion matrix as text.
    print(cm)

    # Plot the confusion matrix as an image.
    plt.matshow(cm)

    # Make various adjustments to the plot.
    plt.colorbar()
    tick_marks = np.arange(num_classes)
    plt.xticks(tick_marks, range(num_classes))
    plt.yticks(tick_marks, range(num_classes))
    plt.xlabel('Predicted')
    plt.ylabel('True')

    # Ensure the plot is shown correctly with multiple plots
    # in a single Notebook cell.
    plt.show()


if __name__ == '__main__':
    # Helper-function to perform optimization iterations
    batch_size = 50
    img_size = 28
    img_size_flat = img_size*img_size
    num_classes = 10
    channel = 1
    # Counter for total number of iterations performed so far.
    total_iterations = 0

    mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
    data_test = mnist.test
    data_train = mnist.train

    # Placeholder variables
    images = tf.placeholder(tf.float32, [batch_size, img_size, img_size, channel])
    labels = tf.placeholder(tf.float32, [batch_size, num_classes])
    net = MyNet({'data': images})

    ip2 = net.layers['ip2']
    pred = tf.nn.softmax(ip2)
    y_pred_cls = tf.argmax(pred, dimension=1)
    y_true_cls = tf.argmax(labels, dimension=1)

    # Cost-function to be optimized
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=ip2, labels=labels)
    loss = tf.reduce_mean(cross_entropy,0)  # cost

    # Optimization Method
    opt = tf.train.RMSPropOptimizer(0.001)
    train_op = opt.minimize(loss)       # Optimizer

    # Performance Measures
    correct_prediction = tf.equal(y_pred_cls, y_true_cls)
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    # Create TensorFlow session
    with tf.Session() as sess:
        # Load the data
        sess.run(tf.global_variables_initializer())
        net.load('mynet.npy', sess)

        data_test.cls = np.argmax(mnist.test.labels, axis=1)

        #train_model(session=sess, num_iterations=10, data=data_train)
        test_model(data=data_test, session=sess)


    print("Hello wolrd")