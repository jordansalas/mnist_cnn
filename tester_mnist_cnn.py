import mnist_cnn
import numpy as np
import tensorflow as tf
import time
from datetime import timedelta

def train_model(net_mnist_cnn,num_iterations,data,session):
    print('\n# PHASE: Training model')
    # Ensure we update the global variable rather than a local copy.
    global total_iterations

    # Start-time used for printing time-usage below.
    start_time = time.time()

    for i in range(total_iterations,
                   total_iterations + num_iterations):

        # Get a batch of training examples.
        # x_batch now holds a batch of images and
        # y_true_batch are the true labels for those images.
        x_batch, y_true_batch = data.next_batch(train_batch_size)

        # Put the batch into a dict with the proper names
        # for placeholder variables in the TensorFlow graph.
        feed_dict_train = {net_mnist_cnn.x: x_batch,
                           net_mnist_cnn.y_true: y_true_batch}

        # Run the optimizer using this batch of training data.
        # TensorFlow assigns the variables in feed_dict_train
        # to the placeholder variables and then runs the optimizer.
        session.run(net_mnist_cnn.optimizer, feed_dict=feed_dict_train)

        # Print status every 100 iterations.
        if i % 100 == 0:
            # Calculate the accuracy on the training-set.
            acc = session.run(net_mnist_cnn.accuracy, feed_dict=feed_dict_train)

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

def test_model(net_mnist_cnn,data,session,show_example_errors=False,
                        show_confusion_matrix=False):
    print('\n# PHASE: TEST model')
    # Split the test-set into smaller batches of this size.
    test_batch_size = 256
    # Number of images in the test-set.
    num_test = len(data.images)

    # Allocate an array for the predicted classes which
    # will be calculated in batches and filled into this array.
    cls_pred = np.zeros(shape=num_test, dtype=np.int)

    # Now calculate the predicted classes for the batches.
    # We will just iterate through all the batches.
    # There might be a more clever and Pythonic way of doing this.

    # The starting index for the next batch is denoted i.
    i = 0

    while i < num_test:
        # The ending index for the next batch is denoted j.
        j = min(i + test_batch_size, num_test)

        # Get the images from the test-set between index i and j.
        images = data.images[i:j, :]

        # Get the associated labels.
        labels = data.labels[i:j, :]

        # Create a feed-dict with these images and labels.
        feed_dict = {net_mnist_cnn.x: images,
                     net_mnist_cnn.y_true: labels}

        # Calculate the predicted class using TensorFlow.
        cls_pred[i:j] = session.run(net_mnist_cnn.y_pred_cls, feed_dict=feed_dict)

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

    # Plot some examples of mis-classifications, if desired.
    if show_example_errors:
        print("Example errors:")
        net_mnist_cnn.plot_example_errors(cls_pred=cls_pred, correct=correct)

    # Plot the confusion matrix, if desired.
    if show_confusion_matrix:
        print("Confusion Matrix:")
        net_mnist_cnn.plot_confusion_matrix(cls_pred=cls_pred)


if __name__ == '__main__':
    # Helper-function to perform optimization iterations
    train_batch_size = 64
    # Counter for total number of iterations performed so far.
    total_iterations = 0

    # Create TensorFlow session
    with tf.Session() as sess:
        net_cnn = mnist_cnn.mnistCNN()
        net_cnn.Build()
        # Initialize variables
        sess.run(tf.global_variables_initializer())

        data_test = net_cnn.data.test
        data_train = net_cnn.data.train

        test_model(net_mnist_cnn=net_cnn,data=data_test,session=sess)
        train_model(net_mnist_cnn=net_cnn,session=sess,num_iterations=100,data=data_train)
        test_model(net_mnist_cnn=net_cnn, data=data_test, session=sess,show_confusion_matrix=True,show_example_errors=True)

    print("FIN!")