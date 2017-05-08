# https://github.com/Hvass-Labs/TensorFlow-Tutorials/blob/master/02_Convolutional_Neural_Network.ipynb
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import os
from sklearn.metrics import confusion_matrix
import time
from datetime import timedelta

from tensorflow.examples.tutorials.mnist import input_data

class mnistCNN:

    def __init__(self):
        # -------------------- CONFIGURATION OF NEURONAL NETWORK -------------------
        self.configuration_NN_data()
        # ----------------------- LOAD DATA - MNIST_data -----------------------------
        self.load_data()
        # ---------------------------- DATA DIMENSIONS -------------------------
        self.data_dimension_NN()

    def configuration_NN_data(self):
        # Convolutional Layer 1.
        self.filter_size1 = 5  # Convolution filters are 5 x 5 pixels.
        self.num_filters1 = 16  # There are 16 of these filters.

        # Convolutional Layer 2.
        self.filter_size2 = 5  # Convolution filters are 5 x 5 pixels.
        self.num_filters2 = 36  # There are 36 of these filters.

        # Fully-connected layer.
        self.fc_size = 128

    def load_data(self):
        self.path_mnist = 'data/MNIST_data/'
        assert os.path.exists(self.path_mnist), 'No existe el directorio de datos ' + self.path_mnist
        self.data = input_data.read_data_sets(self.path_mnist, one_hot=True)

        self.data.test.cls = np.argmax(self.data.test.labels, axis=1)

    def data_dimension_NN(self):
        # We know that MNIST images are 28 pixels in each dimension.
        self.img_size = 28

        # Images are stored in one-dimensional arrays of this length.
        self.img_size_flat = self.img_size * self.img_size

        # Tuple with height and width of images used to reshape arrays.
        self.img_shape = (self.img_size, self.img_size)

        # Number of colour channels for the images: 1 channel for gray-scale.
        self.num_channels = 1

        # Number of classes, one class for each of 10 digits.
        self.num_classes = 10

    # ---------------------- PLOTTING IMAGES -----------------------
    # Helper-function for plotting images - Max 9 images
    def plot_images(self,images, cls_true, cls_pred=None):
        assert len(images) == len(cls_true) == 9

        # Create figure with 3x3 sub-plots.
        fig, axes = plt.subplots(3, 3)
        fig.subplots_adjust(hspace=0.3, wspace=0.3)

        for i, ax in enumerate(axes.flat):
            # Plot image.
            ax.imshow(images[i].reshape(self.img_shape), cmap='binary')

            # Show true and predicted classes.
            if cls_pred is None:
                xlabel = "True: {0}".format(cls_true[i])
            else:
                xlabel = "True: {0}, Pred: {1}".format(cls_true[i], cls_pred[i])

            # Show the classes as the label on the x-axis.
            ax.set_xlabel(xlabel)

            # Remove ticks from the plot.
            ax.set_xticks([])
            ax.set_yticks([])

        # Ensure the plot is shown correctly with multiple plots
        # in a single Notebook cell.
        plt.show()

    # ----------------------------- TENSORFLOW GRAPH ----------------------
    # Helper-functions for creating new variables
    def new_weights(self,shape):
        return tf.Variable(tf.truncated_normal(shape, stddev=0.05))

    def new_biases(self,length):
        return tf.Variable(tf.constant(0.05, shape=[length]))

    # Helper-function for creating a new Convolutional Layer
    def new_conv_layer(self,
                       input,  # The previous layer.
                       num_input_channels,  # Num. channels in prev. layer.
                       filter_size,  # Width and height of each filter.
                       num_filters,  # Number of filters.
                       use_pooling=True):  # Use 2x2 max-pooling.

        # Shape of the filter-weights for the convolution.
        # This format is determined by the TensorFlow API.
        shape = [filter_size, filter_size, num_input_channels, num_filters]

        # Create new weights aka. filters with the given shape.
        weights = self.new_weights(shape=shape)

        # Create new biases, one for each filter.
        biases = self.new_biases(length=num_filters)

        # Create the TensorFlow operation for convolution.
        # Note the strides are set to 1 in all dimensions.
        # The first and last stride must always be 1,
        # because the first is for the image-number and
        # the last is for the input-channel.
        # But e.g. strides=[1, 2, 2, 1] would mean that the filter
        # is moved 2 pixels across the x- and y-axis of the image.
        # The padding is set to 'SAME' which means the input image
        # is padded with zeroes so the size of the output is the same.
        layer = tf.nn.conv2d(input=input,
                             filter=weights,
                             strides=[1, 1, 1, 1],
                             padding='SAME')

        # Add the biases to the results of the convolution.
        # A bias-value is added to each filter-channel.
        layer += biases

        # Use pooling to down-sample the image resolution?
        if use_pooling:
            # This is 2x2 max-pooling, which means that we
            # consider 2x2 windows and select the largest value
            # in each window. Then we move 2 pixels to the next window.
            layer = tf.nn.max_pool(value=layer,
                                   ksize=[1, 2, 2, 1],
                                   strides=[1, 2, 2, 1],
                                   padding='SAME')

        # Rectified Linear Unit (ReLU).
        # It calculates max(x, 0) for each input pixel x.
        # This adds some non-linearity to the formula and allows us
        # to learn more complicated functions.
        layer = tf.nn.relu(layer)

        # Note that ReLU is normally executed before the pooling,
        # but since relu(max_pool(x)) == max_pool(relu(x)) we can
        # save 75% of the relu-operations by max-pooling first.

        # We return both the resulting layer and the filter-weights
        # because we will plot the weights later.
        return layer, weights

    # Helper-function for flattening a layer
    def flatten_layer(self,layer):
        # Get the shape of the input layer.
        layer_shape = layer.get_shape()

        # The shape of the input layer is assumed to be:
        # layer_shape == [num_images, img_height, img_width, num_channels]

        # The number of features is: img_height * img_width * num_channels
        # We can use a function from TensorFlow to calculate this.
        num_features = layer_shape[1:4].num_elements()

        # Reshape the layer to [num_images, num_features].
        # Note that we just set the size of the second dimension
        # to num_features and the size of the first dimension to -1
        # which means the size in that dimension is calculated
        # so the total size of the tensor is unchanged from the reshaping.
        layer_flat = tf.reshape(layer, [-1, num_features])

        # The shape of the flattened layer is now:
        # [num_images, img_height * img_width * num_channels]

        # Return both the flattened layer and the number of features.
        return layer_flat, num_features

    # Helper-function for creating a new Fully-Connected Layer
    def new_fc_layer(self,input,  # The previous layer.
                     num_inputs,  # Num. inputs from prev. layer.
                     num_outputs,  # Num. outputs.
                     use_relu=True):  # Use Rectified Linear Unit (ReLU)?

        # Create new weights and biases.
        weights = self.new_weights(shape=[num_inputs, num_outputs])
        biases = self.new_biases(length=num_outputs)

        # Calculate the layer as the matrix multiplication of
        # the input and weights, and then add the bias-values.
        layer = tf.matmul(input, weights) + biases

        # Use ReLU?
        if use_relu:
            layer = tf.nn.relu(layer)

        return layer

    def Build(self):
        # Placeholder variables
        self.x = tf.placeholder(tf.float32, shape=[None, self.img_size_flat], name='x')
        x_image = tf.reshape(self.x, [-1, self.img_size, self.img_size, self.num_channels])

        self.y_true = tf.placeholder(tf.float32, shape=[None, 10], name='y_true')
        y_true_cls = tf.argmax(self.y_true, dimension=1)

        # Convolutional Layer 1
        layer_conv1, weights_conv1 = self.new_conv_layer(input=x_image,
                                                    num_input_channels=self.num_channels,
                                                    filter_size=self.filter_size1,
                                                    num_filters=self.num_filters1,
                                                    use_pooling=True)

        # Convolutional Layer 2
        layer_conv2, weights_conv2 =self.new_conv_layer(input=layer_conv1,  # NOTA: Esta funcion retorna dos valores
                                                   num_input_channels=self.num_filters1,
                                                   filter_size=self.filter_size2,
                                                   num_filters=self.num_filters2,
                                                   use_pooling=True)
        # Flatten Layer
        layer_flat, num_features = self.flatten_layer(layer_conv2)

        # Fully-Connected Layer 1
        layer_fc1 = self.new_fc_layer(input=layer_flat,
                                 num_inputs=num_features,
                                 num_outputs=self.fc_size,
                                 use_relu=True)
        # Fully-Connected Layer 2
        layer_fc2 = self.new_fc_layer(input=layer_fc1,
                                 num_inputs=self.fc_size,
                                 num_outputs=self.num_classes,
                                 use_relu=False)

        # Predicted Class
        y_pred = tf.nn.softmax(layer_fc2)
        self.y_pred_cls = tf.argmax(y_pred, dimension=1)

        # Cost-function to be optimized
        cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=layer_fc2,
                                                                labels=self.y_true)
        cost = tf.reduce_mean(cross_entropy)

        # Optimization Method
        self.optimizer = tf.train.AdamOptimizer(learning_rate=1e-4).minimize(cost)

        # Performance Measures
        correct_prediction = tf.equal(self.y_pred_cls, y_true_cls)
        self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    # Helper-function to plot confusion matrix
    def plot_confusion_matrix(self,cls_pred):
        # This is called from print_test_accuracy() below.

        # cls_pred is an array of the predicted class-number for
        # all images in the test-set.

        # Get the true classifications for the test-set.
        cls_true = self.data.test.cls

        # Get the confusion matrix using sklearn.
        cm = confusion_matrix(y_true=cls_true,
                              y_pred=cls_pred)

        # Print the confusion matrix as text.
        print(cm)

        # Plot the confusion matrix as an image.
        plt.matshow(cm)

        # Make various adjustments to the plot.
        plt.colorbar()
        tick_marks = np.arange(self.num_classes)
        plt.xticks(tick_marks, range(self.num_classes))
        plt.yticks(tick_marks, range(self.num_classes))
        plt.xlabel('Predicted')
        plt.ylabel('True')

        # Ensure the plot is shown correctly with multiple plots
        # in a single Notebook cell.
        plt.show()

    # Helper-function to plot example errors
    def plot_example_errors(self,cls_pred, correct):
        # This function is called from print_test_accuracy() below.

        # cls_pred is an array of the predicted class-number for
        # all images in the test-set.

        # correct is a boolean array whether the predicted class
        # is equal to the true class for each image in the test-set.

        # Negate the boolean array.
        incorrect = (correct == False)

        # Get the images from the test-set that have been
        # incorrectly classified.
        images = self.data.test.images[incorrect]

        # Get the predicted classes for those images.
        cls_pred = cls_pred[incorrect]

        # Get the true classes for those images.
        cls_true = self.data.test.cls[incorrect]

        # Plot the first 9 images.
        self.plot_images(images=images[0:9],
                    cls_true=cls_true[0:9],
                    cls_pred=cls_pred[0:9])


