import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten
from tensorflow.keras.utils import to_categorical

class Models:
    """The definition of all TensorFlow convolutional Neural Networks used 
    in this project.
    """

    def __init__(self, input_shape=(128, 128, 3), n_classes = None):
        """constructor
        If the number of classess is given, it means that the network uses a ``softmax`` in the output layer and the problem 
        is treated as a classification problem, otherwise a single value between -1 and 1 is returned given
        the ``tanh`` function in the last fully connected layer.

        :param input_shape: the size of input images, defaults to ``(128, 128, 3)``
        :type input_shape: tuple, optional
        :param n_classes: number of output classess in the case of classification, defaults to None
        :type n_classes: ``int``, optional
        """
        
        self.input_shape = input_shape
        self.n_classes   = n_classes

    def getModel(self, model_name):
        """Given the model name, this returns a function that instantiates the desired model.

        :param model_name: name of the desired model
        :type model_name: ``str``
        :return: the model
        :rtype: ``function``
        """

        if model_name == 'model4':
            return self.model4
        if model_name == 'model5':
            return self.model5
        if model_name == 'model6':
            return self.model6

    ## n_classes is None if the network is used for regression
    def model4(self):
        """This is the simplest model in our series of models. 
        The total number of weight number of this model is ``~1.600,000``. It has two sets of double convolutional layers.

        :return: a ``TensorFlow`` model
        :rtype: ``tf.keras.models``
        """

        model = tf.keras.models.Sequential()
        model.add(tf.keras.layers.Conv2D(16, (3, 3), activation='relu', input_shape=self.input_shape, name='conv1'))
        model.add(tf.keras.layers.Conv2D(16, (3, 3), activation='relu', name='conv2'))
        model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))
        model.add(tf.keras.layers.Dropout(0.25))

        model.add(tf.keras.layers.Conv2D(32, (3, 3), activation='relu', name='conv3'))
        model.add(tf.keras.layers.Conv2D(32, (3, 3), activation='relu', name='conv4'))
        model.add(tf.keras.layers.BatchNormalization(axis = 3, name = 'bn_2'))
        model.add(tf.keras.layers.MaxPooling2D(pool_size=(4, 4)))
        model.add(tf.keras.layers.Dropout(0.25))

        ## Fully connected layers
        model.add(tf.keras.layers.Flatten())
        model.add(tf.keras.layers.Dense(256, activation='relu', name='fc_2'))
        model.add(tf.keras.layers.Dense(32, activation='relu', name='fc_4'))
        model.add(tf.keras.layers.Dense(8, activation='relu', name='fc_5'))

        # output layer
        if self.n_classes is None:
            model.add(tf.keras.layers.Dense(1, activation='tanh', name='fc_out2'))
        else:
            model.add(tf.keras.layers.Dense(self.n_classes, activation='softmax'))

        return model

    #################################################################################################################

    ## n_classes is None if the network is used for regression
    def model5(self):
        """This model is the most complex model in our study. 
        It has ``~2,500,000`` free parameters and three sets of double convolutional layers.

        :return: a ``TensorFlow`` model
        :rtype: ``tf.keras.models``
        """

        model = tf.keras.models.Sequential()

        model.add(tf.keras.layers.Conv2D(128, (3, 3), activation='relu', input_shape=self.input_shape, name='conv1'))
        model.add(tf.keras.layers.Conv2D(128, (3, 3), activation='relu', name='conv2'))
        model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))
        model.add(tf.keras.layers.Dropout(0.25))

        model.add(tf.keras.layers.Conv2D(256, (3, 3), activation='relu', name='conv3'))
        model.add(tf.keras.layers.Conv2D(256, (3, 3), activation='relu', name='conv4'))
        model.add(tf.keras.layers.BatchNormalization(axis = 3, name = 'bn_2'))
        model.add(tf.keras.layers.MaxPooling2D(pool_size=(4, 4)))
        model.add(tf.keras.layers.Dropout(0.25))
        
        model.add(tf.keras.layers.Conv2D(256, (3, 3), activation='relu', name='conv5'))
        model.add(tf.keras.layers.Conv2D(256, (3, 3), activation='relu', name='conv6'))
        model.add(tf.keras.layers.BatchNormalization(axis = 3, name = 'bn_3'))
        model.add(tf.keras.layers.MaxPooling2D(pool_size=(4, 4)))

        ## Fully connected layers
        model.add(tf.keras.layers.Flatten())
        model.add(tf.keras.layers.Dense(256, activation='relu', name='fc_2'))
        model.add(tf.keras.layers.Dense(128, activation='relu', name='fc_3')) 
        model.add(tf.keras.layers.Dense(64, activation='relu', name='fc_4'))
        model.add(tf.keras.layers.Dense(8, activation='relu', name='fc_5'))
        
        # output layer
        if self.n_classes is None:
            model.add(tf.keras.layers.Dense(1, activation='tanh', name='fc_out2'))
        else:
            model.add(tf.keras.layers.Dense(self.n_classes, activation='softmax'))

        return model

    #################################################################################################################

    ## n_classes is None if the network is used for regression
    def model6(self):
        """This model is comparable to ``model4`` in terms of complexity, 
        although the number of convolutional units is larger in this model.

        :return: a ``TensorFlow`` model
        :rtype: ``tf.keras.models``
        """

        model = tf.keras.models.Sequential()

        model.add(tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=self.input_shape, name='conv1'))
        model.add(tf.keras.layers.Conv2D(32, (3, 3), activation='relu', name='conv2'))
        model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))
        model.add(tf.keras.layers.Dropout(0.25))

        model.add(tf.keras.layers.Conv2D(64, (3, 3), activation='relu', name='conv3'))
        model.add(tf.keras.layers.Conv2D(64, (3, 3), activation='relu', name='conv4'))
        model.add(tf.keras.layers.BatchNormalization(axis = 3, name = 'bn_2'))
        model.add(tf.keras.layers.MaxPooling2D(pool_size=(4, 4)))
        model.add(tf.keras.layers.Dropout(0.25))

        ## Fully connected layers
        model.add(tf.keras.layers.Flatten())
        model.add(tf.keras.layers.Dense(128, activation='relu', name='fc_3'))
        model.add(tf.keras.layers.Dense(64, activation='relu', name='fc_4'))
        model.add(tf.keras.layers.Dense(8, activation='relu', name='fc_5'))

        # output layer
        # output layer
        if self.n_classes is None:
            model.add(tf.keras.layers.Dense(1, activation='tanh', name='fc_out2'))
        else:
            model.add(tf.keras.layers.Dense(self.n_classes, activation='softmax'))

        return model

    #################################################################################################################