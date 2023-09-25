# Importation des librairies
from keras.models import Sequential
from keras.layers.convolutional import Conv2D
from keras.layers.core import Dense
from keras.layers.core import Flatten
from keras.layers.core import Activation
from keras import backend as K

# creation d'une classe

class ShallowNet:
    @staticmethod
    def build(width, height, depth, classes):

        model =  Sequential()
        inputShape = (height, width, depth)

        # Utilisation de channel first

        if K.image_data_format() == "channels_first":
            inputShape = (depth, height, width)

        # Creation des couches

        model.add(Conv2D(32,(3,3), padding="same", input_shape=inputShape))
        model.add(Activation("relu"))
        model.add(Flatten())
        model.add(Dense(classes))
        model.add(Activation("softmax"))

        return model