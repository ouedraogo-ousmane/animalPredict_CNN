# Importation des librairies
from keras.utils import img_to_array
class ImageToArrayPreprocessor :
    
    def __init__(self, dataFormat=None):
        self.dataFormat = dataFormat

    def preprocess(self, image):

        return img_to_array(image, data_format=self.dataFormat)