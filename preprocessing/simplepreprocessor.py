## Importation des librairies

import cv2

class SimplePreprocessor:
    
    def __init__(self,width, height,inter = cv2.INTER_AREA):
        """Cette classe permet d'instancier la classe

        Args:
            width (_type_): La largeur de l'image
            height (_type_): la hauteur de l'image
            inter (_type_, optional): _description_. Defaults to cv2.INTER_AREA.
        """
        self.width = width
        self.height = height
        self.inter = inter
        
    def preprocess(self,image):
        """Methode pour le redimensionnement d'une image

        Args:
            image (Image): Une image

        Returns:
            _type_: Une nouvelle image redimensionnée à une taille fixé
        """
        
        return cv2.resize(image,(self.width,self.height), interpolation=self.inter)