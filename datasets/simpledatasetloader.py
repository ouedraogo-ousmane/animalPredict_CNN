# Importation des librairies
import cv2
import numpy as np
import os

class SimpleDatasetLoader :

    
    def __init__(self, preprocessors =None):
        
        self.preprocessors = preprocessors
        
        if self.preprocessors is None :
            self.preprocessors = []
            
    def loader(self,imagePaths, verbose =-1):
        data = []
        labels = []
        
        for (i,imagePath) in enumerate(imagePaths) :
            image = cv2.imread(imagePath)
            label = imagePath.split(os.path.sep)[-2]
            
            ## On verifie voir si la methode pour le traitement n'est pas null
            
            if self.preprocessors is not None :
                # On parcourt pour chaque methode de traitement puis l'appliquer à notre image
                for p in self.preprocessors:
                    image = p.preprocess(image)
                    
            data.append(image)
            labels.append(label)
            if verbose >0 and i> 0 and (i+1)%verbose==0 :
                print("[INFO] traitement de {} /{}".format(i + 1,len(imagePaths)))
                
        return (np.array(data),np.array(labels))
    
    