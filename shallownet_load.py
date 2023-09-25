# Importation des packages
from imutils import paths
import argparse
import matplotlib.pyplot as plt
import numpy as np
from preprocessing.imagetoarraypreprocessor import ImageToArrayPreprocessor
from preprocessing.simplepreprocessor import SimplePreprocessor
from datasets.simpledatasetloader import SimpleDatasetLoader
from keras.models import load_model
import cv2

# Construction des arguments du code
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required=True, help="# path to input dataset")
ap.add_argument("-m", "--model", required=True, help="# Path to pre-trained model")
args = vars(ap.parse_args())

# Les classes des données
classLabels = ["cat", "dog", "panda"]

# Recuperation de quelques images pour les tests
print("[INFO] recuperation de quelques images .........")
imagesPaths = np.array(list(paths.list_images(args["dataset"])))
idx = np.random.randint(0,len(imagesPaths), size=(10,))
imagesPaths = imagesPaths[idx]

# Initialisation des methodes de pretraitement des images
sp = SimplePreprocessor(32,32)
iap = ImageToArrayPreprocessor()

# Chargement des images
sdl = SimpleDatasetLoader(preprocessors=[sp,iap])
(data, labels) = sdl.loader(imagePaths=imagesPaths, verbose=500)
data = data.astype("float") / 255

# Chargement du modele
print("[INFO] chargement du modèle deja entrainé ......")
model = load_model(args["model"])

# Predictions
print("[INFO] predictions des données à partir du modèle ....")
predictions = model.predict(data, batch_size=32).argmax(axis=1)

# Affichages des resultats des predictions
for (i, imagePath) in enumerate(imagesPaths):
    # Lecture de l'images
    image = cv2.imread(imagePath)
    cv2.putText(image, "Animal : {}".format(classLabels[predictions[i]]), (10,30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.imshow("Image", image)
    cv2.waitKey(0)