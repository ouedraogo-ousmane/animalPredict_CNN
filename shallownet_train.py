# Importation des packages
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from preprocessing.simplepreprocessor import SimplePreprocessor
from preprocessing.imagetoarraypreprocessor import ImageToArrayPreprocessor
from datasets.simpledatasetloader import SimpleDatasetLoader
from nn.conv.shallownet import ShallowNet
from keras.optimizers import SGD
from imutils import paths
import matplotlib.pyplot as plt
import numpy as np
import argparse

# Creation des arguments
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required=True, help="# path to input dataset")
ap.add_argument("-m", "--model", required=True, help="# Path to output model")
args = vars(ap.parse_args())

# Recuperation de la liste des images
print("[INFO] Recuperation de la liste des images ....")
imagesPaths = list(paths.list_images(args["dataset"]))

# Initialisation des fonctions de pretraitements
iap = ImageToArrayPreprocessor()
sp = SimplePreprocessor(32, 32)

# Chargement des données sur le disque et pretraitement
sdl = SimpleDatasetLoader(preprocessors=[sp,iap])
(data, labels) = sdl.loader(imagePaths=imagesPaths, verbose=500)
data = data.astype("float") / 255

# Division des données
(trainX, testX, trainY, testY) = train_test_split(data, labels, test_size=0.25, random_state=42)

# Encodages des labels
encoder =  LabelBinarizer()
trainY = encoder.fit_transform(trainY)
testY = encoder.fit_transform(testY)

# Initialisation de la fonction d'optimisation et compilation du modèle
print("[INFO] compilation du modèle .....")
sgd = SGD(learning_rate=0.005)
model = ShallowNet.build(32, 32, 3, 3)
model.compile(loss="categorical_crossentropy", optimizer=sgd, metrics=["accuracy"])

#Entrainement du modèle
H = model.fit(trainX, trainY, validation_data=(testX, testY), batch_size=32, epochs=100, verbose=1)

# Enregistrement du modèle sur le disque
print("[INFO] Enregistrement du modèle .....")
model.save(args["model"])

# Evaluation des performances du modele
print("[INFO] Evaluation des performances du modele .......")
predictions = model.predict(testX, batch_size=32)
print(classification_report(predictions.argmax(axis=1), 
                            testY.argmax(axis=1),
                            target_names=["cat", "dog", "panda"]))
 
# Visualisation du loss et accuracy
plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0,100), H.history["loss"], label="train_loss")
plt.plot(np.arange(0,100), H.history["val_loss"], label="val_loss")
plt.plot(np.arange(0, 100), H.history["accuracy"], label="train_accuracy")
plt.plot(np.arange(0,100), H.history["val_accuracy"], label="val_accuracy")
plt.xlabel("Nombre d'epoches ")
plt.ylabel(" Loss / Accuracy")
plt.title("Training Loss and Accuracy")
plt.legend()
plt.show()