# Importation des librairies
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from preprocessing.imagetoarraypreprocessor import ImageToArrayPreprocessor
from nn.conv.shallownet import ShallowNet
from preprocessing.simplepreprocessor import SimplePreprocessor
from datasets.simpledatasetloader import SimpleDatasetLoader
from keras.optimizers import SGD
from imutils import paths
import argparse
import matplotlib.pyplot as plt
import numpy as np

# Construction des parametres 
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset",required=True, help="# path to the input dataset")
args = vars(ap.parse_args())

# Recuperation des images
print("[INFO] chargement des images ....")
imagePaths = list(paths.list_images(args["dataset"]))

# Initialisation du traitement des images
sp = SimplePreprocessor(32,32)
iap =ImageToArrayPreprocessor()

# Chargement des données
sdl = SimpleDatasetLoader(preprocessors=[sp, iap])
(data, labels) = sdl.loader(imagePaths, verbose=500)
data = data.astype("float")  / 255


# Partitionnement des données
(trainX, testX, trainY, testY) = train_test_split(data, labels, random_state=42, test_size=0.25)

# Encodage des données
encoder = LabelBinarizer()
trainY = encoder.fit_transform(trainY)
testY = encoder.fit_transform(testY)

# Initialisation de SGD

sgd = SGD(0.005)

model = ShallowNet.build(width=32, height=32, depth=3, classes=3)
model.compile(loss="categorical_crossentropy", optimizer=sgd, metrics=["accuracy"])

# Entrainement du modèle
print("[INFO] entrainement du modèle . ...")
H = model.fit(trainX, trainY, validation_data=(testX,testY),epochs=100, batch_size=32, verbose=1)

# Evaluation du modèle
print("[INFO] evaluation du modèle .....")
predictions = model.predict(testX, batch_size=32)
print(classification_report(predictions.argmax(axis=1), testY.argmax(axis=1),target_names=["cat", "dog", "panda"]))

# Visualisation

print("[INFO] visualisation graphique...........")
plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0,100), H.history["loss"],label="train_loss")
plt.plot(np.arange(0,100), H.history["val_loss"],label="val_loss")
plt.plot(np.arange(0,100), H.history["accuracy"], label="accuracy")
plt.plot(np.arange(0,100), H.history["val_accuracy"], label="val_accuracy")
plt.title("Training loss and accuracy")
plt.xlabel("# Epochs")
plt.ylabel("Loss / Accuracy")
plt.legend()
plt.show()