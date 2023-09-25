# Projet de Reconnaissance d'Images pour Animaux (Chat, Chien, Panda)

Ce projet utilise un réseau de neurones convolutionnels (CNN) pour la reconnaissance d'images d'animaux, en particulier les chats, les chiens et les pandas.

## Dataset

Nous avons utilisé l'ensemble de données Animals telechargeable sur `Kaggle` qui contient une grande variété d'images de différentes classes, y compris celles de nos trois animaux cibles.

## Architecture du Modèle

Le modèle CNN est composé de couches de convolution, de couches de pooling, de couches denses (fully connected) et une couche de sortie. Cette architecture est conçue pour extraire des caractéristiques pertinentes des images.

## Entraînement

Le modèle a été entraîné sur l'ensemble d'entraînement avec une validation croisée pour éviter le surajustement. L'optimiseur SGD et l'entropie croisée catégorielle ont été utilisés pour l'entraînement.


## Résultats

Le modèle a atteint une précision de 67% sur l'ensemble de test, démontrant son efficacité dans la classification des images de chats, chiens et pandas.

## Remarques

N'hésitez pas à explorer et à adapter ce projet pour vos propres expérimentations en reconnaissance d'images.

## Auteur

OUEDRAOGO Ousmane

oueo5587@gmail.com



