# TP3: Réseaux de Neurones Convolutifs

Ce projet met en œuvre les concepts fondamentaux des Réseaux de Neurones Convolutifs (CNN) pour des tâches de vision par ordinateur, en utilisant TensorFlow et Keras.

## Contenu du Projet

- **`cnn_classification.py`**: Le script principal contenant l'implémentation de :
  1.  La préparation du jeu de données CIFAR-10.
  2.  La construction et l'entraînement d'un CNN de base pour la classification d'images.
  3.  L'implémentation d'un bloc résiduel (ResNet) pour construire un réseau plus profond.
  4.  La configuration pour le transfert de style neuronal avec un modèle VGG16 pré-entraîné.

- **`rapport.pdf`**: Un rapport synthétique répondant aux questions théoriques du TP sur les architectures CNN et ResNet, ainsi que sur les concepts de segmentation et de détection d'objets.

## Prérequis

- Python 3.8+
- TensorFlow 2.x
- NumPy

## Installation

1.  Clonez le dépôt :
    ```bash
    git clone https://github.com/votre-utilisateur/votre-repo.git
    cd votre-repo
    ```

2.  Créez un environnement virtuel et activez-le :
    ```bash
    python -m venv .venv
    source .venv/bin/activate  # Sur Windows: .venv\Scripts\activate
    ```

3.  Installez les dépendances requises :
    ```bash
    pip install tensorflow numpy
    ```

## Exécution

Pour lancer l'entraînement du CNN de base et la construction du modèle ResNet, exécutez le script suivant :
```bash
python cnn_classification.py
```
Le script affichera la progression de l'entraînement, suivi des métriques d'évaluation sur l'ensemble de test.

## Concepts Abordés

- **Convolution & Pooling**: Opérations fondamentales pour l'extraction de caractéristiques et la réduction de dimensionnalité.
- **Classification d'images**: Architecture CNN séquentielle pour classifier les images du dataset CIFAR-10.
- **Réseaux Résiduels (ResNets)**: Utilisation de connexions résiduelles (skip connections) pour faciliter l'entraînement de réseaux profonds et contrer le problème de la dégradation de la performance.
- **Applications Avancées (conceptuel)**: Introduction à la segmentation d'images, la détection d'objets et le transfert de style.

## Résultats

Le CNN de base atteint une précision d'environ **71%** sur l'ensemble de test CIFAR-10 après 10 époques d'entraînement.