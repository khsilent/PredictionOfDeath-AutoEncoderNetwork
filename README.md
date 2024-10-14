
# Autoencoder for Heart Failure Prediction

This project implements an autoencoder neural network for the prediction of heart failure events. The model uses a dataset of patients diagnosed with heart disease to predict mortality outcomes. The main objective of the project is to classify patients who survived or died using various techniques, including data preprocessing, class balancing, and cross-validation.

## Table of Contents
- [Introduction](#introduction)
- [Dataset](#dataset)
- [Model Architecture](#model-architecture)
- [Preprocessing](#preprocessing)
- [Training](#training)
- [Evaluation](#evaluation)
- [Results](#results)
- [Requirements](#requirements)
- [How to Run](#how-to-run)
- [References](#references)

## Introduction
Heart disease is one of the leading causes of death globally. This project aims to address the issue of predicting heart failure events using an autoencoder-based neural network.

## Dataset
The dataset used contains records from 299 patients with 12 features, including age, serum creatinine levels, and ejection fraction. The primary target variable is `death_event`, indicating whether the patient survived or passed away.

## Model Architecture
The model consists of an autoencoder neural network, which is used for feature extraction and dimensionality reduction. After preprocessing the data, the features are fed into a fully connected Multi-Layer Perceptron (MLP) for classification. The model uses K-Fold cross-validation for evaluation.

- Encoder: Multiple layers with decreasing node sizes to compress the input.
- Decoder: Mirror the encoder to reconstruct the input.
- MLP: A classifier used to predict the final output based on the learned features.

## Preprocessing
1. Outlier detection and removal using Box Plot method.
2. Balancing the dataset using `RandomOverSampler`.
3. Normalizing the data between 0 and 1.

## Training
The model is trained using a K-Fold Cross-Validation method. Each fold is trained, and performance metrics such as accuracy, recall, precision, and F1-score are calculated.

## Evaluation
Evaluation is done using confusion matrices and averaged metrics across all folds. The model achieves high accuracy and recall values, indicating good performance in predicting heart failure events.

## Results
- **Average Accuracy**: 0.9392
- **Average Recall**: 0.9720
- **Average Precision**: 0.9080
- **Average F1-Score**: 0.9368

## Requirements
To run the project, you need the following libraries:
- TensorFlow
- Keras
- Numpy
- Pandas
- Seaborn

## How to Run
1. Install the required libraries using pip:
    ```
    pip install tensorflow keras numpy pandas seaborn
    ```
2. Clone the repository:
    ```
    git clone https://github.com/alifaramarziorg/Death
    ```
3. Run the model training script:
    ```
    python train_autoencoder.py
    ```

## References
- [Keras](https://keras.io)
- [TensorFlow](https://tensorflow.org)
- [Seaborn](https://seaborn.pydata.org)
- [NumPy](https://numpy.org)
- [Pandas](https://pandas.pydata.org)
