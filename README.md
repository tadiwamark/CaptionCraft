# Caption Craft

This repository contains a sophisticated Image Captioning Model developed in Python using TensorFlow. It generates descriptive captions for videos by leveraging a dataset of image-caption pairs and training a deep learning model on it.

## Table of Contents

- [Introduction](#introduction)
- [Installation](#installation)
- [Usage](#usage)
- [Model Architecture](#model-architecture)
- [Project Structure](#project-structure)
- [License](#license)

## Introduction

The model is trained on the `coco-2017-dataset` from Kaggle, utilizing a combination of image processing and natural language processing techniques. It uses the features extracted from images and the tokenized form of captions to train and subsequently generate captions for new, unseen images.

## Installation

To set up and run the model, please follow the steps below:

1. Install the necessary libraries and dependencies:
   ```sh
   pip install tensorflow numpy pandas matplotlib keras tqdm nltk
   ```
2. Download the dataset from [Kaggle](https://www.kaggle.com/awsaf49/coco-2017-dataset) and place it in the appropriate directory.

## Usage

1. **Setup Kaggle API Credentials**: Upload your `kaggle.json` file and set up the Kaggle directory in your runtime environment, especially if you're using Google Colab.
2. **Update Paths**: Ensure the `image_path` and `annotations_path` variables in the code point to the correct locations of your dataset.
3. **Run the Script**: Execute the Python script in an environment with TensorFlow installed, preferably with GPU support for better performance.
4. **Generate Captions**: After training, you can use the model to generate captions for any new images using the `generate_caption` function.

## Model Architecture

The model consists of two main components:
1. **Image Model**: Processes the image features, using a Dense layer with ReLU activation and a Dropout layer for regularization.
2. **Text Model**: Handles the processing of text captions through an Embedding layer, an LSTM layer, a Dense layer with ReLU activation, and another Dropout layer for regularization.

The final architecture involves the concatenation of these models and connection to a softmax activation layer. The model employs MobileNetV2 for feature extraction from images, fine-tuning its last few layers during training.

## Project Structure

The project involves several key steps:
1. **Data Retrieval**: Setting up Kaggle directory and downloading the dataset.
2. **Data Preparation**: Adjusting image resolution and using a fraction of the dataset to manage computational requirements.
3. **Text Preprocessing**: Converting captions to lowercase, removing punctuations, and adding 'startseq' and 'endseq' to them.
4. **Tokenization**: Creating tokens from the text captions.
5. **Feature Extraction**: Using MobileNetV2 to extract features from images.
6. **Model Training**: Employing a generator for batch training and utilizing early stopping during the training of the model.
7. **Caption Generation**: Creating descriptive captions for new images post-training.

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details.
