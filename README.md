
# Development and Comparison of Transformer, RNN and LSTM Model for Autonomous Image Captioning

## Introduction

This project focuses on developing an **image captioning system** that can generate natural text description based on the context and content from an image. The ability to effectively generate or caption images requires a model that can understand visual content and translate it into a textual description, bridging the gap between vision and language. This enables machines to interpret and describe images in a way that is understandable to humans.

## Project Goal
The main goal of this project is to be able to **accurately generate captions** that are not only syntactically correct but also semantically relevant to the content of the image. The captions should accurately describe the objects, their relationships and actions/context within the image. 

## Data

*   The dataset used for this project consists of images and their corresponding captions, sourced from **Flickr 8K**.
*   The dataset can be accessed at this link: [https://www.kaggle.com/datasets/adityajn105/flickr8k](https://www.kaggle.com/datasets/adityajn105/flickr8k).
*   The dataset was created by extracting features from images and preprocessing the corresponding captions.
*  The images are stored in the `Images` folder, and the captions are stored in the `captions.txt` file.
*   There are **8091** images in the Flickr 8K dataset.
*   Each image has multiple captions associated with it.
   
#### Other Data 
* MS COCO (Microsoft Common Objects in Context) Dataset
* The data can be accessed through this link MS COCO Dataset
* It is a very large-scale object detection, segmentation, and captioning dataset with almost 330,000 images, where over 200,000 are labelled images. Each of the images are paired to five different captions, providing detailed descriptions of the image’s events and context.
* This dataset was collected by Microsoft Corporation with the main focus on solving problems of everyday scenes and objects, captured from various sources in real-world contexts.

## How It's Done

### 1. Image Feature Extraction
*   **ResNet50**, a pre-trained deep convolutional neural network, is used to extract meaningful features from the images.
*   ResNet50 was pre-trained on large-scale image datasets like ImageNet.
*   The images are loaded and resized to match the input dimensions required by ResNet50.
*   Features are extracted from the penultimate layer (global average pooling output) and are represented as a compact feature vector.
*  The pre-trained ResNet50 model is used as a feature extractor.
*   The extracted features are saved as a CSV file for easier access.
   
### 2. Caption Preprocessing
*   The captions are processed and tokenized into individual words.
*   A vocabulary of unique words present in the dataset is created.
*   The captions are padded or truncated to a fixed length to ensure consistent input dimensions for the model.
*   Special tokens such as "startofseq" and "endofseq" are added to each caption.
*   The captions are converted to lowercase and punctuation and non-English words are removed.
*   Captions are encoded into integer sequences using the created vocabulary.

### 3. Model Training
*   The project utilizes sequence-to-sequence learning, mapping image features (input sequence) to a sequence of words (output captions).
*   The extracted image features are used as the initial input to the model.
*   The model learns to predict the next word in the sequence based on the previous words and the image features.
*   The model's parameters are optimised to generate high-quality captions.
*  The models are trained with a split of training and validation sets (80% and 20% respectively).

## Models

This project explores three different models for image captioning:

### 1. LSTM Model
* A combination of **LSTM (Long Short-Term Memory)** layers are used to capture the sequential dependencies between words in the caption.
*   The model is enhanced with LSTM layers, which are designed to capture the temporal dependencies of the generated captions.
*   The model uses a sequence-to-sequence architecture.
*   The model combines the outputs of the image feature extraction and caption generation models.
*   The model is compiled using categorical cross-entropy loss and the RMSprop optimizer.

### 2. RNN with Attention Model
* A combination of **GRU (Gated Recurrent Unit) + RNN (Recurrent Neural Network)** layers with **Attention mechanisms** are used to generate accurate and contextually relevant captions.
* The model is enhanced with GRU and RNN layers, designed to capture the temporal dependencies of the generated captions.
* An attention mechanism is added to enable the model to focus on specific parts of the image when generating each word of the caption.
* The model is compiled using categorical cross-entropy loss and the Adam optimizer.

### 3. Transformer Model
*   A transformer-based model using an encoder-decoder architecture.
*  The transformer model uses multiple heads and feed forward layers for caption generation.
*  The model is compiled using categorical cross-entropy loss and the Adam optimizer.

## Evaluation
* The models are evaluated using metrics such as accuracy and AUC.
* Additional evaluation metrics used are precision, recall, and F1 score.
* Validation is performed to measure the performance of the models during training.

## Usage

*   The notebook includes code for loading data, preprocessing images and text, model building, training and evaluating, and for generating captions for sample images.
*   You will need to have the required libraries installed to run the notebook.

## Results
*   The best performance was achieved with the RNN model. Below is a table showing the final results achieved by compairing each of the model;
    | Model              | Loss    | Accuracy  | Precision | Recall   | F1 Score | AUC Score |
    |--------------------|---------|-----------|-----------|----------|----------|-----------|
    | Transformer Model  | 3.0040  | 0.3709    | 0.2891    | 0.3709   | 0.3021   | 0.9199    |
    | RNN Model          | 1.6441  | 0.7560    | 0.7643    | 0.7560   | 0.7463   | 0.9415    |
    | LSTM Model         | 5.2362  | 0.3920    | 0.3774    | 0.3920   | 0.3746   | 0.8143    |

* The **RNN Model** demonstrates the best performance based on the metrics, with the lowest loss score and the highest accuracy, precision, recall, F1, and AUC scores. The **transformer model** has the lowest performance, with the highest loss and the lowest accuracy score.
* The project successfully demonstrates image captioning using different models, highlighting the strengths and challenges of each approach.
