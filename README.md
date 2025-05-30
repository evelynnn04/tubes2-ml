# Implementation of CNN, RNN, LSTM

> IF3270 Machine Learning
>
> By Group 36:<br>
> 1. Shazya Audrea Taufik <br>
> 2. Evelyn Yosiana<br>
> 3. Zahira Dina Amalia<br>
>
> School of Electrical Engineering and Informatics<br>
> Bandung Institute of Technology<br>
> 2024/2025

## Table of Contents
* [General Info](#general-information)
* [Libraries Used](#libraries-used)
* [Features](#features)
* [Setup](#setup)
* [Usage](#usage)
* [Project Status](#project-status)
* [Room for Improvement](#room-for-improvement)

## General Information
This system implements:
- Convolutional Neural Network (CNN) for image classification
- Recurrent Neural Network (RNN) for text classification
- Long Short-term Memory (LSTM) for text classification

## Libraries Used
- numpy
- scikit-learn
- tensorflow
- matplotlib, seaborn
- pandas

## Features
- Forward and Backward Propagation
- Comparison of number of layers, neurons, and directions

## Setup

### Prerequisites

- Python 3.8+

### Clone Repository
   ```shell
   git clone https://github.com/evelynnn04/tubes2-ml.git
   ```

## Usage

1. Train the Hard model (the script is already available in the notebook).
2. Extract the weight of the Hard model and load it into the model from scratch.
3. Run feedforward on the from scratch model, compare the prediction result/f1-score with Keras.
4. (Optional) Experiment backward propagation, hyperparameter tuning, and compare performance on validation/test data.

## Project Status
_done_


## Room for Improvement

- Optimization of backward pass speed on large datasets
- Addition of regularization 