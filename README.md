# Leaf Diseases Detection

This project uses a deep learning model to classify leaf diseases. The model is trained on a dataset of leaf images labeled with three classes: Healthy, Powdery, and Rust.

## Dataset Used

The [dataset](https://www.kaggle.com/datasets/rashikrahmanpritom/plant-disease-recognition-dataset "Plant Leaf Dataset") used in this project was taken from Kaggle and was filtered and processed for the specific usecases of this project.

## Technologies Used

1) Keras
2) TensorFlow
3) Matplotlib
4) Numpy

## Running the Code

To run the code, execute the `main.py` script. This script loads a pre-trained model: `SproutIQ_Disease_Detection_Model.keras`, evaluates it on the validation set, and makes predictions on the test set. The predictions are then plotted and displayed.

## Model Evaluation

The model's performance is evaluated in terms of validation loss and validation accuracy, which are printed to the console.

## Making Predictions

The script makes predictions on the first batch of images from the test set. It plots the images along with their true labels and predicted labels.

## Future Work

The project includes commented-out code for building a Kivy app. This is a future direction for the project, allowing users to interact with the model in a graphical user interface on mobile devices.
