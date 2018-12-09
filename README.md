# Implementation of a CNN+LSTM PoS tagger to restore/prediction punctuations in sentences

## Summary:
Using a CNN to extract word/character information and feeding them into a LSTM to make better predictions

`Input`: Sentence without punctuation  
`Output`: Prediction if the words in the sentence should contain punctuations and what they should be

## Requirements:
```
Jupyter notebook
Keras_2.2.4: Framework to create, train and run model
Matplotlib_3.0.1: For Confusion Matrix plotting  
Numpy_1.15.4: For array operations 
Python_3.6.7
Scikit-learn_0.20.1: For the Classification Report and Confusion Matrix functions  
Tensorboard_1.12.0: To view/visualize Tensorflow training log files
Tensorflow-gpu_1.12.0: To train model on GPU
```

## Project folder/files structure:
```
├───data
│   ├───processed
├───results
│   ├───mge
│   └───ted
└───tf_logs
```
* `data`: Contains all (Raw and processed) data used for the model. Sub folder: **processed** holds the Ted Talks dataset (And its processed formats) to be used
* `results`: Contains the screenshots of the model results (Pure LSTM, CNN+LSTM) performed on different datasets
* `src`: Contains the source files for the project
* `tf_logs`: Contains the training log files which can be viewed using tensorboard `tensorbard --logdir=./`
* `PoS Tagger`: Notebook containing all codes required to run the project. Further descriptions are provided within to explain each code block

## Usage:
As a prerequisite, please kindly ensure that [Anaconda](https://www.anaconda.com/download/) has been installed.

1. Clone the repo
2. Create the Conda environment using the yml file: conda env create -f environment.yml
3. Activate the Conda environment: (source or conda) activate cnn
4. Navigate to the src folder
5. Run any of the script files to train the model  

## Results:

# TODO 