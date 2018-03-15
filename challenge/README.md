# Prepare Dataset Challenge

## Overview

This is an entry for [this](https://youtu.be/0xVqLJe9_CY) video by Siraj on Youtube.

The pokemon classifier aims to train a neural network to classify pokemon by their type 1 (i.e fire, water, grass, etc.) using [this](https://www.kaggle.com/abcsds/pokemon) pokemon dataset on Kaggle.

## Dependencies

* tensorflow (pip install tensorflow) 
* numpy (pip install numpy) 


## Demo

Run the following in terminal
```
$ python main.py
```
or with all the variables defined
```
$ python main.py --verbose --trainingIterations 120 --learningRate 0.0005
```

## Results

The python script is able to parse the provided Pokemon dataset and train to an accuracy of around 75% after 120 iterations at a learningRate of 0.0005. 
After the training process, the user is then able to input their own Pokemon stats to see what the network thinks its type 1 is.

## Credits

Credits go to Alberto Barradas (For the dataset), and Siraj (for the idea and starting code).

