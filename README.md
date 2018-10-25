# Mini Project 2
## Goal
The goal of this exercise is to make a Deep learning program which takes up the user provided data to train, on the basis of the labels provided in data, and then provides a probability value on the supplied image for testing
## Requirements
The system runs on Python 3.6.5. Please note that it will not work with Python 3.7.x

The program also requires the following libraries to be installed
* Tensorflow
* Numpy
* Sklearn
* OpenCV
## How to run
1. Download/Clone the repository

2. Install the requirements as stated

3. Run the following command with x replaced by the number of classes you want your system to identify 

```python main.py --train training_data/ --val testing_data --num_classes x```

4. After the system is run and the model is learnt, run the following command with filename.jpg replaced by your test file

```python prediction.py filename.jpg```
