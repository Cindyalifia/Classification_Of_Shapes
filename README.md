# Classification Of Shapes

**Datatrain available in test_set and training_set folder. I do predict with CNN algorithm.**

### Tools that needed to be installed are :
- Keras version 2.2.4
- matplotlib 

### Introduction
In this project, I make some classification to classify shape there are circle, square, and triangle. I use keras library and some other library and there are something you have to do before running the program.

### Installation 
- First step, you have to install keras on your pc   
    you can watch keras installation on https://keras.io/#installation

- Second step, you have to install matplotlib on your pc   
    $ pip install matplotlib
  

### Dataset
In this project, I use shapes for dataset which is circle, square, and triangle. I've got this data from https://www.kaggle.com/cactus3/basicshapes. There are 300 image containing 100 pictures each of circle, rectangle, and triangle. I split data for training set 67% and for test set 33%. 

### Result 
After I train my program, I got the accuracy is 92.04% using 20 epoch.

![](./SS/train.png)

And also I plot my training result :

![](./SS/model_acc.png)

If we look closer to my image, we can see that the accuracy from my train is almost overfitting. and that's why I make 20 epochs.  
And the result for loss is :

![](./SS/model_loss.png)

### My Model Result
I save my model on 'my_model.h5'.

