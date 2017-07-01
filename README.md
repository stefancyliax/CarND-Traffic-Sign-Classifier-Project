# Project: Build a Traffic Sign Recognition Program
[![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)

## Overview
In this project, the task is to model and train a classifier for German traffic signs. A convolutional neural networks is used to trained on a  [provided dataset of German traffic signs](http://benchmark.ini.rub.de/?section=gtsrb&subsection=dataset).

My solution can be found in the [jupyter notebook](https://github.com/stefancyliax/CarND-Traffic-Sign-Classifier-Project/blob/master/Traffic_Sign_Classifier.ipynb) or the [html export](https://github.com/stefancyliax/CarND-Traffic-Sign-Classifier-Project/blob/master/Traffic_Sign_Classifier.html)

## Dataset
The dataset contains color images of 43 differnt German traffic signs. The images are 32x32 pixel RGB. The dataset is pre-split into train-, validatation- and testset.

|                    | n     | %     |
|--------------------|-------|-------|
| Train dataset      | 34799 | 67%   |
| Validation dataset | 4410  | 8,5%  |
| Test dataset       | 12630 | 24,5% |

The folowing signs are included:

![prototypes](https://github.com/stefancyliax/CarND-Traffic-Sign-Classifier-Project/blob/master/pic/prototypes.png)


To get a feeling for the data set, I printed a random set of 32 samples.

![random set](https://github.com/stefancyliax/CarND-Traffic-Sign-Classifier-Project/blob/master/pic/random_set.png)

Some of the images were rather low quality, beeing either very dark, very light or dark with a light background. This is something the preprocessing of the images will have to considert.

![bad](https://github.com/stefancyliax/CarND-Traffic-Sign-Classifier-Project/blob/master/pic/bad_samples.png)

The second observation is, that the images are rather badly distributed over the labels. The most common label is 10 times more frequent than the least common.

![histogram](https://github.com/stefancyliax/CarND-Traffic-Sign-Classifier-Project/blob/master/pic/histogram.png)


## Preprocessing
For preprocessing there are four steps done.
1. Equalizing the histogram of the images. I experimented with cv2.equalizeHist, skimage.exposure.equalize_hist and skimage.exposure.rescale_intensity. Exposure.equalize_hist yielded the best results.
2. Augment the data by generating similar images by rotating, shearing, projecting, blur and noise.
3. Normalize to get a mean of 0 over the data set.
4. Shuffle the data using sklearn

#### Histogram equalizing
Because many of the images are very dark (or very light), I chose to implement a histogram equalization using skimage.exposure.equalize_hist.

![histogram equalization](https://github.com/stefancyliax/CarND-Traffic-Sign-Classifier-Project/blob/master/pic/hist_equ.png)

#### Data augmentation
Because the data is unevenly distributed and there are rather few samples of some traffic signs, a data augmentation was implemnted. For this generated 100 random samples for each traffic sign class and blurred, added noise, projected, rotated and sheared the image.
At first only the less common classes were augmented to even the distribution a bit. But it was found that this degraded the performance slightly.
Better performance was acheived by augmenting images from all classes. For each label 100 random samples were choosen and 8 augmented images generated each. This results in additional 34400 images.

![data augmentation](https://github.com/stefancyliax/CarND-Traffic-Sign-Classifier-Project/blob/master/pic/data_aug.png)

Histgram after data augmentation:

![histogram after augmentation](https://github.com/stefancyliax/CarND-Traffic-Sign-Classifier-Project/blob/master/pic/histogram_after_aug.png)


## Model Architecture
For the model a LeNet5 architecture is used.
To improve the performance dropout was introduced at the first and second fully connected layer. Also the pooling layers were removed and two additional convulutional layers were added. Check the Solution Approach for the steps.

The architecture looks like this:
0. Input layer: 32x32x3
1. Convulutional layer with kernel 5x5x6. Output is 28x28x6.
2. Convulutional layer with kernel 5x5x6. Output is 24x24x16.
3. Convulutional layer with kernel 5x5x6. Output is 20x20x32.
4. Convulutional layer with kernel 5x5x6. Output is 16x16x50.
5. Flatten. Output is 12800.
6. Fully connected layer with 120 output nodes and a dropout of 0.5 during training.
7. Fully connected layer with 84 output nodes and a dropout of 0.5 during training.
8. Fully connected layer with 43 output nodes used as logits.

## Model Training
For training the Adam Optimizer was used since it includes functions like learning rate decay.
The following hyperparameters yielded the best performance.
- Epochs: 10
- Batch Size: 128
- Learning Rate: 0.001
- Keep Probability for Dropout: 0.5

With these an accuracy of **97.3%** on the testset was achieved.

## Solution Approach
As the starting point the model from the LeNet Lab was used.
1. The CNN was subject to overfitting as seen in the difference between the train and validation accuracy. Therefore Dropout as regularization was introduced at the two fully connected layers.
2. Removed the two pooling layers from the network and replaced them with two additinal convulutional layers. This increases the size of the network and the computation needed significantly (from 400x120 to 12800x120 on the first fully connected layer), but increases the performance a lot.
3. Testing if the batch size can by optimized. -> No, best performance with 128 batchsize.
4. Lower learning rate is probably not improving the performance since the model is quite overfit already. - no improvement.
5. The model is still slightly overfit. Test if change to dropout rate improves the performance here.

| Epochs | Batch | Learning rate | ConvLayer | Pooling | FullyConLayer | Dropout | Train | Validation | Test  |
|--------|-------|---------------|-----------|---------|---------------|---------|-------|------------|-------|
| 10     | 128   | 0.001         | 2x        | 2x      | 2x            | 0       | 0.980 | 0.897      | 0.899 |
| 10     | 128   | 0.001         | 2x        | 2x      | 2x            | `2x, 0.5` | 0.946 | 0.956      | 0.928 |
| 10     | 128   | 0.001         | `4x`      | `0`     | 2x            | 2x, 0.5 | 0.994 | 0.985      | **0.973** |
| 10     | `256` | 0.001         | 4x        | 0       | 2x            | 2x, 0.5 | 0.990 | 0.989      | 0.968 |
| 10     | `64`  | 0.001         | 4x        | 0       | 2x            | 2x, 0.5 | 0.993 | 0.976      | 0.968 |
| `20`   | 128   | 0.001         | 4x        | 0       | 2x            | 2x, 0.5 | 0.999 | 0.989      | 0.972 |
| 10     | 128   | `0.0008`      | 4x        | 0       | 2x            | 2x, 0.5 | 0.995 | 0.985      | 0.971 |
| 10     | 128   | 0.001         | 4x        | 0       | 2x            | 2x, `0.45`| 0.992 | 0.979      | 0.973 |
| 10     | 128   | 0.001         | 4x        | 0       | 2x            | 2x, `0.4` | 0.972 | 0.963      | 0.953 |
| 10     | 128   | 0.001         | 4x        | 0       | 2x            | 2x, `0.55`| 0.996 | 0.980      | 0.965 |
| 10     | 128   | 0.001         | 4x        | 0       | 2x            | 2x, `0.6` | 0.996 | 0.976      | 0.964 |


## Acquiring New Images
To test the performance of the trained network on images from the internet, I gathered 8 pictures from Google Streetview.

![Images from internet](https://github.com/stefancyliax/CarND-Traffic-Sign-Classifier-Project/blob/master/pic/new_samples.png)

## Performance on new images -  Softmax Probabilities
The network correctly predicted 8/8 and showed **100%** accuracy.

![softmax probabilities](https://github.com/stefancyliax/CarND-Traffic-Sign-Classifier-Project/blob/master/pic/softmax.png)
