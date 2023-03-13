### Kaggle Bird Classification Competition
Project Created by Seulchan Han and Daniel Tameishi

## **Introduction**
For our project, we competed in the Kaggle bird-classifying competition. The goal of the competition was to create a classifier with the highest accuracy possible. To achieve this, we explored multiple different techniques such as transfer learning with multiple different models, experimenting with hyperparameters, and data augmentation. Our project was essentially a large scale survey on the different ways to improve neural network performance on image classification.

## **Dataset**
The dataset that was used was of 555 different species of birds provided by the [Birds Birds Birds Kaggle Competition](https://www.kaggle.com/competitions/birds23wi)

## **Notebook**
For our project, our entire code runnable is linked in the following Kaggle Notebook:
[Click Here for Kaggle Notebook Link]

## **Approach**
# ****Part 1: Transfer Learning****
Our first course of action was to determine the best neural network architecture for classifying the bird species. We wanted to leverage transfer learning by using a model pretrained on ImageNet, but we were not quite sure which one to use. To begin, we chose several candidate pre-trained models to experimentally evaluate performance. We chose the following models for experimentation.

- [ResNet18](https://pytorch.org/vision/master/models/generated/torchvision.models.resnet18.html)
- [ResNet50](https://pytorch.org/vision/master/models/generated/torchvision.models.resnet50.html)
- [ResNet152](https://pytorch.org/vision/master/models/generated/torchvision.models.resnet152.html)
- [DenseNet201](https://pytorch.org/vision/main/models/generated/torchvision.models.densenet201.html#torchvision.models.densenet201)
- [EfficientNet_v2_s](https://pytorch.org/vision/main/models/generated/torchvision.models.efficientnet_v2_s.html#torchvision.models.efficientnet_v2_s)

For our experiment, we fixed the parameters for each model to be consistent, and trained each model for 3 epochs. We then tested each model's performance on a testing set that the model had not seen before and calculated the accuracy. For brevity, we simply list the accuracy measures below:

<b>Resnet18</b>: 
<b>ResNet50</b>:
<b>ResNet152</b>: 
<b>DenseNet201</b>:
<b>EfficientNet_v2_s</b>:

What we found was that EfficientNet_v2_s had the highest testing accuracy of all the models. This made sense, since EfficientNet also had the highest baseline Top 1 accuracy on ImageNet.

The full experiment code can be found in the experiments directory of the github repository. The exact experiments are also described in full detail in the linked Kaggle Notebook. 

# ****Part 2: Data Augmentation****
Thanks to transfer learning, we now had a very high baseline of accuracy to work from. Next, our attention shifted to working with better data. We read many research papers that outlined the benefits of data augmentation, and how they could be leveraged to prevent overfitting on models. The most effective augmentations that we read about were 

- Random Flipping/Rotation
- Random Crop
- Adjusting Sharpness
- Color Jitter
- Normalization

In particular, we also invented some new data augmentations that were not found in the Pytorch library, and we also implemented some known augmentations that were not well-implemented by the library. The ones in particular that we developed were

- Random Occlusion
- Random Noise
- Random Scaled Crop
- Random Color Muting

This was a large coding component of our project, and the exact code can be found in the augmentations directory. 

For each augmentation, we tested it by transforming the training dataset only with that particular augmentation, an none others. We then trained our EfficientNet model using that particular augmentation, and calculated the testing accuracy. Our goal here was to choose only the data augmentations that seemed to be effective relative to the baseline model with no augmentation. In addition, we upped the number of training epochs to 6, as our effective dataset became a bit larger with data augmentation. Our results were as follows:

<b>Baseline (no augmentation)</b>: 
<b>Random Flipping/Rotation</b>: 
<b>Random Crop</b>: 
<b>Adjusting Sharpness</b>: 
<b>Color Jitter</b>: 
<b>Normalization</b>: 
<b>Random Occlusion</b>: 
<b>Random Noise</b>: 
<b>Random Scaled Crop</b>: 
<b>Random Color Muting</b>: 

We noted that only Normalization, Random Occlusion, and Random Flipping/Rotation seemed to have any benefit on the training accuracy. Quite disappointingly, we did not see any significant boost by any of our invented 



For these, we essentially decided to 

1. To determine what neural network would be the best for our bird classifying, tried multiple pre-trained neural networks to determine which network would be the best "base" network for our bird classification task. These networks included: [ResNet18](https://pytorch.org/vision/master/models/generated/torchvision.models.resnet18.html), [ResNet50](https://pytorch.org/vision/master/models/generated/torchvision.models.resnet50.html), [ResNet152](https://pytorch.org/vision/master/models/generated/torchvision.models.resnet152.html), [EfficientNet_b0](https://pytorch.org/vision/main/models/generated/torchvision.models.efficientnet_b0.html#torchvision.models.efficientnet_b0), [EfficientNet_b1](https://pytorch.org/vision/main/models/generated/torchvision.models.efficientnet_b1.html#torchvision.models.efficientnet_b1), and [EfficientNet_v2_s](https://pytorch.org/vision/main/models/generated/torchvision.models.efficientnet_v2_s.html#torchvision.models.efficientnet_v2_s). After trying all of these networks, we determined that EfficientNet_v2_s was the best suited for our needs.
2. Along the way we also experimented with various data augmentation techniques. These included: random horizontal flipping, random cropping, and color jitter. We found the most success with a combination of random flipping and random cropping. These helped prevent our model from overfitting to the testing set by increasing the variability of the dataset.

### Example of Random Cropping:
![Random Cropped Giraffe](/images/RandomCropping.png)

### Example of Horizontal Flipping
![Example of horizontal flipping](/images/flipping.png)

4. We also found success with a scheduled learning rate that decreased as the epoch we were on increased. This allows the model to make large updates to the parameters early on in training when it is likely necessary and then decrease the size of the updates as we approach optimality. This helps prevent us from overshooting the optimal parameters.

## **Takeways**

### Problems Encountered
1. We ran into quite a few issues with finding the best pre-trained net for our classification task. Specifically, it was hard at the time to interpret the accuracy of our nets and what caused them to increase or decrease. For instance, when testing out ResNet50 and ResNet152, we noticed a drastic decrease in accuracy. After some research, we determined that the decrease in accuracy was likely because we hadn't trained the model for enough epochs since larger nets generally take longer to converge than smaller nets, such as ResNet18. We determined that we simply did not have enough GPU usage time between Google Colab and Kaggle Notebook, so we ultimately opted to choose smaller networks that we could finetune better.
2. We found it very time-consuming to test all of these models and also fine-tune hyper parameters. Even utilizing a GPU for training, models often took several hours to complete training and then test their accuracy. It was at times frustrating to try to fine-tune a parameter, only to realize that our initial hypothesis for increasing accuracy was wrong after several hours. To help combat this, we utilized Kaggle Notebook as well as Google Colab so we could increase our throughput and train multiple models at the same time. We also utilized multiple Google accounts to get access to more GPU time and used checkpoints to save the state of our model to transfer it between accounts when our training was interrupted by GPU over-use


### What Would the Next Steps Have Been?
One area we didn't explore as much as we would have liked was hyperparameter optimization. While we experimented with the learning rate and the number of epochs, we didn't get to dive into other hyperparameters such as decay, batch size, and the choice of an optimizer. It was hard to explore many of these options since we wanted to refrain from changing multiple variables at once to better determine what was improving the accuracy of our model. Now that we have a model we know works well for our classification task and have found some good techniques for data augmentation, we would likely be able to improve our accuracy by adjusting these hyperparameters.

## **Code**
As mentioned before, we utilized both Google Colab and Kaggle Notebook. [Click Here for the Google Colab Code and Graphs](https://github.com/tameidan1/CSE455FinalProject/blob/main/GoogleColabCode.ipynb) and Here for the Kaggle Notebook Code

## **References**
