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
- Random Stitch

For brevity, we will describe two select transformations that we developed here, although all of the above transformations are detailed in full in the Kaggle Notebook. The first is Random Occlusion, which we read about in the following research [paper](https://arxiv.org/pdf/1708.04896.pdf).

This is a technique that randomly chooses a box in the image to cover with noisy pixels. The idea is that it will make the network more robust to occlusion in the testing dataset, as it will have been trained on images that already have various components of the object covered. We implemented a function `RandomOcclusion` to do this transformation, and the transformation can be visualized below:

![Screen Shot 2023-03-12 at 7 36 51 PM](https://user-images.githubusercontent.com/89823879/224595977-f4c42274-7c61-467e-bbbb-f82f6da6f3d1.png)
![Screen Shot 2023-03-12 at 7 36 56 PM](https://user-images.githubusercontent.com/89823879/224595967-06722c4a-079e-474d-9eb5-fb764aeed7e8.png)

Next, we describe the Random Stitch. This transformation was one of our own invention, and it's main idea is that many of the birds are small in area compared to the rest of the image. Thus, the neural network doesn't have much information to work with to identify the birds. Our idea was, that it may be better to take multiple crops of the image centered around the middle, and stitch them together. The hope was that then each bird (or it's body components) would be included in the image multiple times, allowing the network to have more visual information to work with. We implemented the function `RandomStitch` to do just that, and an example can be visualized below:

![Screen Shot 2023-03-12 at 8 41 47 PM](https://user-images.githubusercontent.com/89823879/224603415-ce3d2d27-b847-4074-9c4e-4df330786ee2.png)
![Screen Shot 2023-03-12 at 8 41 44 PM](https://user-images.githubusercontent.com/89823879/224603418-38369fdb-fa72-4606-91d5-c0dd0a21de6a.png)

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

We noted that only Normalization, Random Occlusion, and Random Flipping/Rotation seemed to have any benefit on the training accuracy. Quite disappointingly, we did not see any significant boost by any of our invented augmentations. However, we were still happy about the boost given by the augmentations chosen. 

# ****Part 3: Hyperparameter Tuning****
Now, our bird classifier was nearly complete. One crucial step, though, was the importance of hyperparameters. In particular, the ones we thought would be most important were image size, weight decay, and learning rate.

For the image size, we assumed, intuitively, that larger input sizes would give better results. After all, the more visual acuity the neural has, the better the accuracy it should have. To test this, we defined four input sizes: 224 x 224, 256 x 256, 384 x 384, 512 x 512.

Why 384 x 384? According to the original EfficientNet2 [paper](https://arxiv.org/pdf/2104.00298.pdf), the model was trained using a 384 x 384 central crop on the images. Thus, we thought it would be important to test that particular size. For the experiments, we trained a EfficientNet on each input image size, and calculated the accuracy. The results are given as follows:

<b>Size 224</b>:
<b>Size 256</b>:
<b>Size 384</b>:
<b>Size 512</b>:

According to the results, the 384 size image did the best. However, this may not be that surprising, since the original model used 384 size images, so the transfer learning may be better with that particular size.

Next, we looked at weight decay. We tested four different weight decays: 0.1, 0.01, 0.001, 0.0001. We did the same experimental procedure, where we trained an EfficientNet model for each weight decay and found the testing accuracy. The results were as follows:

<b>Decay 0.1</b>:
<b>Decay 0.01</b>:
<b>Decay 0.001</b>:
<b>Decay 0.0001</b>:

Interestingly, the best weight decay was 0.0001. This indicated to us that the model needed to be quite complex to accurately fit the data, and that we should not limit the model's weights. This also made some sense, since we did have 555 different classes to fit the data on. 

Finally, we needed to try out different learning rate schedules. Since our GPU compute was limited, we elected to try a heuristic approach, rather than an experimental approach using grid search. Earlier, we found success with a scheduled learning rate that decreased as the epoch increased. This allowed the model to make large updates to the parameters early on in training when the loss was large. Later on, the model could then decrease the size of the updates as a local optima was reached. This helped prevent the model from overshooting the optimal parameters.

Thus, we decided to train the model once for 15 epochs, following a schedule with 0.01 learning rate for the first five epochs, then a rate of 0.001 for the next five epochs, and finally a rate of 0.0001 for the final five epochs. For each epoch, we calculated the testing accuracy of the model developed so far. At any point, if the model's accuracy decreased from the previous epoch, then training was stopped. This is a technique called early stopping, and it can help prevent losses from overfitting. 

# ****Part 4: Final Model****
Our final model had the following parameters and augmentations

- input image size: 384 x 384
- pre-trained model: EfficientNet2
- learning rate: 15 epochs with 0:0.01, 5:0.001, 10:0.0001 schedule
- weight decay: 0.0001
- momentum: 0.9
- Normalization, Random Occlusion, Random Flipping/Rotation

With this final model, we ended up getting a testing accuracy of 0.856. Not bad, given the GPU compute and time limits!

## **Takeways**

### Problems Encountered
1. We ran into quite a few issues with finding the best pre-trained net for our classification task. Specifically, it was hard at the time to interpret the accuracy of our nets and what caused them to increase or decrease. For instance, when testing out ResNet50 and ResNet152, we noticed a drastic decrease in accuracy. After some research, we determined that the decrease in accuracy was likely because we hadn't trained the model for enough epochs since larger nets generally take longer to converge than smaller nets, such as ResNet18. We determined that we simply did not have enough GPU usage time between Google Colab and Kaggle Notebook, so we ultimately opted to choose smaller networks that we could finetune better.
2. We found it very time-consuming to test all of these models and also fine-tune hyper parameters. Even utilizing a GPU for training, models often took several hours to complete training and then test their accuracy. It was at times frustrating to try to fine-tune a parameter, only to realize that our initial hypothesis for increasing accuracy was wrong after several hours. To help combat this, we utilized Kaggle Notebook as well as Google Colab so we could increase our throughput and train multiple models at the same time. We also utilized multiple Google accounts to get access to more GPU time and used checkpoints to save the state of our model to transfer it between accounts when our training was interrupted by GPU over-use


### What Would the Next Steps Have Been?
One area we didn't explore as much as we would have liked was hyperparameter optimization. While we experimented with the learning rate and the number of epochs, we didn't get to dive into other hyperparameters such as decay, batch size, and the choice of an optimizer. It was hard to explore many of these options since we wanted to refrain from changing multiple variables at once to better determine what was improving the accuracy of our model. Now that we have a model we know works well for our classification task and have found some good techniques for data augmentation, we would likely be able to improve our accuracy by adjusting these hyperparameters.

## **Code**
As mentioned before, we utilized both Google Colab and Kaggle Notebook. [Click Here for the Google Colab Code and Graphs](https://github.com/tameidan1/CSE455FinalProject/blob/main/GoogleColabCode.ipynb) and Here for the Kaggle Notebook Code

## **References**
