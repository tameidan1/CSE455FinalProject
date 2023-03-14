Project Created by Seulchan Han and Daniel Tameishi

## **Introduction**
For our project, we competed in the Kaggle bird-classifying competition. The goal of the competition was to create a classifier with the highest accuracy possible. To achieve this, we explored multiple different techniques such as transfer learning with multiple different models, experimenting with hyperparameters, and data augmentation. Our project was essentially a large scale survey on the different ways to improve neural network performance on image classification.

## **Dataset**
The dataset that was used was of 555 different species of birds provided by the [Birds Birds Birds Kaggle Competition](https://www.kaggle.com/competitions/birds23wi)

## **Code**
For our project, our entire code runnable is linked in the following [Kaggle Notebook](https://www.kaggle.com/code/seulchanhan/birdsbirdsbirds). Our main code, experiments, and explanations are in the Kaggle Notebook, but additional models and graphs are available in a [Google Colab](https://github.com/tameidan1/CSE455FinalProject/blob/main/GoogleColabCode.ipynb).

For this project, our main coding component came from implementing the three areas of inquiry: tranfer learning, data augmentation, and hyperparameter tuning. Our code consisted of all of the experiments, setup, and augmentation implementations. Our most signficant coding portion was the implementation of the various data augmentation techniques we used.

However, we did leverage pre-trained models, as well as using some of the setup code from the Pytorch tutorials [here](https://github.com/pjreddie/uwimg/blob/main/tutorial1-pytorch-introduction.ipynb).

## **Video**
[Project Video Link](https://youtu.be/tav1ZI8Ontc)

## **Procedure**
### **Part 1: Transfer Learning**
Our first course of action was to determine the best neural network architecture for classifying the bird species. We wanted to leverage transfer learning by using a model pretrained on ImageNet, but we were not quite sure which one to use. To begin, we chose several candidate pre-trained models to experimentally evaluate performance. We chose the following models for experimentation.

- [ResNet18](https://pytorch.org/vision/master/models/generated/torchvision.models.resnet18.html)
- [ResNet50](https://pytorch.org/vision/master/models/generated/torchvision.models.resnet50.html)
- [ResNet152](https://pytorch.org/vision/master/models/generated/torchvision.models.resnet152.html)
- [DenseNet201](https://pytorch.org/vision/main/models/generated/torchvision.models.densenet201.html#torchvision.models.densenet201)
- [EfficientNet_v2_s](https://pytorch.org/vision/main/models/generated/torchvision.models.efficientnet_v2_s.html#torchvision.models.efficientnet_v2_s)

For our experiment, we fixed the parameters for each model to be consistent, and trained each model for 3 epochs. We then tested each model's performance on a testing set that the model had not seen before and calculated the accuracy. For brevity, we simply list the accuracy measures below:

<b>Resnet18</b>: 0.509

<b>ResNet50</b>: 0.517

<b>ResNet152</b>: 0.454

<b>DenseNet201</b>: 0.527

<b>EfficientNet_v2_s</b>: 0.649

What we found was that EfficientNet_v2_s had the highest testing accuracy of all the models. This made sense, since EfficientNet also had the highest baseline Top 1 accuracy on ImageNet.

The full experiment code can be found in the experiments directory of the github repository. The exact experiments are also described in full detail in the linked Kaggle Notebook. 

### **Part 2: Data Augmentation**
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

**Baseline (no augmentation):** 0.786

**Random Occlusion:** 0.791

**Random Flip:** 0.800

**Random Scaled Crop:** 0.754

**Sharpness:** 0.784

**Random Color Jitter:** 0.759

**Normalize:** 0.792

**Random Noise:** 0.619

**Random Stitch:** 0.714

We noted that only Normalization, Random Occlusion, and Random Flipping/Rotation seemed to have any benefit on the training accuracy. Quite disappointingly, we did not see any significant boost by any of our invented augmentations. However, we were still happy about the boost given by the augmentations chosen. 

### **Part 3: Hyperparameter Tuning**
Now, our bird classifier was nearly complete. One crucial step, though, was the importance of hyperparameters. In particular, the ones we thought would be most important were image size, weight decay, and learning rate.

For the image size, we assumed, intuitively, that larger input sizes would give better results. After all, the more visual acuity the neural has, the better the accuracy it should have. To test this, we defined four input sizes: 224 x 224, 256 x 256, 384 x 384, 512 x 512.

Why 384 x 384? According to the original EfficientNet2 [paper](https://arxiv.org/pdf/2104.00298.pdf), the model was trained using a 384 x 384 central crop on the images. Thus, we thought it would be important to test that particular size. For the experiments, we trained a EfficientNet on each input image size, and calculated the accuracy. The results are given as follows:

**Image Size 224:** 0.790

**Image Size 256:** 0.803

**Image Size 384:** 0.840

**Image Size 512:** 0.822

According to the results, the 384 size image did the best. However, this may not be that surprising, since the original model used 384 size images, so the transfer learning may be better with that particular size.

Next, we looked at weight decay. We tested four different weight decays: 0.1, 0.01, 0.001, 0.0001. We did the same experimental procedure, where we trained an EfficientNet model for each weight decay and found the testing accuracy. The results were as follows:

**Decay 0.1:** 0.002

**Decay 0.01:** 0.387

**Decay 0.001:** 0.785

**Decay 0.0001:** 0.786

Interestingly, the best weight decay was 0.0001. This indicated to us that the model needed to be quite complex to accurately fit the data, and that we should not limit the model's weights. This also made some sense, since we did have 555 different classes to fit the data on. 

Finally, we needed to try out different learning rate schedules. Since our GPU compute was limited, we elected to try a heuristic approach, rather than an experimental approach using grid search. Earlier, we found success with a scheduled learning rate that decreased as the epoch increased. This allowed the model to make large updates to the parameters early on in training when the loss was large. Later on, the model could then decrease the size of the updates as a local optima was reached. This helped prevent the model from overshooting the optimal parameters.

Thus, we decided to train the model once for 15 epochs, following a schedule with 0.01 learning rate for the first five epochs, then a rate of 0.001 for the next five epochs, and finally a rate of 0.0001 for the final five epochs. For each epoch, we calculated the testing accuracy of the model developed so far. At any point, if the model's accuracy decreased from the previous epoch, then training was stopped. This is a technique called early stopping, and it can help prevent losses from overfitting. 

### **Part 4: Final Model**
Our final model had the following parameters and augmentations

- input image size: 384 x 384
- pre-trained model: EfficientNet2
- learning rate: 15 epochs with 0:0.01, 5:0.001, 10:0.0001 schedule
- weight decay: 0.0001
- momentum: 0.9
- batch size: 16
- Normalization, Random Occlusion, Random Flipping/Rotation

With this final model, we ended up getting a testing accuracy of 0.856. Not bad, given the GPU compute and time limits! Our loss graph is also given below:

![Screen Shot 2023-03-13 at 10 07 54 PM](https://user-images.githubusercontent.com/89823879/224899787-e8441678-2524-445c-89c2-6ba1436ac63e.png)


## **Takeways**

### Problems Encountered
1. Our most significant issue was the relative quality of the dataset. We noticed right away that many of the training images had low bird area relative to background area. What this meant was that the networks had less information pertaining to the birds for each image. Instead, they were mostly overwhelmed by the amount of background noise that had nothing to do with birds. We tried many data augmentation techniques to try and address this issue, and they worked to some extent. Fundamentally, however, the quality of the dataset makes the classification problem much harder. We believe that with a much higher quality dataset - such as one with birds occupying at least 50% of the image space - the models could easily get >95% accuracy even with minimal augmentation.
2. We found it very time-consuming to test all of the models and also fine-tune hyper parameters. Grid search is very inefficient, especially with as many augmentations we experimented with. Even with utilizing a GPU, models took several hours to train and then test their accuracy. At times, it was frustrating to try to fine-tune a parameter, only to realize that our initial hypothesis for increasing accuracy was wrong after several hours. To help combat this, we utilized both the Kaggle Notebook and a Google Colab so we could increase our throughput and train multiple models at the same time. We also utilized multiple Google accounts to get access to more GPU time and used checkpoints to save the state of our model to transfer it between accounts when our training was interrupted by GPU over-use.
3. We ran into quite a few issues with finding the best pre-trained net for our classification task. Specifically, it was hard at the time to interpret the accuracy of our nets and what caused them to increase or decrease. For instance, when testing out EfficientNet2 versus ResNet152, we noticed a drastic decrease in accuracy. After some research, we determined that the decrease in accuracy was likely because we hadn't trained the model for enough epochs since larger nets generally take longer to converge than smaller nets. Note that EfficientNet2 is much smaller than ResNet152, with about 21 million parameters compared to 60 million for ResNet152. Then, we determined that we simply did not have enough GPU usage time between between the Google Colab and Kaggle Notebook, so we ultimately opted to choose smaller networks like EfficientNet2 that we could finetune better.
4. Finally, we would have liked to explore very deep neural networks, such as WideResNet101. Unfortunately, we ran into a RAM barrier whenever we tried to train such deep networks. Since GPU RAM was limited to 15 GB, we could not load very large networks at all onto the Kaggle/Colab GPUs. Even when we tried lowering the batch size to 1, it did not work. This was very disappointing, since many of the deepest networks had the highest Top 1 accuracy ImageNet scores. 


### What Would the Next Steps Have Been?
1. One area we didn't explore as much as we would have liked was hyperparameter optimization. While we experimented with the learning rate, weight decay, etc. we didn't get to dive into other hyperparameters such as optimizers, momentum, and batch size. It was hard to explore many of these options since we wanted to refrain from changing multiple variables at once to better determine what was improving the accuracy of our model. However, after finding a good model that worked well for the classification task, we would likely be able to improve our accuracy by adjusting these hyperparameters more. Given the time and GPU resources, we would have performed additional experiments to determine the optimal hyperparameters for the EfficientNet model. 
2. Next, we would have liked to explore additional general training techniques. One interesting technique we read about was weight freezing. The basic idea here is that the early layers of pre-trained models could be "frozen", so that backpropogation do not change the weights. The aim is to preserve the general patterns that the early convolutional layers have learned, since they have already been trained on so much ImageNet data. If we had the time and compute, we would have tried freezing some early layers of the EfficientNet model. This may have led to significant performance gains, since the backpropogation during training can stop early.
3. Finally, we would have liked to explore fine-tuning our model with more bird data. We did note that there were additional bird data with more species out in the Internet, but we did not have the compute resources to process all of them in addition to the given Kaggle dataset. Given the time and resources, we believe that fine-tuning our model with different bird species could lead to a significant boost in testing accuracy. At the very least, there would be less overfitting in our model. 

### What Did We Do Differently? Was it Beneficial?
The primary thing that our project did differently was the implementation of many different invented data augmentations. To develop these, we reasoned about the structure and patterns found in the images. We then tried to create modifications that could leverage these patterns to benefit training or hide weaknesses in the dataset.

The augmentations that we had the most hope for were Random Scaled Crop and Random Stitch, since these augmentations leverage the fact that the majority of the bird dataset has 
  1. the bird centered in the middle
  2. the bird area is low compared to the total image area

Unfortunately, most of our augmentations did not work effectively on the dataset. You can see this from the results and data portion of the Kaggle Notebook, but the high level idea is that complicated augmentations seem to not work well during testing time. It seems that if the augmentations are too complex, the model can't generalize them to the testing set, which are unaugmented.
