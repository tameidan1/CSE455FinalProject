# CSE455FinalProject

### Kaggle Bird Classification Competition
Project Created by Seulchan Han and Daniel Tameishi

## Introduction
The project was focused on competiting in a Kaggle bird classifying competion using several computer vision techniques. The goal of the competition was to have the highest accuracy possible. To acheive this, we explored multiple different techniques such as transfer learning with multiple different models, as well as experimenting with hyperparemeters and data augmentation.

## Dataset 
The dataset that was used was of 555 different specifes of birds provided by the [Birds Birds Birds Kaggle Competition](https://www.kaggle.com/competitions/birds23wi)

## Approach
1. To determine what neural network would be the best for our bird classifying, tried multiple pre-trained neural networks to determine which network would be the best "base" network for our bird classification task. These networks included: ResNet18, ResNet50, ResNet152, EfficientNet_b0, EfficientNet_b1, and EfficientNet_v2_s. After trying all of these networks, we determined that EfficientNet_v2_s was the best suited for our needs.
2. Along the way we also experimented with various data augmentation techniques. These included: random horizontal flipping, random cropping, and color jitter. We found the most success with a combination of random flipping and random cropping. These helped prevent our model from overfitting to the testing set by increasing the variability of the dataset.
3. We also found success with a scheduled learning rate that decreased as the epoch we were on increased. This allows the model to make large updates to the parameters early on in training when it is likely necessary and then decrease the size of the updates as we approach optimality. This helps prevent us from overshooting the optimal parameters.


