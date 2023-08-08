# NIH_Chest_XRay

Chest X-Ray multi-label classification

The project uses the NIH Chest X-Ray 8 (CXR8) dataset which comprises 108,948 frontal-view X-ray images of 32,717 (collected from the year 1992 to 2015) unique patients with the text-mined eight common disease labels, mined from the text radiological reports via NLP techniques. The dataset can be found [here]([url](https://nihcc.app.box.com/v/ChestXray-NIHCC)). 

The project aims to implement the [CheXNet]([url](https://arxiv.org/abs/1711.05225)) paper using the PyTorch framework.

Process flow of the project -

**1. Input (train and test) image and label**<br>
Dataloader transforms applied
1. Resize the image to 256
2. Tencrop applied to the resized image - from the original image, take the crops of the 4 corners and the central crop, and then the       horizontal flip of each of these images, resulting in 10 new images of the original image.
3. Convert the image to tensors
4. Normalize the images to bring them on a similar scale and stabilize the network

**2. Model**<br>
Pre-trained DenseNet121 model is used from the PyTorch library with a custom classification layer.<br>
The 3 methodologies followed are -
  1. Baseline testing - calculate the AUROC score on the model without training either the classification or model layers
  2. Classifier training - train only the classifier layer and calculate the AUROC score
  3. Model training - train the entire model's layers including the classifier layer and calculate the AUROC score

**3. Output**<br>
The model is trained on only 10000 images and tested on 500 images.<br>
The AUROC scores of the above 3 methodologies are -
  1. baseline testing - Mean AUROC score of 0.512
  2. train classifier layer - Mean AUROC score of 0.531
  3. train model layers - Mean AUROC score of 0.556

We see an improvement in the AUROC scores as we start training more and more layers of the model. The scores for all three methods are ~0.5 (meaning random guessing) as the training data is very less compared to the total training images (~86000).

