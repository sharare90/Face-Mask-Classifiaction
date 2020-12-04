# Face Mask Classification

TensorFlow-2 implementation of face mask classification. 
Given an image of a face, the code can classify if the face is with mask or without mask.

![Sample image info](./readme_images/samples_from_dataset.png)

For training, we split the dataset into train,validation, test 
with 80%, 10%, 10% ratio, respectively.
We added a global max pooling and a binary classification 
head on top of a pre-trained
[Xception](https://arxiv.org/abs/1610.02357).
Since the dataset is unbalanced, we duplicate not masked images
during training 
such that we have same amount of images for both classes.

## Dataset
Download data from [this link](https://www.kaggle.com/andrewmvd/face-mask-detection).
After downloading the dataset, set `DATASET_ADDRESS` in `settings.py` to the address of the downloaded dataset.
This folder should contain two subdirectories, images and annotations.

## Running the code
Update the `PROJECT_ROOT_ADDRESS` in `settings.py`.
Then run the following command to install the requirements:

```
pip install -r requirments.txt
```

Then you can run `main.py` to train, test and demo the results.

## Citation
Please cite this paper:
S. Zehtabian, S. Khodadadeh, K. Kim, G. Bruder, G. F. Welch, L. Bölöni, and D. Turgut. An Automated Virtual Receptionist for Recognizing Visitors and Assuring Mask Wearing. In International Conference on Artificial Reality and Telexistence / Eurographics Symposium on Virtual Environmnents (ICAT-EGVE-2020), December 2020. 

