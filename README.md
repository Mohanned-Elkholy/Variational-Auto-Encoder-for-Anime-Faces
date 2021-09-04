# Variational-Auto-Encoder-for-Anime-Faces
This is keras implementation of the Variational Auto Encoder on anime data set. You can know more about Variational Auto Encoders here (https://arxiv.org/pdf/1312.6114.pdf). 
![image](https://user-images.githubusercontent.com/47930821/130594480-39502441-cdf3-4916-a22c-456f86955735.png)

---
# Auto-Encoders
Image compression and reconstruction has been a hot topic in AI since the invention of the AutoEncoders. This family of algorithms compresses the image into a smaller latent space and then reconstruct the image from that compressed latent space. This can save space of storing the data.

# Challenges with the Vanilla AutoEncoder
Compressing the data into a smaller latent space usually give little to no control of how the image will be represented into the new space. Most of the time, the latent space is discontinous. This makes the reconstruction sensitive to the latent values which may not be known in advance.
#provide an image here

---
# Variational AutoEncoders
Variational AutoEncoders manages to solve this problem by forcing the model to sample from a uniform gaussian distribution. Since gaussian distribution is easier to follow, it had been the choice for the latent space representation. The sampling layer in the middle is shown in details in this image

![image](https://user-images.githubusercontent.com/47930821/130595122-b917e092-13d4-48a0-9d03-4d16c79e856e.png)

---
# Prerequisites
1- python3 

2- CPU or NVIDIA GPU + CUDA CuDNN (GPU is recommended for faster inversion)

---


# Install dependencies
In this repo, a pretrained biggan in a specified library
```python
pip install torch torchvision matplotlib lpips numpy nltk cv2 pytorch-pretrained-biggan
```
---

# Loss functions
In order to have a good latent representation for the dataset as well as a good reconstruction to the image, two loss functions were combined


# Pixel-wise loss function 
There are multiple pixel-wise loss function that can be chosen for this task, but after multiple trials, L1 loss worked best. The reason behind this is because the gradient in the L1 loss doesn't depend on the value of the loss itself. Thus, the training becomes more consistent and avoids asymptotic behaviours.

![image](https://user-images.githubusercontent.com/47930821/130596676-1cc4bbc7-0afe-4357-99ec-eb26596d2404.png)
# KL-divergence loss
This loss function forces the mean and variance of the latent space in the middle to form a continous gaussian unit distribution. Thus, it would be possible to sample images directly from the latent space

![image](https://user-images.githubusercontent.com/47930821/130596401-a6222954-288b-4049-b8da-1b2dc3ee0fb6.png)

---
# Training
#provide image to work on
```python
python train.py  --num_epochs 2000 --learning_rate 0.007 
```
---
# Results

These images are drawn randomly from the latent space distribution

![image](https://user-images.githubusercontent.com/47930821/132110339-4cb2830a-5299-4582-add8-eb5f156db0d6.png)





