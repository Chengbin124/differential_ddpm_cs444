# differential_ddpm_cs444
This is the final project of our Course: Deep Learning for Computer Vision at University of Illinois at Urbana Champaign. Our team members are Chengbin, Jerry and Ruiying.


Run prepare_data.py to download CIFAR10 and preprocess the dataset

Run /models/ddpm/ddpm_conditional.py to train a ddpm model using the previously downloaded dataset
Run /models/ddpm_differenital/ddpm_differential.py to train a ddpm model which differential self-attention

Each folder also includes an evaluate.py script which can be used to compute FID and IS.

There is a visualization script (visualizations.py) which is used in our project to generate images for our paper.


The runs/ folder includes generated samples generated every N epochs during training