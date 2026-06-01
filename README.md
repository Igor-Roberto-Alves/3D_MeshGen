# 3D_MeshGen


A Diffuse Latent Model in Shapenet dataset

## Current State

Example 1:

![Example1](imgs/Captura%20de%20tela%20de%202026-05-31%2015-36-39.png)

Example 2:

![Example2](imgs/Captura%20de%20tela%20de%202026-05-31%2017-27-38.png)


## I need To

1 - Solve Loss Stagnation

2 - Solve KL Collapse

3 - Init Diffuse

4 - Nksr Incorpore


## History:

### First: [A model can be finded in branch VIT_init]

Simple VAE with PointNet encoder and PointNet decoder. The loss is the Chamfer Distance between the input point cloud and the output point cloud. The model is trained on the Shapenet dataset, only one batch for training.

Results:

![Example1](imgs/Captura%20de%20tela%20de%202026-05-31%2015-36-39.png)

![Example2](imgs/Captura%20de%20tela%20de%202026-05-31%2017-27-38.png)

----------------------------------------------------------------------------------------------------

### Second: [A model can be finded in branch VQ-GAN]

Not yet