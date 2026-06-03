# 3D_MeshGen


A Diffuse Latent Model in Shapenet dataset

## Current State

<div class="container-imagens">
    <img src="imgs/ex1point.png" width = 230>
    <img src="imgs/ex1struc.png"  width = 230>
</div>

<div class="container-imagens">
    <img src="imgs/ex2point.png" width = 230, height= 131>
    <img src="imgs/ex2struc.png"  width = 230>
</div>

## I need To

1 - Solve Loss Stagnation (Normals) (EMD)

2 - Take care with KL collapse

3 - Init Gan structure

4 - Init Diffuse Latent

5 - Nksr Incorpore


## History:

### First: [A model can be finded in branch VIT_init]

Simple VAE with PointNet encoder and PointNet decoder. The loss is the Chamfer Distance between the input point cloud and the output point cloud. The model is trained on the Shapenet dataset, only one batch for training.

Results:

![Example1](imgs/Captura%20de%20tela%20de%202026-05-31%2015-36-39.png)

![Example2](imgs/Captura%20de%20tela%20de%202026-05-31%2017-27-38.png)

----------------------------------------------------------------------------------------------------

### Second: [A model can be finded in branch test]

The model is the same as the first one, but with a more robust MLP decoder. The loss is the Chamfer Distance between the input point cloud and the output point cloud. The model is trained on the Shapenet dataset, only one batch for training.


<div class="container-imagens">
    <img src="imgs/ex1point.png" width = 230>
    <img src="imgs/ex1struc.png"  width = 230>
</div>

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
