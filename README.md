# Visual-Interaction-Networks
An implementation of Deepmind's visual interaction networks in Pytorch.

> From just a glance, humans can make rich predictions about the future state of a wide range of physical systems. On the other hand, modern approaches from engineering, robotics, and graphics are often restricted to narrow domains and require direct measurements of the underlying states. We introduce the Visual Interaction Network, a general-purpose model for learning the dynamics of a physical system from raw visual observations. Our model consists of a perceptual front-end based on convolutional neural networks and a dynamics predictor based on interaction networks. Through joint training, the perceptual front-end learns to parse a dynamic visual scene into a set of factored latent object representations. The dynamics predictor learns to roll these states forward in time by computing their interactions and dynamics, producing a predicted physical trajectory of arbitrary length. We found that from just six input video frames the Visual Interaction Network can generate accurate future trajectories of hundreds of time steps on a wide range of physical systems. Our model can also be applied to scenes with invisible objects, inferring their future states from their effects on the visible objects, and can implicitly infer the unknown mass of objects. Our results demonstrate that the perceptual module and the object-based dynamics predictor module can induce factored latent representations that support accurate dynamical predictions. This work opens new opportunities for model-based decision-making and planning from raw sensory observations in complex physical environments. 

[Watters, N., Tacchetti, A., Weber, T., Pascanu, R., Battaglia, P.W., & Zoran, D. (2017). Visual Interaction Networks. CoRR, abs/1706.01433.](https://arxiv.org/abs/1706.01433)
<div align="center">

<img align="center" hight="256" width="128" src="file://figures/VIN-example.gif">
</div>


## Architecture
<div align="center">
<img hight="600" width="600" src="https://github.com/Mrgemy95/visual-interaction-networks-pytorch/blob/master/figures/2.png?raw=true">
</div>


### Data
Run create_billards_data.py to create a dataset of bouncing billard balls, or supply your own data.

### Dependencies
Required packages:
``` 
Python 3.6
pytorch 0.4
numpy 1.15
scipy 1.1
```

Optional packages for visualization:
```
matploglib 3.0
visdom 0.1
imageio 2.4
```

### Run
- Edit configration file to your needs.
- Supply the data, for instance by running `create_billards_data.py`
- Run `vin.py`

### Thank you!
This repository was primarily created by refactoring and fixing
* https://github.com/MrGemy95/visual-interaction-networks-pytorch

Indirect sources include:
* https://github.com/jaesik817/visual-interaction-networks_tensorflow
* Ilya Sutskever for the [billard data code](http://www.cs.utoronto.ca/~ilya/code/2008/RTRBM.tar)


