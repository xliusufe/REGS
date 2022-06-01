# Relative Entropy Gradient Sampler for Unnormalized Distributions

This repository is the official implementation of [Relative Entropy Gradient Sampler for Unnormalized Distributions].

## Introduction
The **Relative Entropy Gradient Sampler (REGS)** is proposed for sampling from unnormalized densties by the techniques including Wasserstein gradient flows, numerical ODEs,  density-ratio estimation and deep neural networks. REGS achieves a fantastic numerical performance on 2D mixtrues of Gaussian distributions with a large number of modes, a small variance, and a large distance between any two modes.

## Dependencies
* Python 3.7.7
* PyTorch 1.4.0
* torchvision 0.5.0

## Experimental Results

### Toy Examples

To run REGS for the target distribution being a mixtrue of 2 Gaussians on GPU devices, use:

```
python mixGauss2_2d.py train
```

To run REGS for the target distribution being a mixtrue of 8 Gaussians on GPU devices, use:

```
python mixGauss8_2d.py train
```

## Usage

Useful arguments

	max_epoch 		= 50000 # the max epoches
	dataSize 		= 2000  # the size of particles
	gpuDevice		= True  # if GPU is used
	lrd			= 5e-4  # 'learning rate for DNN, default=0.0005'
	batchSize		= 1000  # 'batch size'
	data_target		= 'gm2_2d'
	eta			= 5e-4  # 'learning rate for particle update'
	nlD			= 128   # 'width of hidden layers'
	nm_gauss		= 2     # 'number of components of mixture gaussian'

	radius			= 4.0    # 'radius of mixed Gaussian u(x), the distance of mean from origin'
	target_std		= np.sqrt(0.03)   # 'std of mixed Gaussian u(x), default is 0.4'

	base_mean		= 0.0 	# 'mean of base normal
	base_std		= 3.0   # 'std of base normal'
	init_std 		= 3.0   # 'std of initial normal
	init_mean 		= 0.0   # 'mean of initial normal

Run on CPU devices:

```
python mixGauss2_2d.py train --gpuDevice=False
```

Change the radius:

```
python mixGauss2_2d.py train --radius=3.0
```

Evaluate the numerical performance after 50000 iterations via `E[H(x)]` including mean, variance and `cos(alpha*X + 0.5)`

```
python mixGauss2_2d.py evaluate --Gz_nepoch=50000
```

More details can be found in the codes file.

# References

Feng, X. Gao, Y. Huang, J. Jiao, Y. and Liu, X. Relative Entropy Gradient Sampler for Unnormalized Distributions. Manuscript. [http://arxiv.org/abs/2110.02787](http://arxiv.org/abs/2110.02787).