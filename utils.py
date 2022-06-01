import os
import torch
import random
import torch.nn as nn
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

def makedir(opt):
	try:
		os.mkdir('./figure')
	except:
		pass
	try:
		os.mkdir('./checkpoints')
	except:
		pass

	try:
		os.mkdir('./figure/%s' % opt.data_target)
	except:
		pass
	try:
		os.mkdir('./checkpoints/%s' % opt.data_target)
	except:
		pass

def weights_initD(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)

    elif classname.find('BatchNorm2d') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0.0)

    elif classname.find('Linear') != -1:
        nn.init.xavier_uniform_(m.weight, gain=nn.init.calculate_gain('relu'))
        # m.weight.data.normal_(0.0, 0.02)
        m.bias.data.fill_(0.0)

#--------------------- net --------------------
class D_mlp(nn.Module):
	def __init__(self, nx, nl):
		super(D_mlp, self).__init__()
		main = nn.Sequential(
			nn.Linear(nx, nl),
			nn.LeakyReLU(0.2, inplace=True),

			nn.Linear(nl, nl),
			nn.LeakyReLU(0.2, inplace=True),

			nn.Linear(nl, 1)
			)
		self.main = main

	def forward(self, x):
		out = self.main(x)
		return out.squeeze()

class D_mlp8(nn.Module):
	def __init__(self, nx, nl):
		super(D_mlp8, self).__init__()
		main = nn.Sequential(
			nn.Linear(nx, nl),
			nn.LeakyReLU(0.2, inplace=True),

			nn.Linear(nl, nl),
			nn.LeakyReLU(0.2, inplace=True),

			nn.Linear(nl, nl),
			nn.LeakyReLU(0.2, inplace=True),

			nn.Linear(nl, 1)
			)
		self.main = main

	def forward(self, x):
		out = self.main(x)
		return out.squeeze()

class DRF_def():
	def __init__(self, opt):
		self.device 	= opt.device
		self.base_mean 	= torch.tensor(opt.base_mean).to(opt.device)
		self.base_std	= torch.tensor(opt.base_std).to(opt.device)
		self.nd 		= opt.nd
		if hasattr(opt, 'nm_gauss'):
			self.nm_gauss	= torch.tensor(opt.nm_gauss).int().to(opt.device)
		if hasattr(opt, 'radius'):
			self.radius		= torch.tensor(opt.radius).to(opt.device)
		if hasattr(opt, 'weights'):
			if opt.weights is None:
				weight = self.nm_gauss - torch.arange(self.nm_gauss).to(opt.device)
				weight = weight/torch.sum(weight)
			else:
				self.weights 	= torch.tensor(opt.weights).to(opt.device)
		if hasattr(opt, 'gmm_type'):
			self.gmm_type = opt.gmm_type

	def normal_pdf(self, x, base_mean, base_std):
		if len(x.size()) > 1:
			return 1 / (torch.sqrt(2*torch.pi) * base_std**2) * torch.exp(-((x-base_mean)**2).sum(-1) / (2 * base_std**2))
		else:
			return 1 / (torch.sqrt(2*torch.pi) * base_std**2) * torch.exp(-((x-base_mean)**2) / (2 * base_std**2))

	def target_gmm_circ(self, x, target_std):
		# u(x)
		mu 	= torch.FloatTensor(self.nd).to(self.device)
		ux 	= torch.zeros(x.size(0)).float().to(self.device)
		ks = torch.tensor(range(self.nm_gauss)).to(self.device)
		for k in ks:
			mu[0] 	= self.radius * torch.cos(k*2*torch.pi/(self.nm_gauss))
			mu[1]	= self.radius * torch.sin(k*2*torch.pi/(self.nm_gauss))
			ux 		+= self.normal_pdf(x, mu, target_std)
		return ux

	def dr_circ(self, x, target_std):
		return self.target_gmm_circ(x, target_std)/self.normal_pdf(x, self.base_mean, self.base_std)

	def base_sampler(self, base_mean, base_std, batch_size):
		ux = torch.normal(torch.tensor(0), torch.tensor(base_std), size=(batch_size, self.nd)) + torch.tensor(base_mean)
		return ux.to(self.device)

	def gmm2_sampler_circ(self, std, batch_size):
		true_radius = self.radius.cpu().numpy()
		nm_gauss = self.nm_gauss.cpu().numpy()
		ux 	= torch.zeros(batch_size, self.nd)
		ks = range(nm_gauss)
		ms = int(batch_size/(nm_gauss))
		cov0 =np.diag([std**2,std**2])
		mu 	= torch.FloatTensor(self.nd)
		i_start = 0
		for k in ks:
			i_end = i_start + ms
			mu[0] = true_radius*np.cos(2*k*np.pi/(nm_gauss))
			mu[1] = true_radius*np.sin(2*k*np.pi/(nm_gauss))
			ux[i_start:i_end] = torch.tensor(np.random.multivariate_normal(
				mean 	= mu,
				cov 	= cov0,
				size 	= ms
			))
			i_start = i_end
		return ux.to(self.device)

class MeshDensityGMM():
	def __init__(self, opt):
		self.std 		= opt.target_std
		self.HIGH   	= opt.HIGH_mesh
		self.LOW   		= opt.LOW_mesh
		self.num_mesh	= opt.num_mesh
		self.dim 		= opt.nd
		self.radius 	= opt.radius
		if hasattr(opt, 'nm_gauss'):
			self.nm_gauss	= opt.nm_gauss

	def normal_pdf(self, x, mean=0, std=0.5):
		''' PDF of 2D Normal distribution. '''
		mesh_X, mesh_Y = np.meshgrid(x, x)
		return 1 / (2*np.pi * std**2) * np.exp(-(np.power(mesh_X - mean[0], 2) + np.power(mesh_Y - mean[1], 2)) / (2 * std**2))

	def gmm_circle_pdf(self, x):
		''' Mixture of Gaussians in a 2D circle. '''
		mu 	= np.zeros(self.dim)
		ux 	= np.zeros((self.num_mesh, self.num_mesh))
		for k in range(self.nm_gauss):
			mu[0] 	= self.radius * np.cos(k*2*np.pi / self.nm_gauss)
			mu[1]	= self.radius * np.sin(k*2*np.pi / self.nm_gauss)
			ux 		+= self.normal_pdf(x, mu, self.std)
		return ux/self.nm_gauss

	def get_mesh_density_gmm(self):
		''' Mesh data for contour plots. '''
		x = np.linspace(self.LOW, self.HIGH, self.num_mesh)
		mesh_X, mesh_Y = np.meshgrid(x, x)
		mesh_density = self.gmm_circle_pdf(x)
		return mesh_X, mesh_Y, mesh_density

class update_netD(DRF_def):
	def __init__(self, opt):
		super().__init__(opt)
		self.device 	= opt.device
		self.base_mean 	= opt.base_mean
		self.base_std	= opt.base_std
		self.batchSize 	= opt.batchSize
		self.dataSize 	= opt.dataSize
		self.T			= opt.T

	def update(self, netD, optim_D, Gz, base_X, fake_X, target_std, loss_iter_dr):
		for t in range(self.T):
			# update D
			netD.zero_grad()
			base_X 		= self.base_sampler(self.base_mean, self.base_std, self.batchSize)
			fake_idx 	= random.sample(range(self.dataSize), self.batchSize)
			fake_X 		= Gz[fake_idx].clone()

			fake_X.requires_grad_(True)
			if fake_X.grad is not None:
				fake_X.grad.zero_()
			D_fake_X = netD(fake_X)
			loss_dr = torch.exp(D_fake_X).mean() - (netD(base_X)*self.dr_circ(base_X, target_std)).mean()

			loss_dr.backward()
			optim_D.step()
			loss_iter_dr.append(loss_dr.detach().cpu().item())

class DRF_plots():
    def __init__(self, opt):
        super(DRF_plots, self).__init__()
        self.HIGH   = opt.HIGH
        self.LOW    = opt.LOW
        self.thresh = opt.thresh

    def plot_scatter(self, Gz_plot, data_target, epoch):
        HIGH 	= self.HIGH
        LOW		= self.LOW
        plt.figure(figsize=(HIGH, HIGH))
        plt.scatter(Gz_plot[:, 0], Gz_plot[:, 1], s=0.1)
        plt.xticks([LOW, HIGH])
        plt.yticks([LOW, HIGH])
        plt.savefig("./figure/%s/train_scatter_%s_%s.pdf" % (data_target, data_target, epoch))
        plt.close()

    def plot_scatter_with_contour(self, X_mesh, Y_mesh, density_mesh, Gz_plot, data_target, epoch=0):
        HIGH 	= self.HIGH
        LOW		= self.LOW
        plt.figure(figsize=(HIGH, HIGH))
        plt.scatter(Gz_plot[:, 0], Gz_plot[:, 1], s=0.1, color = 'r')
        plt.contour(X_mesh, Y_mesh, density_mesh, linewidths=0.2)
        plt.xticks([LOW, HIGH])
        plt.yticks([LOW, HIGH])
        plt.savefig("./figure/%s/scatter_with_contour_%s_%s.pdf" % (data_target, data_target, epoch))
        plt.close()