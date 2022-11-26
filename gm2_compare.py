import os
import random
import numpy as np
import pandas as pd
import scipy
import matplotlib.pyplot as plt
from mcmc import Sampler
from svgd import SVGD, LnProb
from utils import makedir_others, ParticlePlots
from utils import MeshDensityGMM_ub as MeshDensityGMM

class Config(object):
	potential           = "gmm_circle"
	num_gaussians       = 2
	radius              = 4
	var_list            = [0.03]
	algorithm           = "MALA" # or 'ULA' or 'SVGD'
	step_size           = 0.02

	weights 		    = 'equal'  # or 'unequal', or np.ones(8)
	dimension           = 2
	burn_in             = 5000
	n_particles_svgd    = 500*num_gaussians
	n_particles         = 2000*num_gaussians
	n_samples           = burn_in + n_particles
	n_samples_plot      = 1000
	num_mesh            = 1000
	single_var          = False
	high                = 5
	low                 = - high
	ns_evaluation       = None
	ns_plot             = 1000

	num_chains          = 1

opt = Config()
#--------------------------------------------------------
def train(**kwargs):
	for k_, v_ in kwargs.items():
		setattr(opt, k_, v_)

	potential       = opt.potential
	dimension       = opt.dimension
	num_gaussians   = opt.num_gaussians
	radius          = opt.radius
	var_list        = opt.var_list
	high, low       = opt.high, opt.low
	burn_in         = opt.burn_in
	n_samples       = opt.n_samples
	algorithm       = opt.algorithm
	step_size       = opt.step_size
	num_chains      = opt.num_chains
	x_init          = np.random.normal(loc=0.0, scale=2.0, size=(num_chains, 2))
	
 
	if opt.weights == 'unequal':
		ms = int(num_gaussians/2)
		ws = np.hstack( (np.ones(ms), 3*np.ones(num_gaussians-ms)) )
		nb_weight = ws/sum(ws)
		potential = "gmm_circle"+"_ub"
		opt.potential = potential
	elif opt.weights == "equal":
		ms = num_gaussians
		ws = np.ones(ms)
		nb_weight = ws/sum(ws)
	else:
		ms = num_gaussians
		nb_weight = np.ones(ms)/sum(np.ones(ms))
	print(nb_weight)

	for kk in range(len(var_list)):
		var = var_list[kk]
		print('var = ', var)
		std = np.sqrt(var)

		if algorithm in ["SVGD"]:
			data_target = "%s_num%s_var%s" % (algorithm, str(num_gaussians), str(kk))
			makedir_others(data_target)
			n_samples_pick = opt.n_particles_svgd
			x0 = np.random.multivariate_normal([0, 0], [[1, 0], [0, 1]], n_samples_pick)
			lnprob = LnProb(potential=potential, dimension=dimension, num_gaussians=num_gaussians, gaussian_sigma=np.array([std, std]), radius=radius, nb_weight=nb_weight)
			svgd = SVGD()
			samples = svgd.update_ub(x0, lnprob, n_iter=2000, stepsize=step_size, bandwidth=-1, alpha=0.9)

		elif algorithm in ["ULA", "MALA", "tULA", "tMALA"]:
			data_target = "%s_num%s_burnin%s_nchain%s_var%s" % (algorithm, str(num_gaussians), str(burn_in), str(num_chains), str(kk))
			makedir_others(data_target)
			n_samples_pick = (n_samples - burn_in) * num_chains
			s = Sampler(potential=potential, dimension=dimension, step=step_size, num_gaussians=num_gaussians,
							gaussian_sigma=np.array([std, std]), radius=radius, nb_weight=nb_weight)
			samples = s.get_samples(algorithm=algorithm, burn_in=burn_in, n_chains=num_chains, x0=x_init, n_samples=n_samples, measuring_points=None, timer=None)
		else:
			raise ValueError("The used algorithm is not found!")

		dataframe = pd.DataFrame(samples)
		dataframe.to_csv('./mixGauss/%s/Gz_gm.csv' %data_target, sep=',')

		idx_plot = random.sample(range(n_samples_pick), opt.n_samples_plot)
		samples_plot = samples[idx_plot]

		mesh_density_gmm = MeshDensityGMM(potential=potential, dimension=dimension, num_gaussians=num_gaussians, gaussian_sigma=np.array([std, std]), radius=radius, HIGH=high, LOW=low, num_mesh=opt.num_mesh, nb_weight=nb_weight)
		X_mesh, Y_mesh, density_mesh = mesh_density_gmm.get_mesh_density_gmm()

		if opt.ns_plot is not None:
			particle_plots = ParticlePlots(HIGH=opt.high, LOW=opt.low, thresh=0.1)
			particle_plots.plot_scatter(samples_plot, data_target)
			particle_plots.plot_kde(samples_plot, data_target)
			particle_plots.plot_scatter_with_contour_eval(X_mesh=X_mesh, Y_mesh=Y_mesh, density_mesh=density_mesh, Gz_plot=samples_plot, data_target=data_target)

		if opt.single_var:
			break
		else:
			pass

def evaluation(**kwargs):
	# Evaluate the generator by H(x) including mean, variance and cos(alpha*X + 0.5)
	for k_, v_ in kwargs.items():
		setattr(opt, k_, v_)

	num_gaussians   = opt.num_gaussians
	if opt.weights == 'unequal':
		ms = int(num_gaussians/2)
		ws = np.hstack( (np.ones(ms), 3*np.ones(num_gaussians-ms)) )
		nb_weight = ws/sum(ws)
		potential = "gmm_circle"+"_ub"
		opt.potential = potential
	elif opt.weights == "equal":
		ms = num_gaussians
		ws = np.ones(ms)
		nb_weight = ws/sum(ws)
	else:
		ms = num_gaussians
		nb_weight = np.ones(ms)/sum(np.ones(ms))
	print(nb_weight)

	dataSize = (opt.n_samples - opt.burn_in) * opt.num_chains
	for kk in range(len(opt.var_list)):
		var = opt.var_list[kk]
		print('var = ', var)
		std = np.sqrt(var)

		mesh_density_gmm = MeshDensityGMM(potential=opt.potential, dimension=opt.dimension, num_gaussians=opt.num_gaussians,
											gaussian_sigma=np.array([std, std]), radius=opt.radius,
											HIGH=opt.high, LOW=opt.low, num_mesh=opt.num_mesh, nb_weight=nb_weight)
		data_target = "%s_num%s_burnin%s_nchain%s_var%s" % (opt.algorithm, str(opt.num_gaussians), str(opt.burn_in), str(opt.num_chains), str(kk))
		makedir_others(data_target)

		Hx_gen_samp = []
		test_vector = np.ones([opt.dimension,1])/np.sqrt(opt.dimension)
		mean_true 	= 0.0
		val_true 	= std**2 + mean_true**2
		cos_true 	= 0.0
		Gz 			= mesh_density_gmm.gmm2_sampler_circ(std, dataSize)
		val_trSam 	= (np.matmul(Gz, test_vector)**2).sum()/dataSize
		mean_trSam 	= np.matmul(Gz, test_vector).mean()
		cos_trSam 	= 10*np.cos(np.matmul(Gz,test_vector)+0.5).mean()
		Hx_gen_samp.append([mean_true, val_true, cos_true])
		Hx_gen_samp.append([mean_trSam, val_trSam, cos_trSam])

		Gz = pd.read_csv('./mixGauss/%s/Gz_gm.csv' %data_target, index_col = 0 ).to_numpy()

		if opt.ns_evaluation is None:
			Gz_eval     = Gz
		else:
			idx_plot    = random.sample(range(dataSize), opt.ns_evaluation)
			Gz_eval          = Gz[idx_plot]

		val_estimated 	= (np.matmul(Gz_eval, test_vector)**2).sum()/dataSize
		mean_estimated 	= np.matmul(Gz_eval,test_vector).mean()
		cos_estimated 	= 10*np.cos(np.matmul(Gz_eval,test_vector)+0.5).mean()
		Hx_gen_samp.append([mean_estimated, val_estimated, cos_estimated])
		saved_loss 		= np.array(Hx_gen_samp, dtype=object)
		columns_name 	= ['moment1', 'moment2', 'cos']
		dataframe 		= pd.DataFrame(saved_loss, columns=columns_name)
		dataframe.to_csv('./evaluation/evaluation_%s.csv' %data_target, sep=',')

		if opt.ns_plot is not None:
			idx_plot = random.sample(range(dataSize), opt.ns_plot)
			samples_plot = Gz[idx_plot]
			X_mesh, Y_mesh, density_mesh = mesh_density_gmm.get_mesh_density_gmm()

			particle_plots = ParticlePlots(HIGH=opt.high, LOW=opt.low, thresh=0.1)
			particle_plots.plot_scatter(samples_plot, data_target)
			particle_plots.plot_kde(samples_plot, data_target)
			particle_plots.plot_scatter_with_contour_eval(X_mesh=X_mesh, Y_mesh=Y_mesh, density_mesh=density_mesh, Gz_plot=samples_plot, data_target=data_target)

def proportion(**kwargs):
	# Evaluate the generator by H(x) including mean, variance and cos(alpha*X + 0.5)
	for k_, v_ in kwargs.items():
		setattr(opt, k_, v_)

	num_gaussians   = opt.num_gaussians
	if opt.weights == 'unequal':
		ms = int(num_gaussians/2)
		ws = np.hstack( (np.ones(ms), 3*np.ones(num_gaussians-ms)) )
		nb_weight = ws/sum(ws)
		potential = "gmm_circle"+"_ub"
		opt.potential = potential
	elif opt.weights == "equal":
		ms = num_gaussians
		ws = np.ones(ms)
		nb_weight = ws/sum(ws)
	else:
		ms = num_gaussians
		nb_weight = np.ones(ms)/sum(np.ones(ms))

	dataSize = (opt.n_samples - opt.burn_in) * opt.num_chains
	for kk in range(len(opt.var_list)):
		var = opt.var_list[kk]
		print('var = ', var)
		std = np.sqrt(var)

		mesh_density_gmm = MeshDensityGMM(potential=opt.potential, dimension=opt.dimension, num_gaussians=opt.num_gaussians,
											gaussian_sigma=np.array([std, std]), radius=opt.radius,
											HIGH=opt.high, LOW=opt.low, num_mesh=opt.num_mesh, nb_weight=nb_weight)
		data_target = "%s_num%s_burnin%s_nchain%s_var%s" % (opt.algorithm, str(opt.num_gaussians), str(opt.burn_in), str(opt.num_chains), str(kk))
		makedir_others(data_target)
		Gz = pd.read_csv('./mixGauss/%s/Gz_gm.csv' %data_target, index_col = 0 ).to_numpy()

		if opt.ns_plot is None:
			Gz_eval     = Gz
		else:
			idx_plot    = random.sample(range(dataSize), opt.ns_plot)
			Gz_eval          = Gz[idx_plot]
		prop = mesh_density_gmm.proportion_circ(Gz_eval, opt.radius)
		prop = 1 - prop

		saved_loss 		= np.array(prop, dtype=object)
		columns_name 	= ['prop']
		dataframe 		= pd.DataFrame(saved_loss, columns=columns_name)
		dataframe.to_csv('./proportion/prop_%s.csv' %data_target, sep=',')

if __name__ == '__main__':
	import fire
	fire.Fire()