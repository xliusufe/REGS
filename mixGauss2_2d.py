import numpy as np
import pandas as pd
import random
import time
import torch
import torch.optim as optim
from torchnet.meter import AverageValueMeter
import utils

class Config(object):
	max_epoch 		= 50000 # the max epoches
	dataSize 		= 2000  # the size of particles
	gpuDevice   	= True  # if GPU is used
	lrd         	= 5e-4  # 'learning rate for DNN, default=0.0005'
	batchSize		= 1000  # 'batch size'
	data_target 	= 'gm2_2d'
	eta				= 5e-4  # 'learning rate for particle update'
	nlD          	= 128   # 'width of hidden layers'
	nm_gauss		= 2     # 'number of components of mixture gaussian'

	radius			= 4.0    # 'radius of mixed Gaussian u(x)'
	target_std		= np.sqrt(0.03)   # 'std of mixed Gaussian u(x), default is 0.03'

	weights 		= 'equal'  # or 'unequal', or np.ones(8)
	base_mean		= 0.0 	# 'mean of base normal
	base_std		= 3.0   # 'std of base normal'
	init_std 		= 3.0   # 'std of initial normal
	init_mean 		= 0.0   # 'mean of initial normal

	is_decay 		= True  # if decay the learning rate
	lr_adjust 		= 10000 # decay the learning rate and step size (eta) after each epoch=3e4
	lrd_rate 		= 0.8   # decay the learning rate and step size (eta) for 0.8 after each epoch=3e4

	plot_period 	= 10000   # the number of epoches to plot scatter plot or other plots
	print_period	= 10000   # the number of epoches to print losses and other values
	save_period 	= 10000   # the number of epoches to save net and Gz
	Gz_nepoch   	= None    # 'checkpoints/gm/Gz_3gm_1000.pth'

	nd 				= 2      # dimension of target samples
	nz 				= 2		 # dimension of base sample
	T 				= 2      # the number of iteration to train R by Bregman divergence
	LOW 			= -5     # the lower limit in the plots
	HIGH 			= 5      # the upper limit in the plots
	thresh 			= 0.6    # None is defalt (purple) or 0.8
	seeds 			= None
	LOW_mesh		= -5     # the lower limit in the plots
	HIGH_mesh		= 5      # the upper limit in the plots
	num_mesh 		= 1000
	ns_plots 		= 500
	ns_evaluation   = None

opt = Config()

#-------------------- main ---------------------
def train(**kwargs):
	for k_, v_ in kwargs.items():
		setattr(opt, k_, v_)
	utils.makedir(opt)

	dataSize 	= opt.dataSize
	data_target = opt.data_target
	eta 		= opt.eta
	lrd 		= opt.lrd

	if opt.weights == 'unequal':
		ms = int(opt.nm_gauss/2)
		ws = np.hstack( (np.ones(ms), 3*np.ones(opt.nm_gauss-ms)) )
		opt.weights = ws/sum(ws)
	elif opt.weights == "equal":
		ms = opt.nm_gauss
		ws = np.ones(ms)
		opt.weights = ws/sum(ws)
	else:
		ms = opt.nm_gauss
		opt.weights = np.ones(ms)/sum(np.ones(ms))
	print(opt.weights)

	device 		= torch.device('cuda') if opt.gpuDevice else torch.device('cpu')
	opt.device 	= device
	# torch.pi 	= torch.tensor(torch.pi)
	target_std 	= torch.tensor(opt.target_std).to(device)
	base_std 	= torch.tensor(opt.base_std).to(device)
	DRF 		= utils.DRF_def(opt)
	DRF_plots 	= utils.DRF_plots(opt)
	MESH_DF 	= utils.MeshDensityGMM(opt)

	netD 		= utils.D_mlp(opt.nd, opt.nlD)
	netD.apply(utils.weights_initD)
	netD.to(device)


	start_epoch = 0
	if opt.Gz_nepoch:
		net_path = 'checkpoints/%s/net_gm_%s.pth' % (data_target, opt.Gz_nepoch)
		map_location = lambda storage, loc: storage
		state = torch.load(net_path, map_location=map_location)
		netD.load_state_dict(state['netD'])
		base_std 		= state['base_std'].clone().detach().to(device)
		target_std 		= state['target_std'].clone().detach().to(device)
		lrd 			= state['lrd']
		eta 			= state['eta']
		start_epoch = opt.Gz_nepoch

	LOSS_DR 	= []
	Grad_norm 	= []

	Gz 			= torch.FloatTensor(dataSize, opt.nd).to(device)
	fake_X 		= torch.FloatTensor(opt.batchSize, opt.nd).to(device)
	base_X 		= torch.FloatTensor(opt.batchSize, opt.nd).to(device)
	Gz_tmp 		= torch.FloatTensor(dataSize, opt.nd).to(device)
	X_mesh, Y_mesh, density_mesh = MESH_DF.get_mesh_density_gmm()

	if opt.seeds is None:
		opt.seeds = random.randint(1, 10000)
	random.seed(opt.seeds)
	torch.manual_seed(opt.seeds)

	# set optimizer
	optim_D 	= optim.RMSprop(netD.parameters(), lr=lrd)
	schedulerD 	= optim.lr_scheduler.StepLR(optim_D, step_size = opt.lr_adjust, gamma=opt.lrd_rate)
	error_meter = AverageValueMeter()

	if opt.Gz_nepoch:
		source_data = pd.read_csv('./checkpoints/%s/Gz_gm_%s.csv' % (data_target, opt.Gz_nepoch), index_col = 0 )
		Gz = torch.from_numpy(source_data.to_numpy()).float().to(device)
		loss_save = pd.read_csv('./checkpoints/%s/loss_%sgm_%s.csv' % (data_target, opt.nm_gauss, opt.Gz_nepoch), index_col = 0 )
		loss = loss_save.to_numpy()
		LOSS_DR 	= loss[:,1].tolist()
		Grad_norm 	= loss[:,2].tolist()
	else:
		Gz = DRF.base_sampler(opt.init_mean, opt.init_std, dataSize)
	epochs = range(start_epoch, start_epoch + opt.max_epoch)
	for epoch in iter(epochs):
		netD.train()
		loss_iter_dr = []
		update_netD = utils.update_netD(opt)
		update_netD.update(netD, optim_D, Gz, base_X, fake_X, target_std, loss_iter_dr)

		if opt.is_decay:
			schedulerD.step()

		# update particles
		Gz_tmp = Gz.clone()
		Gz_tmp.requires_grad_(True)
		if Gz_tmp.grad is not None:
			Gz_tmp.grad.zero_()
		Gz_ratio = - netD(Gz_tmp)
		s = torch.ones_like(Gz_ratio.detach())
		s.unsqueeze_(1).expand_as(Gz).to(device)
		Gz_ratio.backward(torch.ones(len(Gz)).to(device))
		Gz = Gz - eta * s * Gz_tmp.grad

		Grad_norm.append(Gz_tmp.grad.norm(p=2).detach().cpu().item())
		LOSS_DR.append(np.mean(loss_iter_dr))

		if (epoch +1) % opt.print_period == 0 or epoch == 0:
			print('epoch(%s)%d: %.4f | %.4f | %.4f | %.4f|'
				% (data_target, epoch + 1, LOSS_DR[-1], Grad_norm[-1], opt.radius, target_std),
				time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
				)
			idx_epoch 	= np.arange(epoch+1) + 1
			saved_loss 	= np.vstack((idx_epoch, np.array(LOSS_DR), np.array(Grad_norm))).transpose()
			columns_name = ['epoch', 'Loss_dr', 'Grad_norm']
			dataframe 	= pd.DataFrame(saved_loss, index=idx_epoch, columns=columns_name)
			dataframe.to_csv('./checkpoints/%s/loss_%sgm_%s.csv' % (data_target, opt.nm_gauss, epoch+1), sep=',')

		if (epoch + 1) % opt.plot_period == 0 or epoch == 0:
			DRF_plots 	= utils.DRF_plots(opt)
			Gz_plot 	= Gz.detach().cpu().numpy()
			DRF_plots.plot_scatter(Gz_plot, data_target, epoch+1)
			DRF_plots.plot_scatter_with_contour(X_mesh, Y_mesh, density_mesh, Gz_plot, data_target, epoch+1)

		if (epoch+1) % opt.save_period == 0:
			Gz_save = Gz.cpu().numpy()
			dataframe = pd.DataFrame(Gz_save)
			dataframe.to_csv('checkpoints/%s/Gz_gm_%s.csv' % (data_target, 1+epoch), sep=',')
			net_path = 'checkpoints/%s/net_gm_%s.pth' % (data_target, 1+epoch)
			state = {
				'netD': 		netD.state_dict(),
				'target_std':	target_std,
				'base_std':		base_std,
				'lrd':			schedulerD.get_last_lr()[0],
				'eta':			eta,
				'epoch': 		epoch
			}
			torch.save(state, net_path)
			error_meter.reset()

def evaluate(**kwargs):
	# Evaluate the generator by H(x) including mean, variance and cos(alpha*X + 0.5)
	for k_, v_ in kwargs.items():
		setattr(opt, k_, v_)
	#device		= torch.device('cuda') if opt.gpuDevice else torch.device('cpu')
	device 		= torch.device('cpu')
	opt.device 	= device
	DRF 		= utils.DRF_def(opt)
	if opt.Gz_nepoch is None: opt.Gz_nepoch = 1
	data_target = opt.data_target
	if opt.weights == 'unequal':
		ms = int(opt.nm_gauss/2)
		ws = np.hstack( (np.ones(ms), 3*np.ones(opt.nm_gauss-ms)) )
		opt.weights = ws/sum(ws)
	elif opt.weights == "equal":
		ms = opt.nm_gauss
		ws = np.ones(ms)
		opt.weights = ws/sum(ws)
	else:
		ms = opt.nm_gauss
		opt.weights = np.ones(ms)/sum(np.ones(ms))

	Hx_gen_samp = []
	test_vector = (torch.ones(opt.nd,1)/np.sqrt(opt.nd)).to(device)
	Gz 			= DRF.gmm2_sampler_circ(opt.target_std, opt.dataSize)

	val_trSam 	= (torch.mm(Gz, test_vector)**2).sum()/opt.dataSize
	mean_trSam 	= Gz.mm(test_vector).mean()
	cos_trSam 	= 10*torch.cos(Gz.mm(test_vector)+0.5).mean()
	Hx_gen_samp.append([mean_trSam.numpy(), val_trSam.numpy(), cos_trSam.numpy()])

	source_data = pd.read_csv('./checkpoints/%s/Gz_gm_%s.csv' % (data_target, opt.Gz_nepoch), index_col = 0 )
	Gz = torch.from_numpy(source_data.to_numpy()).float().to(device)

	if opt.ns_evaluation is None:
		Gz_eval     = Gz
	else:
		idx_plot    = random.sample(range(opt.dataSize), opt.ns_evaluation)
		Gz_eval          = Gz[idx_plot]
	val_estimated 	= (torch.mm(Gz_eval, test_vector)**2).sum()/opt.dataSize
	mean_estimated 	= Gz_eval.mm(test_vector).mean()
	cos_estimated 	= 10*torch.cos(Gz_eval.mm(test_vector)+0.5).mean()
	Hx_gen_samp.append([mean_estimated.numpy(), val_estimated.numpy(), cos_estimated.numpy()])
	saved_loss 		= np.array(Hx_gen_samp, dtype=object)
	columns_name 	= ['moment1', 'moment2', 'cos']
	dataframe 		= pd.DataFrame(saved_loss, columns=columns_name)
	dataframe.to_csv('./checkpoints/%s/evaluation_%s_%s.csv' % (data_target, data_target, opt.Gz_nepoch), sep=',')

	ns_plots = opt.ns_plots
	idx_plot = random.sample(range(opt.dataSize), ns_plots)
	Gz = Gz[idx_plot]
	MESH_DF 	= utils.MeshDensityGMM(opt)
	DRF_plots 	= utils.DRF_plots(opt)
	X_mesh, Y_mesh, density_mesh = MESH_DF.get_mesh_density_gmm()
	DRF_plots.plot_scatter_with_contour(X_mesh, Y_mesh, density_mesh, Gz, data_target, opt.Gz_nepoch)

if __name__ == '__main__':
    import fire
    fire.Fire()
