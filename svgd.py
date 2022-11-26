import numpy as np
from scipy.spatial.distance import pdist, squareform

class SVGD():
    def __init__(self):
        pass

    def svgd_kernel(self, theta, h = -1):
        sq_dist = pdist(theta)
        pairwise_dists = squareform(sq_dist)**2
        if h < 0: # if h < 0, using median trick
            h = np.median(pairwise_dists)
            h = np.sqrt(0.5 * h / np.log(theta.shape[0]+1))

        # compute the rbf kernel
        Kxy = np.exp( -pairwise_dists / h**2 / 2)

        dxkxy = -np.matmul(Kxy, theta)
        sumkxy = np.sum(Kxy, axis=1)
        for i in range(theta.shape[1]):
            dxkxy[:, i] = dxkxy[:,i] + np.multiply(theta[:,i],sumkxy)
        dxkxy = dxkxy / (h**2)
        return (Kxy, dxkxy)

    def update(self, x0, lnprob, n_iter = 1000, stepsize = 1e-3, bandwidth = -1, alpha = 0.9, debug = False):
        # Check input
        if x0 is None or lnprob is None:
            raise ValueError('x0 or lnprob cannot be None!')

        theta = np.copy(x0)

        # adagrad with momentum
        fudge_factor = 1e-6
        historical_grad = 0
        for iter in range(n_iter):
            if debug and (iter+1) % 1000 == 0:
                print('iter ' + str(iter+1))
            lnpgrad = lnprob.get_lnpgrad(theta)
            # calculating the kernel matrix
            kxy, dxkxy = self.svgd_kernel(theta, h = -1)
            grad_theta = (np.matmul(kxy, lnpgrad) + dxkxy) / x0.shape[0]

            # adagrad
            if iter == 0:
                historical_grad = historical_grad + grad_theta ** 2
            else:
                historical_grad = alpha * historical_grad + (1 - alpha) * (grad_theta ** 2)
            adj_grad = np.divide(grad_theta, fudge_factor+np.sqrt(historical_grad))
            theta = theta + stepsize * adj_grad

        return theta

    def update_ub(self, x0, lnprob, n_iter = 1000, stepsize = 1e-3, bandwidth = -1, alpha = 0.9, debug = False):
        # Check input
        if x0 is None or lnprob is None:
            raise ValueError('x0 or lnprob cannot be None!')

        theta = np.copy(x0)

        # adagrad with momentum
        fudge_factor = 1e-6
        historical_grad = 0
        for iter in range(n_iter):
            if debug and (iter+1) % 1000 == 0:
                print('iter ' + str(iter+1))
            lnpgrad = lnprob.get_lnpgrad_ub(theta)
            # calculating the kernel matrix
            kxy, dxkxy = self.svgd_kernel(theta, h = -1)
            grad_theta = (np.matmul(kxy, lnpgrad) + dxkxy) / x0.shape[0]

            # adagrad
            if iter == 0:
                historical_grad = historical_grad + grad_theta ** 2
            else:
                historical_grad = alpha * historical_grad + (1 - alpha) * (grad_theta ** 2)
            adj_grad = np.divide(grad_theta, fudge_factor+np.sqrt(historical_grad))
            theta = theta + stepsize * adj_grad

        return theta

class LnProb():
    def __init__(self, potential="gaussian", dimension=2, num_gaussians=2, gaussian_sigma=None, radius=4, nb_weight=np.array([1])):
        self.name, self.dim, self.num_gaussians, self.radius = potential, dimension, num_gaussians, radius
        self.std = gaussian_sigma[0]
        self.nb_weight = nb_weight

    def normal_pdf(self, x, mean=[0,0], std=0.5):
        ''' PDF of 2D Normal distribution. '''
        return 1 / (2*np.pi * std**2) * np.exp(-((x - mean)**2).sum(-1, keepdims=True) / (2 * std**2))

    def normal_pdf_grad(self, x, mean=[0,0], std=0.5):
        ''' Gradient - PDF of 2D Normal distribution. '''
        return - (x - mean) / (2*np.pi * std**4) * np.exp(-((x - mean)**2).sum(-1, keepdims=True) / (2 * std**2))

    def gmm_circle_pdf(self, x):
        ''' Mixture of Gaussians in a circle. '''
        mu 	= np.zeros((1, self.dim))
        ux 	= np.zeros((1, 1))
        for k in range(self.num_gaussians):
            mu[0, 0] = self.radius * np.cos(k*2*np.pi / self.num_gaussians)
            mu[0, 1] = self.radius * np.sin(k*2*np.pi / self.num_gaussians)
            ux 		 = ux + self.normal_pdf(x, mu, self.std)
        return ux / self.num_gaussians

    def gmm_circle(self, x):
        ''' Mixture of Gaussians in a circle. '''
        return np.log(self.gmm_circle_pdf(x))

    def gmm_circle_grad(self, x):
        ''' Gradient - Mixture of Gaussians in a circle. '''
        mu 	= np.zeros((1, self.dim))
        ux_grad = np.zeros((1, self.dim))
        for k in range(self.num_gaussians):
            mu[0, 0] 	= self.radius * np.cos(k*2*np.pi / self.num_gaussians)
            mu[0, 1]	= self.radius * np.sin(k*2*np.pi / self.num_gaussians)
            ux_grad     = ux_grad + self.normal_pdf_grad(x, mu, self.std)
        return ux_grad / self.num_gaussians / (self.gmm_circle_pdf(x))

    def get_lnpgrad(self, x):
        if self.name == "gmm_circle":
            lnprob = self.gmm_circle_grad(x)
        elif self.name == "gmm_square":
            lnprob = self.gmm_square_grad(x)
        else:
            raise ValueError("The Gaussian mixture model is not found!")
        return lnprob





    # ------------------- unbalanced weight ----------------------
    def gmm_circle_ub_pdf(self, x):
        ''' Mixture of Gaussians in a circle. '''
        mu 	= np.zeros((1, self.dim))
        ux 	= np.zeros((1, 1))
        for k in range(self.num_gaussians):
            mu[0, 0]   = self.radius * np.cos(k*2*np.pi / self.num_gaussians)
            mu[0, 1]   = self.radius * np.sin(k*2*np.pi / self.num_gaussians)
            ux         = ux + self.normal_pdf(x, mu, self.std) * self.nb_weight[k]
        return ux / self.num_gaussians

    def gmm_circle_ub(self, x):
        ''' Mixture of Gaussians in a circle. '''
        return -np.log(self.gmm_circle_ub_pdf(x))

    def gmm_circle_ub_grad(self, x):
        ''' Gradient - Mixture of Gaussians in a circle. '''
        mu 	= np.zeros((1, self.dim))
        ux_grad = np.zeros((1, self.dim))
        for k in range(self.num_gaussians):
            mu[0, 0]   = self.radius * np.cos(k*2*np.pi / self.num_gaussians)
            mu[0, 1]   = self.radius * np.sin(k*2*np.pi / self.num_gaussians)
            ux_grad    = ux_grad + self.normal_pdf_grad(x, mu, self.std) * self.nb_weight[k]
        return ux_grad / self.gmm_circle_ub_pdf(x)

    def get_lnpgrad_ub(self, x):
        if self.name == "gmm_circle":
            lnprob = self.gmm_circle_grad(x)
        elif self.name == "gmm_square":
            lnprob = self.gmm_square_grad(x)
        elif self.name == "gmm_circle_ub":
            lnprob = self.gmm_circle_ub_grad(x)
        elif self.name == "gmm_square_ub":
            lnprob = self.gmm_square_ub_grad(x)
        else:
            raise ValueError("The Gaussian mixture model is not found!")
        return lnprob
