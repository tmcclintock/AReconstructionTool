import numpy as np
import george
from george import kernels
import pyDOE2
from scipy.optimize import minimize

class resampler(object):
    """A class used for enabling importance sampling of an MCMC chain.

    Note: the 'scale' parameter is not used if you use the `lnlikes_binning`
    method of choosing training points.

    Args:
        chain: an MCMC chain that we want to draw samples from
        lnlikes: the log-likelihoods of the samples in the MCMC chain
        scale: 'spread' of the training points. Default is 6.
    """
    def __init__(self, chain, lnlikes, scale=5):
        self.scale = scale
        
        chain = np.asarray(chain)
        lnlikes = np.asarray(lnlikes).copy()
        
        if chain.ndim > 2:
            raise Exception("chain cannot be more than two dimensional.")
        if chain.ndim < 1:
            raise Exception("chain must be a list of parameters.")
        if lnlikes.ndim > 1:
            raise Exception("lnlikes must be a 1D array.")
        
        self.chain = np.atleast_2d(chain)
        if len(self.chain) < len(self.chain[0]):
            raise Exception("More samples than parameters in chain.")

        #Remove the max lnlike. This can help numerical stability
        self.lnlike_max = np.max(lnlikes)
        lnlikes -= self.lnlike_max
        
        self.lnlikes = lnlikes

        #Compute the rotation matrix of the chain
        self.chain_means = np.mean(chain, 0)
        self.chain_cov = np.cov(self.chain.T)
        w, R = np.linalg.eig(self.chain_cov)
        self.eigenvalues = w
        self.rotation_matrix = R

        self.chain_rotated = np.array([np.dot(R.T, ci) for ci in chain])
        self.rotated_chain_means = np.mean(self.chain_rotated, 0)
        self.rotated_chain_stddevs = np.std(self.chain_rotated, 0)
        self.chain_rotated_regularized = self.chain_rotated[:] - \
            self.rotated_chain_means
        self.chain_rotated_regularized /= self.rotated_chain_stddevs
        self.rotated_chain_mins = np.min(self.chain_rotated_regularized, 0)
        self.rotated_chain_maxs = np.max(self.chain_rotated_regularized, 0)

    def _select_lnlike_based_training_points(self, Nsamples, Nbins=5):
        """Select samples from the histogramed loglikelihoods.
        Note that this method is currently unused.

        Args:
            Nsamples (int): number of samples to use
            Nbins (int): number of histogram bins to chop up the likelihoods; 
                default is 5

        Returns:
            (array-like): indices of the chain points to use for trianing

        """
        if Nbins > Nsamples:
            raise Exception("Cannot have more bins than samples.")

        N_per_bin = Nsamples / Nbins
        N_extra = Nsamples % Nbins #extra for the last bin

        lnlikes = self.lnlikes
        N_in_bins, edges = np.histogram(lnlikes, bins=Nbins)
        all_inds = np.arange(len(lnlikes)) #individual indices

        #indices to return
        ret_inds = np.array([])

        #Sort the lnlikes
        sorted_inds = np.argsort(lnlikes)
        llsorted = lnlikes[sorted_inds]
        for i in range(0, Nbins):
            ii = (llsorted >= edges[i]) * (llsorted < edges[i+1])
            sii = sorted_inds[ii]
            #If the bin is not populated (the tail), use the whole bin
            if len(sii) < N_per_bin:
                ret_inds = np.append(ret_inds, sii)
            else:
                #Otherwise, take a random selection from the bin
                ret_inds = np.append(ret_inds, np.random.choice(sii,
                                                                N_per_bin,
                                                                replace=False))
            continue
        #If we don't have enough samples, take more samples
        #but make sure we have no duplicates
        if len(ret_inds) < Nsamples:
            Nleft = Nsamples - len(ret_inds)
            for i in range(0, Nleft):
                new_sample = np.random.choice(all_inds)
                if new_sample not in ret_inds: #Enforce no replacement
                    ret_inds = np.append(ret_inds, new_sample)
                else:
                    i -= 1
                continue
        return ret_inds.astype(int)
        
    def select_training_points(self, Nsamples=40, method="LH"):
        """Select training points from the chain to train the GPs.

        Note: this method does not use the "scale" parameter.
        
        Args:
            Nsamples (int): number of samples to use; defualt is 40
            method (string): keyword for selecting different ways of
                obtaining training points. Currently unused.
        
        """
        #Create LH training samples
        x = pyDOE2.lhs(len(self.chain_cov), samples=Nsamples,
                       criterion="center", iterations=5)
        
        #Transform them correctly
        x -= 0.5 #center the training points
        s = self.scale
        w = self.eigenvalues
        R = self.rotation_matrix

        #Snap the training points to the MCMC chain
        samples = np.dot(s*x[:]*np.sqrt(w), R.T)[:] + self.chain_means
        cov = self.chain_cov
        def sqdists(chain, s, cov):
            X = chain[:] - s
            r = np.linalg.solve(cov, X.T).T
            d = np.sum(X * r, axis=1)
            return d
        indices = np.array([np.argmin(sqdists(self.chain, s, cov)) \
                            for s in samples])
            
        #Include the max liklihood point
        best_ind = np.argmax(self.lnlikes)
        self.training_inds = np.append(indices, best_ind)
        return
    
    def get_training_data(self):
        """Obtain the currently used training points

        return:
            Tuple of chain and lnlikelihood values.
        """
        inds = self.training_inds
        return (self.chain[inds], self.lnlikes[inds])

    def train(self, kernel=None):
        """Train a Gaussian Process to interpolate the log-likelihood
        of the training samples.

        Args:
            kernel (george.kernels.Kernel object): kernel to use, or any 
                acceptable object that can be accepted by the george.GP object

        """
        inds = self.training_inds
        x = self.chain_rotated_regularized[inds]
        lnL = self.lnlikes[inds]
        _guess = 4.5
        if kernel is None:
            kernel = kernels.ExpSquaredKernel(metric=_guess, ndim=len(x[0]))
        #Note: the mean is set slightly lower that the minimum lnlike
        lnPmin = np.min(self.lnlikes)
        gp = george.GP(kernel, mean=lnPmin-np.fabs(lnPmin*3))
        gp.compute(x)
        def neg_ln_likelihood(p):
            gp.set_parameter_vector(p)
            return -gp.log_likelihood(lnL)

        def grad_neg_ln_likelihood(p):
            gp.set_parameter_vector(p)
            return -gp.grad_log_likelihood(lnL)

        result = minimize(neg_ln_likelihood, gp.get_parameter_vector(),
                          jac=grad_neg_ln_likelihood)
        print(result)
        gp.set_parameter_vector(result.x)
        self.gp = gp
        self.lnL_training = lnL
        return

    def _transform_data(self, x):
        #Get x into the eigenbasis
        R = self.rotation_matrix.T
        xR = np.array([np.dot(R, xi) for xi in x])
        xR -= self.rotated_chain_means
        xR /= self.rotated_chain_stddevs
        return xR

    def predict(self, x):
        """Given a set of parameters, return the predicted log probability.

        Args:
            x (array-like): parameters to interpolate at
        
        Returns:
            interpolated log probability at x.

        """
        #Make it the correct format
        x = np.atleast_2d(x).copy()
        #Get x into the eigenbasis
        x = self._transform_data(x)

        pred, pred_var = self.gp.predict(self.lnL_training, x)
        return pred + self.lnlike_max 
        
if __name__ == "__main__":
    import scipy.stats
    x_mean, y_mean = 3, 0
    means = np.array([x_mean, y_mean])
    cov = np.array([[1,0.1],[0.1,0.5]])
    icov = np.linalg.inv(cov)
    chain = np.random.multivariate_normal(mean=means,
                                          cov=cov,
                                          size=(10000))
    likes = scipy.stats.multivariate_normal.pdf(chain, mean=means, cov=cov)
    lnlikes = np.log(likes)
    IS = resampler(chain, lnlikes)
    IS.select_training_points(1000, Nbins = 5)
    IS.train()
    x, y = chain.T

    import matplotlib
    matplotlib.use('TkAgg')
    import matplotlib.pyplot as plt
    plt.scatter(x, y, c='b', s=0.5, alpha=0.2)
    points, _ = IS.get_training_data()
    plt.scatter(points[:,0], points[:,1], c='k', s=5)
    plt.show()
