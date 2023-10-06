import numpy as np
import matplotlib.pyplot as plt

class Gaussian:

    def pdf(self, x, mu, sigma):
        """
        Returns the probability density function given some mu, sigma and a list of points x.
        """
        out = (1 / (sigma * np.sqrt(2*np.pi))) * np.exp(-0.5 * (((x - mu) / sigma) ** 2))

        return out

    def KL_divergence(self, distr1, distr2):
        """
        Computes the Kullback-Leibler Divergence between 2 distributions
        """

        return np.mean(distr1 * np.log(distr1 / distr2))

    def KL_divergence_against_gaussian(self, sample):
        """
        Computes the Kullback-Leibler Divergence of a given set of samples against a standard normal gaussian. Useful in analysing how
        the distribution shifts along the timesteps of diffusion.
        """

        mu = np.mean(sample)
        sigma = np.var(sample)

        x = np.linspace(-10, 10, int(1e+6))

        distr1 = self.pdf(x, 0, 1) 
        distr2 = self.pdf(x, mu, sigma)

        return self.KL_divergence(distr1, distr2)

    def get_limits(self, mu, sigma, edge_factor = 0.01):
        """
        Gets the upper and lower limit of the bell curve of the distribution, used for fixing limits for plotting the distribution.
        """

        p_min = edge_factor * self.pdf(mu, mu, sigma)
        x_1 = -1 * sigma * np.sqrt(2 * np.log(1 / (p_min * sigma * np.sqrt(2*np.pi)))) + mu
        x_2 = sigma * np.sqrt(2 * np.log(1 / (p_min * sigma * np.sqrt(2*np.pi)))) + mu

        return x_1, x_2

    def multivariate_pdf(self, mean, var, size = 1024, limits = (-1, 1)):
        """
        Inputs:
        mean => 1D array of size k, in a 2D trajectory it is k=2 and for 3D k=3
        covariance => 2D array of (k x k)
        size => Integer; Per axis length of output
        limits => Upper and lower limits for each axis
        Output:
        Array of 'k' dimensions of shape (size x size x .... x size)
        """

        mean = np.array(mean)
        k = mean.size

        axis_div = np.repeat([np.linspace(limits[0], limits[1], size)], repeats = k, axis = 0)
        x = np.array(np.meshgrid(*axis_div))

        mu = np.reshape(mean, (k, *np.ones(k, dtype = np.int32)))

        pdf = (1 / (((2 * np.pi) ** (k/2)) * np.sqrt(var**k))) * np.exp(-0.5 * np.sum((x - mu)**2, axis = 0) / var)

        return pdf

