"""
This code is taken from the Garg et. al paper(2022)

Samples from X from a Gaussian
"""

import math

import torch

##################################################
################# Abstract Class #################
##################################################
class DataSampler:
    def __init__(self, n_dims):
        self.n_dims = n_dims

    def sample_xs(self):
        raise NotImplementedError

##################################################
################ Helper Functions ################
##################################################
def get_data_sampler(data_name, n_dims, **kwargs):
    """
    Function to get a random sampler

    Args:
        data_name (String) : the name of the sampling technique you want to use
        n_dims (int) : size of each data point
    Returns:
        DataSampler : the corresponding data sampler object
    """
    # Supported samplers
    names_to_classes = {
        "gaussian": GaussianSampler,
    }
    if data_name in names_to_classes:
        sampler_cls = names_to_classes[data_name]
        return sampler_cls(n_dims, **kwargs)
    else:
        print("Unknown sampler")
        raise NotImplementedError

def sample_transformation(eigenvalues, normalize=False):
    """
    Creates a random, full rank matrix with the given eigenvalues

    Args:
        eigenvalues (list) : the eigenvalues wanted from the output matrix
        normalize (bool) : whether the output matrix should be normalized or not 
    Returns:
        np.ndarray : a 2D array of shape (n_dim, n_dim) where n_dims is the number of eigenvalues
    """

    n_dims = len(eigenvalues)
    U, _, _ = torch.linalg.svd(torch.randn(n_dims, n_dims))
    t = U @ torch.diag(eigenvalues) @ torch.transpose(U, 0, 1)
    if normalize:
        norm_subspace = torch.sum(eigenvalues**2)
        t *= math.sqrt(n_dims / norm_subspace)
    return t


##################################################
################ Sampler Classes #################
##################################################
class GaussianSampler(DataSampler):
    def __init__(self, n_dims, bias=None, scale=None):
        """
        Args:
            n_dims (int) : the size of each x in the ICL task 
            bias (torch.Tensor) : the bias to add to the sampled x's
            scale (torch.Tensor) : the scaling tensor to multiply to the sampled x's
        """
        super().__init__(n_dims)
        self.bias = bias
        self.scale = scale

    def sample_xs(self, n_points, b_size, n_dims_truncated=None, seeds=None):
        """
        Samples specified number of X's for a given batch size and number of points for each data point

        Args:
            n_points (int) : the number of points for each example of the ICL task 
            b_size (int) : the batch size
            n_dims_truncated (int) : number of feature dimensions from the beginning for each point to zero out
            seeds (list) : the list of seeds to generate the points from for each item in the batch; should be the same size as b_size
        Returns:
            torch.Tensor : the sampled x
        """
        if seeds is None:
            xs_b = torch.randn(b_size, n_points, self.n_dims)
        else:
            xs_b = torch.zeros(b_size, n_points, self.n_dims)
            generator = torch.Generator()
            assert len(seeds) == b_size
            for i, seed in enumerate(seeds):
                generator.manual_seed(seed)
                xs_b[i] = torch.randn(n_points, self.n_dims, generator=generator)
        if self.scale is not None:
            xs_b = xs_b @ self.scale
        if self.bias is not None:
            xs_b += self.bias
        if n_dims_truncated is not None:
            xs_b[:, :, n_dims_truncated:] = 0
        return xs_b
