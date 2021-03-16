import numpy as np

import tensorflow as tf
import tensorflow_probability as tfp
tfb = tfp.bijectors

class QEns():
    def unpack_params(self, param_vec, K, M, tau_groups):
        """
        Utility function to convert from a vector of parameters to a dictionary
        of parameter values.

        Parameters
        ----------
        param_vec: 1D tensor of length K*(M-1)
            parameters vector
        K: integer
            number of quantile levels
        M: integer
            number of component models
        tau_groups: 1D numpy array of integers of length K
            vector defining groups of quantile levels that have shared
            parameter values.  For example, [0, 0, 0, 1, 1, 1, 2, 2, 2]
            indicates that the component weights are shared within the first
            three, middle three, and last three quantile levels.

        Returns
        -------
        params_dict: dictionary
            Dictionary with one element, ‘w’, which is a 2D tensor of shape
            (K, M) with model weights at each quantile level
        """
        # get number of different tau groups
        groups_unique, groups_unique_idx = np.unique(tau_groups,
                                                     return_inverse=True)
        num_tau_groups = len(groups_unique)

        # reshape param_vec to (num_tau_groups, M - 1)
        v_by_group = tf.reshape(param_vec, (num_tau_groups, M - 1))

        # apply centered softmax forward transform; output has shape (num_tau_groups, M)
        w_by_group = tfb.SoftmaxCentered().forward(v_by_group)

        # expand by groups to shape (K, M)
        w = tf.gather(w_by_group, groups_unique_idx)

        # return as dictionary
        return {'w': w}

