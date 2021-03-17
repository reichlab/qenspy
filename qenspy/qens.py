import numpy as np

import tensorflow as tf
import tensorflow_probability as tfp
tfb = tfp.bijectors

class QEns():
    def fill_missing_and_renormalize(self, q, w):
        """
        Generate prediction from a weighted mean quantile forecast ensemble.

        Parameters
        ----------
        q: 3D tensor with shape (N, K, M)
            Component prediction quantiles for observation cases i = 1, ..., N,
            quantile levels k = 1, ..., K, and models m = 1, ..., M
        w: 2D tensor with shape (K, M)
            Component model weights, where `w[m, k]` is the weight given to
            model m for quantile level k

        Returns
        -------
        (q, w): tuple with two 3D tensors with shape (N, K, M)
            - q is a copy of the argument q with missing values replaced by 0
            - w is N copies of the argument w with weights w[i,k,m] set to 0
            at indices where q[i,k,m] is 0, and the weights re-normalized to
            sum to 1 within each combination of i and k.
        """
        # note! handling of missingness and renormalization is not done yet!
        q_shape = tf.shape(q).numpy()
        expanded_w = tf.broadcast_to(w, q_shape)
        return (q, expanded_w)

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



class MeanQEns(QEns):
    def predict(self, q, w):
        """
        Generate prediction from a weighted mean quantile forecast ensemble.

        Parameters
        ----------
        q: 3D tensor with shape (N, K, M)
            Component prediction quantiles for observation cases i = 1, ..., N,
            quantile levels k = 1, ..., K, and models m = 1, ..., M
        w: 2D tensor with shape (K, M)
            Component model weights, where `w[m, k]` is the weight given to
            model m for quantile level k

        Returns
        -------
        ensemble_q: 2D tensor with shape (N, K)
            Ensemble forecasts for each observation case i = 1, ..., N and
            quantile level k = 1, ..., K
        """
        # TODO: handle missing values in q
        (q, w) = super().fill_missing_and_renormalize(q, w)

        # calculate weighted mean along the M axis for each combination of N, K
        ensemble_q = tf.reduce_sum(q * w, axis = 2)

        # return
        return ensemble_q
