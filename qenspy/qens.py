import numpy as np

import tensorflow as tf
import tensorflow_probability as tfp
tfb = tfp.bijectors

class QEns():
    def broadcast_w_and_renormalize(self, q, w):
        """
        Broadcast w to the same shape as q, set weights corresponding to missing
        entries of q to 0, and re-normalize so that the weights sum to 1.

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
        broadcast_w: 3D tensor with shape (N, K, M)
            broadcast_w has N copies of the argument w with weights w[i,k,m] set
            to 0 at indices where q[i,k,m] is nan. The weights are then
            re-normalized to sum to 1 within each combination of i and k.
        """
        # broadcast w to the same shape as q, creating N copies of w
        q_shape = tf.shape(q).numpy()
        broadcast_w = tf.broadcast_to(w, q_shape)

        # if there is missingness, adjust entries of broadcast_w
        missing_mask = tf.math.is_nan(q)
        if tf.reduce_any(missing_mask):
            # nonmissing mask has shape (N, K, M), with entries
            # 1 where q had missing values and 0 elsewhere
            nonmissing_mask = tf.cast(
                tf.logical_not(missing_mask),
                dtype = broadcast_w.dtype)

            # set weights corresponding to missing entries of q to 0
            broadcast_w = tf.math.multiply(broadcast_w, nonmissing_mask)

            # renormalize weights to sum to 1 along the model axis
            (broadcast_w, _) = tf.linalg.normalize(broadcast_w, ord = 1, axis = 2)

        return broadcast_w

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
        # adjust w to handle missing values in q
        w = super().broadcast_w_and_renormalize(q, w)

        # calculate weighted mean along the M axis for each combination of N, K
        ensemble_q = tf.reduce_sum(tf.math.multiply_no_nan(q, w), axis = 2)

        # return
        return ensemble_q
