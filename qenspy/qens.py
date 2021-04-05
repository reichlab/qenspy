import numpy as np
import abc

import tensorflow as tf
import tensorflow_probability as tfp
tfb = tfp.bijectors
tfd = tfp.distributions

class QEns(abc.ABC):
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
            # 0 where q had missing values and 1 elsewhere
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
    
    def pinball_loss(self, y, q, tau):
        """
        Calculate pinball loss of predictions from a single model.

        Parameters
        ----------
        y: 1D tensor of length N
            observed values
        q: 2D tensor with shape (N, K)
            forecast values
        tau: 1D tensor of length K: Each slice `q[:, k, :]` corresponds 
        to predictions at quantile level `tau[k]`

        Returns
        -------
        Sum of pinball loss over all predictions as scalar tensor 
        (sum over all i = 1, …, N and k = 1, …, K)
        """
        # broadcast y to shape (N, K)
        y_broadcast = tf.transpose(tf.broadcast_to(y, tf.transpose(q).shape))
        loss = tf.reduce_sum(tf.maximum(tau*(y_broadcast - q), (tau-1)*(y_broadcast-q)))
        return loss

    def pinball_loss_objective(self, param_vec, y, q, tau, tau_groups):
        """
        Pinball loss objective function for use during parameter estimation:
        a function of component weights

        Parameters
        ----------
        param_vec: 1D tensor of length K*(M-1)
            parameters vector
        y: 1D tensor of length N
            observed values
        q: 3D tensor or array
            model forecasts of shape (N, K, M)
        tau: 1D tensor of quantile levels (probabilities) of length K
            Each slice `q[..., k,:]` corresponds to predictions at quantile level `tau[k]`
        tau_groups: 1D numpy array of integers of length K
            vector defining groups of quantile levels that have shared
            parameter values.  For example, [0, 0, 0, 1, 1, 1, 2, 2, 2]
            indicates that the component weights are shared within the first
            three, middle three, and last three quantile levels.
        
        Returns
        -------
        Total pinball loss over all predictions as scalar tensor
        """
        K = q.shape[1]
        M = q.shape[2]
        w = unpack_params(param_vec, K, M, tau_groups)

        q = tf.constant(q)
        w_value = tf.constant(w["w"])
        ensemble_q = predict(q, w)

        loss = pinball_loss(y, ensemble_q, tau)

        return loss

    @abc.abstractmethod
    def predict(self, q, w):
        """
        Generate prediction from a quantile forecast ensemble.

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
    
    def get_param_estimates_vec(self):
        """
        Get parameter estimates in vector form 

        Returns
        ----------
        param_estimates_vec: 1D tensor of length K*(M-1)
        """
        return self.param_estimates_vec
    
    def set_param_estimates_vec(self, param_estimates_vec):
        """
        Set parameter estimates in vector form 

        Parameters
        ----------
        param_estimates_vec: 1D tensor of length K*(M-1)
        """
        self.param_estimates_vec = param_estimates_vec
    

    def fit(self, y, q, tau, tau_groups, init_param_vec, optim_method, num_iter, learning_rate):
        """
        Estimate model parameters
        
        Parameters
        ----------
        y: 1D tensor of length N
            observed values
        q: 3D tensor or array
            model forecasts of shape (N, K, M)
        tau: 1D tensor of quantile levels (probabilities) of length K
            Each slice `q[..., k,:]` corresponds to predictions at quantile level `tau[k]`
        tau_groups: 1D numpy array of integers of length K
            vector defining groups of quantile levels that have shared
            parameter values.  For example, [0, 0, 0, 1, 1, 1, 2, 2, 2]
            indicates that the component weights are shared within the first
            three, middle three, and last three quantile levels.
        init_param_vec: optional 1D tensor of length K*(M-1)
            optional initial values for the weights during estimation
        optim_method: string
            optional method for optimization.  For now, only support "adam" or "sgd".
        num_iter: integer
            number of iterations for optimization
        learning_rate: Tensor or a floating point value. 
            The learning rate
        """
        params_vec_var = tf.Variable(
            initial_value=init_param_vec,
            name='params_vec',
            dtype=np.float64)
        
        # create optimizer
        if optim_method == "adam":
            optimizer = tf.optimizers.Adam(learning_rate = learning_rate)
        elif optim_method == "sgd":
            optimizer = tf.optimizers.SGD(learning_rate =learning_rate)
        
        # initiate loss trace
        lls_ = np.zeros(num_iter, np.float64)
        
        # create a list of trainable variables
        trainable_variables = [params_vec_var]

        # apply gradient descent with num_iter times
        for i in range(num_iter):
            with tf.GradientTape() as tape:
                loss = pinball_loss_objective(params_vec_var, y, q, tau, tau_groups)
            grads = tape.gradient(loss, trainable_variables)
            optimizer.apply_gradients(zip(grads, trainable_variables))
            lls_[i] = loss

        # set parameter estimates
        set_param_estimates_vec(params_vec_var.numpy())
        self.loss_trace = lls_


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

class MedianQEns(QEns):
    def __init__(self, bw_method):
        self.bw_method = bw_method

    def calc_bandwidth(self, q, w = None):
        """
        Calculate bandwidth

        Parameters
        ----------
        q: 3D tensor with shape (N, K, M)
            Predictive quantiles from component models where N is the number of 
            location/forecast date/horizon triples, M is number of models, 
            and K is number of quantile levels
        w: 2D tensor with shape (K, M)
            Model weight. Not required if self.bw_method is “unweighted”

        Returns
        -------
        bw: 2D tensor with shape (N, K)
            bandwidths
        """
        if self.bw_method == "silverman_unweighted":
            M = q.shape[2]
            w_unweighted = tf.broadcast_to(tf.constant([1/M], dtype = q.dtype),[q.shape[1], q.shape[2]])
            broadcast_w = super().broadcast_w_and_renormalize(q, w_unweighted)
        
        elif self.bw_method == "silverman_weighted":
            if w is None:
                raise ValueError("Please provide w.")
            M = w.shape[1]
            broadcast_w = super().broadcast_w_and_renormalize(q, w)

        # calculate weighted mean along the M axis for each combination of N, K but keep extra dims
        weighted_mean = tf.reduce_sum(tf.math.multiply_no_nan(q, broadcast_w), axis = 2, keepdims=True)
            
        squared_diff = tf.square(tf.subtract(q, weighted_mean))

        weighted_sd = tf.sqrt(tf.reduce_sum(tf.math.multiply_no_nan(squared_diff, broadcast_w), axis = 2)/(M))
            
        bw = 0.9 * weighted_sd * (3**(-0.2))
        
        return bw

    def weighted_cdf(self, x, q, w):
        """
        Calculate weighted KDE cdf

        Parameters
        ----------
        x:  3D tensor with shape (N, K, ...)
            Points at which to evaluate the cdf
        q: 3D tensor with shape (N, K, M)
            Predictive quantiles from component models where N is the number of 
            location/forecast date/horizon triples, M is number of models, 
            and K is number of quantile levels
        w: 2D tensor with shape (K, M)
            Model weight. Not required if self.bw_method is “unweighted”

        Returns
        -------
        weighted_cdf: 3D tensor with shape (N, K, ...)
            weighted_cdf
        """

        bw = self.calc_bandwidth(q = q, w = w)
        rectangle_bw = tf.sqrt(12 * (bw ** 2))

        # to handle missing values
        # (N, K, M)
        broadcast_w = super().broadcast_w_and_renormalize(q, w)

        #(N, K, ...)
        weighted_cdf = tf.zeros(shape = x.shape, dtype = x.dtype)
        M = q.shape[2]
        for i in range(M):
            # (N, K, 1)
            low = tf.expand_dims(tf.subtract(q[:,:,i], rectangle_bw/2), -1)
            high = tf.expand_dims(tf.add(q[:,:,i], rectangle_bw/2), -1)
            curr_broadcast_w = tf.expand_dims(broadcast_w[:,:,i],-1)
            # case1
            weighted_cdf = tf.where(tf.less(x, low), weighted_cdf, weighted_cdf)
            # case2
            weighted_cdf = tf.where(tf.logical_and(tf.less_equal(x, high), \
                                tf.greater_equal(x, low)), \
                                tf.add(weighted_cdf, curr_broadcast_w * tf.subtract(x, low) * (1/  tf.expand_dims(rectangle_bw,-1))),\
                                weighted_cdf)
            # case3
            weighted_cdf = tf.where(tf.greater(x, high), tf.add(weighted_cdf, curr_broadcast_w), weighted_cdf)

        return weighted_cdf
       
    def predict(self, q, w):
        return q
