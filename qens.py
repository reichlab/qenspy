import pickle

import numpy as np
import abc

import tensorflow as tf
import tensorflow_probability as tfp
tfb = tfp.bijectors
tfd = tfp.distributions

class QEns(abc.ABC):
    """
    Base class for a quantile ensemble with estimation by optimizing quantile
    score.
    """
    def __init__(self, M, tau, tau_groups, init_method = "xavier") -> None:
        """
        Initialize a QEns model
        
        Parameters
        ----------
        M: integer
            number of component models
        tau: 1D tensor of quantile levels (probabilities) of length K
            For example, `tau = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]`
            means that the model will work with predictive quantiles at those
            nine probability levels.
        tau_groups: 1D numpy array of integers of length K
            vector defining groups of quantile levels that have shared
            parameter values.  For example, [0, 0, 0, 1, 1, 1, 2, 2, 2]
            indicates that the component weights are shared within the first
            three, middle three, and last three quantile levels.
        init_method: string
            initialization method for model weights: "xavier" or "equal"
        
        Returns
        -------
        None
        """
        self.M = M
        self.K = len(tau_groups)
        self.tau = tf.convert_to_tensor(tau, dtype=tf.float32)
        self.tau_groups = tau_groups
        
        # get number of different tau groups
        tau_groups_unique, tau_groups_idx = np.unique(tau_groups,
                                                      return_inverse=True)
        num_tau_groups = len(tau_groups_unique)
        self.num_tau_groups = num_tau_groups
        self.tau_groups_idx = tau_groups_idx
        
        # bijector to map to simplex of model probabilities
        softmax_bijector = tfb.SoftmaxCentered()
        
        # define a function, `init_w`, that initializes the model weights
        if init_method == "xavier":
            # Xavier initialization for w
            w_xavier_hw = 1.0 / tf.math.sqrt(tf.constant(M-1, dtype=tf.float32))
            def init_w():
                return softmax_bijector.forward(
                    tf.random.uniform((num_tau_groups, M - 1), -w_xavier_hw, w_xavier_hw, dtype=tf.float32))
        elif init_method == "equal":
            # initialize to equal weights, zeros on raw scale
            def init_w():
                return softmax_bijector.forward(tf.zeros((num_tau_groups, M - 1), dtype=tf.float32))
        else:
            raise ValueError("init_method must be 'xavier' or 'equal'")
        
        # dictionary of transformed variables for ensemble parameters
        # the parameter is the vector of model weights within each tau group
        self.parameters = {
            'w_tau_groups': tfp.util.TransformedVariable(
                initial_value=init_w(),
                bijector=softmax_bijector,
                name='w_tau_groups',
                dtype=np.float32)
        }
        
        # initialize loss trace
        self.loss_trace = np.zeros((0,), np.float32)
    
    
    def model_q_ordered(model_df, tau_lvls):
        """
        Helper function intended for internal use only. Convert a data frame of
        predictive quantiles for a single model to a 3d array suitable for
        concatenation.
        """
        model_df = model_df.droplevel(list(range(len(model_cols) + 1)), axis = 1)
        missing_cols = np.setdiff1d(tau_lvls, model_df.columns.values)
        for col_name in missing_cols:
            model_df[col_name] = np.NaN
        return model_df[tau_lvls].values[..., np.newaxis]
    
    
    def df_to_array(self, q_df, y_df=None,
                    model_cols=['model'],
                    task_cols=None,
                    tau_col='quantile',
                    value_col='value'):
        """
        Convert predictive quantiles from a tidy data frame to a 3d array.

        Parameters
        ----------
        q_df: a pandas dataframe with predictive quantiles from component models
            It should contain:
            - one or more columns identifying the model
            - one or more columns identifying the prediction task
            - a column identifying the quantile level
            - a column identifying the predictive value
        y_df: an optional pandas dataframe with observed values
            It should contain:
            - one or more columns identifying the prediction task
            - a column identifying the observed value
        model_cols: list of character strings naming columns that identify the model
        task_cols: list of character strings naming columns that identify a
            prediction task
        tau_col: character string naming a column with quantile levels
        value_col: character string naming a column with predictive values
        
        Returns
        -------
        A 3D tensorflow tensor with shape (N, K, M)
            Component prediction quantiles for prediction tasks i = 1, ..., N,
            quantile levels k = 1, ..., K, and models m = 1, ..., M
        """
        tau_lvls = list(df[tau_col].unique())
        tau_lvls.sort()
        
        q_wide = df \
            .set_index(keys = task_cols + model_cols + [tau_col]) \
            [[value_col]] \
            .unstack(model_cols + [tau_col]) \
            .groupby(by = model_cols, axis = 1)
        
        q_arr = np.concatenate(
            [self.model_q_ordered(model_df, tau_lvls) for _, model_df in q_wide],
            axis = 2
        )
        
        return tf.constant(q_arr, dtype=tf.float32)
    
    
    def handle_missingness(self, q, w):
        """
        Broadcast w to the same shape as q, set weights corresponding to missing
        entries of q to 0, and re-normalize so that the weights sum to 1.

        Replace nans in q with 0

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
        q_nans_replaced: 3D tensor with shape (N, K, M)
            Component prediction quantiles with nans replaced with 0
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
        
        # replace nan with 0 in q
        q_nans_replaced = tf.where(missing_mask, np.float32(0.0), np.float32(q))
        
        return q_nans_replaced, broadcast_w
    
    
    @property
    def w(self):
        """
        Model weights for each quantile level

        Returns
        -------
        w: 2d tensor
            A tensor of shape (K, M) with model weights at each quantile level
        """
        # expand by groups to shape (K, M)
        return tf.gather(self.parameters['w_tau_groups'], self.tau_groups_idx)
    
    
    def pinball_loss(self, y, q):
        """
        Calculate pinball loss of predictions from a single model.

        Parameters
        ----------
        y: 1D tensor of length N
            observed values
        q: 2D tensor with shape (N, K)
            forecast values

        Returns
        -------
        Mean pinball loss over all predictions as scalar tensor 
        (mean over all i = 1, …, N and k = 1, …, K)
        """
    
        # add an extra dimension to y --> (N,1)
        y_broadcast = tf.expand_dims(y, -1)
        # broadcast y to shape (N, K)
        y_broadcast = tf.broadcast_to(y_broadcast, q.shape)
        loss = tf.reduce_mean(tf.maximum(self.tau*(y_broadcast - q), (self.tau-1)*(y_broadcast-q)))

        return loss

    def pinball_loss_objective(self, y, q):
        """
        Pinball loss objective function for use during parameter estimation:
        a function of component weights

        Parameters
        ----------
        y: 1D tensor of length N
            observed values
        q: 3D tensor or array
            model forecasts of shape (N, K, M)
        
        Returns
        -------
        Total pinball loss over all predictions as scalar tensor
        """
        return self.pinball_loss(y=y, q=self.predict(q = q))
    
    
    @abc.abstractmethod
    def predict(self, q):
        """
        Generate prediction from a quantile forecast ensemble.

        Parameters
        ----------
        q: 3D tensor with shape (N, K, M)
            Component prediction quantiles for observation cases i = 1, ..., N,
            quantile levels k = 1, ..., K, and models m = 1, ..., M

        Returns
        -------
        ensemble_q: 2D tensor with shape (N, K)
            Ensemble forecasts for each observation case i = 1, ..., N and
            quantile level k = 1, ..., K
        """
    
    
    def fit(self,
            y,
            q,
            optim_method = "adam",
            num_iter = 1000,
            learning_rate = 0.1,
            verbose = False,
            save_frequency = None,
            save_path = None):
        """
        Estimate model parameters
        
        Parameters
        ----------
        y: 1D tensor or array with observed data
            Tensor or array of length N containing observed values for each
            training set prediction task.
        q: 3D tensor or array with predictive quantiles
            Tensor or array with shape (N, K, M), where entry (i, k, m) is a
            predictive quantile at level tau_k for prediction task i from model m
        optim_method: string
            optional method for optimization.  For now, only support "adam" or "sgd".
        num_iter: integer
            number of iterations for optimization
        learning_rate: Tensor or a floating point value.
            The learning rate
        verbose: boolean
            Indicator for whether to print output during parameter estimation
        save_frequency: integer or None
            Frequency with which to save intermediate model fits during
            parameter estimation
        save_path: string
            Path to file where intermediate model fits are saved
        """
        # convert inputs to float tensors
        y = tf.convert_to_tensor(y, dtype=tf.float32)
        q = tf.convert_to_tensor(q, dtype=tf.float32)
        
        # TODO: validate shapes of y and q
        
        # validate num_iter and save_frequency
        if not isinstance(num_iter, int):
            raise ValueError('num_iter must be an int')
        
        if save_frequency == None:
            save_frequency = num_iter + 1
        
        if not isinstance(save_frequency, int):
            raise ValueError('save_frequency must be None or an int')
        
        # trainable variables representing parameters to estimate
        trainable_variables = [self.parameters[v].trainable_variables[0] \
            for v in self.parameters.keys()]
        
        # create optimizer
        if optim_method == "adam":
            optimizer = tf.optimizers.Adam(learning_rate = learning_rate)
        elif optim_method == "sgd":
            optimizer = tf.optimizers.SGD(learning_rate = learning_rate)
        
        # initiate loss trace
        loss_tr = np.zeros(num_iter, np.float32)
        
        # apply gradient descent with num_iter times
        for i in range(num_iter):
            with tf.GradientTape() as tape:
                loss = self.pinball_loss_objective(
                    y = y,
                    q = q)
            grads = tape.gradient(loss, trainable_variables)
            grads, _ = tf.clip_by_global_norm(grads, 10.0)
            optimizer.apply_gradients(zip(grads, trainable_variables))
            loss_tr[i] = loss

            if verbose:
                print(i)
                print("w_tau_groups = ")
                print(self.parameters['w_tau_groups'].numpy())
                print("loss = ")
                print(loss.numpy())
                print("grads = ")
                print(grads)
            
            if (i + 1) % save_frequency == 0:
                # save parameter estimates and loss trace
                to_save = self.parameters.copy().update({
                    'loss_trace': loss_tr
                })
                
                with open(str(save_path), "wb") as f:
                    pickle.dump(to_save, f)
        
        # update loss trace
        self.loss_trace = np.concatenate([self.loss_trace, loss_tr])


class MeanQEns(QEns):
    def predict(self, q):
        """
        Generate prediction from a weighted mean quantile forecast ensemble.

        Parameters
        ----------
        q: 3D tensor with shape (N, K, M)
            Component prediction quantiles for observation cases i = 1, ..., N,
            quantile levels k = 1, ..., K, and models m = 1, ..., M

        Returns
        -------
        ensemble_q: 2D tensor with shape (N, K)
            Ensemble forecasts for each observation case i = 1, ..., N and
            quantile level k = 1, ..., K
        """
        # if w == None:
        #     # load weights from object
        #     w = self.w
       
        # adjust w and q to handle missing values
        q, w = super().handle_missingness(q, self.w)
        
        # calculate weighted mean along the M axis for each combination of N, K
        ensemble_q = tf.reduce_sum(tf.math.multiply_no_nan(q, w), axis = 2)
        
        return ensemble_q


class MedianQEns(QEns):
    # def __init__(self, rectangle_bw = None):
    #     self.rectangle_bw = rectangle_bw

    def calc_kde_rectangle_width(self, q, train_q):
        """
        Calculate kde rectangle width

        Parameters
        ----------
        q: 3D tensor with shape (N, K, M)
            Predictive quantiles from component models where N is the number of
            location/forecast date/horizon triples, M is number of models,
            and K is number of quantile levels
        train_q: boolean
            Indicator for calculating bandwidth during training or not
        Returns
        -------
        rectangle_bw: 3D tensor with shape (N, K, M)
            bandwidths
        """
        # if train_q and self.rectangle_bw != None:
        #     return self.rectangle_bw
        # else:
        # sort the third axis of q
        sorted_indx = tf.argsort(q, axis = -1)
        q_sorted = tf.gather(q, sorted_indx, batch_dims = 2)
        # calculate difference (N, K, M-1)
        q_diff = q_sorted[:,:, 1:] - q_sorted[:,:,:-1]
        # (N, K, M+1)
        q_diff = tf.concat((tf.expand_dims(q_sorted[:,:, 1] - q_sorted[:,:, 0], -1), \
                q_diff, \
                tf.expand_dims(q_sorted[:,:,-1] - q_sorted[:,:,-2], -1)),\
                axis = -1)
        # calculate shortest distance
        min_diff = tf.minimum(q_diff[:,:,1:], q_diff[:, :, :-1])
        # replace nans and any value smaller than 1 with 1
        min_diff = tf.where(tf.logical_or(min_diff < 1, tf.math.is_nan(min_diff)), np.float32(1.0), min_diff)
        # rearrange distance 
        unsorted_indx = np.argsort(sorted_indx, axis = -1)
        bw = tf.gather(min_diff, unsorted_indx, batch_dims = 2)
        self.rectangle_bw = bw *2
        return self.rectangle_bw

    def weighted_cdf(self, x, q, w, rectangle_bw):
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
        rectangle_bw: 3D tensor with shape (N, K, M)
            Rectangle bandwidth values

        Returns
        -------
        weighted_cdf: 3D tensor with shape (N, K, ...)
            weighted_cdf
        """
        # print("rectangle bw")
        # print(rectangle_bw)
        # to handle missing values
        # (N, K, M)
        q, broadcast_w = super().handle_missingness(q, w)

        #(N, K, ...)
        weighted_cdf = tf.zeros(shape = x.shape, dtype = x.dtype)
        M = q.shape[2]
        for i in range(M):
            # (N, K, 1)
            low = tf.expand_dims(tf.subtract(q[:,:,i], rectangle_bw[:, :, i]/2), -1)
            high = tf.expand_dims(tf.add(q[:,:,i], rectangle_bw[:, :, i]/2), -1)
            curr_broadcast_w = tf.expand_dims(broadcast_w[:,:,i],-1)
            # print("curr_broadcast_w = ")
            # print(curr_broadcast_w)
            # case1 (no calculation needed): when x is on the left hand side of the rectangular kernel
            # case2: when x is in the middle of the rectangular kernel
            weighted_cdf = tf.where(
                tf.logical_and(tf.less_equal(x, high), tf.greater_equal(x, low)), \
                tf.add(weighted_cdf, curr_broadcast_w * tf.subtract(x, low) * (1/ tf.expand_dims(rectangle_bw[:, :, i],-1))),\
                weighted_cdf)
            # case3: when x is on the right hand side of the rectangular kernel
            weighted_cdf = tf.where(tf.greater(x, high), tf.add(weighted_cdf, curr_broadcast_w), weighted_cdf)
            # print("i = " + str(i))
            # print(weighted_cdf[8, 10, :])
        
        return weighted_cdf

    def predict(self, q):
        """
        Calculate weighted median at different quantile levels

        Parameters
        ----------
        q: 3D tensor with shape (N, K, M)
            Predictive quantiles from component models where N is the number of
            location/forecast date/horizon triples, M is number of models,
            and K is number of quantile levels
        
        Returns
        -------
        median: 3D tensor with shape (N, K)
            weighted median
        """
        # if w == None:
        #     # load weights from object
        #     w = self.w
        
        rectangle_bw = self.calc_kde_rectangle_width(q = q, train_q = False)
        
        q_nans_replaced = tf.where(tf.math.is_nan(q), np.float32(0.0), np.float32(q))
        
        low = q_nans_replaced - rectangle_bw/2
        high = q_nans_replaced + rectangle_bw/2
        
        # (N, K, 2M)
        slope_changepoints = tf.sort(tf.concat([low, high], axis = 2))
        
        # (N, K, 2M)
        changepoint_cdf_values = self.weighted_cdf(x = slope_changepoints, q = q, w = self.w, rectangle_bw = rectangle_bw)
        
        # the smallest index that has cdf >= 0.5 (N, K)
        inds = tf.math.argmax(changepoint_cdf_values >= np.float32(0.5), axis = 2)
        start_inds = inds - 1
        
        # change point value (N, K)
        p = tf.gather_nd(slope_changepoints, tf.expand_dims(inds, -1), batch_dims = 2)
        p_start = tf.gather_nd(slope_changepoints, tf.expand_dims(start_inds, -1), batch_dims = 2)
        # corresponding cdf value (N, K)
        c = tf.gather_nd(changepoint_cdf_values, tf.expand_dims(inds, -1), batch_dims = 2)
        c_start = tf.gather_nd(changepoint_cdf_values, tf.expand_dims(start_inds, -1), batch_dims = 2)
        
        # slope (N, K)
        m = tf.subtract(c, c_start) / tf.subtract(p, p_start)
        
        median = (np.float32(0.5) - c_start + m * p_start) / m
        
        # print("p =")
        # print(p)
        # print("p_start =")
        # print(p_start)
        # print("c =")
        # print(c[8, 10])
        # print("c_start =")
        # print(c_start)
        # print("median =")
        # print(median)
        
        return median