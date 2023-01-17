import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

import numpy as np
from scipy.spatial.distance import pdist
from scipy.spatial.distance import squareform
import tensorflow as tf
import tensorflow_probability as tfp
tfd = tfp.distributions
import unittest

from qenspy import qens
import timeit

class Test_MedianQEns(unittest.TestCase):
    def test_MedianQEns_calc_kde_rectangle_bw_equal_distance(self):
        q = np.linspace(1, 4 * 3 * 4, 4 * 3 * 5).reshape((4, 3, 5))
        actual_rectangle_bw = qens.MedianQEns(M=5,
                                              tau=[0.25, 0.5, 0.75],
                                              tau_grps=[0, 1, 2]) \
            .calc_kde_rectangle_width(q = tf.constant(q), train_q = True)
        # each value has 0.797 as the closest distance to its neighbor
        # but will round  0.797 to 1
        expected_rectangle_bw = np.ones(shape = (4,3,5)) * 2
        for i in range(actual_rectangle_bw.shape[0]):
            for j in range(actual_rectangle_bw.shape[1]):
                for k in range(actual_rectangle_bw.shape[2]):
                    self.assertAlmostEqual(actual_rectangle_bw[i,j,k],expected_rectangle_bw[i,j,k], places=7)
    
    def test_MedianQEns_calc_kde_rectangle_bw_unequal_distance(self):
        q = np.random.uniform(low=-5, high=15, size=(20,)).reshape((2, 2, 5))
        actual_rectangle_bw = qens.MedianQEns(M=5,
                                              tau=[0.25, 0.75],
                                              tau_grps=[0, 1]) \
            .calc_kde_rectangle_width(q = tf.constant(q), train_q = True)

        expected_rectangle_bw = np.zeros(shape = (2, 2, 5))
        for i in range(actual_rectangle_bw.shape[0]):
            for j in range(actual_rectangle_bw.shape[1]):
                l = np.array_split(q[i,j,:], actual_rectangle_bw.shape[2])
                # pariwise distance for every point/subarray in l
                distance = squareform(pdist(l, 'euclidean'))
                # replace 0 with infinity
                distance[distance == 0] = np.inf
                # replace value smaller than 1 with 1
                distance[distance < 1] = 1
                # get closest distance for each point in the original array q[i, j,:]
                min_distance =  np.min(distance, axis = 1)
                expected_rectangle_bw = min_distance * 2
                for k in range(actual_rectangle_bw.shape[2]):
                    self.assertAlmostEqual(expected_rectangle_bw[k],actual_rectangle_bw[i,j,k], places=7)
    
    def test_MedianQEns_calc_kde_rectangle_bw_missing_values(self):
        q = np.array([[[-1., -2.5, -5, np.nan ]]])
        actual_rectangle_bw = qens.MedianQEns(M=4, tau=[0.5], tau_grps=[0]) \
            .calc_kde_rectangle_width(q = tf.constant(q), train_q = True)
        expected_rectangle_bw = np.array([[[2, 3,  5,  2]]])
        for i in range(actual_rectangle_bw.shape[0]):
            for j in range(actual_rectangle_bw.shape[1]):
                for k in range(actual_rectangle_bw.shape[2]):
                    self.assertAlmostEqual(expected_rectangle_bw[i,j,k],actual_rectangle_bw[i,j,k], places=7)

    def test_MedianQEns_weighted_cdf(self):
        q = tf.constant(np.linspace(1, 4 * 3 * 2, 4 * 3 * 2).reshape((4, 3, 2)), dtype="float32")
        w = tf.constant(np.array(
            [[0.75, 0.25],
            [0.10, 0.90],
            [0.20, 0.80]]
        ), dtype="float32")

        # 5 is sample_shape = # of points evaluating at CDF
        # 4 is N = # of forecasts at location, horizon, target, forecast_date....
        # 3 is K = # of quantiles
        # 2 is M = # of models
        x = np.linspace(-5 * 3 * 2, 5 * 3 * 2, num = 5 * 3 *2 * 2).reshape((4, 3, 5))
        qe = qens.MedianQEns(M=5, tau=[0.25, 0.5, 0.75], tau_grps=[0, 1, 2])
        rectangle_bw = qe.calc_kde_rectangle_width(q = tf.constant(q), train_q = True)
        actual_weighted_cdf = qe.weighted_cdf(tf.constant(x, dtype="float32"),
                                              tf.constant(q),
                                              tf.constant(w),
                                              rectangle_bw)
        actual_weighted_cdf = actual_weighted_cdf.numpy()
        
        q = tf.constant(q)
        w = tf.constant(w)
        #transpose x to [5, 4, 3]. sample_shape comes first in tensorflow
        x = np.transpose(x.T, (0, 2, 1))
        
        rectangle_bw = qe.calc_kde_rectangle_width(q = q, train_q = True)
        q, broadcast_w = qe.handle_missingness(q, w)
        
        low = q - rectangle_bw/2
        high = q + rectangle_bw/2
        
        # create a mixture of uniform distributions
        # batch shape is (N, K)
        model = tfd.MixtureSameFamily(
            # batch shape is (N, K)
            mixture_distribution=tfd.Categorical(probs = broadcast_w), 
            # batch shape is (N, K, M)
            components_distribution=tfd.Uniform(low = low, high = high))
        expected_weighted_cdf = model.cdf(x).numpy()
        # transpose back to [4, 3, 5]
        expected_weighted_cdf = np.transpose(expected_weighted_cdf.T, (1, 0, 2))
        
        for i in range(expected_weighted_cdf.shape[0]):
            for j in range(expected_weighted_cdf.shape[1]):
                 for k in range(expected_weighted_cdf.shape[2]):
                    self.assertAlmostEqual(expected_weighted_cdf[i,j,k],actual_weighted_cdf[i,j,k], places=7)
    
    def test_MedianQEns_predict(self):
        q = tf.constant(np.linspace(1, 4 * 3 * 2, 4 * 3 * 2).reshape((4, 3, 2)), dtype="float32")
        w = tf.constant(np.array(
            [[0.50, 0.50],
            [0.50, 0.50],
            [0.50, 0.50]]
        ), dtype="float32")
        # make predictions based on pre-saved rectangle_width
        qe = qens.MedianQEns(M=2, tau=[0.25, 0.5, 0.75], tau_grps=[0, 1, 2])
        junk = qe.parameters['w_tau_grps'].assign(w)
        rectangle_bw = qe.calc_kde_rectangle_width(q = tf.constant(q), train_q = True)
        prediction = qe.predict(tf.constant(q))
        
        actual_cdf = qe.weighted_cdf(tf.expand_dims(prediction,-1),tf.constant(q), tf.constant(w), rectangle_bw)
        actual_cdf = actual_cdf.numpy()
        
        for i in range(actual_cdf.shape[0]):
            for j in range(actual_cdf.shape[1]):
                 for k in range(actual_cdf.shape[2]):
                    self.assertAlmostEqual(actual_cdf[i,j,k],0.5, places=7)

        

if __name__ == '__main__':
  unittest.main()
