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
        q = np.linspace(1, 4 * 3 * 5, 4 * 3 * 5).reshape((4, 3, 5))
        actual_rectangle_bw = qens.MedianQEns().calc_kde_rectangle_width(q = tf.constant(q), train_q = False)
        # each value has 1 as the closest distance to its neighbor
        expected_rectangle_bw = np.ones(shape = (4,3,5)) * np.sqrt(12 * (1 ** 2))
        for i in range(actual_rectangle_bw.shape[0]):
            for j in range(actual_rectangle_bw.shape[1]):
                for k in range(actual_rectangle_bw.shape[2]):
                    self.assertAlmostEqual(actual_rectangle_bw[i,j,k],expected_rectangle_bw[i,j,k], places=7)
    
    def test_MedianQEns_calc_kde_rectangle_bw_unequal_distance(self):
        print("unequal distance")
        q = np.random.uniform(low=-5, high=15, size=(20,)).reshape((2, 2, 5))
        actual_rectangle_bw = qens.MedianQEns().calc_kde_rectangle_width(q = tf.constant(q), train_q = False)

        expected_rectangle_bw = np.zeros(shape = (2, 2, 5))
        for i in range(actual_rectangle_bw.shape[0]):
            for j in range(actual_rectangle_bw.shape[1]):
                l = np.array_split(q[i,j,:], actual_rectangle_bw.shape[2])
                # pariwise distance for every point/subarray in l
                distance = squareform(pdist(l, 'euclidean'))
                # replace 0 with infinity
                distance[distance == 0] = np.inf
                # get closest distance for each point in the original array q[i, j,:]
                min_distance =  np.min(distance, axis = 1)
                expected_rectangle_bw = tf.sqrt(12 * (min_distance ** 2))
                for k in range(actual_rectangle_bw.shape[2]):
                    self.assertAlmostEqual(expected_rectangle_bw[k],actual_rectangle_bw[i,j,k], places=7)

    def test_MedianQEns_weighted_cdf(self):
        q = np.linspace(1, 4 * 3 * 2, 4 * 3 * 2).reshape((4, 3, 2))
        w = np.array(
        [[0.75, 0.25],
        [0.10, 0.90],
        [0.20, 0.80]]
        )

        # 5 is sample_shape = # of points evaluating at CDF
        # 4 is N = # of forecasts at location, horizon, target, forecast_date....
        # 3 is K = # of quantiles
        # 2 is M = # of models
        x = np.linspace(-5 * 3 * 2, 5 * 3 * 2, num = 5 * 3 *2 * 2).reshape((4, 3, 5))
        rectangle_bw = qens.MedianQEns().calc_kde_rectangle_width(q = tf.constant(q), train_q = False, w = tf.constant(w))
        actual_weighted_cdf = qens.MedianQEns().weighted_cdf(tf.constant(x), tf.constant(q), tf.constant(w), rectangle_bw)
        actual_weighted_cdf = actual_weighted_cdf.numpy()


        q = tf.constant(q)
        w = tf.constant(w)
        #transpose x to [5, 4, 3]. sample_shape comes first in tensorflow
        x = np.transpose(x.T, (0, 2, 1))
    
        rectangle_bw = qens.MedianQEns().calc_kde_rectangle_width(q = q, train_q = False, w = w)
        q, broadcast_w = qens.MedianQEns().handle_missingness(q, w)

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
        q = np.linspace(1, 4 * 3 * 2, 4 * 3 * 2).reshape((4, 3, 2))
        w = np.array(
        [[0.50, 0.50],
        [0.50, 0.50],
        [0.50, 0.50]]
        )

        prediction = qens.MedianQEns().predict(tf.constant(q), train_q = False, w = {'w': tf.constant(w)})

        rectangle_bw = qens.MedianQEns().calc_kde_rectangle_width(q = tf.constant(q), train_q = False, w = tf.constant(w))
        actual_cdf = qens.MedianQEns().weighted_cdf(tf.expand_dims(prediction,-1),tf.constant(q), tf.constant(w), rectangle_bw)
        actual_cdf = actual_cdf.numpy() 

        for i in range(actual_cdf.shape[0]):
            for j in range(actual_cdf.shape[1]):
                 for k in range(actual_cdf.shape[2]):
                    self.assertAlmostEqual(actual_cdf[i,j,k],0.5, places=7)

        

if __name__ == '__main__':
  unittest.main()
