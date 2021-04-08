import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
tfd = tfp.distributions
import unittest

from qenspy import qens
import timeit

class Test_MedianQEns(unittest.TestCase):
    def test_MedianQEns_calc_bandwidth_unweighted(self):
        q = np.linspace(1, 4 * 3 * 2, 4 * 3 * 2).reshape((4, 3, 2))
        w = np.array(
        [[0.50, 0.50],
        [0.50, 0.50],
        [0.50, 0.50]]
        )

        actual_unweighted_bw = qens.MedianQEns("silverman_unweighted").calc_bandwidth(tf.constant(q))
        actual_unweighted_bw = actual_unweighted_bw.numpy()

        unweighted_mean = q[..., 0] * w[:, 0] + q[..., 1] * w[:, 1]
        variance = (((q[..., 0] - unweighted_mean)**2) * w[:, 0]  + ((q[..., 1] - unweighted_mean)**2) * w[:, 1])/2
        unweighted_sd = np.sqrt(variance)
        expected_unweighted_bw = 0.9 * unweighted_sd * (2**(-0.2))

        for i in range(actual_unweighted_bw.shape[0]):
            for j in range(actual_unweighted_bw.shape[1]):
                self.assertAlmostEqual(actual_unweighted_bw[i,j],expected_unweighted_bw[i,j], places=7)
    
    def test_MedianQEns_calc_bandwidth_weighted(self):
        q = np.linspace(1, 4 * 3 * 2, 4 * 3 * 2).reshape((4, 3, 2))
        w = np.array(
        [[0.75, 0.25],
        [0.10, 0.90],
        [0.20, 0.80]]
        )

        actual_weighted_bw = qens.MedianQEns("silverman_weighted").calc_bandwidth(tf.constant(q), tf.constant(w))
        actual_weighted_bw = actual_weighted_bw.numpy()
        weighted_mean = q[..., 0] * w[:, 0] + q[..., 1] * w[:, 1]
        variance = (((q[..., 0] - weighted_mean)**2) * w[:, 0]  + ((q[..., 1] - weighted_mean)**2) * w[:, 1])/2
        weighted_sd = np.sqrt(variance)
        expected_weighted_bw = 0.9 * weighted_sd * (2**(-0.2))

        for i in range(actual_weighted_bw.shape[0]):
            for j in range(actual_weighted_bw.shape[1]):
                self.assertAlmostEqual(actual_weighted_bw[i,j],expected_weighted_bw[i,j], places=7)

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
        bw = qens.MedianQEns("silverman_weighted").calc_bandwidth(tf.constant(q), tf.constant(w))
        rectangle_bw = tf.sqrt(12 * (bw ** 2))
        actual_weighted_cdf = qens.MedianQEns("silverman_weighted").weighted_cdf(tf.constant(x), tf.constant(q), tf.constant(w), rectangle_bw)
        actual_weighted_cdf = actual_weighted_cdf.numpy()


        q = tf.constant(q)
        w = tf.constant(w)
        #transpose x to [5, 4, 3]. sample_shape comes first in tensorflow
        x = np.transpose(x.T, (0, 2, 1))
    
        bw = qens.MedianQEns("silverman_weighted").calc_bandwidth(q, w)
        rectangle_bw = tf.sqrt(12 * (bw ** 2))
        broadcast_w, q = qens.MedianQEns("silverman_weighted").handle_missingness(q, w)

        low = q - tf.reshape(rectangle_bw/2, [rectangle_bw.shape[0], rectangle_bw.shape[1], 1])
        high = q + tf.reshape(rectangle_bw/2, [rectangle_bw.shape[0], rectangle_bw.shape[1], 1])
      
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

        prediction = qens.MedianQEns("silverman_unweighted").predict(tf.constant(q), tf.constant(w))

        bw = qens.MedianQEns("silverman_unweighted").calc_bandwidth(tf.constant(q), tf.constant(w))
        rectangle_bw = tf.sqrt(12 * (bw ** 2))
        actual_cdf = qens.MedianQEns("silverman_unweighted").weighted_cdf(tf.expand_dims(prediction,-1),tf.constant(q), tf.constant(w), rectangle_bw)
        actual_cdf = actual_cdf.numpy() 

        for i in range(actual_cdf.shape[0]):
            for j in range(actual_cdf.shape[1]):
                 for k in range(actual_cdf.shape[2]):
                    self.assertAlmostEqual(actual_cdf[i,j,k],0.5, places=7)

        

if __name__ == '__main__':
  unittest.main()
