import numpy as np
import tensorflow as tf
import unittest

from qenspy import qens

class Test_MedianQEns(unittest.TestCase):
    def test_MedianQEns_calc_bandwith_unweighted(self):
        q = np.linspace(1, 4 * 3 * 2, 4 * 3 * 2).reshape((4, 3, 2))
        w = np.array(
        [[0.50, 0.50],
        [0.50, 0.50],
        [0.50, 0.50]]
        )

        actual_unweighted_bw = qens.MedianQEns("silverman_unweighted").calc_bandwith(tf.constant(q))
        actual_unweighted_bw = actual_unweighted_bw.numpy()

        unweighted_mean = q[..., 0] * w[:, 0] + q[..., 1] * w[:, 1]
        variance = (((q[..., 0] - unweighted_mean)**2) * w[:, 0]  + ((q[..., 1] - unweighted_mean)**2) * w[:, 1])/2
        unweighted_sd = np.sqrt(variance)
        expected_unweighted_bw = 0.9 * unweighted_sd * (3**(-0.2))

        for i in range(actual_unweighted_bw.shape[0]):
            for j in range(actual_unweighted_bw.shape[1]):
                self.assertAlmostEqual(actual_unweighted_bw[i,j],expected_unweighted_bw[i,j], places=7)
    
    def test_MedianQEns_calc_bandwith_weighted(self):
        q = np.linspace(1, 4 * 3 * 2, 4 * 3 * 2).reshape((4, 3, 2))
        w = np.array(
        [[0.75, 0.25],
        [0.10, 0.90],
        [0.20, 0.80]]
        )

        actual_weighted_bw = qens.MedianQEns("silverman_weighted").calc_bandwith(tf.constant(q), tf.constant(w))
        actual_weighted_bw = actual_weighted_bw.numpy()
        weighted_mean = q[..., 0] * w[:, 0] + q[..., 1] * w[:, 1]
        variance = (((q[..., 0] - weighted_mean)**2) * w[:, 0]  + ((q[..., 1] - weighted_mean)**2) * w[:, 1])/2
        weighted_sd = np.sqrt(variance)
        expected_weighted_bw = 0.9 * weighted_sd * (3**(-0.2))

        for i in range(actual_weighted_bw.shape[0]):
            for j in range(actual_weighted_bw.shape[1]):
                self.assertAlmostEqual(actual_weighted_bw[i,j],expected_weighted_bw[i,j], places=7)