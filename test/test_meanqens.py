import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

import numpy as np
import tensorflow as tf
import unittest

import qens


class Test_MeanQEns(unittest.TestCase):
  def test_MeanQEns_predict_none_missing(self):
    # (N=4, K=3, M=2): 4 prediction tasks, 3 quantile levels, 2 models
    q = tf.constant(np.linspace(1, 4 * 3 * 2, 4 * 3 * 2).reshape((4, 3, 2)), dtype="float32")
    w = tf.constant(np.array(
      [[0.75, 0.25],
      [0.10, 0.90],
      [0.20, 0.80]]
    ), dtype="float32")

    qe = qens.MeanQEns(M=2, tau=[0.25, 0.5, 0.75], tau_grps=[0,1,2])
    junk = qe.parameters['w_tau_grps'].assign(w)
    ensemble_q = qe.predict(q=q).numpy()

    q_np = q.numpy()
    w_np = w.numpy()
    expected_ensemble_q = q[..., 0] * w[:, 0] + q[..., 1] * w[:, 1]

    # all entries (almost) equal
    self.assertTrue(
      np.all(np.abs(ensemble_q - expected_ensemble_q) < 1e-6))


  def test_MeanQEns_predict_with_missing(self):
    q_np = np.linspace(1, 4 * 3 * 2, 4 * 3 * 2).reshape((4, 3, 2))
    q_np[[0, 1, 2], [0, 0, 1], [0, 0, 1]] = np.nan
    q = tf.constant(q_np, dtype = "float32")

    w_np = np.array(
      [[0.75, 0.25],
        [0.10, 0.90],
        [0.20, 0.80]]
    )
    w = tf.constant(w_np, dtype = "float32")

    # actual ensemble prediction
    qe = qens.MeanQEns(M=2, tau=[0.25, 0.5, 0.75], tau_grps=[0,1,2])
    junk = qe.parameters['w_tau_grps'].assign(w)
    ensemble_q = qe.predict(q=q).numpy()
    # ensemble predictions if missingness weren't a problem
    expected_ensemble_q = q[..., 0] * w[:, 0] + q[..., 1] * w[:, 1]
    expected_ensemble_q = expected_ensemble_q.numpy()

    # set predictions at entries with missingness from one model
    # to the prediction from the model without missingness
    expected_ensemble_q[[0, 1, 2], [0, 0, 1]] = \
      q_np[[0, 1, 2], [0, 0, 1], [1, 1, 0]]

    # all entries (almost) equal
    self.assertTrue(
      np.all(np.abs(ensemble_q - expected_ensemble_q) < 1e-6))


if __name__ == '__main__':
  unittest.main()
