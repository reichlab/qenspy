import numpy as np
import tensorflow as tf
import unittest

from qenspy import qens


class Test_MeanQEns(unittest.TestCase):
  def test_MeanQEns_predict_none_missing(self):
    q = tf.constant(np.linspace(1, 4 * 3 * 2, 4 * 3 * 2).reshape((4, 3, 2)))
    w = tf.constant(np.array(
      [[0.75, 0.25],
      [0.10, 0.90],
      [0.20, 0.80]]
    ))

    ensemble_q = qens.MeanQEns().predict(q,w).numpy()

    q_np = q.numpy()
    w_np = w.numpy()
    expected_ensemble_q = q[..., 0] * w[:, 0] + q[..., 1] * w[:, 1]

    # all entries equal
    self.assertTrue(np.all(ensemble_q == expected_ensemble_q))


if __name__ == '__main__':
  unittest.main()
