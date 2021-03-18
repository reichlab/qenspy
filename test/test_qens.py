import numpy as np
import tensorflow as tf
import unittest

from qenspy import qens


class Test_QEns(unittest.TestCase):
  def test_fill_missing_and_renormalize_none_missing(self):
    tau_groups = [0, 0, 0, 1, 1, 1, 1, 2, 2, 2]
    param_vec = tf.constant(np.log([2,1,3,2,4,3]), dtype = "float64")
    params_dict = qens.MeanQEns().unpack_params(
      param_vec=param_vec,
      K=10,
      M=3,
      tau_groups=tau_groups
    )
    w = params_dict['w']
    q = tf.constant(np.linspace(1, 5 * 10 * 3, 5 * 10 * 3).reshape((5, 10, 3)))

    (result_q, result_w) = qens.MeanQEns().fill_missing_and_renormalize(q, w)

    # no change to q
    self.assertTrue(np.all(q.numpy() == result_q.numpy()))

    # 5 copies of the original w
    for i in range(5):
      self.assertTrue(np.all(w.numpy() == result_w.numpy()[i, ...]))


  def test_unpack_params(self):
    tau_groups = [0, 0, 0, 1, 1, 1, 1, 2, 2, 2]
    param_vec = tf.constant(np.log([2,1,3,2,4,3]), dtype = "float64")
    params_dict = qens.MeanQEns().unpack_params(
      param_vec=param_vec,
      K=10,
      M=3,
      tau_groups=tau_groups
    )
    w = params_dict['w']

    # Assert w has shape (K, M) == (10, 3)
    self.assertTrue(np.all(tf.shape(w).numpy() == np.array([10, 3])))

    # Assert row sums are all approximately 1
    self.assertTrue(
      np.all(np.abs(tf.math.reduce_sum(w, axis = 1).numpy() - np.ones(10)) < 1e-7))

    # Assert rows defined by tau_groups are equal to each other
    for i in [1,2]:
      self.assertTrue(np.all(w[0, :].numpy() == w[i, :].numpy()))

    for i in [4,5,6]:
      self.assertTrue(np.all(w[3, :].numpy() == w[i, :].numpy()))

    for i in [8,9]:
      self.assertTrue(np.all(w[7, :].numpy() == w[i, :].numpy()))


  def test_pinball_loss(self):
    y = np.array([0.1, 0.2, 0.3])
    q = np.concatenate((
        np.array([0.2, -0.5, 0.4]).reshape(3,1), 
        np.array([0.1, 0.7, -0.5]).reshape(3,1)), axis = 1)
    tau = np.array([0.2, 0.5])

    actual = qens.MeanQEns().pinball_loss(tf.constant(y), tf.constant(q), tf.constant(tau))

    # calculate expected 
    expected = 0
    for i in range(q.shape[0]):
        for k in range(q.shape[1]):
            if y[i] < q[i,k]:
                expected += (1 - tau[k]) * (q[i,k] - y[i])
            else:
                expected += (0 - tau[k]) * (q[i,k] - y[i])

    self.assertAlmostEqual(actual.numpy(),expected,places=2)

if __name__ == '__main__':
  unittest.main()
