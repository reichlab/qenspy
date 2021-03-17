import numpy as np
import tensorflow as tf
import unittest

from qenspy import qens


class Test_QEns(unittest.TestCase):
  def test_broadcast_w_and_renormalize_none_missing(self):
    tau_groups = [0, 0, 0, 1, 1, 1, 1, 2, 2, 2]
    param_vec = tf.constant(np.log([2,1,3,2,4,3]), dtype = "float64")
    params_dict = qens.QEns().unpack_params(
      param_vec=param_vec,
      K=10,
      M=3,
      tau_groups=tau_groups
    )
    w = params_dict['w']
    q = tf.constant(np.linspace(1, 5 * 10 * 3, 5 * 10 * 3).reshape((5, 10, 3)))

    result_w = qens.QEns().broadcast_w_and_renormalize(q, w)

    # 5 copies of the original w
    for i in range(5):
      self.assertTrue(np.all(w.numpy() == result_w.numpy()[i, ...]))

    # Assert sum of weights across models are all approximately 1
    self.assertTrue(
      np.all(np.abs(tf.math.reduce_sum(result_w, axis = 2).numpy() - np.ones((5, 10))) < 1e-7))


  def test_broadcast_w_and_renormalize_with_missing(self):
    tau_groups = [0, 0, 0, 1, 1, 1, 1, 2, 2, 2]
    param_vec = tf.constant(np.log([2,1,3,2,4,3]), dtype = "float64")
    params_dict = qens.QEns().unpack_params(
      param_vec=param_vec,
      K=10,
      M=3,
      tau_groups=tau_groups
    )
    w = params_dict['w']
    q_np = np.linspace(1, 5 * 10 * 3, 5 * 10 * 3).reshape((5, 10, 3))
    q_np[[0, 0, 0, 3], [0, 0, 1, 3], [0, 1, 1, 2]] = np.nan
    q = tf.constant(q_np)

    result_w = qens.QEns().broadcast_w_and_renormalize(q, w)

    # entries at indices i with no missingness are copies of the original w
    for i in [1,2,4]:
      self.assertTrue(
        np.all(np.abs(w.numpy() - result_w.numpy()[i, ...]) < 1e-7))

    # for values of i with missingness,
    # no change in rows k without missingness
    self.assertTrue(
      np.all(np.abs(w[2:10, :].numpy() - result_w.numpy()[0, 2:10, :]) < 1e-7))
    self.assertTrue(
      np.all(np.abs(w[0:2, :].numpy() - result_w.numpy()[3, 0:2, :]) < 1e-7))
    self.assertTrue(
      np.all(np.abs(w[4:10, :].numpy() - result_w.numpy()[3, 4:10, :]) < 1e-7))

    # for values of i with missingness,
    # entries [i, k, m] with missingness are 0
    self.assertTrue(
      np.all(result_w.numpy()[[0, 0, 0, 3], [0, 0, 1, 3], [0, 1, 1, 2]] == np.zeros(4)))

    # Assert sum of weights across models are all approximately 1
    self.assertTrue(
      np.all(np.abs(tf.math.reduce_sum(result_w, axis = 2).numpy() - np.ones((5, 10))) < 1e-7))


  def test_unpack_params(self):
    tau_groups = [0, 0, 0, 1, 1, 1, 1, 2, 2, 2]
    param_vec = tf.constant(np.log([2,1,3,2,4,3]), dtype = "float64")
    params_dict = qens.QEns().unpack_params(
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


if __name__ == '__main__':
  unittest.main()
