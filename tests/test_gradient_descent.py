from unittest import TestCase

import numpy as np

from gradient_descent import stochastic_gradient_descent
from gradient_descent import batch_gradient_descent
from gradient_descent import minibatch_gradient_descent
from gradient_descent import newton_raphson_method

def gradloss(d, l):
    return sum(d.T @ l)

def hessianloss(d, l):
    return np.eye(d.shape[1]) / 2

class GradientDescentTest(TestCase):
    def test_stochastic_gradient_descent(self):
        stoch_res = list(stochastic_gradient_descent(
            np.array([1, 2, 3])[None].T,
            np.array([-1, 0, 1]),
            gradloss,
            learning_rate=0.1))
        self.assertEqual(len(stoch_res), 3)
        for c, w in zip(sorted([-0.1, 0, 0.3]), sorted(stoch_res)):
            self.assertEqual(len(w), 1)
            self.assertAlmostEqual(w[0], c, 8)
    
    def test_minibatch_gradient_descent_batch_size_1(self):
        minibatch_res = list(minibatch_gradient_descent(
            np.array([1, 2, 3])[None].T,
            np.array([-1, 0, 1]),
            gradloss,
            batch_size=1,
            learning_rate=0.1))

        self.assertEqual(len(minibatch_res), 3)
        for c, w in zip(sorted([-0.1, 0, 0.3]), sorted(minibatch_res)):
            self.assertEqual(len(w), 1)
            self.assertAlmostEqual(w[0], c, 8)
        
    def test_minibatch_gradient_descent(self):
        self.assertEqual(len(list(
            minibatch_gradient_descent(np.array([1, 2, 3])[None].T,
                                       np.array([-1, 0, 1]),
                                       gradloss,
                                       batch_size=2,
                                       learning_rate=0.1))),
            2)
    
    def test_batch_gradient_descent(self):
        res = list(batch_gradient_descent(np.random.random((3, 2)),
                                          np.random.randint(0, 2, 3),
                                          gradloss,
                                          learning_rate=0))
        self.assertEqual(len(res), 1)
        self.assertEqual(sum(res[0] ** 2), 0)
