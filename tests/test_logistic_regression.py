from unittest import TestCase

import numpy as np

from logistic_regression import LogisticRegression

def generate_dummy_update(n):
    def dummy_update(*args, **kwargs):
        yield np.ones(n)
        yield -np.ones(n)
    return dummy_update

class LogisticRegressionTest(TestCase):
    def test_fit(self):
        model = LogisticRegression(2, epochs=1,
                                   update_method=generate_dummy_update(2))
        ws = [list(w) for w in model.fit(np.array([1, 2, 3])[None].T,
                                         np.array([1, 0, 1]))]
        self.assertEqual(len(ws), 3)
        self.assertListEqual(ws[0], ws[2])
        self.assertFalse(list(ws[0]) == list(ws[1]))
    
    def test_loss(self):
        model = LogisticRegression(2)
        model.w = np.random.random(2) * 2 - 1
        random_data = np.random.random((3, 2)) * 2 - 1
        random_labels = np.random.randint(0, 2, 2)
        self.assertTrue(model.loss(random_data, random_labels) >= -1e-8)
        self.assertTrue(model.loss(random_data, 1 - random_labels) >= -1e-8)
    
    def test_gradloss(self):
        np.random.seed(0)
        model = LogisticRegression(2)
        model.w = np.zeros(2)
        self.assertListEqual(list(model.gradloss(np.random.random((2, 2)),
                                                 np.zeros(2))),
                             list(np.zeros(2)))
        self.assertListEqual(list(model.gradloss(np.random.random((2, 2)),
                                                 np.zeros(2))),
                             list(np.zeros(2)))
        model.w = np.ones(2)
        self.assertFalse(list(model.gradloss(np.random.random((100, 2)),
                                             np.zeros(100)))
                         == list(np.zeros(2)))

    def test_calculate_probabilities(self):
        model = LogisticRegression(2)
        model.w = np.zeros(2)
        self.assertListEqual(
            model.calculate_probabilities(np.random.random((100, 2))),
            list(np.ones(100) * 0.5)
        )

    def test_predict(self):
        model = LogisticRegression(2)
        model.w = np.array([1, 0])
        self.assertListEqual(list(model.predict(np.array([[3, 7], [-2, 3]]))),
                             [1, 0])
