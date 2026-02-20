import unittest
from nn.utils import *

class TestUtils(unittest.TestCase):
    def setUp(self):
        self.vector_dim1_1 = np.array([1,1])
        self.vector_dim1_2 = np.array([0,0])

        self.batch_vectors_1 = np.array([[ 1, 1],
                                       [ 1, 2],
                                       [-1, 1]])
        self.batch_vectors_2 = np.zeros((3,2))

        self.y_true = np.array([1, 0, 1])
        self.y_true_batch = np.array([[1,0,1],
                                      [1,1,0]])

    def test_loss_MSE(self):
        loss_f = MSE()
        self.assertEqual(loss_f(self.vector_dim1_1,self.vector_dim1_2),1)

        # Check partial respect X
        r = loss_f.partial(self.vector_dim1_1,self.vector_dim1_2, respect_to = "x")
        self.assertTrue(np.array_equal(r, [1,1]), msg = f"\n\033[31mError in partial respect to x\033[0m, should be: {[1, 1]}\n\treturned: {r}")

        # Check partial respect Y
        r = loss_f.partial(self.vector_dim1_1,self.vector_dim1_2, respect_to = "y")
        self.assertTrue(np.array_equal(r, [-1,-1]), msg = f"\n\033[31mError in partial respect to y\033[0m, should be: {[-1, -1]}\n\treturned: {r}")

        # Check for batch
        r = loss_f(self.batch_vectors_1,self.batch_vectors_2)
        real = np.array([1, 2.5, 1])

        self.assertTrue(np.array_equal(r,real),
                        msg=f"\n\033[31mError\033[0m\tExpected: {real}\n\tGiven: {r}")

        # Checl partial batch
        rx = loss_f.partial(self.batch_vectors_1, 
                           self.batch_vectors_2, 
                           respect_to="x")
        ry = loss_f.partial(self.batch_vectors_1, 
                           self.batch_vectors_2, 
                           respect_to="y")
        self.assertTrue(np.array_equal(rx, self.batch_vectors_1),
                        msg=f"\n\033[31mError\033[0m\tExpected: {self.batch_vectors_1}\n\tGiven: {rx}")
        self.assertTrue(np.array_equal(ry, -self.batch_vectors_1),
                        msg=f"\n\033[31mError\033[0m\tExpected: {-self.batch_vectors_1}\n\tGiven: {ry}")

    def test_loss_CrossEntropy(self):
        loss_f = CrossEntropy()
        y_pred = np.array([1,1,0])
        r = loss_f(y_pred,self.y_true)
        expected = - np.log(1e-13)
        # Assert run for 1
        self.assertTrue(np.isclose(r,expected,rtol=1e-12), msg=f"\033[31mError\033[0m\t Given: {r}; Expected: {expected}")
        # Assert partial for 1
        r = loss_f.partial(y_pred, self.y_true)
        expected = np.array([-1,0,-1/1e-13])
        self.assertTrue(np.allclose(r, expected, rtol=1e-12),
                        msg=f"\033[31mError\033[0m\tGiven: {r}; Expected {expected}")


        # Assert for batched Entrys
        y_pred = np.array([[1,1,0],
                           [1,1,0]])
        r = loss_f(y_pred, self.y_true_batch)
        expected = np.array([-np.log(1e-13), 0])
        self.assertTrue(np.allclose(r,expected,rtol = 1e-12),
                        msg=f"\033[31mError\033[0m\tGiven: {r}; Expected {expected}")
        
        r = loss_f.partial(y_pred, self.y_true_batch)
        expected = np.array([[-1,0,-1/1e-13],
                             [-1, -1, 0]])
        self.assertTrue(np.allclose(r, expected, rtol = 1e-12),
                        msg=f"\033[31mError\033[0m\tGiven: {r}; Expected {expected}")
        
        
if __name__ == '__main__':
    unittest.main()