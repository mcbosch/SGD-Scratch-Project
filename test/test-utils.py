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

    def test_ReLU(self):
        relu = ReLU()
        self.assertEqual(relu(-1),0)
        self.assertEqual(relu(12),12)
        self.assertEqual(relu.partial(-1), 0)
        self.assertEqual(relu.partial(12), 1)

        v = np.array([[-1, 1, 2],
                      [0.2, -1, -2]])
        
        self.assertTrue(np.array_equal(relu(v),np.array([[0,1,2],[0.2, 0, 0]])))
        self.assertTrue(np.array_equal(relu.partial(v),np.array([[0,1,1],[1,0,0]])))

    def test_Sigmoind(self):
        S = Sigmoind()
        self.assertEqual(S(0),0.5)
        self.assertTrue(np.isclose(S(np.log(1/3)),0.25,rtol=1e-12), 
                        msg=f"\033[31mError\033[0m\nExpected: {0.25}\nResult: {S(np.log(1/3))}")
        
        self.assertEqual(S.partial(0),0.25)
        
        
        self.assertTrue(np.allclose(S(np.array([0,np.log(1/3)])),
                                    np.array([0.5,0.25]), 
                                    rtol = 1e-12) )

        self.assertTrue(np.allclose(S(np.array([[0,np.log(1/3)],[0,np.log(1/3)]])),
                                    np.array([[0.5,0.25],[0.5,0.25]]),
                                    rtol=1e-12))

        partial_batch = S.partial(np.array([[0,np.log(1/3)],
                                            [0,np.log(1/3)]]))
        
        expected = np.array([[0.25, 0.25*0.75],[0.25,0.25*0.75]])

        self.assertTrue(np.allclose(partial_batch, expected, rtol=1e-12))

        # Test for overflow
    
    def test_Softmax(self):
        S = Softmax()
        a = np.array([5, 1, -2])
        a1 = np.exp(5)
        a2 = np.exp(1)
        a3 = np.exp(-2)
        t = a1+a2+a3
        t1,t2,t3 = a1/t, a2/t, a3/t

        true = np.array([t1, t2, t3])
        self.assertEqual(np.sum(S(a)),1)

        self.assertTrue(np.allclose(S(a),true,rtol=1e-12))
        ab = np.array([[5, 1, -2],
                       [5, 1, -2]])
        self.assertTrue(np.allclose(S(ab),np.array([true,true]),rtol=1e-12))

        partial1 = S.partial(a)
        expected = np.array([[t1*(1-t1), -t1*t2, - t1*t3],
                             [-t2*t1, t2*(1-t2), -t2*t3],
                             [-t3*t1, -t3*t2, t3*(1-t3)]])
        self.assertTrue(np.allclose(partial1, expected, rtol=1e-12),
                        msg=f"\033[31mError\033[0m\nResult: \n{partial1}\nExpected:\n{expected}")

        partialb = S.partial(ab)
        expectedb = np.array([expected, expected])
        self.assertTrue(np.allclose(partialb,expectedb),
                        msg=f"\033[31mError\033[0m\nResult: \n{partialb}\nExpected:\n{expectedb}")
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