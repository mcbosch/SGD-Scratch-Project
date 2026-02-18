from nn.NN import *
import unittest 

class TestLayer(unittest.TestCase):

    def setUp(self):
        self.database = [((0,0),1),
                         ((0,1),0),
                         ((1,1),1),
                         ((1,0),0)]
        
        self.layer = LinearLayer(2,3,activation='ReLU', random_seed=87925)
    
        
    def test_init(self):
        """Tests that the dimensionality of the layer it's correctly
        -----------------------------
        IMPORTANT NOTE
        This package works with the following oreder of operations:
        
            INPUT @ WEIGHTS + BIAS
        
        And works with vectors as rows. Why is this? 
        Because we are working with numpy.arrays, of shape (n,), and we want to be able to scale this models to work with batches. This means that we'll have a bunch of rows (b,n). 

        For numpy it's better to treat vectors as rows. As mathematician I am not comfortable with it, but it is what it is.
        """

        self.assertEqual(self.layer.weights.shape, (2,3))
        self.assertEqual(self.layer.bias.shape, (3,))
        
    def test_forward(self):
        """
        We set as seed 87925. Thus, the weights should be:
        W = [[ 0.34293932, -0.23579481,  1.32306839],
             [ 0.08207777,  0.97098021, -0.34122417]]
        b = [ 0.0855185 , -1.09668207, -0.93902462]
        Then we have to assert basic operations
        """
        sol1 = self.layer.forward([1,0])
        self.assertEqual(list(sol1),[np.float64(0.42845781277754424), np.float64(0.0)        , np.float64(0.3840437644609407)] )
        
        sol2 = self.layer.forward([0,1])
        self.assertEqual(list(sol2),[np.float64(0.16759626952487727),np.float64(0.0),np.float64(0.0)])

        # We check it for a batch
        result_by_batches = self.layer.forward([[1,0],[0,1]])
        for i in range(3):
            self.assertEqual(result_by_batches[0][i],sol1[i])
            self.assertEqual(result_by_batches[1][i],sol2[i])

    def test_backpropagation(self):
        """
        For testing backpropagation, we have to backpropagate a error of neurons, verify it's backproagating correctly and verify it's updating our parameters correctly.

        We give a vector as the error of another layer and we test if it returns what is expected.
        """
        W = self.layer.weights.T
        b = self.layer.bias.T
        x = np.array([1,1])
        self.layer.forward(x) # update neurons layer
        d2 = np.array([1,2,3]) # Vector to backpropagate
        
        result = self.layer.backpropagation(d2)
        # Check cache is correct
        self.assertTrue(np.array_equal(self.layer.cache,np.array([1.,0.,3.])))

        # Check if it returns what expected
        ret = (d2*np.array([1,0,1]))@W
        self.assertTrue(np.array_equal(result, (d2*np.array([1,0,1]))@W))

        # Check for batch
        d2 = np.array([[1,2,3],
                       [1,2,3]])
        self.layer.forward(np.array([[1,1],
                                     [1,1]]))
        result = self.layer.backpropagation(d2)

        # Test if Cache is correct
        self.assertTrue(np.array_equal(self.layer.cache,np.array([[1.,0.,3.],[1.,0.,3.]])))

        # Test if what returns is correct
        self.assertTrue(np.array_equal(result, np.array([ret, ret])))

# We didn't build a Test for Sequence because it's simply a list of layers, maybe in future updates.

class TestNN(unittest.TestCase):
    def setUp(self):
        self.nn = NeuralNetwork(dimensions=[2,3,2],
                                activations=["ReLU","Softmax"])
    
    def test_forward():
        pass

if __name__ == '__main__':
    unittest.main()