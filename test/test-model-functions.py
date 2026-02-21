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
        
        self.assertEqual(self.layer.weights.shape, (2,3))
        self.assertEqual(self.layer.bias.shape, (3,))
        
    def test_forward(self):
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
    unittest.main(verbosity=2)