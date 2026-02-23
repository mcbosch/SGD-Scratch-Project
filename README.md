# Neural Network Package from Scratch

In this repository I try to build a basic neural network package builded from scratch for learning purposes. The objective is to really understand the mechanism of a Python package, build tests, and really learn the SGD algorithm to make future projects with larger data. By "from scratch", I mean that I only used the following python packages to make the model and perfom operations:
1. `numpy`: All the NN structure uses numpy
2. `pandas`: Only used to load, store datasets and save our results 

Note that making an atomic neural network it's easy. It's just a bunch of afins transformations with a non-linear function in the middle. I will build an atomic version for the people that want to understand a simple nn. Nevertheless, I've done a more complex version similar to torch to be able to work with complex data, batches, autoencoders, etc.  

Other packages may be used only to load datasets and study our models performence. Moreover, we build a NN with pytorch to compare results. 

I know there's a lot of repositories with neural networks from scratch. As I said, is for learning purposes. My final objective is to introduce myself into Variational Inerence and build Variational Autoencoders from scratch. I already tryed, you can see the repository [here](). Nevertheess, the learning was strange and slow, so I want to do the basics better, so I have a more scalable project to build Variational Autoencoders.

Here we can find a good explanation on backpropagation in [*neuralnetworksanddeeplearning.com*](http://neuralnetworksanddeeplearning.com/about.html). I recomend to make the dirty work with a formal notation. I recomend it because when we attack bigger problems such as build a Variational Autoencoder, backpropagation it's not as trivial as saying "oh! you only have to pass the error information through the neurons". I have a **document in the other repository with all the dirty work**. 


## Future Updates

-   Compare with a PyTorch NN 
-   Analyze graphs. I have overparams?
-   Scale this class to a VAE class
-   Build a VAE class with Pytorch and compare results.