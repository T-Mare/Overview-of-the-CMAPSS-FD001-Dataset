import tensorflow as tf
from tensorflow import keras

#ttps://www.tensorflow.org/api_docs/python/tf/custom_gradient
@tf.custom_gradient #this is a tensrflow function enablesus to define how we want the gradient to be computed
def gradient_reversal(x, lambda_):
    def grad(dy):
        return -lambda_ * dy, None
    
    return x, grad

class GradientReversalLayer(keras.layers.Layer):
    #define lamda
    def __init__(self, lambda_=1.0, **kwargs): 
   
        super(GradientReversalLayer, self).__init__(**kwargs) #parent class to make sure it behaves like normal keras layer

        self.lambda_ = lambda_
        
    #this is the forward pass with the gradient reversal
    def call(self, x):
        
        return gradient_reversal(x, self.lambda_)
    
    #this is the serialization of the layer which so thatthe layer can be saved
    def get_config(self):
        config = super(GradientReversalLayer, self).get_config()
        config.update({'lambda_': self.lambda_})
        return config

