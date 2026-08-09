import tensorflow as tf
from tensorflow import keras

@tf.custom_gradient #this is a tensrflow function enablesus to define how we want the gradient to be computed
def gradient_reversal(x, lambda_):
    def grad(dy):
        return -lambda_ * dy, None
    
    return x, grad

class GradientReversalLayer(keras.layers.Layer):
    
    def __init__(self, lambda_=1.0, **kwargs): 
        
        super(GradientReversalLayer, self).__init__(**kwargs) #parent class to make sure it behaves like normal keras layer
        self.lambda_ = lambda_
        
    def call(self, x):
        return gradient_reversal(x, self.lambda_)
    
    def get_config(self):
        config = super(GradientReversalLayer, self).get_config()
        config.update({'lambda_': self.lambda_})
        return config

