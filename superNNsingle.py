# -*- coding: utf-8 -*-
"""
Created on Mon Nov  2 17:46:04 2020

@author: sunmodza
"""
import numpy as np

def shuffle(*args):
    ind=np.random.permutation(len(args[0]))
    for arg in args:
        arg=arg[ind]
    return args

class Gradient_Descent:
    def __init__(self,lr):
        self.lr=lr
    
    def update_weight(self,m_gradient):
        return -self.lr*np.mean(m_gradient)
    
    def update_bias(self,b_gradient):
        return -self.lr*np.mean(b_gradient)
    
class SGD(Gradient_Descent):
    def __init__(self,lr=10e-2):
        super().__init__(lr)
        
    def update_weight(self,m_gradient):
        return -self.lr*m_gradient
    
    def update_bias(self,b_gradient):
        return -self.lr*b_gradient

class Loss:#Square error as templated
    def __init__(self):
        pass
    def loss(self,actual,predict):
        self.previous_error=actual-predict
        self.derivative=-2*(self.previous_error)
        return (self.previous_error)**2
    

class single_neural_layer:
    def __init__(self,next_layer=None,prev_layer=None):
        self.next_layer=next_layer
        self.prev_layer=prev_layer
        if self.prev_layer is not None:
            self.prev_layer.next_layer=self
        self.weight=np.random.rand(1)
    
    def feed_forward(self,x,chain_feed=False):
        self.last_input=x
        self.a=self.weight*x
        if chain_feed and self.next_layer is not None:
            return self.next_layer.feed_forward(self.a,chain_feed=True)
        return self.a
    
    def update_weight(self):
        self.weight+=self.optimizer.update_weight(self.gradient)
    
    def feed_backward(self,loss=None):
        if self.next_layer is None:
            self.gradient=self.a*loss.derivative #output layer
        else:
            self.gradient=self.a*self.next_layer.gradient*self.last_input #hidden layer and output layer
        
        #update_weight
        self.update_weight()
        
        if self.prev_layer is not None:
            self.prev_layer.feed_backward()
    
    def apply_optimizer(self,optimizer,lr):
        self.optimizer=optimizer(lr)
        return self

class SingularNN:
    def __init__(self,lr=10e-3,optimizer=Gradient_Descent,loss=Loss):
        self.optimizer=optimizer
        self.loss=Loss()
        self.input_layer=single_neural_layer().apply_optimizer(self.optimizer,lr)
        self.hidden_layer1=single_neural_layer(prev_layer=self.input_layer).apply_optimizer(self.optimizer,lr)
        self.hidden_layer2=single_neural_layer(prev_layer=self.hidden_layer1).apply_optimizer(self.optimizer,lr)
        self.hidden_layer3=single_neural_layer(prev_layer=self.hidden_layer2).apply_optimizer(self.optimizer,lr)
        self.output_layer=single_neural_layer(prev_layer=self.hidden_layer3).apply_optimizer(self.optimizer,lr)
    
    def predict(self,x):
        predicted_value=self.input_layer.feed_forward(x,chain_feed=True)
        return predicted_value
        
    def backpropagation(self,x,y):
        predict_value=self.predict(x)
        self.loss_value=self.loss.loss(y,predict_value)
        self.output_layer.feed_backward(loss=self.loss)
        return self.loss_value
    
        
    def fit(self,x,y,epochs):
        for _ in range(epochs):
            x,y=shuffle(x,y)
            for xt,yt in zip(x,y):
                print(nn.backpropagation(xt, yt))
    
nn=SingularNN()
nn.fit(np.array([1,2]),np.array([2,4]),300)
print(nn.predict([1]))
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        