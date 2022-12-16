import numpy as np

class LinearReg(object):
    
    def __init__(self,
                 X,
                 y):
        '''
            :param X: feature values
            :param y: known target values
        '''
        self.X = X
        self.y = y
        
    def fitted_values(self, w):
        '''
        Calculates fitted values based on 
        design matrix X and weights.
        
            :param w: weights (or coefficients)
            
        '''
        return np.dot(self.X, w.T)
    
    def gradient(self, w):
        '''
        Calculates the gradient for the intercept 
        and the features.
        
            :param w: weights (or coefficients)
            
        '''
        w_features = np.mean((self.fitted_values(w)-self.y
                  )*self.X,axis=0)
        return w_features

    def gradient_descent(self, 
                         learning_rate, 
                         n_iter, 
                         threshold=1e-06):
        '''
        Performs gradient descent to the weights, which minimize
        the squared loss function which yields in the best line of fit.
        In the iterative process the weights are updated as long as the 
        difference between the old and the updated weights are >= 1e-06.
            :param learning_rate: rate which applies gradient for update
            :param n_iter: number of iterations in optimisation
            :param threshold: minimum difference between old and new weights
        '''
        self.y = np.array(self.y).reshape((len(self.y),1))
        one = np.ones((len(self.X),1))
        self.X = np.append(one, self.X, axis=1)
        weight_vector = np.random.rand(1, self.X.shape[1])
        for _ in range(n_iter):
            update = -learning_rate * np.array(self.gradient(
                weight_vector))
            if np.all(np.abs(update) <= threshold):
                break
            weight_vector += update
        return weight_vector