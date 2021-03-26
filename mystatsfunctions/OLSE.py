# Some statistical functions I wrote
# Nick Leach 2021

"""
Basic methods for fitting least squares estimators.
"""

import numpy as np
from scipy.stats import t

# Simple & multiple least squares estimation in numpy

# TODO add in partial correlations

## auxiliary functions

# autocorrelation function of input X
def ACF(X):
    
    """
    Calculates autocorrelation function over first dimension of X
    
    X : np.ndarray
    
    returns np.ndarray of ACF at each lag of the first dimension of X
    """
    
    N = X.shape[0]
    
    # compute mean of X
    _X = X.mean(axis=0)
    
    # autocovariance
    ACV = np.array([np.sum((X-_X)[:N-k]*(X[k:]-_X),axis=0) for k in np.arange(N)]) / N
    
    # autocorrelation
    ACF = ACV / np.std(X,axis=0)**2
    
    return ACF


## estimator classes ##

class simple():
    """
    This function calculates a (weighted) simple least-squares regression over the first dimension of x & y.
    
    The regression is broadcast over all the other dimensions (so you can eg. compute point-by-point simple regressions of two variables).
    
    Compatible with masked arrays.
    
    Y: np.ndarray
    Array of dependent variable values.
    First dimension is the axis over which the regression will be computed.
    
    W: np.ndarray
    Weights array.
    Must be broadcastable with Y (easiest way is to ensure they have the same number of dimensions).
    """
    
    ## attributes
    
    def __init__(self, Y, W=None):
        
        self.Y = Y
        self.n = Y.shape[0]
        
        self.W = W
        
        self._fit = False
        
    def fit(self, X):
        """
        Fit the dependent variable array(s) to the explanatory variable array(s) with simple (weighted) least squares.
        
        X: np.ndarray
        Array of explanatory variable values.
        Must be broadcastable with Y (easiest way is to ensure they have the same number of dimensions).
        
        Creates class objects:
        
        res: np.ndarray
        Array of residuals. Same shape as X/Y broadcasting.
        
        s2: np.ndarray
        Variance of errors.
        
        b0: np.ndarray
        Estimated intercept.
        
        b1: np.ndarray
        Estimated slope.
        
        err_b0(1): np.ndarray
        Standard errors in intercept (slope) parameters.
        """
        
        self.X = X
        
        if self.W is None:
            w = np.ones(X.shape)
            self.W = w
        else:
            w = self.W
        
        x = self.X
        y = self.Y
        
        n = self.n
        
        # compute weighted means
        _x = np.ma.sum(w * x, axis=0) / np.ma.sum(w, axis=0)
        _y = np.ma.sum(w * y, axis=0) / np.ma.sum(w, axis=0)
        
        # compute sums of squares
        Sxx = np.ma.sum(w * (x-_x)**2, axis=0)
        Syy = np.ma.sum(w * (y-_y)**2, axis=0)
        Sxy = np.ma.sum(w * (x-_x)*(y-_y), axis=0)
        
        # parameter estimates
        b1 = Sxy / Sxx
        b0 = _y - b1 * _x
        
        # residuals
        e = y - b0 - b1 * x
        
        # standard error
        s2 = np.ma.sum(w * e**2, axis=0) / (n-2)
    
        # parameter variances
        s2_b1 = s2 / Sxx
        s2_b0 = s2 * ( 1/np.ma.sum(w, axis=0) + _x**2/Sxx )
        
        # set class residuals
        self.res = e
        
        # set class standard error
        self.s2 = s2
        
        # set class parameters
        self.b0 = b0
        self.b1 = b1
        
        # set parameter errors
        self.err_b1 = np.sqrt(s2_b1)
        self.err_b0 = np.sqrt(s2_b0)
        
        # set "hidden" variables for use later
        self._x = _x
        self._y = _y
        self._Sxx = Sxx
        self._Syy = Syy
        self._Sxy = Sxy
        
        # specify the model has been fit
        self._fit = True
        
    def cov(self):
        """
        Returns the covariance of the dependent and explanatory variable arrays.
        """
        
        if self._fit:
            return self._Sxy / np.ma.sum(self.W, axis=0)
        
        else:
            raise TypeError('Linear model not yet specified with self.fit()')
            
    def cor(self):
        """
        Returns the correlation of the dependent and explanatory variable arrays.
        """
        
        if self._fit:
            return self._Sxy / np.sqrt( self._Sxx * self._Syy )
        
        else:
            raise TypeError('Linear model not yet specified with self.fit()')
            
    def pred(self, X0=None):
        """
        Returns predicted values of the fit linear model based on input explanatory variable arrays, X0.
        
        If no X0 supplied, the training data will be used.
        
        X0: np.ndarray
        Array of explanatory variable values.
        Must be broadcastable with model parameters.
        """
        
        if X0 is None:
            X0 = self.X
        
        if not self._fit:
            raise TypeError('Linear model not yet specified with self.fit()')
            
        return self.b0 + self.b1*X0
            
    def CI(self, X0=None, interval = 0.9):
        """
        Returns confidence interval in predicted values of the fit linear model mean based on input explanatory variable arrays, X0.
        
        If no X0 supplied, the training data will be used.
        
        X0: np.ndarray
        Array of explanatory variable values.
        Must be broadcastable with model parameters.
        
        interval: float
        Confidence interval desired.
        """
        
        if X0 is None:
            X0 = self.X
        
        if not self._fit:
            raise TypeError('Linear model not yet specified with self.fit()')
            
        var_model = self.s2 * ( 1/np.ma.sum(self.W, axis=0) + (X0 - self._x)**2/self._Sxx )
        
        t_val = t(self.n-2).ppf(0.5+interval/2)
        
        model = self.pred(X0)
        
        return np.stack([model-t_val*np.sqrt(var_model), model+t_val*np.sqrt(var_model)],axis=0)
    
    def PI(self, X0=None, interval = 0.9):
        
        """
        Returns prediction interval in predicted values of the fit linear model actuals based on input explanatory variable arrays, X0.
        
        If no X0 supplied, the training data will be used.
        
        X0: np.ndarray
        Array of explanatory variable values.
        Must be broadcastable with model parameters.
        
        interval: float
        Prediction interval desired.
        """
        
        if X0 is None:
            X0 = self.X
        
        if not self.fit:
            raise TypeError('Linear model not yet specified with self.fit()')
            
        var_model = self.s2 * ( 1 + 1/np.ma.sum(self.W, axis=0) + (X0 - self._x)**2/self._Sxx )
        
        t_val = t(self.n-2).ppf(0.5+interval/2)
        
        model = self.pred(X0)
        
        return np.stack([model-t_val*np.sqrt(var_model), model+t_val*np.sqrt(var_model)],axis=0)
    
    
class multiple():
    """
    This function calculates an ordinary least squares estimator.
    
    Any weighting (eg. accounting for heteroscedasticity, pre-whitening) must be done prior to input.
    
    Not currently compatible with masked arrays.
    
    Y : np.ndarray
    dimensions: (samples, targets)
    """
    
    ## attributes
    
    def __init__(self, Y):
        
        self.Y = Y
        self.n = Y.shape[0]
        
        self._fit = False
        
    def fit(self, X, add_intercept=True):
        """
        X : np.ndarray
        dimensions: (samples, features)
        
        Creates class objects:
        
        res: np.ndarray
        Array of residuals. Same shape as Y.
        
        s2: np.ndarray
        Variance of errors.
        
        B: np.ndarray
        Estimated OLSE parameters.
        
        err_B: np.ndarray
        Standard errors in OLSE parameters.
        """
        
        if add_intercept:
            X = np.concatenate( [np.ones(self.n)[:,None], X], axis=1 )
        
        self.X = X
        k = X.shape[1]
        self.k = k
        Y = self.Y
        n = self.n
        
        # OLSE estimator of model parameters:
        B = np.linalg.inv( X.T @ X ) @ X.T @ Y
        
        # compute residuals
        e = Y - X@B
        
        # standard error estimate
        s2 = np.sum(e**2, axis=0) / (n-k)
        
        # parameter error estimate
        # by default this is arranged with dimensions (targets, features)
        s2_B = np.einsum( 'i,j->ij', s2, np.diag( np.linalg.inv(X.T @ X) ) )
        
        # set class residuals
        self.res = e
        
        # set class standard error
        self.s2 = s2
        
        # set class parameters
        self.B = B
        
        # set parameter errors
        self.err_B = np.sqrt(s2_B)
        
        # specify the model has been fit
        self._fit = True
        
    def pred(self, X0=None):
        """
        Returns predicted values of the OLSE based on input features, X0.
        
        If no X0 supplied, the training data will be used.
        
        X0: np.ndarray
        Features array, with samples in the first dimension.
        """
        
        if X0 is None:
            X0 = self.X
        
        if not self._fit:
            raise TypeError('Linear model not yet specified with self.fit()')
            
        return X0@self.B
    
    def CI(self, X0=None, interval = 0.9):
        """
        Returns confidence interval in mean predicted values of the OLSE about input features, X0.
        
        If no X0 supplied, the training data will be used.
        
        X0: np.ndarray
        Features array, with samples in the first dimension.
        
        interval: float
        Confidence interval desired.
        """
        
        if X0 is None:
            X0 = self.X
        
        if not self._fit:
            raise TypeError('Linear model not yet specified with self.fit()')
            
        var_model = np.einsum('i,j->ij',np.diag( self.X @ np.linalg.inv(self.X.T @ self.X) @ self.X.T ), self.s2)
        
        model = self.pred(X0)
        
        t_val = t(self.n-2).ppf(0.5+interval/2)
        
        return np.stack([model-t_val*np.sqrt(var_model), model+t_val*np.sqrt(var_model)],axis=0)
    
    def PI(self, X0=None, interval = 0.9):
        """
        Returns prediction interval in predicted values of the OLSE about input features, X0.
        
        If no X0 supplied, the training data will be used.
        
        X0: np.ndarray
        Features array, with samples in the first dimension.
        
        interval: float
        Prediction interval desired.
        """
        
        if X0 is None:
            X0 = self.X
        
        if not self._fit:
            raise TypeError('Linear model not yet specified with self.fit()')
            
        var_model = np.einsum('i,j->ij',1 + np.diag( self.X @ np.linalg.inv(self.X.T @ self.X) @ self.X.T ), self.s2)
        
        model = self.pred(X0)
        
        t_val = t(self.n-2).ppf(0.5+interval/2)
        
        return np.stack([model-t_val*np.sqrt(var_model), model+t_val*np.sqrt(var_model)],axis=0)
    
    def R2(self):
        """
        Returns the coefficient of determination of the OLSE.
        """
        
        if self._fit:
            return 1 - np.sum((self.Y - self.pred())**2, axis=0) / np.sum((self.Y - np.mean(self.Y,axis=0))**2, axis=0)
        
        else:
            raise TypeError('Linear model not yet specified with self.fit()')


# partial correlations of X in Y
# def PARCOR(X , Y):
    
#     """
#     Uses OLSE_NORM to remove the unbiased best-estimate regression model using all but one feature of X, then computes the correlation of that feature against the residuals.
#     """
    
#     correlations = np.zeros((X.shape[-1],Y.shape[-1]))
    
#     for i in np.arange(X.shape[-1]):
        
#         X_roll = np.roll(X,-i,axis=-1)
#         resids = OLSE_NORM(X_roll[:,1:],Y)['res']
#         correlations[i,:] = calculate_stats(X_roll[:,[0]],resids)['cor']
        
#     return correlations
