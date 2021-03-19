# Some statistical functions I wrote
# Nick Leach 2021

import numpy as np

# simple linear regression that utilises numpy array operations
def SLR(x,y,w=False,mask=False):
    
    """
    This function calculates the (weighted) least-squares regression over the first dimension of x & y.
    
    x,y,(w,mask) must have the same number of dimensions, but the dimension length of x (or y) over degenerate dimensions can be 1.
    
    x , y , w , mask : np.ndarray
    
    x = array of independent variable (predictor)
    
    y = array of dependent variable
    
    w = array of weights
    
    mask = mask array
    
    returns:
    dict of regression parameters, errors and tests
    """
    
    if w is False: # create appropriate weight matrix if none provided (all one)
        w = np.ones(x.shape)
    
    if not mask is False: # mask the arrays in the calculation
        x = np.ma.array(x,mask=mask)
        y = np.ma.array(y,mask=mask)
        w = np.ma.array(w,mask=mask)
    
    n = x.shape[0] - np.sum(mask , axis=0) # calculate the data points at each pixel, taking masked values into account
    
    x_mean = np.ma.sum(x*w,axis=0) / np.ma.sum(w,axis=0)
    y_mean = np.ma.sum(y*w,axis=0) / np.ma.sum(w,axis=0)
    Sxy = np.ma.sum( w * ( y - y_mean ) * ( x - x_mean ) , axis=0 )
    Sxx = np.ma.sum( w * ( x - x_mean )**2 , axis=0 )
    Syy = np.ma.sum( w * ( y - y_mean )**2 , axis=0 )
    
    slope = Sxy / Sxx
    intercept = y_mean - x_mean * slope
    
    res =  y - x * slope - intercept
    
    std_err = np.sqrt( np.ma.sum(w*res**2 , axis=0) / (n-2) )
    
    slope_err = std_err / np.sqrt( Sxx )
    intercept_err = np.sqrt( std_err**2 / np.ma.sum( w , axis=0 ) + ( slope_err * x_mean )**2 )
    
    cov = Sxy / np.ma.sum(w,axis=0)
    cor = Sxy / np.sqrt( Syy * Sxx )
    
    return {'cov':cov, 'cor':cor, 'slope':slope, 'intercept':intercept, 'slope_err':slope_err, 'intercept_err':intercept_err, 'res':res, 'std_err':std_err}

# autocorrelation function calculator
def ACF(X):
    
    """
    Calculates autocorrelation function over first dimension of X
    
    X : np.ndarray
    
    returns np.ndarray of ACF at each lag (lags in first dimension) of the first dimension of X
    """
    
    N = X.shape[0]
    
    mean = X.mean(axis=0)[np.newaxis,:]
    
    autocov = np.array([np.sum((X-mean)[:N-k]*(X[k:]-mean),axis=0) for k in np.arange(N)]) / N
    
    autocorr = autocov / np.std(X,axis=0)**2
    
    return autocorr


# ordinary least squares regression
def OLSE_NORM(X,Y,add_intercept=True):
    
    """
    Multiple OLS regression. First dimension is regression dimension, second is features (X), or targets (Y).
    """
    
    if add_intercept:
    
        X_1 = np.concatenate((np.ones(X.shape[0])[:,np.newaxis],X),axis=1)
        
    else:
        
        X_1 = X.copy()
    
    B = np.dot( np.linalg.inv( np.dot( X_1.T , X_1 ) ) , np.dot( X_1.T , Y ) )
    
    e = Y - np.dot(X_1,B)
    
    SSE = np.sum(e**2,axis=0)

    SST = np.sum((Y - np.mean(Y,axis=0))**2,axis=0)

    R2 = 1 - SSE / SST

    MSE_var = SSE / (X_1.shape[0] - X_1.shape[-1])

    SE_B = np.sqrt( np.diag( np.linalg.inv( np.dot( X_1.T , X_1 ) ) )[:,np.newaxis] * MSE_var[np.newaxis,:] )
    
    if add_intercept:
        return {'coefs':B[1:],'coef_err':SE_B[1:],'res':e,'intercept':B[0],'intercept_err':SE_B[0],'R2':R2}
    else:
        return {'coefs':B,'coef_err':SE_B,'res':e,'R2':R2}

# partial correlations of X in Y
def PARCOR(X , Y):
    
    """
    Uses OLSE_NORM to remove the unbiased best-estimate regression model using all but one feature of X, then computes the correlation of that feature against the residuals.
    """
    
    correlations = np.zeros((X.shape[-1],Y.shape[-1]))
    
    for i in np.arange(X.shape[-1]):
        
        X_roll = np.roll(X,-i,axis=-1)
        resids = OLSE_NORM(X_roll[:,1:],Y)['res']
        correlations[i,:] = calculate_stats(X_roll[:,[0]],resids)['cor']
        
    return correlations
