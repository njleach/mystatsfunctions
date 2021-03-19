# Some statistical functions I wrote
# Nick Leach 2021

import numpy as np
import scipy as sp

# Method of L-moments for fitting a number of set statistical distributions

## auxiliary functions ##
def b_r(r,a):
    
    """
    Helper function for lmom_to_3
    """
    
    n = a.shape[0]

    return np.sum([np.prod(i+1-np.arange(1,r+1))/np.prod(n-np.arange(1,r+1))*a[i] for i in np.arange(n)],axis=0)/n

def lmom_to_3(a):
    
    """
    returns first three l moments, a is the sorted array
    """
    
    b = [b_r(i,a) for i in np.arange(3)]
    
    return b[0] , -b[0] + 2 * b[1] , b[0] + 6 * ( b[2] - b[1] ) #, -b[0] + 12 * b[1] - 30 * b[2] + 20 * b[3] , b[0] + 10 * (-2*b[1]+9*b[2]-14*b[3]+7*b[4])

## distribution fit functions ##

def fit_gev_pwm(a):
    
    """
    Fits the (3) parameters of a GEV distribution over the first dimension of a
    
    a : np.ndarray
    """
    
    a_sort = np.sort(a,axis=0)
    
    b = [b_r(i,a_sort) for i in np.arange(3)]
    
    c = (2*b[1]-b[0])/(3*b[2]-b[0]) - np.log(2)/np.log(3)
    
    k = 7.8590*c + 2.9554*c**2
    
    a = ( 2*b[1]-b[0] )*k / ( sp.special.gamma(1+k) * (1-2**(-k)) )
    
    X = b[0] + a*( sp.special.gamma(1+k) - 1 ) / k
    
    return k,X,a

def get_gev(x,params,function='cdf'):
    
    """
    Returns either cdf or pdf of GEV distribution, based on parameter array 'params'
    
    x : np.ndarray
    
    params : np.ndarray, identical to the output from 'fit_gev_pwm'
    
    function : 'cdf' or 'pdf'
    """
    
    k,X,a = params
    
    if function=='cdf':
        cdf = np.exp(-1*(1-k*((x-X)/a))**(1/k))
        cdf[np.isnan(cdf)] = 1 # set values above the GEV limit to be 1 rather than undefined
        return cdf
    
    if function=='pdf':
        pdf = (1/a) * ( 1-k*( (x-X)/a ) )**( (1/k)-1 ) * np.exp( -1*(1-k*( (x-X)/a) )**(1/k) )
        pdf[np.isnan(pdf)] = 0 # set values above the GEV limit to be 1 rather than undefined
        return pdf

def fit_glo_pwm(a):
    
    """
    Fits the (3) parameters of a GL distribution over the first dimension of a
    
    a : np.ndarray
    """
    
    a_sort = np.sort(a,axis=0)
    
    l = lmom_to_3(a_sort)
    
    k = -l[2]/l[1]
    
    a = l[1] / (sp.special.gamma(1+k)*sp.special.gamma(1-k))
    
    X = l[0]+(l[1]-a)/k
    
    return k,X,a

def get_glo(x,params,function='cdf'):
    
    """
    Returns either cdf or pdf of GL distribution, based on parameter array 'params'
    
    x : np.ndarray
    
    params : np.ndarray, identical to the output from 'fit_glo_pwm'
    
    function : 'cdf' or 'pdf'
    """
    
    k,X,a = params
    
    if function=='cdf':
        cdf = 1/( 1 + (1-k*(x-X)/a)**(1/k) )
        return cdf
    
    if function=='pdf':
        pdf = (1-k*(x-X)/a)**(1/k-1) / ( a * ( 1 + (1-k*(x-X)/a)**(1/k) )**2 )
        return pdf

def fit_gpd_pwm(a):
    
    """
    Fits the (3) parameters of a GL distribution over the first dimension of a
    
    a : np.ndarray
    """
    
    a_sort = np.sort(a,axis=0)
    
    l = lmom_to_3(a_sort)
    
    k = ( 1-3*l[2]/l[1] ) / ( 1+l[2]/l[1] )
    
    a = l[1]*(1+k)*(2+k)
    
    X = l[0] - l[1]*(2+k)
    
    return k,X,a

def get_gpd(x,params,function='cdf'):
    
    """
    Returns either cdf or pdf of GL distribution, based on parameter array 'params'
    
    x : np.ndarray
    
    params : np.ndarray, identical to the output from 'fit_glo_pwm'
    
    function : 'cdf' or 'pdf'
    """
    
    k,X,a = params
    
    if function=='cdf':
        cdf = 1 - ( 1 - k*(x-X)/a )**(1/k)
        return cdf
    
    if function=='pdf':
        pdf = ( 1 - k*(x-X)/a )**(1/k-1)
        return pdf

def fit_norm_pwm(a):
    
    """
    Fits the (2) parameters of a normal distribution over the first dimension of a
    
    a : np.ndarray
    """
    
    a_sort = np.sort(a,axis=0)
    
    l = lmom_to_3(a_sort)
    
    s = np.pi**(1/2) * l[1]
    
    m = l[0]
    
    return m,s

def get_norm(x,params,function='cdf'):
    
    """
    Returns either cdf or pdf of normal distribution, based on parameter array 'params'
    
    x : np.ndarray
    
    params : np.ndarray, identical to the output from 'fit_norm_pwm'
    
    function : 'cdf' or 'pdf'
    """
    
    m,s = params
    
    y = (x-m)/s
    
    if function=='cdf':
        cdf = (1/2) * (1 + sp.special.erf(y/np.sqrt(2)))
        return cdf
    
    if function=='pdf':
        pdf = ( 1/( s * np.sqrt(2*np.pi) ) ) * np.exp( -y**2 / 2 )
        return pdf
