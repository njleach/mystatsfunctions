# Some statistical functions I wrote
# Nick Leach 2021

import numpy as np
import scipy as sp

# Method of L-moments for fitting a number of set statistical distributions

# TODO: make sure the override functions are compatible with the shape parameter fits (ie. setting GEV CDF to 1 above max threshold)
# TODO: add gamma distribution class
# TODO: add quantile function method

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

class gev:
    ## attributes
    
    def __init__(self, k=None, X=None, a=None):
        self.k = k
        self.X = X
        self.a = a
        
    def fit(self, x):
        
        """
        Fits the (3) parameters of a GEV distribution over the first dimension of x

        x : np.ndarray
        """

        x_sort = np.sort(x,axis=0)

        l = lmom_to_3(x_sort)

        c = 2/(3+l[2]/l[1]) - np.log(2)/np.log(3)

        k = 7.8590*c + 2.9554*c**2

        a = l[1]*k / ( sp.special.gamma(1+k) * (1-2**(-k)) )

        X = l[0] + a*( sp.special.gamma(1+k) - 1 ) / k

        self.k=k
        self.X=X
        self.a=a

    def pdf(self, x):
        
        """
        Returns the pdf of a GEV based on the set parameters. 
        Raises exception if parameters not set or fit to data.
        """
        
        if any(value is None for value in self.__dict__.values()):
            raise TypeError('GEV parameters not fully set.')
        
        pdf = (1/self.a) * ( 1-self.k*( (x-self.X)/self.a ) )**( (1/self.k)-1 ) * np.exp( -1*(1-self.k*( (x-self.X)/self.a) )**(1/self.k) )
        # set values above the GEV limit to be 1 rather than undefined
        return np.where(np.isnan(pdf),0,pdf)

    def cdf(self, x):
        
        """
        Returns the cdf of a GEV based on the set parameters. 
        Raises exception if parameters not set or fit to data.
        """
        
        if any(value is None for value in self.__dict__.values()):
            raise TypeError('GEV parameters not fully set.')
        
        cdf = np.exp(-1*(1-self.k*((x-self.X)/self.a))**(1/self.k))
        # set values above the GEV limit to be 1 rather than undefined
        return np.where(np.isnan(cdf),1,cdf)
    
    
class glo:
    ## attributes
    
    def __init__(self, k=None, X=None, a=None):
        self.k = k
        self.X = X
        self.a = a
        
    def fit(self, x):
        
        """
        Fits the (3) parameters of a GEV distribution over the first dimension of x

        x : np.ndarray
        """

        x_sort = np.sort(x,axis=0)

        l = lmom_to_3(x_sort)

        k = -l[2]/l[1]
    
        a = l[1] / (sp.special.gamma(1+k)*sp.special.gamma(1-k))

        X = l[0]+(l[1]-a)/k

        self.k=k
        self.X=X
        self.a=a

    def pdf(self, x):
        
        """
        Returns the pdf of a GL based on the set parameters. 
        Raises exception if parameters not set or fit to data.
        """
        
        if any(value is None for value in self.__dict__.values()):
            raise TypeError('GLo parameters not fully set.')
        
        pdf = (1-self.k*(x-self.X)/self.a)**(1/self.k-1) / ( self.a * ( 1 + (1-self.k*(x-self.X)/self.a)**(1/self.k) )**2 )
        return pdf

    def cdf(self, x):
        
        """
        Returns the cdf of a GL based on the set parameters. 
        Raises exception if parameters not set or fit to data.
        """
        
        if any(value is None for value in self.__dict__.values()):
            raise TypeError('GLo parameters not fully set.')
        
        cdf = 1/( 1 + (1-self.k*(x-self.X)/self.a)**(1/self.k) )
        return cdf


class gpd:
    ## attributes
    
    def __init__(self, k=None, X=None, a=None):
        self.k = k
        self.X = X
        self.a = a
        
    def fit(self, x):
        
        """
        Fits the (3) parameters of a GP distribution over the first dimension of x

        x : np.ndarray
        """

        x_sort = np.sort(x,axis=0)

        l = lmom_to_3(x_sort)

        k = ( 1-3*l[2]/l[1] ) / ( 1+l[2]/l[1] )
    
        a = l[1]*(1+k)*(2+k)

        X = l[0] - l[1]*(2+k)

        self.k=k
        self.X=X
        self.a=a

    def pdf(self, x):
        
        """
        Returns the pdf of a GPD based on the set parameters. 
        Raises exception if parameters not set or fit to data.
        """
        
        if any(value is None for value in self.__dict__.values()):
            raise TypeError('GPD parameters not fully set.')
        
        pdf = (1/self.a) * ( 1 - self.k*(x-self.X)/self.a )**(1/self.k-1)
        return np.where(x>X , pdf , 0)

    def cdf(self, x):
        
        """
        Returns the cdf of a GPD based on the set parameters. 
        Raises exception if parameters not set or fit to data.
        """
        
        if any(value is None for value in self.__dict__.values()):
            raise TypeError('GPD parameters not fully set.')
        
        cdf = 1 - ( 1 - self.k*(x-self.X)/self.a )**(1/self.k)
        return np.where(x>X , cdf , 0)

    
class norm:
    ## attributes
    
    def __init__(self, X=None, a=None):
        self.X = X
        self.a = a
        
    def fit(self, x):
        
        """
        Fits the (2) parameters of a normal distribution over the first dimension of x

        x : np.ndarray
        """

        x_sort = np.sort(x,axis=0)

        l = lmom_to_3(x_sort)

        a = np.pi**(1/2) * l[1]
    
        X = l[0]

        self.X=X
        self.a=a

    def pdf(self, x):
        
        """
        Returns the pdf of a normal based on the set parameters. 
        Raises exception if parameters not set or fit to data.
        """
        
        if any(value is None for value in self.__dict__.values()):
            raise TypeError('norm parameters not fully set.')
            
        y = (x-X)/a
        pdf = ( 1/( self.a * np.sqrt(2*np.pi) ) ) * np.exp( -y**2 / 2 )
        return pdf

    def cdf(self, x):
        
        """
        Returns the cdf of a normal based on the set parameters. 
        Raises exception if parameters not set or fit to data.
        """
        
        if any(value is None for value in self.__dict__.values()):
            raise TypeError('norm parameters not fully set.')
        
        y = (x-X)/a
        cdf = (1/2) * (1 + sp.special.erf(y/np.sqrt(2)))
        return cdf