# Some statistical functions I wrote
# Nick Leach 2021

"""
Methods for fitting statistical distributions efficiently using L-moments.

For the mathematics behind this module, see:

Hosking, J. R. M. (1990). L-Moments: Analysis and Estimation of Distributions Using Linear Combinations of Order Statistics. Journal of the Royal Statistical Society. Series B (Methodological), 52(1), 105–124. http://www.jstor.org/stable/2345653

Far faster & more convenient than scipy.fit if fitting distributions many times or over multiple dimensions.
"""

import numpy as np
import scipy as sp

# Method of L-moments for fitting a number of set statistical distributions

# TODO: add gamma distribution class
# TODO add rvs method
# TODO: add likelihood method
# TODO create parent class for distributions
# TODO add AIC, BIC, AICc, Anderson-Darling, KS methods
# TODO add self.data for data fit
# TODO improve way in which specified params are checked

## auxiliary functions ##
def b_r(r,a):
    
    """
    Helper function for get_lmoments.
    """
    
    n = a.shape[0]

    return np.sum([np.prod(i+1-np.arange(1,r+1))/np.prod(n-np.arange(1,r+1))*a[i] for i in np.arange(n)],axis=0)/n

def get_lmoments(a,r=3):
    
    """
    returns first r l moments, a is the *sorted* array.
    
    r must be less than the size of a's first dimension
    """
    
    _r = np.arange(r)[None]
    
    _k = np.arange(r)[:,None]
    
    # generates the r x r coefficients array
    br_coefs = (-1)**(_r+_k)*sp.special.comb(_r,_k) * sp.special.comb(_r+_k,_k)
    
    # generates the first r b values
    b_to_r = np.array([b_r(i,a) for i in np.arange(r)])
    
    # string used by the einsum operation
    dimstr = 'jklmnopqrstuvwxyz' # maxes out at 17 dimensions...
    
    # einstein summation over 2nd dimension
    b = np.einsum('ij,i'+dimstr[1:b_to_r.ndim]+'->'+dimstr[:b_to_r.ndim],br_coefs,b_to_r)
    
    return b


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

        l = get_lmoments(x_sort)

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
        # set values outside the GEV limits to be 0 rather than undefined
        return np.where(np.isnan(pdf),0,pdf)

    def cdf(self, x):
        
        """
        Returns the cdf of a GEV based on the set parameters. 
        Raises exception if parameters not set or fit to data.
        """
        
        if any(value is None for value in self.__dict__.values()):
            raise TypeError('GEV parameters not fully set.')
        
        cdf = np.exp(-1*(1-self.k*((x-self.X)/self.a))**(1/self.k))
        # set values outside the GEV limits to be 1 (0 for negative shape parameter) rather than undefined
        return np.where(np.isnan(cdf), np.where(self.k<0, 0, 1), cdf)
    
    def qf(self, F):
        
        """
        Returns the quantile function of a GEV based on the set parameters. 
        Raises exception if parameters not set or fit to data.
        """
        
        if any(value is None for value in self.__dict__.values()):
            raise TypeError('GEV parameters not fully set.')
            
        if np.any(np.abs(F)>1):
            raise ValueError('Input probabilities must be 0<F<=1.')
            
        qf = self.X+self.a*(1-(-np.log(F))**self.k)/self.k
        
        return qf
    
    
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

        l = get_lmoments(x_sort)

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
        
        ## set values outside the glo limits equal to zero
        pdf = np.where(self.k*(x-self.X)/self.a<1 , pdf , 0)
        
        return pdf

    def cdf(self, x):
        
        """
        Returns the cdf of a GL based on the set parameters. 
        Raises exception if parameters not set or fit to data.
        """
        
        if any(value is None for value in self.__dict__.values()):
            raise TypeError('GLo parameters not fully set.')
        
        cdf = 1/( 1 + (1-self.k*(x-self.X)/self.a)**(1/self.k) )
        
        ## set values outside the glo limits equal to zero
        cdf = np.where( self.k*(x-self.X)/self.a<1, cdf, np.where(self.k<0, 0, 1) )
        
        return cdf
    
    def qf(self, F):
        
        """
        Returns the quantile function of a GL based on the set parameters. 
        Raises exception if parameters not set or fit to data.
        """
        
        if any(value is None for value in self.__dict__.values()):
            raise TypeError('GLo parameters not fully set.')
            
        if np.any(np.abs(F)>1):
            raise ValueError('Input probabilities must be 0<F<=1.')
            
        qf = self.X+self.a*(1-((1-F)/F)**self.k)/self.k
        
        return qf


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

        l = get_lmoments(x_sort)

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
        
        ## set values outside the gpd limits equal to zero
        pdf = np.where(self.k*(x-self.X)/self.a<1 , pdf , 0)
        
        ## set values smaller than the location parameter equal to zero
        pdf = np.where(x>self.X , pdf , 0)
        
        return pdf

    def cdf(self, x):
        
        """
        Returns the cdf of a GPD based on the set parameters. 
        Raises exception if parameters not set or fit to data.
        """
        
        if any(value is None for value in self.__dict__.values()):
            raise TypeError('GPD parameters not fully set.')
        
        cdf = 1 - ( 1 - self.k*(x-self.X)/self.a )**(1/self.k)
        
        ## set values outside the gpd limits equal to one
        cdf = np.where(self.k*(x-self.X)/self.a<1 , cdf , 1)
        
        ## set values smaller than the location parameter equal to zero
        cdf = np.where(x>self.X , cdf , 0)
        
        return cdf
    
    def qf(self, F):
        
        """
        Returns the quantile function of a GPD based on the set parameters. 
        Raises exception if parameters not set or fit to data.
        """
        
        if any(value is None for value in self.__dict__.values()):
            raise TypeError('GPD parameters not fully set.')
            
        if np.any(np.abs(F)>1):
            raise ValueError('Input probabilities must be 0<F<=1.')
            
        qf = self.X+self.a*(1-(1-F)**self.k)/self.k
        
        return qf

    
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

        l = get_lmoments(x_sort,r=2)

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
            
        y = (x-self.X)/self.a
        pdf = ( 1/( self.a * np.sqrt(2*np.pi) ) ) * np.exp( -y**2 / 2 )
        return pdf

    def cdf(self, x):
        
        """
        Returns the cdf of a normal based on the set parameters. 
        Raises exception if parameters not set or fit to data.
        """
        
        if any(value is None for value in self.__dict__.values()):
            raise TypeError('norm parameters not fully set.')
        
        y = (x-self.X)/self.a
        cdf = (1/2) * (1 + sp.special.erf(y/np.sqrt(2)))
        return cdf
    
    def qf(self, F):
        
        """
        Returns the quantile function of a norm based on the set parameters. 
        Raises exception if parameters not set or fit to data.
        """
        
        if any(value is None for value in self.__dict__.values()):
            raise TypeError('norm parameters not fully set.')
            
        if np.any(np.abs(F)>1):
            raise ValueError('Input probabilities must be 0<F<=1.')
            
        qf = self.X+self.a*np.sqrt(2)*sp.special.erfinv(2*F-1)
        
        return qf