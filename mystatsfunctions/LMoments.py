# Some statistical functions I wrote
# Nick Leach 2021

"""
Methods for fitting statistical distributions efficiently using L-moments.

For the mathematics & definitions of distributions used in this module, see:

Hosking, J. R. M. (1990). L-Moments: Analysis and Estimation of Distributions Using Linear Combinations of Order Statistics. Journal of the Royal Statistical Society: Series B (Methodological), 52(1), 105–124. https://doi.org/10.1111/j.2517-6161.1990.tb01775.x

Hosking, J. R. M., & Wallis, J. R. (1997). Regional Frequency Analysis. In Regional Frequency Analysis. Cambridge University Press. https://doi.org/10.1017/cbo9780511529443

Hosking, J. R. M. (2005). Research Report: Fortran Routines for use with the Method of L-Moments. http://ftp.uni-bayreuth.de/math/statlib/general/lmoments.pdf

Far faster & more convenient than scipy.fit if fitting distributions many times or over multiple dimensions.

LIMITATIONS: since the first dimension is used as the sample dimension, you cannot fit differently-sized samples simultaneously. However, for bootstrapping and geospatial applications this isn't a common issue.
"""

import numpy as np
import scipy as sp
import math

# Method of L-moments for fitting a number of set statistical distributions

## auxiliary functions ##
def b_r(r,a):
    
    """
    Calculates the rth U-statistic for estimating L-moments.
    
    Args:
        r: L-moment order.
        a: *sorted* sample.
    
    Returns:
        rth normalized U-statistic
    """

    # get the sample size
    n = np.shape(a)[0]
    
    # calculate the denominator
    _denom = np.prod(n-np.arange(1,r+1))*n
    
    # generate array for summation & compute sum
    _to_sum = np.zeros(n)
    _to_sum[r:] = np.multiply.reduce([np.arange(i,n-r+i) for i in np.arange(1,r+1)])
        
    # sum (put sample index into final dimension for correct broadcasting)
    summed = np.array(np.sum(_to_sum*a.T,axis=-1)/_denom)
        
    return np.transpose(summed)


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


## distribution classes ##

# generic distribution class
class _dist:
    ## attributes
            
    ## placeholder classes that are specific to each distribution type. ##
    def _check_params(self):
        """Placeholder method."""
        pass
        
    def pdf(self, x):
        """Placeholder method."""
        pass
    
    def cdf(self, x):
        """Placeholder method."""
        pass
    
    def qf(self, F):
        """Placeholder method."""
        pass
    
    def fit(self, x):
        """Placeholder method."""
        pass
    
    ## generic methods applicable across distribution types. ##
    def rvs(self, n):
        
        """
        Generates n random variables based on the model parameters.

        n : int
        """
        
        self._check_params()
        
        u = np.random.random((n,*np.ones(np.ndim(self.X),dtype=int)))
        
        return self.qf(u)
        
    
    def nll(self):
        
        """
        Computes the model negative log likelihood based on the fit data and parameters.

        x : np.ndarray
        """
        
        if self.data is None:
            raise Exception('Model not fit to data. Use self.fit(data) to fit parameters to data.')
            
        return -1 * np.sum( np.log( self.pdf(self.data) ),axis=0 )
    
    def AIC(self):
        
        """
        Returns Akaike information criterion of fit parameters given model data. Smaller = better.
        """
        
        nll = self.nll()
        
        return 2 * ( self._no_of_params + nll )
    
    def AICc(self):
        
        """
        Returns corrected Akaike information criterion of fit parameters given model data. Smaller = better.
        """
        
        AIC = self.AIC()
        
        penalty = (2*self._no_of_params**2 + 2*self._no_of_params) / (self.data.shape[0]-self._no_of_params-1)
        
        return AIC + penalty
    
    def BIC(self):
        
        """
        Returns Bayesian information criterion of fit parameters given model data. Smaller = better.
        """
        
        nll = self.nll()
        
        return self._no_of_params * self.data.shape[0] + 2 * nll
    
    def ADts(self):
        
        """
        Returns Anderson-Darling test statistic. Smaller = better.
        """
        
        F_data = self.cdf(np.sort(self.data,axis=0))
        
        n = self.data.shape[0]
        
        # string used by the einsum operation
        dimstr = 'jklmnopqrstuvwxyz' # maxes out at 17 dimensions...
        
        S = np.einsum('i,i'+dimstr[:self.data.ndim-1]+'->'+dimstr[:self.data.ndim-1], (2*np.arange(1,n+1)-1)/n, np.log(F_data)+np.log(1-F_data[::-1]))
        
        return -n-S
    
    def CvMts(self):
        
        """
        Returns Cramer-von Mises test statistic. Smaller = better.
        """
        
        F_data = self.cdf(np.sort(self.data,axis=0))
        
        n = self.data.shape[0]
        
        T = 1/(12*n) + np.sum( ((2*np.arange(1,n+1).reshape(-1,*np.ones(self.data.ndim-1,dtype=int))-1)/(2*n) - F_data)**2, axis=0 )
        
        return T
    
    def KSts(self):
        
        """
        Returns Kolmogorov–Smirnov test statistic. Smaller = better.
        """
        
        F_data = self.cdf(np.sort(self.data,axis=0))
        
        n = self.data.shape[0]
        
        F_emp = np.arange(1,n+1).reshape(-1,*np.ones(self.data.ndim-1,dtype=int)) / n
        
        D = np.max(np.abs(F_emp - F_data),axis=0)
        
        return D
    
# specific Generalised Extreme Value class
class gev(_dist):
    ## attributes
    
    def __init__(self, k=None, X=None, a=None):
        self.k = k
        self.X = X
        self.a = a
        
        self.data = None
        
        self._no_of_params=3
        
    def _check_params(self):
        
        """
        Checks the model parameters are fully specified.
        
        Raises an exception if not all parameters are set.
        """
        
        if any([x is None for x in [self.k,self.X,self.a]]):
            raise TypeError('parameters not (fully) set.')

    def pdf(self, x):
        
        """
        Returns the pdf of a GEV based on the set parameters. 
        Raises exception if parameters not set or fit to data.
        """
        
        self._check_params()
        
        pdf = (1/self.a) * ( 1-self.k*( (x-self.X)/self.a ) )**( (1/self.k)-1 ) * np.exp( -1*(1-self.k*( (x-self.X)/self.a) )**(1/self.k) )
        # set values outside the GEV limits to be 0 rather than undefined
        return np.where(np.isnan(pdf),0,pdf)

    def cdf(self, x):
        
        """
        Returns the cdf of a GEV based on the set parameters. 
        Raises exception if parameters not set or fit to data.
        """
        
        self._check_params()
        
        cdf = np.exp(-1*(1-self.k*((x-self.X)/self.a))**(1/self.k))
        # set values outside the GEV limits to be 1 (0 for negative shape parameter) rather than undefined
        return np.where(np.isnan(cdf), np.where(self.k<0, 0, 1), cdf)
    
    def qf(self, F):
        
        """
        Returns the quantile function of a GEV based on the set parameters. 
        Raises exception if parameters not set or fit to data.
        """
        
        self._check_params()
            
        if np.any(np.abs(F)>1):
            raise ValueError('Input probabilities must be 0<F<=1.')
            
        qf = self.X+self.a*(1-(-np.log(F))**self.k)/self.k
        
        return qf
        
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
        
        self.data = x[:]

# specific Gumbel class
class gum(_dist):
    ## attributes
    
    def __init__(self, X=None, a=None):
        self.X = X
        self.a = a
        
        self.data = None
        
        self._no_of_params=2
        
    def _check_params(self):
        
        """
        Checks the model parameters are fully specified.
        
        Raises an exception if not all parameters are set.
        """
        
        if any([x is None for x in [self.X,self.a]]):
            raise TypeError('parameters not (fully) set.')

    def pdf(self, x):
        
        """
        Returns the pdf of a Gumbel based on the set parameters. 
        Raises exception if parameters not set or fit to data.
        """
        
        self._check_params()
        
        pdf = (1/self.a) * np.exp(-((x-self.X)/self.a+np.exp(-(x-self.X)/self.a)))
        
        return pdf

    def cdf(self, x):
        
        """
        Returns the cdf of a Gumbel based on the set parameters. 
        Raises exception if parameters not set or fit to data.
        """
        
        self._check_params()
        
        cdf = np.exp(-np.exp(-(x-self.X)/self.a))
        
        return cdf
    
    def qf(self, F):
        
        """
        Returns the quantile function of a Gumbel based on the set parameters. 
        Raises exception if parameters not set or fit to data.
        """
        
        self._check_params()
            
        if np.any(np.abs(F)>1):
            raise ValueError('Input probabilities must be 0<F<=1.')
            
        qf = self.X-self.a*np.log(-np.log(F))
        
        return qf
        
    def fit(self, x):
        
        """
        Fits the (2) parameters of a Gumbel distribution over the first dimension of x

        x : np.ndarray
        """

        x_sort = np.sort(x,axis=0)

        l = get_lmoments(x_sort)

        a = l[1]/np.log(2)

        X = l[0] - np.euler_gamma * a
        
        self.X=X
        self.a=a
        
        self.data = x[:]
    
# specific Generalised Logistic class    
class glo(_dist):
    ## attributes
    
    def __init__(self, k=None, X=None, a=None):
        self.k = k
        self.X = X
        self.a = a
        
        self.data = None
        
        self._no_of_params = 3
        
    def _check_params(self):
        
        """
        Checks the model parameters are fully specified.
        
        Raises an exception if not all parameters are set.
        """
        
        if any([x is None for x in [self.k,self.X,self.a]]):
            raise TypeError('parameters not (fully) set.')

    def pdf(self, x):
        
        """
        Returns the pdf of a GL based on the set parameters. 
        Raises exception if parameters not set or fit to data.
        """
        
        self._check_params()
        
        pdf = (1-self.k*(x-self.X)/self.a)**(1/self.k-1) / ( self.a * ( 1 + (1-self.k*(x-self.X)/self.a)**(1/self.k) )**2 )
        
        ## set values outside the glo limits equal to zero
        pdf = np.where(self.k*(x-self.X)/self.a<1 , pdf , 0)
        
        return pdf

    def cdf(self, x):
        
        """
        Returns the cdf of a GL based on the set parameters. 
        Raises exception if parameters not set or fit to data.
        """
        
        self._check_params()
        
        cdf = 1/( 1 + (1-self.k*(x-self.X)/self.a)**(1/self.k) )
        
        ## set values outside the glo limits equal to zero
        cdf = np.where( self.k*(x-self.X)/self.a<1, cdf, np.where(self.k<0, 0, 1) )
        
        return cdf
    
    def qf(self, F):
        
        """
        Returns the quantile function of a GL based on the set parameters. 
        Raises exception if parameters not set or fit to data.
        """
        
        self._check_params()
            
        if np.any(np.abs(F)>1):
            raise ValueError('Input probabilities must be 0<F<=1.')
            
        qf = self.X+self.a*(1-((1-F)/F)**self.k)/self.k
        
        return qf
    
    def fit(self, x):
        
        """
        Fits the (3) parameters of a GLo distribution over the first dimension of x

        x : np.ndarray
        """

        x_sort = np.sort(x,axis=0)

        l = get_lmoments(x_sort)

        k = -l[2]/l[1]
    
        a = l[1] * np.sin(k*np.pi) / (k*np.pi)

        X = l[0] - a * (1 /k - np.pi/np.sin(k*np.pi) )

        self.k=k
        self.X=X
        self.a=a
        
        self.data = x[:]

# specific Generalised Pareto Distribution class
class gpd(_dist):
    ## attributes
    
    def __init__(self, k=None, X=None, a=None):
        self.k = k
        self.X = X
        self.a = a
        
        self.data = None
        
        self._no_of_params = 3
        
    def _check_params(self):
        
        """
        Checks the model parameters are fully specified.
        
        Raises an exception if not all parameters are set.
        """
        
        if any([x is None for x in [self.k,self.X,self.a]]):
            raise TypeError('parameters not (fully) set.')

    def pdf(self, x):
        
        """
        Returns the pdf of a GPD based on the set parameters. 
        Raises exception if parameters not set or fit to data.
        """
        
        self._check_params()
        
        pdf = (1/self.a) * ( 1 - self.k*(x-self.X)/self.a )**(1/self.k-1)
        
        ## set values outside the gpd limits equal to zero
        pdf = np.where(self.k*(x-self.X)/self.a<1 , pdf , 0)
        
        ## set values smaller than the location parameter equal to zero
        pdf = np.where(x>=self.X , pdf , 0)
        
        return pdf

    def cdf(self, x):
        
        """
        Returns the cdf of a GPD based on the set parameters. 
        Raises exception if parameters not set or fit to data.
        """
        
        self._check_params()
        
        cdf = 1 - ( 1 - self.k*(x-self.X)/self.a )**(1/self.k)
        
        ## set values outside the gpd limits equal to one
        cdf = np.where(self.k*(x-self.X)/self.a<1 , cdf , 1)
        
        ## set values smaller than the location parameter equal to zero
        cdf = np.where(x>=self.X , cdf , 0)
        
        return cdf
    
    def qf(self, F):
        
        """
        Returns the quantile function of a GPD based on the set parameters. 
        Raises exception if parameters not set or fit to data.
        """
        
        self._check_params()
            
        if np.any(np.abs(F)>1):
            raise ValueError('Input probabilities must be 0<F<=1.')
            
        qf = self.X+self.a*(1-(1-F)**self.k)/self.k
        
        return qf
        
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
        
        self.data = x[:]
        
    def fit_X0(self, x):
        
        """
        Fits the (2) parameters of a GP distribution over the first dimension of x
        
        Assumes X = 0

        x : np.ndarray
        """

        x_sort = np.sort(x,axis=0)

        l = get_lmoments(x_sort)

        k = l[0]/l[1]-2
    
        a = l[0]*(1+k)

        X = 0

        self.k=k
        self.X=X
        self.a=a
        
        self.data = x[:]
        
# specific Weibull distribution class
class weib(_dist):
    
    """
    Follows distribution definition in Goda, Y., Kudaka, M., & Kawai, H. (2011). INCORPORATION OF WEIBULL DISTRIBUTION IN L-MOMENTS METHOD FOR. Coastal Engineering Proceedings, 1(32), 62. https://doi.org/10.9753/icce.v32.waves.62
    """
    
    ## attributes
    
    def __init__(self, k=None, X=None, a=None):
        self.k = k
        self.X = X
        self.a = a
        
        self.data = None
        
        self._no_of_params = 3
        
    def _check_params(self):
        
        """
        Checks the model parameters are fully specified.
        
        Raises an exception if not all parameters are set.
        """
        
        if any([x is None for x in [self.k,self.X,self.a]]):
            raise TypeError('parameters not (fully) set.')

    def pdf(self, x):
        
        """
        Returns the pdf of a Weibull based on the set parameters. 
        Raises exception if parameters not set or fit to data.
        """
        
        self._check_params()
        
        pdf = self.k/self.a*( (x-self.X)/self.a )**(self.k-1) * np.exp( -((x-self.X)/self.a)**self.k )
        
        ## set values smaller than the location parameter equal to zero
        pdf = np.where(x>=self.X , pdf , 0)
        
        return pdf

    def cdf(self, x):
        
        """
        Returns the cdf of a Weibull based on the set parameters. 
        Raises exception if parameters not set or fit to data.
        """
        
        self._check_params()
        
        cdf = 1 - np.exp( -((x-self.X)/self.a)**self.k )
        
        ## set values smaller than the location parameter equal to zero
        cdf = np.where(x>=self.X , cdf , 0)
        
        return cdf
    
    def qf(self, F):
        
        """
        Returns the quantile function of a Weibull based on the set parameters. 
        Raises exception if parameters not set or fit to data.
        """
        
        self._check_params()
            
        if np.any(np.abs(F)>1):
            raise ValueError('Input probabilities must be 0<F<=1.')
            
        qf = self.X + self.a*( -np.log(1-F) )**(1/self.k)
        
        return qf
        
    def fit(self, x):
        
        """
        Fits the (3) parameters of a Weibull distribution over the first dimension of x

        x : np.ndarray
        """

        x_sort = np.sort(x,axis=0)

        l = get_lmoments(x_sort)
        
        t3 = l[2]/l[1]

        # I use a 10-order polynomial, fit over the t3 / k relation between 0<k<5
        k = 7.64145855e+02*t3**10 + -4.78351190e+03*t3**9 + 1.23664931e+04*t3**8 + -1.74853312e+04*t3**7 + 1.50061498e+04*t3**6 + -8.20115240e+03*t3**5 + 2.93892208e+03*t3**4 + -7.20630145e+02*t3**3 + 1.33908727e+02*t3**2 + -2.24593687e+01*t3 + 3.52289017e+00
    
        a = l[1] / ( (1-2**(-1/k))*sp.special.gamma(1+1/k) )

        X = l[0] - a*sp.special.gamma(1+1/k)

        self.k=k
        self.X=X
        self.a=a
        
        self.data = x[:]
        
    def fit_X0(self, x):
        
        """
        Fits the (3) parameters of a Weibull distribution over the first dimension of x

        x : np.ndarray
        """

        x_sort = np.sort(x,axis=0)

        l = get_lmoments(x_sort)

        k = -np.log(2) / np.log(1 - l[1]/l[0])
    
        a = l[0] / sp.special.gamma(1+1/k)

        self.k=k
        self.X=0
        self.a=a
        
        self.data = x[:]
        
# specific Gamma distribution class
class gam(_dist):
    ## attributes
    
    def __init__(self, a=None, B=None):
        self.a = a
        self.B = B
        
        self.data = None
        
        self._no_of_params = 2
        
    def _check_params(self):
        
        """
        Checks the model parameters are fully specified.
        
        Raises an exception if not all parameters are set.
        """
        
        if any([x is None for x in [self.a,self.B]]):
            raise TypeError('parameters not (fully) set.')

    def pdf(self, x):
        
        """
        Returns the pdf of a gamma based on the set parameters. 
        Raises exception if parameters not set or fit to data.
        """
        
        self._check_params()
            
        pdf = x**(self.a-1) * np.exp(-x/self.B) / (self.B**self.a * sp.special.gamma(self.a))
        
        ## set x<0 equal to zero
        pdf = np.where(x>0 , pdf , 0)
        
        return pdf

    def cdf(self, x):
        
        """
        Returns the cdf of a gamma based on the set parameters. 
        Raises exception if parameters not set or fit to data.
        """
        
        self._check_params()
        
        cdf = sp.special.gammainc(self.a, x/self.B)
        
        ## set x<0 equal to zero
        cdf = np.where(x>0 , cdf , 0)
        
        return cdf
    
    def qf(self, F):
        
        """
        Returns the quantile function of a gamma based on the set parameters. 
        Raises exception if parameters not set or fit to data.
        """
        
        self._check_params()
            
        if np.any(np.abs(F)>1):
            raise ValueError('Input probabilities must be 0<F<=1.')
            
        qf = sp.special.gammaincinv(self.a, F) * self.B
        
        return qf
        
    def fit(self, x):
        
        """
        Fits the (2) parameters of a gamma distribution over the first dimension of x.
        
        Rational approximation from Hosking, 1990.

        x : np.ndarray
        """

        x_sort = np.sort(x,axis=0)

        l = get_lmoments(x_sort,r=2)

        t = l[1]/l[0]
        
        z1 = np.pi*t**2
        z2 = 1-t
        
        a = np.where( t<1/2, (1-0.3080*z1)/(z1-0.05812*z1**2+0.01765*z1**3), (0.7213*z2-0.5947*z2**2)/(1-2.1817*z2+1.2113*z2**2))[()]
        
        B = l[0]/a

        self.a=a
        self.B=B
        
        self.data = x[:]

# specific Normal distribution class
class norm(_dist):
    ## attributes
    
    def __init__(self, X=None, a=None):
        self.X = X
        self.a = a
        
        self.data = None
        
        self._no_of_params = 2
        
    def _check_params(self):
        
        """
        Checks the model parameters are fully specified.
        
        Raises an exception if not all parameters are set.
        """
        
        if any([x is None for x in [self.X,self.a]]):
            raise TypeError('parameters not (fully) set.')

    def pdf(self, x):
        
        """
        Returns the pdf of a normal based on the set parameters. 
        Raises exception if parameters not set or fit to data.
        """
        
        self._check_params()
            
        y = (x-self.X)/self.a
        pdf = ( 1/( self.a * np.sqrt(2*np.pi) ) ) * np.exp( -y**2 / 2 )
        
        return pdf

    def cdf(self, x):
        
        """
        Returns the cdf of a normal based on the set parameters. 
        Raises exception if parameters not set or fit to data.
        """
        
        self._check_params()
        
        y = (x-self.X)/self.a
        cdf = (1/2) * (1 + sp.special.erf(y/np.sqrt(2)))
        
        return cdf
    
    def qf(self, F):
        
        """
        Returns the quantile function of a norm based on the set parameters. 
        Raises exception if parameters not set or fit to data.
        """
        
        self._check_params()
            
        if np.any(np.abs(F)>1):
            raise ValueError('Input probabilities must be 0<F<=1.')
            
        qf = self.X+self.a*np.sqrt(2)*sp.special.erfinv(2*F-1)
        
        return qf
        
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
        
        self.data = x[:]
