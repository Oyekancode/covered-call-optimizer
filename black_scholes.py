"""
Black-Scholes Option Pricing Model with Greeks
"""

import numpy as np
from scipy.stats import norm

class BlackScholes:
    """
    Black-Scholes model for European option pricing with full Greeks suite.
    """
    
    def __init__(self, S, K, T, r, sigma):
        """
        Initialize Black-Scholes parameters.
        
        Parameters:
        -----------
        S : float
            Current stock price
        K : float
            Strike price
        T : float
            Time to expiration (in years)
        r : float
            Risk-free interest rate (annualized)
        sigma : float
            Volatility (annualized)
        """
        self.S = S
        self.K = K
        self.T = T
        self.r = r
        self.sigma = sigma
        
    def _d1(self):
        """Calculate d1 parameter"""
        return (np.log(self.S / self.K) + (self.r + 0.5 * self.sigma**2) * self.T) / (self.sigma * np.sqrt(self.T))
    
    def _d2(self):
        """Calculate d2 parameter"""
        return self._d1() - self.sigma * np.sqrt(self.T)
    
    def call_price(self):
        """
        Calculate European call option price.
        
        Returns:
        --------
        float : Call option price
        """
        d1 = self._d1()
        d2 = self._d2()
        
        call = self.S * norm.cdf(d1) - self.K * np.exp(-self.r * self.T) * norm.cdf(d2)
        return call
    
    def put_price(self):
        """
        Calculate European put option price.
        
        Returns:
        --------
        float : Put option price
        """
        d1 = self._d1()
        d2 = self._d2()
        
        put = self.K * np.exp(-self.r * self.T) * norm.cdf(-d2) - self.S * norm.cdf(-d1)
        return put
    
    def delta(self, option_type='call'):
        """
        Calculate Delta: rate of change of option price with respect to stock price.
        
        Parameters:
        -----------
        option_type : str
            'call' or 'put'
            
        Returns:
        --------
        float : Delta value
        """
        d1 = self._d1()
        
        if option_type == 'call':
            return norm.cdf(d1)
        else:  # put
            return norm.cdf(d1) - 1
    
    def gamma(self):
        """
        Calculate Gamma: rate of change of delta with respect to stock price.
        Gamma is the same for both calls and puts.
        
        Returns:
        --------
        float : Gamma value
        """
        d1 = self._d1()
        return norm.pdf(d1) / (self.S * self.sigma * np.sqrt(self.T))
    
    def theta(self, option_type='call'):
        """
        Calculate Theta: rate of change of option price with respect to time.
        
        Parameters:
        -----------
        option_type : str
            'call' or 'put'
            
        Returns:
        --------
        float : Theta value (typically negative, representing time decay)
        """
        d1 = self._d1()
        d2 = self._d2()
        
        term1 = -(self.S * norm.pdf(d1) * self.sigma) / (2 * np.sqrt(self.T))
        
        if option_type == 'call':
            term2 = -self.r * self.K * np.exp(-self.r * self.T) * norm.cdf(d2)
            return (term1 + term2) / 365  # Convert to daily theta
        else:  # put
            term2 = self.r * self.K * np.exp(-self.r * self.T) * norm.cdf(-d2)
            return (term1 + term2) / 365  # Convert to daily theta
    
    def vega(self):
        """
        Calculate Vega: rate of change of option price with respect to volatility.
        Vega is the same for both calls and puts.
        
        Returns:
        --------
        float : Vega value (per 1% change in volatility)
        """
        d1 = self._d1()
        return self.S * norm.pdf(d1) * np.sqrt(self.T) / 100  # Divide by 100 for 1% change
    
    def rho(self, option_type='call'):
        """
        Calculate Rho: rate of change of option price with respect to interest rate.
        
        Parameters:
        -----------
        option_type : str
            'call' or 'put'
            
        Returns:
        --------
        float : Rho value (per 1% change in interest rate)
        """
        d2 = self._d2()
        
        if option_type == 'call':
            return self.K * self.T * np.exp(-self.r * self.T) * norm.cdf(d2) / 100
        else:  # put
            return -self.K * self.T * np.exp(-self.r * self.T) * norm.cdf(-d2) / 100
    
    def all_greeks(self, option_type='call'):
        """
        Calculate all Greeks at once.
        
        Parameters:
        -----------
        option_type : str
            'call' or 'put'
            
        Returns:
        --------
        dict : Dictionary containing all Greeks
        """
        return {
            'delta': self.delta(option_type),
            'gamma': self.gamma(),
            'theta': self.theta(option_type),
            'vega': self.vega(),
            'rho': self.rho(option_type)
        }


def example_usage():
    """Example usage of the Black-Scholes model"""
    
    # Example parameters
    S = 100      # Current stock price
    K = 105      # Strike price
    T = 0.25     # Time to expiration (3 months)
    r = 0.05     # Risk-free rate (5%)
    sigma = 0.2  # Volatility (20%)
    
    bs = BlackScholes(S, K, T, r, sigma)
    
    print("Black-Scholes Option Pricing")
    print("=" * 50)
    print(f"Stock Price: ${S}")
    print(f"Strike Price: ${K}")
    print(f"Time to Expiration: {T} years")
    print(f"Risk-Free Rate: {r*100}%")
    print(f"Volatility: {sigma*100}%")
    print("\n" + "=" * 50)
    
    print(f"\nCall Option Price: ${bs.call_price():.2f}")
    print(f"Put Option Price: ${bs.put_price():.2f}")
    
    print("\nCall Option Greeks:")
    greeks = bs.all_greeks('call')
    for greek, value in greeks.items():
        print(f"  {greek.capitalize()}: {value:.4f}")


if __name__ == "__main__":
    example_usage()
