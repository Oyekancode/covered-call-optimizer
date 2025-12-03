"""
Covered Call Optimizer
Finds optimal strike price for covered call strategy based on risk-return objectives
"""

import numpy as np
import pandas as pd
from black_scholes import BlackScholes


class CoveredCallOptimizer:
    """
    Optimizer for covered call strategy that maximizes expected return
    subject to downside protection constraints.
    """
    
    def __init__(self, S, T, r, sigma, min_downside_protection=0.05):
        """
        Initialize optimizer parameters.
        
        Parameters:
        -----------
        S : float
            Current stock price
        T : float
            Time to expiration (in years)
        r : float
            Risk-free interest rate
        sigma : float
            Implied volatility
        min_downside_protection : float
            Minimum required downside protection as percentage (e.g., 0.05 = 5%)
        """
        self.S = S
        self.T = T
        self.r = r
        self.sigma = sigma
        self.min_downside_protection = min_downside_protection
        
    def calculate_covered_call_return(self, K):
        """
        Calculate expected return for a covered call at strike K.
        
        Parameters:
        -----------
        K : float
            Strike price
            
        Returns:
        --------
        dict : Dictionary containing return metrics
        """
        bs = BlackScholes(self.S, K, self.T, self.r, self.sigma)
        
        # Premium received from selling call
        call_premium = bs.call_price()
        
        # Maximum profit (if stock >= strike at expiration)
        max_profit = (K - self.S) + call_premium
        max_return = max_profit / self.S
        
        # Downside protection from premium
        downside_protection = call_premium / self.S
        
        # Break-even price
        breakeven = self.S - call_premium
        
        # Probability of being called away (stock > strike at expiration)
        prob_called = 1 - bs._d2()  # Using norm.cdf(-d2) = 1 - norm.cdf(d2)
        
        return {
            'strike': K,
            'call_premium': call_premium,
            'max_return': max_return,
            'annualized_return': max_return / self.T,
            'downside_protection': downside_protection,
            'breakeven': breakeven,
            'prob_called': prob_called,
            'moneyness': K / self.S  # >1 OTM, <1 ITM, =1 ATM
        }
    
    def optimize_strike(self, strike_range=None, num_strikes=50):
        """
        Find optimal strike price that maximizes return while meeting
        downside protection constraints.
        
        Parameters:
        -----------
        strike_range : tuple
            (min_strike, max_strike). If None, uses reasonable defaults.
        num_strikes : int
            Number of strike prices to evaluate
            
        Returns:
        --------
        pd.DataFrame : Results for all strikes evaluated
        dict : Optimal strike details
        """
        
        # Default strike range: 90% to 110% of current price
        if strike_range is None:
            strike_range = (self.S * 0.90, self.S * 1.10)
        
        strikes = np.linspace(strike_range[0], strike_range[1], num_strikes)
        
        results = []
        for K in strikes:
            result = self.calculate_covered_call_return(K)
            results.append(result)
        
        df = pd.DataFrame(results)
        
        # Filter by downside protection constraint
        valid_strikes = df[df['downside_protection'] >= self.min_downside_protection]
        
        if len(valid_strikes) == 0:
            print(f"Warning: No strikes meet minimum downside protection of {self.min_downside_protection*100}%")
            print("Consider lowering the constraint or increasing time to expiration.")
            return df, None
        
        # Find strike with maximum return among valid strikes
        optimal_idx = valid_strikes['max_return'].idxmax()
        optimal_strike = valid_strikes.loc[optimal_idx].to_dict()
        
        return df, optimal_strike
    
    def analyze_strategy(self, optimal_strike_info):
        """
        Print detailed analysis of the optimal covered call strategy.
        
        Parameters:
        -----------
        optimal_strike_info : dict
            Dictionary containing optimal strike information
        """
        if optimal_strike_info is None:
            print("No optimal strike found.")
            return
        
        print("\n" + "="*70)
        print("OPTIMAL COVERED CALL STRATEGY ANALYSIS")
        print("="*70)
        
        print(f"\nCurrent Stock Price: ${self.S:.2f}")
        print(f"Time to Expiration: {self.T*365:.0f} days ({self.T:.2f} years)")
        print(f"Implied Volatility: {self.sigma*100:.1f}%")
        print(f"Risk-Free Rate: {self.r*100:.2f}%")
        
        print("\n" + "-"*70)
        print("OPTIMAL STRIKE SELECTION")
        print("-"*70)
        
        K = optimal_strike_info['strike']
        print(f"Optimal Strike Price: ${K:.2f}")
        print(f"Moneyness: {optimal_strike_info['moneyness']:.3f} ", end="")
        if optimal_strike_info['moneyness'] > 1:
            print("(Out-of-the-Money)")
        elif optimal_strike_info['moneyness'] < 1:
            print("(In-the-Money)")
        else:
            print("(At-the-Money)")
        
        print("\n" + "-"*70)
        print("RETURN METRICS")
        print("-"*70)
        
        print(f"Call Premium Received: ${optimal_strike_info['call_premium']:.2f}")
        print(f"Maximum Return: {optimal_strike_info['max_return']*100:.2f}%")
        print(f"Annualized Return: {optimal_strike_info['annualized_return']*100:.2f}%")
        
        print("\n" + "-"*70)
        print("RISK METRICS")
        print("-"*70)
        
        print(f"Downside Protection: {optimal_strike_info['downside_protection']*100:.2f}%")
        print(f"Breakeven Price: ${optimal_strike_info['breakeven']:.2f}")
        print(f"Downside to Breakeven: {(optimal_strike_info['breakeven'] - self.S)/self.S*100:.2f}%")
        print(f"Probability of Assignment: {optimal_strike_info['prob_called']*100:.1f}%")
        
        print("\n" + "-"*70)
        print("GREEKS AT OPTIMAL STRIKE")
        print("-"*70)
        
        bs = BlackScholes(self.S, K, self.T, self.r, self.sigma)
        greeks = bs.all_greeks('call')
        print(f"Delta: {greeks['delta']:.4f} (exposure to stock movement)")
        print(f"Gamma: {greeks['gamma']:.4f} (delta sensitivity)")
        print(f"Theta: {greeks['theta']:.4f} (daily time decay)")
        print(f"Vega: {greeks['vega']:.4f} (volatility sensitivity)")
        print(f"Rho: {greeks['rho']:.4f} (interest rate sensitivity)")
        
        print("\n" + "="*70 + "\n")


def example_optimization():
    """Example usage of the covered call optimizer"""
    
    # Example: Own 100 shares of stock trading at $150
    S = 150                 # Current stock price
    T = 30/365             # 30 days to expiration
    r = 0.05               # 5% risk-free rate
    sigma = 0.25           # 25% implied volatility
    min_protection = 0.02  # Require at least 2% downside protection
    
    # Initialize optimizer
    optimizer = CoveredCallOptimizer(S, T, r, sigma, min_protection)
    
    # Find optimal strike
    results_df, optimal_strike = optimizer.optimize_strike()
    
    # Print analysis
    optimizer.analyze_strategy(optimal_strike)
    
    # Show comparison of different strikes
    print("COMPARISON OF ALTERNATIVE STRIKES")
    print("="*70)
    display_cols = ['strike', 'moneyness', 'call_premium', 'max_return', 
                   'annualized_return', 'downside_protection', 'prob_called']
    
    print(results_df[display_cols].round(4).to_string(index=False))
    
    return results_df, optimal_strike


if __name__ == "__main__":
    results_df, optimal = example_optimization()
