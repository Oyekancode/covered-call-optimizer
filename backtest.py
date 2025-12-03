"""
Backtest Module for Covered Call Strategy
Tests the optimizer on historical data
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from covered_call_optimizer import CoveredCallOptimizer


class CoveredCallBacktest:
    """
    Backtest covered call strategy using historical price data.
    """
    
    def __init__(self, prices, volatility=0.25, risk_free_rate=0.05, 
                 option_duration_days=30, min_downside_protection=0.02):
        """
        Initialize backtest parameters.
        
        Parameters:
        -----------
        prices : pd.Series or np.array
            Historical price data (daily)
        volatility : float
            Assumed implied volatility (in practice, would use historical or implied vol)
        risk_free_rate : float
            Risk-free interest rate
        option_duration_days : int
            Days to expiration for each covered call
        min_downside_protection : float
            Minimum downside protection required
        """
        self.prices = pd.Series(prices) if not isinstance(prices, pd.Series) else prices
        self.volatility = volatility
        self.risk_free_rate = risk_free_rate
        self.option_duration_days = option_duration_days
        self.min_downside_protection = min_downside_protection
        
    def run_backtest(self):
        """
        Run backtest by selling covered calls at optimal strikes.
        
        Returns:
        --------
        pd.DataFrame : Backtest results with trade details
        dict : Performance summary
        """
        trades = []
        position = None  # Current covered call position
        
        for i in range(len(self.prices) - self.option_duration_days):
            current_date = i
            current_price = self.prices.iloc[i]
            
            # If no position, sell a covered call
            if position is None:
                T = self.option_duration_days / 365
                
                optimizer = CoveredCallOptimizer(
                    current_price, T, self.risk_free_rate, 
                    self.volatility, self.min_downside_protection
                )
                
                _, optimal_strike = optimizer.optimize_strike()
                
                if optimal_strike is not None:
                    position = {
                        'entry_date': current_date,
                        'entry_price': current_price,
                        'strike': optimal_strike['strike'],
                        'premium': optimal_strike['call_premium'],
                        'expiration_date': current_date + self.option_duration_days
                    }
            
            # Check if option expired
            elif current_date >= position['expiration_date']:
                expiration_price = self.prices.iloc[current_date]
                
                # Calculate P&L
                stock_pl = expiration_price - position['entry_price']
                
                # If stock above strike, stock is called away
                if expiration_price >= position['strike']:
                    total_pl = (position['strike'] - position['entry_price']) + position['premium']
                    outcome = 'assigned'
                else:
                    total_pl = stock_pl + position['premium']
                    outcome = 'expired'
                
                trade_return = total_pl / position['entry_price']
                
                trades.append({
                    'entry_date': position['entry_date'],
                    'exit_date': current_date,
                    'entry_price': position['entry_price'],
                    'exit_price': expiration_price,
                    'strike': position['strike'],
                    'premium': position['premium'],
                    'outcome': outcome,
                    'stock_pl': stock_pl,
                    'total_pl': total_pl,
                    'return': trade_return,
                    'hold_days': self.option_duration_days
                })
                
                position = None  # Close position
        
        if len(trades) == 0:
            print("No trades executed in backtest period.")
            return None, None
        
        results_df = pd.DataFrame(trades)
        
        # Calculate performance metrics
        performance = self._calculate_performance(results_df)
        
        return results_df, performance
    
    def _calculate_performance(self, trades_df):
        """Calculate performance metrics from trades."""
        
        total_return = trades_df['return'].sum()
        avg_return = trades_df['return'].mean()
        win_rate = (trades_df['total_pl'] > 0).sum() / len(trades_df)
        
        # Annualized metrics
        total_days = trades_df['hold_days'].sum()
        annualized_return = (1 + total_return) ** (365 / total_days) - 1
        
        # Risk metrics
        volatility = trades_df['return'].std()
        sharpe_ratio = (avg_return * (365/self.option_duration_days)) / (volatility * np.sqrt(365/self.option_duration_days))
        
        max_drawdown = self._calculate_max_drawdown(trades_df)
        
        return {
            'total_return': total_return,
            'annualized_return': annualized_return,
            'avg_return_per_trade': avg_return,
            'win_rate': win_rate,
            'num_trades': len(trades_df),
            'sharpe_ratio': sharpe_ratio,
            'volatility': volatility,
            'max_drawdown': max_drawdown,
            'total_premium_collected': trades_df['premium'].sum(),
            'assignment_rate': (trades_df['outcome'] == 'assigned').sum() / len(trades_df)
        }
    
    def _calculate_max_drawdown(self, trades_df):
        """Calculate maximum drawdown from cumulative returns."""
        cumulative_returns = (1 + trades_df['return']).cumprod()
        running_max = cumulative_returns.expanding().max()
        drawdown = (cumulative_returns - running_max) / running_max
        return drawdown.min()
    
    def print_performance(self, performance):
        """Print performance summary."""
        if performance is None:
            return
        
        print("\n" + "="*70)
        print("BACKTEST PERFORMANCE SUMMARY")
        print("="*70)
        
        print(f"\nTotal Return: {performance['total_return']*100:.2f}%")
        print(f"Annualized Return: {performance['annualized_return']*100:.2f}%")
        print(f"Average Return per Trade: {performance['avg_return_per_trade']*100:.2f}%")
        
        print(f"\nNumber of Trades: {performance['num_trades']}")
        print(f"Win Rate: {performance['win_rate']*100:.1f}%")
        print(f"Assignment Rate: {performance['assignment_rate']*100:.1f}%")
        
        print(f"\nTotal Premium Collected: ${performance['total_premium_collected']:.2f}")
        
        print(f"\nSharpe Ratio: {performance['sharpe_ratio']:.2f}")
        print(f"Volatility: {performance['volatility']*100:.2f}%")
        print(f"Maximum Drawdown: {performance['max_drawdown']*100:.2f}%")
        
        print("\n" + "="*70 + "\n")


def generate_sample_data(initial_price=100, days=365, drift=0.10, volatility=0.25):
    """
    Generate sample price data using geometric Brownian motion.
    
    Parameters:
    -----------
    initial_price : float
        Starting price
    days : int
        Number of days to simulate
    drift : float
        Annual drift (expected return)
    volatility : float
        Annual volatility
        
    Returns:
    --------
    pd.Series : Simulated price series
    """
    dt = 1/365
    np.random.seed(42)  # For reproducibility
    
    returns = np.random.normal(drift * dt, volatility * np.sqrt(dt), days)
    price_path = initial_price * np.exp(np.cumsum(returns))
    
    dates = pd.date_range(start='2023-01-01', periods=days, freq='D')
    return pd.Series(price_path, index=dates)


def example_backtest():
    """Example backtest using simulated data."""
    
    print("Generating sample price data...")
    prices = generate_sample_data(initial_price=150, days=365, 
                                  drift=0.12, volatility=0.25)
    
    print(f"Testing covered call strategy on {len(prices)} days of data")
    print(f"Initial Price: ${prices.iloc[0]:.2f}")
    print(f"Final Price: ${prices.iloc[-1]:.2f}")
    print(f"Buy & Hold Return: {(prices.iloc[-1]/prices.iloc[0] - 1)*100:.2f}%")
    
    # Run backtest
    backtest = CoveredCallBacktest(
        prices, 
        volatility=0.25,
        option_duration_days=30,
        min_downside_protection=0.02
    )
    
    trades_df, performance = backtest.run_backtest()
    
    if performance is not None:
        backtest.print_performance(performance)
        
        print("\nSample Trades:")
        print(trades_df.head(10).to_string(index=False))
        
        # Compare to buy and hold
        buy_hold_return = (prices.iloc[-1] / prices.iloc[0]) - 1
        print(f"\nCovered Call Strategy Return: {performance['total_return']*100:.2f}%")
        print(f"Buy & Hold Return: {buy_hold_return*100:.2f}%")
        print(f"Outperformance: {(performance['total_return'] - buy_hold_return)*100:.2f}%")


if __name__ == "__main__":
    example_backtest()
