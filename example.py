"""
Example Script - Demonstrating All Features
Run this to see the covered call optimizer in action
"""

from black_scholes import BlackScholes
from covered_call_optimizer import CoveredCallOptimizer
from backtest import CoveredCallBacktest, generate_sample_data
from visualizations import plot_option_greeks, plot_covered_call_returns, plot_payoff_diagram
import matplotlib.pyplot as plt

def main():
    """Run complete example demonstrating all features."""
    
    print("="*70)
    print("BLACK-SCHOLES COVERED CALL OPTIMIZER - COMPLETE DEMONSTRATION")
    print("="*70)
    
    # ============================================================
    # PART 1: BLACK-SCHOLES PRICING WITH GREEKS
    # ============================================================
    print("\n" + "="*70)
    print("PART 1: BLACK-SCHOLES OPTION PRICING")
    print("="*70)
    
    S = 150      # Stock price
    K = 155      # Strike price
    T = 30/365   # 30 days
    r = 0.05     # 5% risk-free rate
    sigma = 0.25 # 25% volatility
    
    bs = BlackScholes(S, K, T, r, sigma)
    
    print(f"\nMarket Parameters:")
    print(f"  Stock Price: ${S}")
    print(f"  Strike Price: ${K}")
    print(f"  Time to Expiration: {T*365:.0f} days")
    print(f"  Risk-Free Rate: {r*100}%")
    print(f"  Volatility: {sigma*100}%")
    
    print(f"\nOption Prices:")
    print(f"  Call: ${bs.call_price():.2f}")
    print(f"  Put: ${bs.put_price():.2f}")
    
    print(f"\nCall Greeks:")
    greeks = bs.all_greeks('call')
    for greek, value in greeks.items():
        print(f"  {greek.capitalize()}: {value:.4f}")
    
    # ============================================================
    # PART 2: COVERED CALL OPTIMIZATION
    # ============================================================
    print("\n" + "="*70)
    print("PART 2: COVERED CALL STRIKE OPTIMIZATION")
    print("="*70)
    
    optimizer = CoveredCallOptimizer(
        S=S,
        T=T,
        r=r,
        sigma=sigma,
        min_downside_protection=0.02
    )
    
    results_df, optimal_strike = optimizer.optimize_strike()
    optimizer.analyze_strategy(optimal_strike)
    
    print("\nStrike Comparison (Top 5 by Return):")
    top_5 = results_df.nlargest(5, 'max_return')[
        ['strike', 'call_premium', 'max_return', 'annualized_return', 
         'downside_protection', 'prob_called']
    ]
    print(top_5.to_string(index=False))
    
    # ============================================================
    # PART 3: BACKTESTING
    # ============================================================
    print("\n" + "="*70)
    print("PART 3: STRATEGY BACKTESTING")
    print("="*70)
    
    print("\nGenerating 1 year of simulated price data...")
    prices = generate_sample_data(
        initial_price=150,
        days=365,
        drift=0.12,
        volatility=0.25
    )
    
    print(f"Initial Price: ${prices.iloc[0]:.2f}")
    print(f"Final Price: ${prices.iloc[-1]:.2f}")
    print(f"Buy & Hold Return: {(prices.iloc[-1]/prices.iloc[0] - 1)*100:.2f}%")
    
    print("\nRunning backtest...")
    backtest = CoveredCallBacktest(
        prices=prices,
        volatility=0.25,
        option_duration_days=30,
        min_downside_protection=0.02
    )
    
    trades_df, performance = backtest.run_backtest()
    backtest.print_performance(performance)
    
    if performance:
        print("Sample Trades (First 5):")
        sample_trades = trades_df.head(5)[
            ['entry_price', 'strike', 'premium', 'outcome', 'return']
        ]
        print(sample_trades.to_string(index=False))
    
    # ============================================================
    # PART 4: VISUALIZATIONS
    # ============================================================
    print("\n" + "="*70)
    print("PART 4: GENERATING VISUALIZATIONS")
    print("="*70)
    
    try:
        print("\nGenerating Greeks analysis chart...")
        fig1 = plot_option_greeks(S, T, r, sigma)
        plt.savefig('/home/claude/greeks_chart.png', dpi=300, bbox_inches='tight')
        print("✓ Saved: greeks_chart.png")
        plt.close()
        
        print("Generating covered call returns chart...")
        fig2 = plot_covered_call_returns(S, T, r, sigma)
        plt.savefig('/home/claude/returns_chart.png', dpi=300, bbox_inches='tight')
        print("✓ Saved: returns_chart.png")
        plt.close()
        
        if optimal_strike:
            print("Generating payoff diagram...")
            fig3 = plot_payoff_diagram(S, optimal_strike['strike'], optimal_strike['call_premium'])
            plt.savefig('/home/claude/payoff_chart.png', dpi=300, bbox_inches='tight')
            print("✓ Saved: payoff_chart.png")
            plt.close()
        
        print("\n✓ All visualizations generated successfully!")
        
    except Exception as e:
        print(f"Note: Visualization generation skipped ({e})")
    
    # ============================================================
    # SUMMARY
    # ============================================================
    print("\n" + "="*70)
    print("DEMONSTRATION COMPLETE")
    print("="*70)
    
    print("\nKey Takeaways:")
    print("1. Black-Scholes model accurately prices options with full Greeks")
    print("2. Optimizer finds strikes balancing returns and downside protection")
    if optimal_strike:
        print(f"3. Optimal strike: ${optimal_strike['strike']:.2f} with {optimal_strike['max_return']*100:.2f}% return")
    if performance:
        print(f"4. Backtest shows {performance['annualized_return']*100:.2f}% annualized return")
        print(f"5. Strategy achieved {performance['win_rate']*100:.1f}% win rate")
    
    print("\nNext Steps:")
    print("- Customize parameters for your portfolio")
    print("- Test with your actual stock holdings")
    print("- Adjust risk constraints as needed")
    print("- Consider implementing in production with real-time data")
    
    print("\n" + "="*70)
    print("Thank you for using the Black-Scholes Covered Call Optimizer!")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
