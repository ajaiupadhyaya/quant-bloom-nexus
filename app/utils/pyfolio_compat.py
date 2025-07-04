"""
Pyfolio compatibility layer for modern Python versions.
This module provides alternatives to pyfolio functions that work with Python 3.11+
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Optional
import warnings

# Suppress warnings from deprecated packages
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=DeprecationWarning)

def calculate_performance_metrics(returns: pd.Series, benchmark_returns: pd.Series = None) -> Dict[str, float]:
    """
    Calculate portfolio performance metrics similar to pyfolio.
    This replaces pyfolio.timeseries.perf_stats
    """
    try:
        # Basic calculations
        total_return = (1 + returns).prod() - 1
        annual_return = (1 + returns).resample('Y').apply(lambda x: (1 + x).prod() - 1).mean()
        volatility = returns.std() * np.sqrt(252)
        
        # Sharpe ratio (assuming 2% risk-free rate)
        risk_free_rate = 0.02
        excess_returns = returns - risk_free_rate / 252
        sharpe_ratio = excess_returns.mean() / excess_returns.std() * np.sqrt(252) if excess_returns.std() > 0 else 0
        
        # Sortino ratio
        downside_returns = returns[returns < 0]
        downside_std = downside_returns.std() * np.sqrt(252) if len(downside_returns) > 0 else 0
        sortino_ratio = (annual_return - risk_free_rate) / downside_std if downside_std > 0 else 0
        
        # Maximum drawdown
        cumulative = (1 + returns).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        max_drawdown = drawdown.min()
        
        # Calmar ratio
        calmar_ratio = annual_return / abs(max_drawdown) if max_drawdown != 0 else 0
        
        # Win rate
        positive_returns = returns[returns > 0]
        win_rate = len(positive_returns) / len(returns) if len(returns) > 0 else 0
        
        # Profit factor
        gross_profit = positive_returns.sum() if len(positive_returns) > 0 else 0
        gross_loss = abs(returns[returns < 0].sum()) if len(returns[returns < 0]) > 0 else 1
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else 0
        
        # Beta and Alpha (if benchmark provided)
        beta = 0
        alpha = 0
        information_ratio = 0
        treynor_ratio = 0
        
        if benchmark_returns is not None and len(benchmark_returns) > 0:
            # Align returns
            aligned_returns = returns.reindex(benchmark_returns.index).dropna()
            aligned_benchmark = benchmark_returns.reindex(aligned_returns.index).dropna()
            
            if len(aligned_returns) > 0 and len(aligned_benchmark) > 0:
                # Beta calculation
                covariance = np.cov(aligned_returns, aligned_benchmark)[0][1]
                benchmark_variance = np.var(aligned_benchmark)
                beta = covariance / benchmark_variance if benchmark_variance > 0 else 0
                
                # Alpha calculation (CAPM)
                benchmark_annual_return = (1 + aligned_benchmark).resample('Y').apply(lambda x: (1 + x).prod() - 1).mean()
                alpha = annual_return - (risk_free_rate + beta * (benchmark_annual_return - risk_free_rate))
                
                # Information ratio
                active_returns = aligned_returns - aligned_benchmark
                tracking_error = active_returns.std() * np.sqrt(252)
                information_ratio = active_returns.mean() * np.sqrt(252) / tracking_error if tracking_error > 0 else 0
                
                # Treynor ratio
                treynor_ratio = (annual_return - risk_free_rate) / beta if beta > 0 else 0
        
        return {
            'total_return': float(total_return),
            'annualized_return': float(annual_return),
            'volatility': float(volatility),
            'sharpe_ratio': float(sharpe_ratio),
            'sortino_ratio': float(sortino_ratio),
            'max_drawdown': float(max_drawdown),
            'calmar_ratio': float(calmar_ratio),
            'win_rate': float(win_rate),
            'profit_factor': float(profit_factor),
            'alpha': float(alpha),
            'beta': float(beta),
            'information_ratio': float(information_ratio),
            'treynor_ratio': float(treynor_ratio)
        }
        
    except Exception as e:
        # Return default values if calculation fails
        return {
            'total_return': 0.0,
            'annualized_return': 0.0,
            'volatility': 0.0,
            'sharpe_ratio': 0.0,
            'sortino_ratio': 0.0,
            'max_drawdown': 0.0,
            'calmar_ratio': 0.0,
            'win_rate': 0.0,
            'profit_factor': 0.0,
            'alpha': 0.0,
            'beta': 0.0,
            'information_ratio': 0.0,
            'treynor_ratio': 0.0
        }

def create_returns_tearsheet(returns: pd.Series, benchmark_returns: pd.Series = None) -> Dict[str, Any]:
    """
    Create a simplified returns tearsheet similar to pyfolio.
    """
    metrics = calculate_performance_metrics(returns, benchmark_returns)
    
    # Add additional analysis
    monthly_returns = returns.resample('M').apply(lambda x: (1 + x).prod() - 1)
    
    return {
        'performance_metrics': metrics,
        'monthly_returns': monthly_returns.to_dict(),
        'total_periods': len(returns),
        'start_date': returns.index[0].strftime('%Y-%m-%d') if len(returns) > 0 else None,
        'end_date': returns.index[-1].strftime('%Y-%m-%d') if len(returns) > 0 else None
    }

def simple_backtest_analysis(returns: pd.Series, positions: pd.Series = None) -> Dict[str, Any]:
    """
    Simple backtest analysis without pyfolio dependencies.
    """
    try:
        # Calculate basic metrics
        cumulative_returns = (1 + returns).cumprod()
        
        # Drawdown analysis
        running_max = cumulative_returns.expanding().max()
        drawdown = (cumulative_returns - running_max) / running_max
        
        # Find drawdown periods
        is_drawdown = drawdown < 0
        drawdown_periods = []
        
        if is_drawdown.any():
            # Group consecutive drawdown periods
            drawdown_groups = (is_drawdown != is_drawdown.shift()).cumsum()
            for group in drawdown_groups[is_drawdown].unique():
                period_data = drawdown[drawdown_groups == group]
                drawdown_periods.append({
                    'start': period_data.index[0].strftime('%Y-%m-%d'),
                    'end': period_data.index[-1].strftime('%Y-%m-%d'),
                    'duration': len(period_data),
                    'max_drawdown': float(period_data.min())
                })
        
        # Rolling metrics
        rolling_sharpe = returns.rolling(252).apply(
            lambda x: x.mean() / x.std() * np.sqrt(252) if x.std() > 0 else 0
        )
        
        return {
            'cumulative_returns': cumulative_returns.to_dict(),
            'drawdown_periods': drawdown_periods,
            'rolling_sharpe': rolling_sharpe.dropna().to_dict(),
            'final_value': float(cumulative_returns.iloc[-1]) if len(cumulative_returns) > 0 else 1.0
        }
        
    except Exception as e:
        return {
            'cumulative_returns': {},
            'drawdown_periods': [],
            'rolling_sharpe': {},
            'final_value': 1.0,
            'error': str(e)
        }

# Monkey patch for configparser compatibility
def patch_configparser():
    """Fix configparser compatibility issues"""
    import configparser
    if not hasattr(configparser, 'SafeConfigParser'):
        configparser.SafeConfigParser = configparser.ConfigParser

# Apply the patch
patch_configparser() 