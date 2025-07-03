import numpy as np
import pandas as pd
from scipy import stats, optimize
from scipy.stats import norm, t, chi2
import math
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime, timedelta
import warnings
from dataclasses import dataclass
from enum import Enum
import logging

warnings.filterwarnings('ignore')
logger = logging.getLogger(__name__)

class OptionType(Enum):
    CALL = "call"
    PUT = "put"

@dataclass
class GreeksResult:
    delta: float
    gamma: float
    theta: float
    vega: float
    rho: float

@dataclass
class RiskMetrics:
    var_95: float
    var_99: float
    cvar_95: float
    cvar_99: float
    max_drawdown: float
    sharpe_ratio: float
    sortino_ratio: float
    calmar_ratio: float
    volatility: float
    skewness: float
    kurtosis: float

@dataclass
class PerformanceMetrics:
    total_return: float
    annualized_return: float
    volatility: float
    sharpe_ratio: float
    sortino_ratio: float
    max_drawdown: float
    calmar_ratio: float
    win_rate: float
    profit_factor: float
    alpha: float
    beta: float
    information_ratio: float
    treynor_ratio: float

class QuantitativeEngine:
    """Complete quantitative finance and mathematical engine"""
    
    def __init__(self):
        self.risk_free_rate = 0.02  # 2% default risk-free rate
        
    # =================== OPTIONS PRICING ===================
    
    def black_scholes_price(self, S: float, K: float, T: float, r: float, 
                           sigma: float, option_type: OptionType) -> float:
        """Black-Scholes option pricing model"""
        try:
            d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
            d2 = d1 - sigma * np.sqrt(T)
            
            if option_type == OptionType.CALL:
                price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
            else:  # PUT
                price = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
                
            return max(price, 0)
        except Exception as e:
            logger.error(f"Black-Scholes calculation failed: {e}")
            return 0.0
    
    def calculate_greeks(self, S: float, K: float, T: float, r: float, 
                        sigma: float, option_type: OptionType) -> GreeksResult:
        """Calculate option Greeks"""
        try:
            d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
            d2 = d1 - sigma * np.sqrt(T)
            
            # Delta
            if option_type == OptionType.CALL:
                delta = norm.cdf(d1)
            else:
                delta = norm.cdf(d1) - 1
            
            # Gamma
            gamma = norm.pdf(d1) / (S * sigma * np.sqrt(T))
            
            # Theta
            if option_type == OptionType.CALL:
                theta = (-S * norm.pdf(d1) * sigma / (2 * np.sqrt(T)) 
                        - r * K * np.exp(-r * T) * norm.cdf(d2)) / 365
            else:
                theta = (-S * norm.pdf(d1) * sigma / (2 * np.sqrt(T)) 
                        + r * K * np.exp(-r * T) * norm.cdf(-d2)) / 365
            
            # Vega
            vega = S * norm.pdf(d1) * np.sqrt(T) / 100
            
            # Rho
            if option_type == OptionType.CALL:
                rho = K * T * np.exp(-r * T) * norm.cdf(d2) / 100
            else:
                rho = -K * T * np.exp(-r * T) * norm.cdf(-d2) / 100
                
            return GreeksResult(delta, gamma, theta, vega, rho)
            
        except Exception as e:
            logger.error(f"Greeks calculation failed: {e}")
            return GreeksResult(0, 0, 0, 0, 0)
    
    def implied_volatility(self, market_price: float, S: float, K: float, 
                          T: float, r: float, option_type: OptionType) -> float:
        """Calculate implied volatility using Brent's method"""
        try:
            def objective(sigma):
                return self.black_scholes_price(S, K, T, r, sigma, option_type) - market_price
            
            result = optimize.brentq(objective, 0.001, 5.0, xtol=1e-6)
            return result
        except Exception as e:
            logger.error(f"Implied volatility calculation failed: {e}")
            return 0.2  # Default 20% volatility
    
    def binomial_tree_price(self, S: float, K: float, T: float, r: float, 
                           sigma: float, n_steps: int, option_type: OptionType, 
                           american: bool = False) -> float:
        """Binomial tree option pricing"""
        try:
            dt = T / n_steps
            u = np.exp(sigma * np.sqrt(dt))
            d = 1 / u
            p = (np.exp(r * dt) - d) / (u - d)
            
            # Initialize asset prices at maturity
            asset_prices = np.zeros(n_steps + 1)
            for i in range(n_steps + 1):
                asset_prices[i] = S * (u ** (n_steps - i)) * (d ** i)
            
            # Initialize option values at maturity
            option_values = np.zeros(n_steps + 1)
            for i in range(n_steps + 1):
                if option_type == OptionType.CALL:
                    option_values[i] = max(asset_prices[i] - K, 0)
                else:
                    option_values[i] = max(K - asset_prices[i], 0)
            
            # Work backwards through the tree
            for j in range(n_steps - 1, -1, -1):
                for i in range(j + 1):
                    # Calculate option value
                    option_values[i] = np.exp(-r * dt) * (p * option_values[i] + (1 - p) * option_values[i + 1])
                    
                    # For American options, check early exercise
                    if american:
                        asset_price = S * (u ** (j - i)) * (d ** i)
                        if option_type == OptionType.CALL:
                            exercise_value = max(asset_price - K, 0)
                        else:
                            exercise_value = max(K - asset_price, 0)
                        option_values[i] = max(option_values[i], exercise_value)
            
            return option_values[0]
        except Exception as e:
            logger.error(f"Binomial tree calculation failed: {e}")
            return 0.0
    
    # =================== RISK MANAGEMENT ===================
    
    def calculate_var(self, returns: np.ndarray, confidence_level: float = 0.95) -> float:
        """Calculate Value at Risk"""
        try:
            return np.percentile(returns, (1 - confidence_level) * 100)
        except Exception as e:
            logger.error(f"VaR calculation failed: {e}")
            return 0.0
    
    def calculate_cvar(self, returns: np.ndarray, confidence_level: float = 0.95) -> float:
        """Calculate Conditional Value at Risk (Expected Shortfall)"""
        try:
            var = self.calculate_var(returns, confidence_level)
            return returns[returns <= var].mean()
        except Exception as e:
            logger.error(f"CVaR calculation failed: {e}")
            return 0.0
    
    def calculate_risk_metrics(self, returns: np.ndarray) -> RiskMetrics:
        """Calculate comprehensive risk metrics"""
        try:
            # Convert to numpy array if pandas series
            if isinstance(returns, pd.Series):
                returns = returns.values
            
            # Remove NaN values
            returns = returns[~np.isnan(returns)]
            
            if len(returns) == 0:
                return RiskMetrics(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)
            
            # VaR calculations
            var_95 = self.calculate_var(returns, 0.95)
            var_99 = self.calculate_var(returns, 0.99)
            cvar_95 = self.calculate_cvar(returns, 0.95)
            cvar_99 = self.calculate_cvar(returns, 0.99)
            
            # Cumulative returns for drawdown
            cumulative_returns = (1 + returns).cumprod()
            running_max = np.maximum.accumulate(cumulative_returns)
            drawdown = (cumulative_returns - running_max) / running_max
            max_drawdown = drawdown.min()
            
            # Performance ratios
            volatility = np.std(returns) * np.sqrt(252)
            mean_return = np.mean(returns) * 252
            
            sharpe_ratio = (mean_return - self.risk_free_rate) / volatility if volatility != 0 else 0
            
            # Sortino ratio (downside deviation)
            downside_returns = returns[returns < 0]
            downside_std = np.std(downside_returns) * np.sqrt(252) if len(downside_returns) > 0 else 0
            sortino_ratio = (mean_return - self.risk_free_rate) / downside_std if downside_std != 0 else 0
            
            # Calmar ratio
            calmar_ratio = mean_return / abs(max_drawdown) if max_drawdown != 0 else 0
            
            # Higher moments
            skewness = stats.skew(returns)
            kurtosis = stats.kurtosis(returns)
            
            return RiskMetrics(
                var_95=var_95,
                var_99=var_99,
                cvar_95=cvar_95,
                cvar_99=cvar_99,
                max_drawdown=max_drawdown,
                sharpe_ratio=sharpe_ratio,
                sortino_ratio=sortino_ratio,
                calmar_ratio=calmar_ratio,
                volatility=volatility,
                skewness=skewness,
                kurtosis=kurtosis
            )
        except Exception as e:
            logger.error(f"Risk metrics calculation failed: {e}")
            return RiskMetrics(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)
    
    # =================== PORTFOLIO OPTIMIZATION ===================
    
    def mean_variance_optimization(self, returns: pd.DataFrame, target_return: float = None) -> Dict[str, float]:
        """Markowitz mean-variance optimization"""
        try:
            mean_returns = returns.mean() * 252
            cov_matrix = returns.cov() * 252
            n_assets = len(mean_returns)
            
            # Constraints and bounds
            constraints = [{'type': 'eq', 'fun': lambda x: np.sum(x) - 1}]
            if target_return is not None:
                constraints.append({'type': 'eq', 'fun': lambda x: np.sum(x * mean_returns) - target_return})
            
            bounds = tuple((0, 1) for _ in range(n_assets))
            
            # Objective function (minimize variance)
            def objective(weights):
                return np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
            
            # Initial guess (equal weights)
            x0 = np.array([1/n_assets] * n_assets)
            
            # Optimize
            result = optimize.minimize(objective, x0, method='SLSQP', bounds=bounds, constraints=constraints)
            
            if result.success:
                weights = dict(zip(returns.columns, result.x))
                return weights
            else:
                # Return equal weights if optimization fails
                return dict(zip(returns.columns, [1/n_assets] * n_assets))
                
        except Exception as e:
            logger.error(f"Portfolio optimization failed: {e}")
            return {}
    
    def black_litterman_optimization(self, returns: pd.DataFrame, market_caps: Dict[str, float], 
                                   views: Dict[str, float] = None, tau: float = 0.05) -> Dict[str, float]:
        """Black-Litterman portfolio optimization"""
        try:
            # Calculate market-implied expected returns
            mean_returns = returns.mean() * 252
            cov_matrix = returns.cov() * 252
            
            # Market capitalization weights
            total_market_cap = sum(market_caps.values())
            market_weights = np.array([market_caps.get(asset, 0) / total_market_cap for asset in returns.columns])
            
            # Implied equilibrium returns
            risk_aversion = 3.0  # Typical value
            pi = risk_aversion * np.dot(cov_matrix, market_weights)
            
            if views:
                # Incorporate views
                P = np.zeros((len(views), len(returns.columns)))
                Q = np.zeros(len(views))
                
                for i, (asset, view) in enumerate(views.items()):
                    if asset in returns.columns:
                        asset_idx = list(returns.columns).index(asset)
                        P[i, asset_idx] = 1
                        Q[i] = view
                
                # Uncertainty matrix for views
                omega = np.diag(np.diag(np.dot(P, np.dot(tau * cov_matrix, P.T))))
                
                # Black-Litterman formula
                M1 = np.linalg.inv(tau * cov_matrix)
                M2 = np.dot(P.T, np.dot(np.linalg.inv(omega), P))
                M3 = np.dot(np.linalg.inv(tau * cov_matrix), pi)
                M4 = np.dot(P.T, np.dot(np.linalg.inv(omega), Q))
                
                mu_bl = np.dot(np.linalg.inv(M1 + M2), M3 + M4)
                cov_bl = np.linalg.inv(M1 + M2)
            else:
                mu_bl = pi
                cov_bl = cov_matrix
            
            # Optimize portfolio
            def objective(weights):
                return 0.5 * risk_aversion * np.dot(weights.T, np.dot(cov_bl, weights)) - np.dot(weights, mu_bl)
            
            constraints = [{'type': 'eq', 'fun': lambda x: np.sum(x) - 1}]
            bounds = tuple((0, 1) for _ in range(len(returns.columns)))
            x0 = market_weights
            
            result = optimize.minimize(objective, x0, method='SLSQP', bounds=bounds, constraints=constraints)
            
            if result.success:
                return dict(zip(returns.columns, result.x))
            else:
                return dict(zip(returns.columns, market_weights))
                
        except Exception as e:
            logger.error(f"Black-Litterman optimization failed: {e}")
            return {}
    
    # =================== TECHNICAL ANALYSIS ===================
    
    def calculate_bollinger_bands(self, prices: pd.Series, window: int = 20, num_std: float = 2) -> Dict[str, pd.Series]:
        """Calculate Bollinger Bands"""
        try:
            sma = prices.rolling(window=window).mean()
            std = prices.rolling(window=window).std()
            
            upper_band = sma + (std * num_std)
            lower_band = sma - (std * num_std)
            
            return {
                'sma': sma,
                'upper_band': upper_band,
                'lower_band': lower_band,
                'bandwidth': (upper_band - lower_band) / sma,
                'percent_b': (prices - lower_band) / (upper_band - lower_band)
            }
        except Exception as e:
            logger.error(f"Bollinger Bands calculation failed: {e}")
            return {}
    
    def calculate_rsi(self, prices: pd.Series, window: int = 14) -> pd.Series:
        """Calculate Relative Strength Index"""
        try:
            delta = prices.diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
            return rsi
        except Exception as e:
            logger.error(f"RSI calculation failed: {e}")
            return pd.Series()
    
    def calculate_macd(self, prices: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> Dict[str, pd.Series]:
        """Calculate MACD"""
        try:
            ema_fast = prices.ewm(span=fast).mean()
            ema_slow = prices.ewm(span=slow).mean()
            macd = ema_fast - ema_slow
            signal_line = macd.ewm(span=signal).mean()
            histogram = macd - signal_line
            
            return {
                'macd': macd,
                'signal': signal_line,
                'histogram': histogram
            }
        except Exception as e:
            logger.error(f"MACD calculation failed: {e}")
            return {}
    
    def calculate_stochastic(self, high: pd.Series, low: pd.Series, close: pd.Series, 
                           k_window: int = 14, d_window: int = 3) -> Dict[str, pd.Series]:
        """Calculate Stochastic Oscillator"""
        try:
            lowest_low = low.rolling(window=k_window).min()
            highest_high = high.rolling(window=k_window).max()
            
            k_percent = 100 * ((close - lowest_low) / (highest_high - lowest_low))
            d_percent = k_percent.rolling(window=d_window).mean()
            
            return {
                'k_percent': k_percent,
                'd_percent': d_percent
            }
        except Exception as e:
            logger.error(f"Stochastic calculation failed: {e}")
            return {}
    
    # =================== ECONOMETRIC MODELS ===================
    
    def calculate_beta(self, asset_returns: pd.Series, market_returns: pd.Series) -> float:
        """Calculate beta coefficient"""
        try:
            covariance = np.cov(asset_returns.dropna(), market_returns.dropna())[0, 1]
            market_variance = np.var(market_returns.dropna())
            return covariance / market_variance if market_variance != 0 else 0
        except Exception as e:
            logger.error(f"Beta calculation failed: {e}")
            return 1.0
    
    def calculate_alpha(self, asset_returns: pd.Series, market_returns: pd.Series, 
                       risk_free_rate: float = None) -> float:
        """Calculate Jensen's alpha"""
        try:
            if risk_free_rate is None:
                risk_free_rate = self.risk_free_rate
            
            beta = self.calculate_beta(asset_returns, market_returns)
            asset_mean = asset_returns.mean() * 252
            market_mean = market_returns.mean() * 252
            
            alpha = asset_mean - (risk_free_rate + beta * (market_mean - risk_free_rate))
            return alpha
        except Exception as e:
            logger.error(f"Alpha calculation failed: {e}")
            return 0.0
    
    def information_ratio(self, portfolio_returns: pd.Series, benchmark_returns: pd.Series) -> float:
        """Calculate Information Ratio"""
        try:
            excess_returns = portfolio_returns - benchmark_returns
            tracking_error = excess_returns.std() * np.sqrt(252)
            active_return = excess_returns.mean() * 252
            
            return active_return / tracking_error if tracking_error != 0 else 0
        except Exception as e:
            logger.error(f"Information ratio calculation failed: {e}")
            return 0.0
    
    def treynor_ratio(self, portfolio_returns: pd.Series, market_returns: pd.Series, 
                     risk_free_rate: float = None) -> float:
        """Calculate Treynor Ratio"""
        try:
            if risk_free_rate is None:
                risk_free_rate = self.risk_free_rate
            
            beta = self.calculate_beta(portfolio_returns, market_returns)
            excess_return = portfolio_returns.mean() * 252 - risk_free_rate
            
            return excess_return / beta if beta != 0 else 0
        except Exception as e:
            logger.error(f"Treynor ratio calculation failed: {e}")
            return 0.0
    
    # =================== MONTE CARLO SIMULATIONS ===================
    
    def monte_carlo_var(self, returns: np.ndarray, num_simulations: int = 10000, 
                       time_horizon: int = 1, confidence_level: float = 0.95) -> float:
        """Monte Carlo Value at Risk simulation"""
        try:
            mean_return = np.mean(returns)
            std_return = np.std(returns)
            
            # Generate random scenarios
            random_returns = np.random.normal(mean_return, std_return, 
                                            (num_simulations, time_horizon))
            
            # Calculate portfolio values
            portfolio_values = np.prod(1 + random_returns, axis=1) - 1
            
            # Calculate VaR
            var = np.percentile(portfolio_values, (1 - confidence_level) * 100)
            return var
        except Exception as e:
            logger.error(f"Monte Carlo VaR calculation failed: {e}")
            return 0.0
    
    def geometric_brownian_motion(self, S0: float, mu: float, sigma: float, 
                                 T: float, dt: float, num_paths: int = 1000) -> np.ndarray:
        """Simulate Geometric Brownian Motion paths"""
        try:
            num_steps = int(T / dt)
            paths = np.zeros((num_paths, num_steps + 1))
            paths[:, 0] = S0
            
            for i in range(1, num_steps + 1):
                dW = np.random.normal(0, np.sqrt(dt), num_paths)
                paths[:, i] = paths[:, i-1] * np.exp((mu - 0.5 * sigma**2) * dt + sigma * dW)
            
            return paths
        except Exception as e:
            logger.error(f"GBM simulation failed: {e}")
            return np.array([[S0]])
    
    # =================== PERFORMANCE ATTRIBUTION ===================
    
    def calculate_performance_metrics(self, returns: pd.Series, benchmark_returns: pd.Series = None) -> PerformanceMetrics:
        """Calculate comprehensive performance metrics"""
        try:
            if benchmark_returns is None:
                benchmark_returns = pd.Series([0] * len(returns))
            
            # Basic metrics
            total_return = (1 + returns).prod() - 1
            annualized_return = (1 + returns.mean())**252 - 1
            volatility = returns.std() * np.sqrt(252)
            
            # Risk-adjusted metrics
            excess_returns = returns - self.risk_free_rate / 252
            sharpe_ratio = excess_returns.mean() / returns.std() * np.sqrt(252) if returns.std() != 0 else 0
            
            downside_returns = returns[returns < 0]
            downside_std = downside_returns.std() if len(downside_returns) > 0 else 0
            sortino_ratio = excess_returns.mean() / downside_std * np.sqrt(252) if downside_std != 0 else 0
            
            # Drawdown metrics
            cumulative = (1 + returns).cumprod()
            running_max = cumulative.expanding().max()
            drawdown = (cumulative - running_max) / running_max
            max_drawdown = drawdown.min()
            calmar_ratio = annualized_return / abs(max_drawdown) if max_drawdown != 0 else 0
            
            # Trading metrics
            winning_trades = returns[returns > 0]
            losing_trades = returns[returns < 0]
            win_rate = len(winning_trades) / len(returns) if len(returns) > 0 else 0
            
            avg_win = winning_trades.mean() if len(winning_trades) > 0 else 0
            avg_loss = abs(losing_trades.mean()) if len(losing_trades) > 0 else 0
            profit_factor = avg_win / avg_loss if avg_loss != 0 else 0
            
            # Market-relative metrics
            alpha = self.calculate_alpha(returns, benchmark_returns)
            beta = self.calculate_beta(returns, benchmark_returns)
            information_ratio = self.information_ratio(returns, benchmark_returns)
            treynor_ratio = self.treynor_ratio(returns, benchmark_returns)
            
            return PerformanceMetrics(
                total_return=total_return,
                annualized_return=annualized_return,
                volatility=volatility,
                sharpe_ratio=sharpe_ratio,
                sortino_ratio=sortino_ratio,
                max_drawdown=max_drawdown,
                calmar_ratio=calmar_ratio,
                win_rate=win_rate,
                profit_factor=profit_factor,
                alpha=alpha,
                beta=beta,
                information_ratio=information_ratio,
                treynor_ratio=treynor_ratio
            )
        except Exception as e:
            logger.error(f"Performance metrics calculation failed: {e}")
            return PerformanceMetrics(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)

# Global quantitative engine instance
quantitative_engine = QuantitativeEngine() 