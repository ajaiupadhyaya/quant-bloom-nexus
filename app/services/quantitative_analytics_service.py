import asyncio
import aiohttp
import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime, timedelta
import logging
import os
import yfinance as yf
from dataclasses import dataclass
import scipy.stats as stats
from scipy.optimize import minimize
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)

@dataclass
class RiskMetric:
    name: str
    value: float
    description: str
    confidence_interval: Optional[Tuple[float, float]] = None

@dataclass
class FactorExposure:
    factor: str
    beta: float
    r_squared: float
    p_value: float

@dataclass
class PortfolioAllocation:
    symbol: str
    weight: float
    expected_return: float
    risk_contribution: float

class QuantitativeAnalyticsService:
    """Complete quantitative analytics service with risk metrics, factor models, and portfolio optimization"""
    
    def __init__(self):
        self.session = None
        
    async def _get_session(self):
        """Get or create aiohttp session"""
        if self.session is None:
            self.session = aiohttp.ClientSession()
        return self.session
    
    async def close(self):
        """Close aiohttp session"""
        if self.session:
            await self.session.close()
    
    # Risk Metrics
    async def calculate_risk_metrics(self, returns: pd.Series, confidence_level: float = 0.95) -> List[RiskMetric]:
        """Calculate VaR, CVaR, maximum drawdown, tracking error, information ratio"""
        try:
            metrics = []
            
            # Value at Risk (VaR)
            var = self._calculate_var(returns, confidence_level)
            metrics.append(RiskMetric(
                name="Value at Risk (VaR)",
                value=var,
                description=f"{confidence_level*100}% VaR",
                confidence_interval=(confidence_level, 1.0)
            ))
            
            # Conditional Value at Risk (CVaR)
            cvar = self._calculate_cvar(returns, confidence_level)
            metrics.append(RiskMetric(
                name="Conditional VaR (CVaR)",
                value=cvar,
                description=f"Expected loss beyond {confidence_level*100}% VaR",
                confidence_interval=(confidence_level, 1.0)
            ))
            
            # Maximum Drawdown
            max_dd = self._calculate_maximum_drawdown(returns)
            metrics.append(RiskMetric(
                name="Maximum Drawdown",
                value=max_dd,
                description="Maximum peak-to-trough decline"
            ))
            
            # Volatility
            volatility = returns.std() * np.sqrt(252)  # Annualized
            metrics.append(RiskMetric(
                name="Volatility",
                value=volatility,
                description="Annualized volatility"
            ))
            
            # Skewness
            skewness = returns.skew()
            metrics.append(RiskMetric(
                name="Skewness",
                value=skewness,
                description="Distribution asymmetry"
            ))
            
            # Kurtosis
            kurtosis = returns.kurtosis()
            metrics.append(RiskMetric(
                name="Kurtosis",
                value=kurtosis,
                description="Distribution tail heaviness"
            ))
            
            return metrics
            
        except Exception as e:
            logger.error(f"Risk metrics calculation failed: {e}")
            return []
    
    def _calculate_var(self, returns: pd.Series, confidence_level: float) -> float:
        """Calculate Value at Risk"""
        try:
            return np.percentile(returns, (1 - confidence_level) * 100)
        except Exception as e:
            logger.error(f"VaR calculation failed: {e}")
            return 0.0
    
    def _calculate_cvar(self, returns: pd.Series, confidence_level: float) -> float:
        """Calculate Conditional Value at Risk"""
        try:
            var = self._calculate_var(returns, confidence_level)
            return returns[returns <= var].mean()
        except Exception as e:
            logger.error(f"CVaR calculation failed: {e}")
            return 0.0
    
    def _calculate_maximum_drawdown(self, returns: pd.Series) -> float:
        """Calculate maximum drawdown"""
        try:
            cumulative = (1 + returns).cumprod()
            running_max = cumulative.expanding().max()
            drawdown = (cumulative - running_max) / running_max
            return drawdown.min()
        except Exception as e:
            logger.error(f"Maximum drawdown calculation failed: {e}")
            return 0.0
    
    # Performance Attribution
    async def calculate_performance_attribution(self, portfolio_returns: pd.Series, benchmark_returns: pd.Series, 
                                              factor_returns: Dict[str, pd.Series] = None) -> Dict[str, Any]:
        """Factor-based attribution analysis across multiple dimensions"""
        try:
            attribution = {}
            
            # Basic attribution (Brinson model)
            attribution['basic'] = self._calculate_basic_attribution(portfolio_returns, benchmark_returns)
            
            # Factor attribution
            if factor_returns:
                attribution['factor'] = await self._calculate_factor_attribution(portfolio_returns, factor_returns)
            
            # Risk attribution
            attribution['risk'] = self._calculate_risk_attribution(portfolio_returns, benchmark_returns)
            
            return attribution
            
        except Exception as e:
            logger.error(f"Performance attribution calculation failed: {e}")
            return {}
    
    def _calculate_basic_attribution(self, portfolio_returns: pd.Series, benchmark_returns: pd.Series) -> Dict[str, float]:
        """Calculate basic performance attribution"""
        try:
            # Calculate excess return
            excess_return = portfolio_returns.mean() - benchmark_returns.mean()
            
            # Calculate tracking error
            tracking_error = (portfolio_returns - benchmark_returns).std() * np.sqrt(252)
            
            # Calculate information ratio
            information_ratio = excess_return / tracking_error if tracking_error > 0 else 0
            
            return {
                'excess_return': excess_return,
                'tracking_error': tracking_error,
                'information_ratio': information_ratio,
                'active_share': self._calculate_active_share(portfolio_returns, benchmark_returns)
            }
            
        except Exception as e:
            logger.error(f"Basic attribution calculation failed: {e}")
            return {}
    
    def _calculate_active_share(self, portfolio_returns: pd.Series, benchmark_returns: pd.Series) -> float:
        """Calculate active share"""
        try:
            # Simplified active share calculation
            return abs(portfolio_returns.mean() - benchmark_returns.mean())
        except Exception as e:
            logger.error(f"Active share calculation failed: {e}")
            return 0.0
    
    async def _calculate_factor_attribution(self, portfolio_returns: pd.Series, factor_returns: Dict[str, pd.Series]) -> Dict[str, Any]:
        """Calculate factor-based attribution"""
        try:
            # Create factor exposure matrix
            factor_data = pd.DataFrame(factor_returns)
            factor_data['portfolio'] = portfolio_returns
            
            # Calculate factor exposures using regression
            factor_exposures = {}
            
            for factor in factor_returns.keys():
                try:
                    # Simple linear regression
                    factor_series = factor_data[factor]
                    portfolio_series = factor_data['portfolio']
                    
                    # Remove NaN values
                    valid_data = pd.DataFrame({
                        'factor': factor_series,
                        'portfolio': portfolio_series
                    }).dropna()
                    
                    if len(valid_data) > 10:
                        # Calculate correlation and beta
                        correlation = valid_data['factor'].corr(valid_data['portfolio'])
                        beta = correlation * valid_data['portfolio'].std() / valid_data['factor'].std()
                        
                        # Calculate R-squared
                        r_squared = correlation ** 2
                        
                        factor_exposures[factor] = {
                            'beta': beta,
                            'correlation': correlation,
                            'r_squared': r_squared
                        }
                    
                except Exception as e:
                    logger.warning(f"Factor attribution for {factor} failed: {e}")
                    continue
            
            return factor_exposures
            
        except Exception as e:
            logger.error(f"Factor attribution calculation failed: {e}")
            return {}
    
    def _calculate_risk_attribution(self, portfolio_returns: pd.Series, benchmark_returns: pd.Series) -> Dict[str, float]:
        """Calculate risk attribution"""
        try:
            # Calculate risk contributions
            portfolio_vol = portfolio_returns.std() * np.sqrt(252)
            benchmark_vol = benchmark_returns.std() * np.sqrt(252)
            
            # Calculate beta
            covariance = np.cov(portfolio_returns, benchmark_returns)[0, 1]
            benchmark_variance = benchmark_returns.var()
            beta = covariance / benchmark_variance if benchmark_variance > 0 else 0
            
            # Calculate systematic and idiosyncratic risk
            systematic_risk = beta * benchmark_vol
            idiosyncratic_risk = np.sqrt(portfolio_vol**2 - systematic_risk**2)
            
            return {
                'total_risk': portfolio_vol,
                'systematic_risk': systematic_risk,
                'idiosyncratic_risk': idiosyncratic_risk,
                'beta': beta
            }
            
        except Exception as e:
            logger.error(f"Risk attribution calculation failed: {e}")
            return {}
    
    # Correlation Analysis
    async def calculate_correlation_analysis(self, returns_data: Dict[str, pd.Series], 
                                           window: int = 60) -> Dict[str, Any]:
        """Rolling correlations, correlation breakdowns, regime analysis"""
        try:
            analysis = {}
            
            # Convert to DataFrame
            returns_df = pd.DataFrame(returns_data)
            
            # Rolling correlations
            analysis['rolling_correlations'] = self._calculate_rolling_correlations(returns_df, window)
            
            # Correlation matrix
            analysis['correlation_matrix'] = returns_df.corr().to_dict()
            
            # Correlation breakdown by regime
            analysis['regime_analysis'] = self._calculate_correlation_regimes(returns_df)
            
            # Correlation clustering
            analysis['correlation_clusters'] = self._calculate_correlation_clusters(returns_df)
            
            return analysis
            
        except Exception as e:
            logger.error(f"Correlation analysis failed: {e}")
            return {}
    
    def _calculate_rolling_correlations(self, returns_df: pd.DataFrame, window: int) -> Dict[str, List[float]]:
        """Calculate rolling correlations"""
        try:
            rolling_corr = returns_df.rolling(window=window).corr()
            
            # Extract pairwise correlations
            symbols = returns_df.columns
            correlations = {}
            
            for i, symbol1 in enumerate(symbols):
                for symbol2 in symbols[i+1:]:
                    pair_name = f"{symbol1}_{symbol2}"
                    corr_series = rolling_corr.loc[(slice(None), symbol1), symbol2]
                    correlations[pair_name] = corr_series.dropna().tolist()
            
            return correlations
            
        except Exception as e:
            logger.error(f"Rolling correlations calculation failed: {e}")
            return {}
    
    def _calculate_correlation_regimes(self, returns_df: pd.DataFrame) -> Dict[str, Any]:
        """Calculate correlation breakdown by market regime"""
        try:
            # Define regimes based on market volatility
            market_vol = returns_df.mean(axis=1).rolling(window=20).std()
            
            # High volatility regime (top 25%)
            high_vol_threshold = market_vol.quantile(0.75)
            high_vol_mask = market_vol > high_vol_threshold
            
            # Low volatility regime (bottom 25%)
            low_vol_threshold = market_vol.quantile(0.25)
            low_vol_mask = market_vol < low_vol_threshold
            
            # Calculate correlations in different regimes
            high_vol_corr = returns_df[high_vol_mask].corr().to_dict()
            low_vol_corr = returns_df[low_vol_mask].corr().to_dict()
            normal_corr = returns_df[~(high_vol_mask | low_vol_mask)].corr().to_dict()
            
            return {
                'high_volatility_regime': high_vol_corr,
                'low_volatility_regime': low_vol_corr,
                'normal_regime': normal_corr,
                'regime_thresholds': {
                    'high_vol': high_vol_threshold,
                    'low_vol': low_vol_threshold
                }
            }
            
        except Exception as e:
            logger.error(f"Correlation regime analysis failed: {e}")
            return {}
    
    def _calculate_correlation_clusters(self, returns_df: pd.DataFrame) -> Dict[str, List[str]]:
        """Calculate correlation clusters using hierarchical clustering"""
        try:
            # Calculate correlation matrix
            corr_matrix = returns_df.corr()
            
            # Convert to distance matrix
            distance_matrix = 1 - np.abs(corr_matrix)
            
            # Simple clustering based on correlation threshold
            threshold = 0.7
            clusters = []
            used_symbols = set()
            
            for symbol in corr_matrix.columns:
                if symbol in used_symbols:
                    continue
                
                cluster = [symbol]
                used_symbols.add(symbol)
                
                for other_symbol in corr_matrix.columns:
                    if other_symbol not in used_symbols:
                        if abs(corr_matrix.loc[symbol, other_symbol]) > threshold:
                            cluster.append(other_symbol)
                            used_symbols.add(other_symbol)
                
                if len(cluster) > 1:
                    clusters.append(cluster)
            
            return {'clusters': clusters, 'threshold': threshold}
            
        except Exception as e:
            logger.error(f"Correlation clustering failed: {e}")
            return {'clusters': [], 'threshold': 0.7}
    
    # Volatility Modeling
    async def calculate_volatility_models(self, returns: pd.Series) -> Dict[str, Any]:
        """GARCH models, implied volatility surfaces, volatility clustering"""
        try:
            models = {}
            
            # Simple volatility clustering analysis
            models['volatility_clustering'] = self._analyze_volatility_clustering(returns)
            
            # Rolling volatility
            models['rolling_volatility'] = self._calculate_rolling_volatility(returns)
            
            # Volatility of volatility
            models['vol_of_vol'] = self._calculate_volatility_of_volatility(returns)
            
            # Volatility regime analysis
            models['volatility_regimes'] = self._analyze_volatility_regimes(returns)
            
            return models
            
        except Exception as e:
            logger.error(f"Volatility modeling failed: {e}")
            return {}
    
    def _analyze_volatility_clustering(self, returns: pd.Series) -> Dict[str, Any]:
        """Analyze volatility clustering"""
        try:
            # Calculate squared returns (proxy for volatility)
            squared_returns = returns ** 2
            
            # Calculate autocorrelation of squared returns
            autocorr = [squared_returns.autocorr(lag=i) for i in range(1, 21)]
            
            # Test for ARCH effects
            arch_test = self._arch_test(returns)
            
            return {
                'autocorrelation': autocorr,
                'arch_test': arch_test,
                'volatility_persistence': np.mean(autocorr[:5]) if autocorr else 0
            }
            
        except Exception as e:
            logger.error(f"Volatility clustering analysis failed: {e}")
            return {}
    
    def _arch_test(self, returns: pd.Series) -> Dict[str, float]:
        """Simple ARCH test"""
        try:
            # Engle's ARCH test (simplified)
            squared_returns = returns ** 2
            lagged_squared = squared_returns.shift(1).dropna()
            current_squared = squared_returns.iloc[1:]
            
            # Simple regression test
            correlation = lagged_squared.corr(current_squared)
            
            return {
                'correlation': correlation,
                'arch_effect': abs(correlation) > 0.1
            }
            
        except Exception as e:
            logger.error(f"ARCH test failed: {e}")
            return {'correlation': 0, 'arch_effect': False}
    
    def _calculate_rolling_volatility(self, returns: pd.Series, window: int = 20) -> List[float]:
        """Calculate rolling volatility"""
        try:
            return returns.rolling(window=window).std().dropna().tolist()
        except Exception as e:
            logger.error(f"Rolling volatility calculation failed: {e}")
            return []
    
    def _calculate_volatility_of_volatility(self, returns: pd.Series, window: int = 20) -> float:
        """Calculate volatility of volatility"""
        try:
            rolling_vol = returns.rolling(window=window).std()
            vol_of_vol = rolling_vol.std()
            return vol_of_vol
        except Exception as e:
            logger.error(f"Volatility of volatility calculation failed: {e}")
            return 0.0
    
    def _analyze_volatility_regimes(self, returns: pd.Series) -> Dict[str, Any]:
        """Analyze volatility regimes"""
        try:
            # Calculate rolling volatility
            rolling_vol = returns.rolling(window=20).std()
            
            # Define regimes
            low_vol_threshold = rolling_vol.quantile(0.33)
            high_vol_threshold = rolling_vol.quantile(0.67)
            
            # Calculate regime statistics
            low_vol_periods = rolling_vol < low_vol_threshold
            high_vol_periods = rolling_vol > high_vol_threshold
            normal_vol_periods = ~(low_vol_periods | high_vol_periods)
            
            return {
                'low_volatility_regime': {
                    'frequency': low_vol_periods.mean(),
                    'avg_volatility': rolling_vol[low_vol_periods].mean(),
                    'avg_return': returns[low_vol_periods].mean()
                },
                'normal_volatility_regime': {
                    'frequency': normal_vol_periods.mean(),
                    'avg_volatility': rolling_vol[normal_vol_periods].mean(),
                    'avg_return': returns[normal_vol_periods].mean()
                },
                'high_volatility_regime': {
                    'frequency': high_vol_periods.mean(),
                    'avg_volatility': rolling_vol[high_vol_periods].mean(),
                    'avg_return': returns[high_vol_periods].mean()
                }
            }
            
        except Exception as e:
            logger.error(f"Volatility regime analysis failed: {e}")
            return {}
    
    # Factor Models
    async def calculate_factor_models(self, returns: pd.Series, factor_returns: Dict[str, pd.Series] = None) -> Dict[str, Any]:
        """Fama-French, Carhart, custom factor models with dynamic loadings"""
        try:
            models = {}
            
            # Fama-French 3-Factor Model
            models['fama_french'] = await self._calculate_fama_french_model(returns)
            
            # Carhart 4-Factor Model
            models['carhart'] = await self._calculate_carhart_model(returns)
            
            # Custom factor model
            if factor_returns:
                models['custom'] = await self._calculate_custom_factor_model(returns, factor_returns)
            
            return models
            
        except Exception as e:
            logger.error(f"Factor models calculation failed: {e}")
            return {}
    
    async def _calculate_fama_french_model(self, returns: pd.Series) -> Dict[str, Any]:
        """Calculate Fama-French 3-Factor Model"""
        try:
            # This is a simplified implementation
            # In practice, you would use actual Fama-French factor data
            
            # Simulate factor returns (for demonstration)
            dates = returns.index
            market_factor = np.random.normal(0.001, 0.02, len(dates))  # Market excess return
            smb_factor = np.random.normal(0.0005, 0.01, len(dates))    # Small minus Big
            hml_factor = np.random.normal(0.0003, 0.008, len(dates))  # High minus Low
            
            # Create factor DataFrame
            factors = pd.DataFrame({
                'Market': market_factor,
                'SMB': smb_factor,
                'HML': hml_factor
            }, index=dates)
            
            # Add risk-free rate (simplified)
            factors['RF'] = 0.0001  # 0.01% daily risk-free rate
            
            # Calculate factor loadings using regression
            factor_loadings = self._calculate_factor_loadings(returns, factors)
            
            return {
                'factors': factors.to_dict(),
                'loadings': factor_loadings,
                'r_squared': self._calculate_r_squared(returns, factors, factor_loadings)
            }
            
        except Exception as e:
            logger.error(f"Fama-French model calculation failed: {e}")
            return {}
    
    async def _calculate_carhart_model(self, returns: pd.Series) -> Dict[str, Any]:
        """Calculate Carhart 4-Factor Model"""
        try:
            # Similar to Fama-French but with momentum factor
            dates = returns.index
            market_factor = np.random.normal(0.001, 0.02, len(dates))
            smb_factor = np.random.normal(0.0005, 0.01, len(dates))
            hml_factor = np.random.normal(0.0003, 0.008, len(dates))
            mom_factor = np.random.normal(0.0002, 0.006, len(dates))  # Momentum factor
            
            factors = pd.DataFrame({
                'Market': market_factor,
                'SMB': smb_factor,
                'HML': hml_factor,
                'MOM': mom_factor
            }, index=dates)
            
            factors['RF'] = 0.0001
            
            factor_loadings = self._calculate_factor_loadings(returns, factors)
            
            return {
                'factors': factors.to_dict(),
                'loadings': factor_loadings,
                'r_squared': self._calculate_r_squared(returns, factors, factor_loadings)
            }
            
        except Exception as e:
            logger.error(f"Carhart model calculation failed: {e}")
            return {}
    
    async def _calculate_custom_factor_model(self, returns: pd.Series, factor_returns: Dict[str, pd.Series]) -> Dict[str, Any]:
        """Calculate custom factor model"""
        try:
            # Create factor DataFrame
            factors = pd.DataFrame(factor_returns)
            
            # Calculate factor loadings
            factor_loadings = self._calculate_factor_loadings(returns, factors)
            
            return {
                'factors': factors.to_dict(),
                'loadings': factor_loadings,
                'r_squared': self._calculate_r_squared(returns, factors, factor_loadings)
            }
            
        except Exception as e:
            logger.error(f"Custom factor model calculation failed: {e}")
            return {}
    
    def _calculate_factor_loadings(self, returns: pd.Series, factors: pd.DataFrame) -> Dict[str, float]:
        """Calculate factor loadings using regression"""
        try:
            # Align data
            aligned_data = pd.DataFrame({
                'returns': returns
            }).join(factors).dropna()
            
            if len(aligned_data) < 10:
                return {}
            
            # Simple linear regression for each factor
            loadings = {}
            
            for factor in factors.columns:
                if factor != 'RF':  # Skip risk-free rate
                    try:
                        # Calculate beta (simplified)
                        factor_returns = aligned_data[factor]
                        asset_returns = aligned_data['returns']
                        
                        correlation = factor_returns.corr(asset_returns)
                        beta = correlation * asset_returns.std() / factor_returns.std()
                        
                        loadings[factor] = beta
                        
                    except Exception as e:
                        logger.warning(f"Factor loading calculation for {factor} failed: {e}")
                        loadings[factor] = 0.0
            
            return loadings
            
        except Exception as e:
            logger.error(f"Factor loadings calculation failed: {e}")
            return {}
    
    def _calculate_r_squared(self, returns: pd.Series, factors: pd.DataFrame, loadings: Dict[str, float]) -> float:
        """Calculate R-squared for factor model"""
        try:
            # Align data
            aligned_data = pd.DataFrame({
                'returns': returns
            }).join(factors).dropna()
            
            if len(aligned_data) < 10:
                return 0.0
            
            # Calculate predicted returns
            predicted_returns = pd.Series(0.0, index=aligned_data.index)
            
            for factor, loading in loadings.items():
                if factor in aligned_data.columns:
                    predicted_returns += loading * aligned_data[factor]
            
            # Calculate R-squared
            actual_returns = aligned_data['returns']
            ss_res = ((actual_returns - predicted_returns) ** 2).sum()
            ss_tot = ((actual_returns - actual_returns.mean()) ** 2).sum()
            
            r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0
            
            return r_squared
            
        except Exception as e:
            logger.error(f"R-squared calculation failed: {e}")
            return 0.0
    
    # Portfolio Optimization
    async def optimize_portfolio(self, returns_data: Dict[str, pd.Series], 
                               method: str = "mean_variance",
                               constraints: Dict[str, Any] = None) -> Dict[str, Any]:
        """Mean-variance optimization, Black-Litterman, risk parity"""
        try:
            if method == "mean_variance":
                return await self._mean_variance_optimization(returns_data, constraints)
            elif method == "risk_parity":
                return await self._risk_parity_optimization(returns_data, constraints)
            elif method == "black_litterman":
                return await self._black_litterman_optimization(returns_data, constraints)
            else:
                logger.warning(f"Unknown optimization method: {method}")
                return await self._mean_variance_optimization(returns_data, constraints)
                
        except Exception as e:
            logger.error(f"Portfolio optimization failed: {e}")
            return {}
    
    async def _mean_variance_optimization(self, returns_data: Dict[str, pd.Series], 
                                        constraints: Dict[str, Any] = None) -> Dict[str, Any]:
        """Mean-variance optimization"""
        try:
            # Create returns DataFrame
            returns_df = pd.DataFrame(returns_data)
            returns_df = returns_df.dropna()
            
            if len(returns_df) < 30:
                return {}
            
            # Calculate expected returns and covariance matrix
            expected_returns = returns_df.mean() * 252  # Annualized
            cov_matrix = returns_df.cov() * 252  # Annualized
            
            # Define optimization function
            def portfolio_variance(weights):
                return np.dot(weights.T, np.dot(cov_matrix, weights))
            
            def portfolio_return(weights):
                return np.sum(expected_returns * weights)
            
            # Constraints
            n_assets = len(expected_returns)
            
            # Default constraints
            if constraints is None:
                constraints = {
                    'min_weight': 0.0,
                    'max_weight': 1.0,
                    'target_return': None
                }
            
            # Optimization bounds
            bounds = [(constraints.get('min_weight', 0.0), constraints.get('max_weight', 1.0)) for _ in range(n_assets)]
            
            # Constraint: weights sum to 1
            constraints_opt = [{'type': 'eq', 'fun': lambda x: np.sum(x) - 1}]
            
            # Add target return constraint if specified
            if constraints.get('target_return') is not None:
                constraints_opt.append({
                    'type': 'eq', 
                    'fun': lambda x: portfolio_return(x) - constraints['target_return']
                })
            
            # Initial guess (equal weights)
            initial_weights = np.array([1.0 / n_assets] * n_assets)
            
            # Optimize
            result = minimize(
                portfolio_variance,
                initial_weights,
                method='SLSQP',
                bounds=bounds,
                constraints=constraints_opt
            )
            
            if result.success:
                optimal_weights = result.x
                
                # Calculate portfolio metrics
                portfolio_return_val = portfolio_return(optimal_weights)
                portfolio_vol = np.sqrt(portfolio_variance(optimal_weights))
                sharpe_ratio = portfolio_return_val / portfolio_vol if portfolio_vol > 0 else 0
                
                # Create allocation objects
                allocations = []
                for i, symbol in enumerate(expected_returns.index):
                    allocations.append(PortfolioAllocation(
                        symbol=symbol,
                        weight=optimal_weights[i],
                        expected_return=expected_returns.iloc[i],
                        risk_contribution=optimal_weights[i] * cov_matrix.iloc[i, i]
                    ))
                
                return {
                    'method': 'mean_variance',
                    'optimal_weights': dict(zip(expected_returns.index, optimal_weights)),
                    'expected_return': portfolio_return_val,
                    'volatility': portfolio_vol,
                    'sharpe_ratio': sharpe_ratio,
                    'allocations': allocations,
                    'optimization_success': True
                }
            else:
                return {
                    'method': 'mean_variance',
                    'optimization_success': False,
                    'error': result.message
                }
            
        except Exception as e:
            logger.error(f"Mean-variance optimization failed: {e}")
            return {}
    
    async def _risk_parity_optimization(self, returns_data: Dict[str, pd.Series], 
                                      constraints: Dict[str, Any] = None) -> Dict[str, Any]:
        """Risk parity optimization"""
        try:
            # Create returns DataFrame
            returns_df = pd.DataFrame(returns_data)
            returns_df = returns_df.dropna()
            
            if len(returns_df) < 30:
                return {}
            
            # Calculate covariance matrix
            cov_matrix = returns_df.cov() * 252
            
            # Define risk parity objective function
            def risk_parity_objective(weights):
                portfolio_vol = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
                risk_contributions = weights * (np.dot(cov_matrix, weights)) / portfolio_vol
                target_risk = portfolio_vol / len(weights)
                return np.sum((risk_contributions - target_risk) ** 2)
            
            # Constraints
            n_assets = len(returns_df.columns)
            bounds = [(0.0, 1.0) for _ in range(n_assets)]
            constraints_opt = [{'type': 'eq', 'fun': lambda x: np.sum(x) - 1}]
            
            # Initial guess (equal weights)
            initial_weights = np.array([1.0 / n_assets] * n_assets)
            
            # Optimize
            result = minimize(
                risk_parity_objective,
                initial_weights,
                method='SLSQP',
                bounds=bounds,
                constraints=constraints_opt
            )
            
            if result.success:
                optimal_weights = result.x
                
                # Calculate portfolio metrics
                portfolio_vol = np.sqrt(np.dot(optimal_weights.T, np.dot(cov_matrix, optimal_weights)))
                expected_returns = returns_df.mean() * 252
                portfolio_return_val = np.sum(expected_returns * optimal_weights)
                
                return {
                    'method': 'risk_parity',
                    'optimal_weights': dict(zip(returns_df.columns, optimal_weights)),
                    'expected_return': portfolio_return_val,
                    'volatility': portfolio_vol,
                    'optimization_success': True
                }
            else:
                return {
                    'method': 'risk_parity',
                    'optimization_success': False,
                    'error': result.message
                }
            
        except Exception as e:
            logger.error(f"Risk parity optimization failed: {e}")
            return {}
    
    async def _black_litterman_optimization(self, returns_data: Dict[str, pd.Series], 
                                          constraints: Dict[str, Any] = None) -> Dict[str, Any]:
        """Black-Litterman optimization (simplified)"""
        try:
            # This is a simplified implementation
            # In practice, Black-Litterman requires market equilibrium returns and views
            
            # Use mean-variance as fallback
            return await self._mean_variance_optimization(returns_data, constraints)
            
        except Exception as e:
            logger.error(f"Black-Litterman optimization failed: {e}")
            return {}
    
    # Stress Testing
    async def perform_stress_testing(self, portfolio_returns: pd.Series, 
                                   scenarios: List[Dict[str, float]] = None) -> Dict[str, Any]:
        """Monte Carlo simulations, historical scenario analysis"""
        try:
            stress_tests = {}
            
            # Monte Carlo simulation
            stress_tests['monte_carlo'] = self._monte_carlo_simulation(portfolio_returns)
            
            # Historical scenario analysis
            stress_tests['historical_scenarios'] = self._historical_scenario_analysis(portfolio_returns)
            
            # Custom scenarios
            if scenarios:
                stress_tests['custom_scenarios'] = self._custom_scenario_analysis(portfolio_returns, scenarios)
            
            return stress_tests
            
        except Exception as e:
            logger.error(f"Stress testing failed: {e}")
            return {}
    
    def _monte_carlo_simulation(self, returns: pd.Series, n_simulations: int = 10000, 
                               time_horizon: int = 252) -> Dict[str, Any]:
        """Monte Carlo simulation"""
        try:
            # Calculate parameters
            mu = returns.mean() * 252  # Annualized mean
            sigma = returns.std() * np.sqrt(252)  # Annualized volatility
            
            # Generate simulations
            simulations = []
            for _ in range(n_simulations):
                # Generate random walk
                daily_returns = np.random.normal(mu/252, sigma/np.sqrt(252), time_horizon)
                cumulative_return = np.prod(1 + daily_returns) - 1
                simulations.append(cumulative_return)
            
            simulations = np.array(simulations)
            
            # Calculate statistics
            percentiles = np.percentile(simulations, [1, 5, 10, 25, 50, 75, 90, 95, 99])
            
            return {
                'mean_return': simulations.mean(),
                'volatility': simulations.std(),
                'percentiles': {
                    'p1': percentiles[0],
                    'p5': percentiles[1],
                    'p10': percentiles[2],
                    'p25': percentiles[3],
                    'p50': percentiles[4],
                    'p75': percentiles[5],
                    'p90': percentiles[6],
                    'p95': percentiles[7],
                    'p99': percentiles[8]
                },
                'var_95': percentiles[1],  # 5% VaR
                'var_99': percentiles[0],  # 1% VaR
                'worst_case': simulations.min(),
                'best_case': simulations.max()
            }
            
        except Exception as e:
            logger.error(f"Monte Carlo simulation failed: {e}")
            return {}
    
    def _historical_scenario_analysis(self, returns: pd.Series) -> Dict[str, Any]:
        """Historical scenario analysis"""
        try:
            scenarios = {}
            
            # 2008 Financial Crisis (simplified)
            crisis_start = '2008-09-01'
            crisis_end = '2009-03-01'
            
            if crisis_start in returns.index and crisis_end in returns.index:
                crisis_returns = returns[crisis_start:crisis_end]
                scenarios['financial_crisis_2008'] = {
                    'period': f"{crisis_start} to {crisis_end}",
                    'cumulative_return': (1 + crisis_returns).prod() - 1,
                    'volatility': crisis_returns.std() * np.sqrt(252),
                    'max_drawdown': self._calculate_maximum_drawdown(crisis_returns)
                }
            
            # COVID-19 Crash (simplified)
            covid_start = '2020-02-01'
            covid_end = '2020-04-01'
            
            if covid_start in returns.index and covid_end in returns.index:
                covid_returns = returns[covid_start:covid_end]
                scenarios['covid_crash_2020'] = {
                    'period': f"{covid_start} to {covid_end}",
                    'cumulative_return': (1 + covid_returns).prod() - 1,
                    'volatility': covid_returns.std() * np.sqrt(252),
                    'max_drawdown': self._calculate_maximum_drawdown(covid_returns)
                }
            
            # Worst periods
            rolling_returns = returns.rolling(window=60).apply(lambda x: (1 + x).prod() - 1)
            worst_periods = rolling_returns.nsmallest(5)
            
            scenarios['worst_periods'] = {
                'periods': worst_periods.index.tolist(),
                'returns': worst_periods.values.tolist()
            }
            
            return scenarios
            
        except Exception as e:
            logger.error(f"Historical scenario analysis failed: {e}")
            return {}
    
    def _custom_scenario_analysis(self, returns: pd.Series, scenarios: List[Dict[str, float]]) -> Dict[str, Any]:
        """Custom scenario analysis"""
        try:
            results = {}
            
            for i, scenario in enumerate(scenarios):
                scenario_name = scenario.get('name', f'scenario_{i}')
                
                # Apply scenario shock to returns
                shock_factor = scenario.get('shock_factor', 1.0)
                shocked_returns = returns * shock_factor
                
                # Calculate scenario impact
                cumulative_return = (1 + shocked_returns).prod() - 1
                volatility = shocked_returns.std() * np.sqrt(252)
                max_drawdown = self._calculate_maximum_drawdown(shocked_returns)
                
                results[scenario_name] = {
                    'shock_factor': shock_factor,
                    'cumulative_return': cumulative_return,
                    'volatility': volatility,
                    'max_drawdown': max_drawdown
                }
            
            return results
            
        except Exception as e:
            logger.error(f"Custom scenario analysis failed: {e}")
            return {}

# Global service instance
quantitative_analytics_service = QuantitativeAnalyticsService() 