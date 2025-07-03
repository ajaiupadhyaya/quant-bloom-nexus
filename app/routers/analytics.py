from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
from datetime import datetime
import pandas as pd
import numpy as np
import logging

from ..services.quantitative_engine import quantitative_engine, OptionType, GreeksResult, RiskMetrics, PerformanceMetrics
from ..services.technical_analysis import technical_analysis_engine, TechnicalSignal, PatternRecognition
from ..services.market_data_service import get_market_data

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/analytics", tags=["Analytics"])

# =================== REQUEST MODELS ===================

class OptionPricingRequest(BaseModel):
    underlying_price: float
    strike_price: float
    time_to_expiry: float  # in years
    risk_free_rate: float
    volatility: float
    option_type: str  # "call" or "put"

class PortfolioOptimizationRequest(BaseModel):
    symbols: List[str]
    target_return: Optional[float] = None
    optimization_method: str = "mean_variance"  # mean_variance, black_litterman
    market_caps: Optional[Dict[str, float]] = None
    views: Optional[Dict[str, float]] = None

class RiskAnalysisRequest(BaseModel):
    symbols: List[str]
    weights: List[float]
    confidence_levels: List[float] = [0.95, 0.99]

class TechnicalAnalysisRequest(BaseModel):
    symbol: str
    indicators: List[str] = ["all"]
    period: str = "1y"

class BacktestRequest(BaseModel):
    symbols: List[str]
    strategy_type: str
    start_date: str
    end_date: str
    initial_capital: float = 100000
    parameters: Dict[str, Any] = {}

class TechnicalIndicators(BaseModel):
    symbol: str
    rsi: float
    macd: str
    sma_20: float
    bollinger_bands: str
    volume_trend: str
    timestamp: str

# =================== OPTIONS ANALYTICS ===================

@router.post("/options/price")
async def calculate_option_price(request: OptionPricingRequest):
    """Calculate option price using Black-Scholes model"""
    try:
        option_type = OptionType.CALL if request.option_type.lower() == "call" else OptionType.PUT
        
        # Black-Scholes price
        bs_price = quantitative_engine.black_scholes_price(
            S=request.underlying_price,
            K=request.strike_price,
            T=request.time_to_expiry,
            r=request.risk_free_rate,
            sigma=request.volatility,
            option_type=option_type
        )
        
        # Greeks calculation
        greeks = quantitative_engine.calculate_greeks(
            S=request.underlying_price,
            K=request.strike_price,
            T=request.time_to_expiry,
            r=request.risk_free_rate,
            sigma=request.volatility,
            option_type=option_type
        )
        
        # Binomial tree price for comparison
        binomial_price = quantitative_engine.binomial_tree_price(
            S=request.underlying_price,
            K=request.strike_price,
            T=request.time_to_expiry,
            r=request.risk_free_rate,
            sigma=request.volatility,
            n_steps=100,
            option_type=option_type
        )
        
        return {
            "black_scholes_price": bs_price,
            "binomial_price": binomial_price,
            "greeks": {
                "delta": greeks.delta,
                "gamma": greeks.gamma,
                "theta": greeks.theta,
                "vega": greeks.vega,
                "rho": greeks.rho
            },
            "pricing_parameters": {
                "underlying_price": request.underlying_price,
                "strike_price": request.strike_price,
                "time_to_expiry": request.time_to_expiry,
                "risk_free_rate": request.risk_free_rate,
                "volatility": request.volatility,
                "option_type": request.option_type
            },
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Option pricing failed: {e}")
        raise HTTPException(status_code=500, detail=f"Option pricing calculation failed: {str(e)}")

@router.get("/options/implied-volatility")
async def calculate_implied_volatility(
    market_price: float = Query(..., description="Market price of the option"),
    underlying_price: float = Query(..., description="Current underlying price"),
    strike_price: float = Query(..., description="Strike price"),
    time_to_expiry: float = Query(..., description="Time to expiry in years"),
    risk_free_rate: float = Query(0.02, description="Risk-free rate"),
    option_type: str = Query(..., description="call or put")
):
    """Calculate implied volatility from market price"""
    try:
        opt_type = OptionType.CALL if option_type.lower() == "call" else OptionType.PUT
        
        implied_vol = quantitative_engine.implied_volatility(
            market_price=market_price,
            S=underlying_price,
            K=strike_price,
            T=time_to_expiry,
            r=risk_free_rate,
            option_type=opt_type
        )
        
        return {
            "implied_volatility": implied_vol,
            "implied_volatility_percent": implied_vol * 100,
            "market_price": market_price,
            "calculated_price": quantitative_engine.black_scholes_price(
                underlying_price, strike_price, time_to_expiry, risk_free_rate, implied_vol, opt_type
            ),
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Implied volatility calculation failed: {e}")
        raise HTTPException(status_code=500, detail=f"Implied volatility calculation failed: {str(e)}")

# =================== PORTFOLIO ANALYTICS ===================

@router.post("/portfolio/optimize")
async def optimize_portfolio(request: PortfolioOptimizationRequest):
    """Optimize portfolio using modern portfolio theory"""
    try:
        # Get historical data for all symbols
        returns_data = {}
        for symbol in request.symbols:
            data = await get_market_data(symbol, period="2y")
            if data:
                df = pd.DataFrame(data)
                df['date'] = pd.to_datetime(df['timestamp'], unit='ms')
                df = df.set_index('date').sort_index()
                returns = df['close'].pct_change().dropna()
                returns_data[symbol] = returns
        
        if not returns_data:
            raise HTTPException(status_code=400, detail="No data available for symbols")
        
        # Create returns DataFrame
        returns_df = pd.DataFrame(returns_data)
        returns_df = returns_df.dropna()
        
        # Optimize based on method
        if request.optimization_method == "mean_variance":
            weights = quantitative_engine.mean_variance_optimization(returns_df, request.target_return)
        elif request.optimization_method == "black_litterman":
            if not request.market_caps:
                raise HTTPException(status_code=400, detail="Market caps required for Black-Litterman")
            weights = quantitative_engine.black_litterman_optimization(
                returns_df, request.market_caps, request.views
            )
        else:
            raise HTTPException(status_code=400, detail="Invalid optimization method")
        
        # Calculate portfolio metrics
        portfolio_returns = sum(weights[symbol] * returns_df[symbol] for symbol in weights.keys())
        risk_metrics = quantitative_engine.calculate_risk_metrics(portfolio_returns.values)
        
        return {
            "optimal_weights": weights,
            "portfolio_metrics": {
                "expected_return": portfolio_returns.mean() * 252,
                "volatility": portfolio_returns.std() * np.sqrt(252),
                "sharpe_ratio": risk_metrics.sharpe_ratio,
                "max_drawdown": risk_metrics.max_drawdown,
                "var_95": risk_metrics.var_95,
                "var_99": risk_metrics.var_99
            },
            "optimization_method": request.optimization_method,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Portfolio optimization failed: {e}")
        raise HTTPException(status_code=500, detail=f"Portfolio optimization failed: {str(e)}")

@router.post("/portfolio/risk-analysis")
async def analyze_portfolio_risk(request: RiskAnalysisRequest):
    """Comprehensive portfolio risk analysis"""
    try:
        if len(request.symbols) != len(request.weights):
            raise HTTPException(status_code=400, detail="Symbols and weights must have same length")
        
        if abs(sum(request.weights) - 1.0) > 0.01:
            raise HTTPException(status_code=400, detail="Weights must sum to 1.0")
        
        # Get historical data
        returns_data = {}
        for symbol in request.symbols:
            data = await get_market_data(symbol, period="2y")
            if data:
                df = pd.DataFrame(data)
                df['date'] = pd.to_datetime(df['timestamp'], unit='ms')
                df = df.set_index('date').sort_index()
                returns = df['close'].pct_change().dropna()
                returns_data[symbol] = returns
        
        # Calculate portfolio returns
        returns_df = pd.DataFrame(returns_data).dropna()
        portfolio_returns = sum(request.weights[i] * returns_df[request.symbols[i]] 
                               for i in range(len(request.symbols)))
        
        # Risk metrics
        risk_metrics = quantitative_engine.calculate_risk_metrics(portfolio_returns.values)
        
        # Monte Carlo VaR
        mc_var_95 = quantitative_engine.monte_carlo_var(portfolio_returns.values, confidence_level=0.95)
        mc_var_99 = quantitative_engine.monte_carlo_var(portfolio_returns.values, confidence_level=0.99)
        
        # Component analysis
        component_analysis = {}
        for i, symbol in enumerate(request.symbols):
            asset_returns = returns_df[symbol]
            component_analysis[symbol] = {
                "weight": request.weights[i],
                "volatility": asset_returns.std() * np.sqrt(252),
                "contribution_to_risk": request.weights[i] * asset_returns.std() * np.sqrt(252),
                "beta": quantitative_engine.calculate_beta(asset_returns, portfolio_returns)
            }
        
        return {
            "portfolio_risk_metrics": {
                "var_95_historical": risk_metrics.var_95,
                "var_99_historical": risk_metrics.var_99,
                "var_95_monte_carlo": mc_var_95,
                "var_99_monte_carlo": mc_var_99,
                "cvar_95": risk_metrics.cvar_95,
                "cvar_99": risk_metrics.cvar_99,
                "max_drawdown": risk_metrics.max_drawdown,
                "volatility": risk_metrics.volatility,
                "sharpe_ratio": risk_metrics.sharpe_ratio,
                "sortino_ratio": risk_metrics.sortino_ratio,
                "skewness": risk_metrics.skewness,
                "kurtosis": risk_metrics.kurtosis
            },
            "component_analysis": component_analysis,
            "correlation_matrix": returns_df[request.symbols].corr().to_dict(),
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Portfolio risk analysis failed: {e}")
        raise HTTPException(status_code=500, detail=f"Portfolio risk analysis failed: {str(e)}")

# =================== TECHNICAL ANALYSIS ===================

@router.post("/technical/indicators")
async def calculate_technical_indicators(request: TechnicalAnalysisRequest):
    """Calculate comprehensive technical indicators"""
    try:
        # Get market data
        data = await get_market_data(request.symbol, period=request.period)
        if not data:
            raise HTTPException(status_code=404, detail=f"No data available for {request.symbol}")
        
        df = pd.DataFrame(data)
        df['date'] = pd.to_datetime(df['timestamp'], unit='ms')
        df = df.set_index('date').sort_index()
        
        indicators = {}
        
        # Calculate requested indicators
        if "all" in request.indicators or "sma" in request.indicators:
            indicators["sma_20"] = technical_analysis_engine.simple_moving_average(df['close'], 20).tail(50).to_dict()
            indicators["sma_50"] = technical_analysis_engine.simple_moving_average(df['close'], 50).tail(50).to_dict()
        
        if "all" in request.indicators or "ema" in request.indicators:
            indicators["ema_12"] = technical_analysis_engine.exponential_moving_average(df['close'], 12).tail(50).to_dict()
            indicators["ema_26"] = technical_analysis_engine.exponential_moving_average(df['close'], 26).tail(50).to_dict()
        
        if "all" in request.indicators or "rsi" in request.indicators:
            indicators["rsi"] = technical_analysis_engine.rsi(df['close']).tail(50).to_dict()
        
        if "all" in request.indicators or "macd" in request.indicators:
            macd_data = technical_analysis_engine.macd(df['close'])
            indicators["macd"] = {
                "macd": macd_data['macd'].tail(50).to_dict(),
                "signal": macd_data['signal'].tail(50).to_dict(),
                "histogram": macd_data['histogram'].tail(50).to_dict()
            }
        
        if "all" in request.indicators or "bollinger" in request.indicators:
            bb_data = technical_analysis_engine.bollinger_bands(df['close'])
            indicators["bollinger_bands"] = {
                "upper": bb_data['upper'].tail(50).to_dict(),
                "middle": bb_data['middle'].tail(50).to_dict(),
                "lower": bb_data['lower'].tail(50).to_dict(),
                "bandwidth": bb_data['bandwidth'].tail(50).to_dict()
            }
        
        if "all" in request.indicators or "stochastic" in request.indicators:
            stoch_data = technical_analysis_engine.stochastic_oscillator(df['high'], df['low'], df['close'])
            indicators["stochastic"] = {
                "k_percent": stoch_data['%K'].tail(50).to_dict(),
                "d_percent": stoch_data['%D'].tail(50).to_dict()
            }
        
        if "all" in request.indicators or "atr" in request.indicators:
            indicators["atr"] = technical_analysis_engine.average_true_range(
                df['high'], df['low'], df['close']
            ).tail(50).to_dict()
        
        # Generate signals
        signals = technical_analysis_engine.generate_comprehensive_signals(
            df['high'], df['low'], df['close'], df['volume']
        )
        
        # Pattern recognition
        patterns = []
        hs_pattern = technical_analysis_engine.detect_head_and_shoulders(df['close'])
        if hs_pattern:
            patterns.append(hs_pattern.__dict__)
        
        dt_pattern = technical_analysis_engine.detect_double_top_bottom(df['close'])
        if dt_pattern:
            patterns.append(dt_pattern.__dict__)
        
        # Support and resistance
        support_resistance = technical_analysis_engine.detect_support_resistance(df['close'])
        
        return {
            "symbol": request.symbol,
            "indicators": indicators,
            "signals": [signal.__dict__ for signal in signals],
            "patterns": patterns,
            "support_resistance": support_resistance,
            "current_price": float(df['close'].iloc[-1]),
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Technical analysis failed for {request.symbol}: {e}")
        raise HTTPException(status_code=500, detail=f"Technical analysis failed: {str(e)}")

# =================== PERFORMANCE ANALYTICS ===================

@router.get("/performance/metrics")
async def calculate_performance_metrics(
    symbol: str = Query(..., description="Symbol to analyze"),
    benchmark: str = Query("SPY", description="Benchmark symbol"),
    period: str = Query("1y", description="Analysis period")
):
    """Calculate comprehensive performance metrics"""
    try:
        # Get data for symbol and benchmark
        symbol_data = await get_market_data(symbol, period=period)
        benchmark_data = await get_market_data(benchmark, period=period)
        
        if not symbol_data or not benchmark_data:
            raise HTTPException(status_code=404, detail="Data not available")
        
        # Process data
        symbol_df = pd.DataFrame(symbol_data)
        symbol_df['date'] = pd.to_datetime(symbol_df['timestamp'], unit='ms')
        symbol_df = symbol_df.set_index('date').sort_index()
        symbol_returns = symbol_df['close'].pct_change().dropna()
        
        benchmark_df = pd.DataFrame(benchmark_data)
        benchmark_df['date'] = pd.to_datetime(benchmark_df['timestamp'], unit='ms')
        benchmark_df = benchmark_df.set_index('date').sort_index()
        benchmark_returns = benchmark_df['close'].pct_change().dropna()
        
        # Align dates
        common_dates = symbol_returns.index.intersection(benchmark_returns.index)
        symbol_returns = symbol_returns.loc[common_dates]
        benchmark_returns = benchmark_returns.loc[common_dates]
        
        # Calculate performance metrics
        performance_metrics = quantitative_engine.calculate_performance_metrics(symbol_returns, benchmark_returns)
        
        # Additional calculations
        correlation = symbol_returns.corr(benchmark_returns)
        tracking_error = (symbol_returns - benchmark_returns).std() * np.sqrt(252)
        
        return {
            "symbol": symbol,
            "benchmark": benchmark,
            "period": period,
            "performance_metrics": {
                "total_return": performance_metrics.total_return,
                "annualized_return": performance_metrics.annualized_return,
                "volatility": performance_metrics.volatility,
                "sharpe_ratio": performance_metrics.sharpe_ratio,
                "sortino_ratio": performance_metrics.sortino_ratio,
                "max_drawdown": performance_metrics.max_drawdown,
                "calmar_ratio": performance_metrics.calmar_ratio,
                "win_rate": performance_metrics.win_rate,
                "profit_factor": performance_metrics.profit_factor,
                "alpha": performance_metrics.alpha,
                "beta": performance_metrics.beta,
                "information_ratio": performance_metrics.information_ratio,
                "treynor_ratio": performance_metrics.treynor_ratio
            },
            "additional_metrics": {
                "correlation_with_benchmark": correlation,
                "tracking_error": tracking_error,
                "current_price": float(symbol_df['close'].iloc[-1]),
                "benchmark_price": float(benchmark_df['close'].iloc[-1])
            },
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Performance analysis failed: {e}")
        raise HTTPException(status_code=500, detail=f"Performance analysis failed: {str(e)}")

# =================== MARKET MICROSTRUCTURE ===================

@router.get("/microstructure/analysis")
async def analyze_market_microstructure(
    symbol: str = Query(..., description="Symbol to analyze"),
    period: str = Query("1d", description="Analysis period")
):
    """Analyze market microstructure"""
    try:
        data = await get_market_data(symbol, period=period, interval="5m")
        if not data:
            raise HTTPException(status_code=404, detail="Intraday data not available")
        
        df = pd.DataFrame(data)
        df['date'] = pd.to_datetime(df['timestamp'], unit='ms')
        df = df.set_index('date').sort_index()
        
        # Calculate microstructure metrics
        returns = df['close'].pct_change().dropna()
        
        # Bid-Ask Spread proxy (High-Low spread)
        spread = (df['high'] - df['low']) / df['close']
        avg_spread = spread.mean()
        
        # Price impact
        volume_buckets = pd.qcut(df['volume'], q=5, labels=['Very Low', 'Low', 'Medium', 'High', 'Very High'])
        price_impact = {}
        for bucket in volume_buckets.cat.categories:
            bucket_data = df[volume_buckets == bucket]
            if len(bucket_data) > 1:
                bucket_returns = bucket_data['close'].pct_change().abs().mean()
                price_impact[bucket] = bucket_returns
        
        # Volatility clustering
        squared_returns = returns ** 2
        volatility_autocorr = squared_returns.autocorr(lag=1)
        
        # Trading intensity
        volume_profile = df.groupby(df.index.hour)['volume'].mean().to_dict()
        
        return {
            "symbol": symbol,
            "microstructure_metrics": {
                "average_spread": float(avg_spread),
                "price_impact_by_volume": price_impact,
                "volatility_clustering": float(volatility_autocorr),
                "intraday_volume_profile": volume_profile,
                "total_volume": int(df['volume'].sum()),
                "average_trade_size": float(df['volume'].mean()),
                "price_volatility": float(returns.std())
            },
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Microstructure analysis failed: {e}")
        raise HTTPException(status_code=500, detail=f"Microstructure analysis failed: {str(e)}")

# =================== STATISTICAL TESTS ===================

@router.get("/statistics/tests")
async def run_statistical_tests(
    symbol: str = Query(..., description="Symbol to test"),
    period: str = Query("1y", description="Test period")
):
    """Run comprehensive statistical tests on returns"""
    try:
        data = await get_market_data(symbol, period=period)
        if not data:
            raise HTTPException(status_code=404, detail="Data not available")
        
        df = pd.DataFrame(data)
        df['date'] = pd.to_datetime(df['timestamp'], unit='ms')
        df = df.set_index('date').sort_index()
        returns = df['close'].pct_change().dropna()
        
        from scipy import stats
        
        # Normality tests
        shapiro_stat, shapiro_p = stats.shapiro(returns)
        jarque_bera_stat, jarque_bera_p = stats.jarque_bera(returns)
        
        # Stationarity test (Augmented Dickey-Fuller)
        from statsmodels.tsa.stattools import adfuller
        adf_result = adfuller(returns.dropna())
        
        # Autocorrelation test (Ljung-Box)
        from statsmodels.stats.diagnostic import acorr_ljungbox
        ljung_box = acorr_ljungbox(returns, lags=10, return_df=True)
        
        # ARCH effect test
        from statsmodels.stats.diagnostic import het_arch
        arch_test = het_arch(returns.dropna())
        
        return {
            "symbol": symbol,
            "statistical_tests": {
                "normality_tests": {
                    "shapiro_wilk": {
                        "statistic": float(shapiro_stat),
                        "p_value": float(shapiro_p),
                        "is_normal": shapiro_p > 0.05
                    },
                    "jarque_bera": {
                        "statistic": float(jarque_bera_stat),
                        "p_value": float(jarque_bera_p),
                        "is_normal": jarque_bera_p > 0.05
                    }
                },
                "stationarity_test": {
                    "adf_statistic": float(adf_result[0]),
                    "adf_p_value": float(adf_result[1]),
                    "is_stationary": adf_result[1] < 0.05
                },
                "autocorrelation_test": {
                    "ljung_box_statistic": float(ljung_box['lb_stat'].iloc[-1]),
                    "ljung_box_p_value": float(ljung_box['lb_pvalue'].iloc[-1]),
                    "has_autocorrelation": ljung_box['lb_pvalue'].iloc[-1] < 0.05
                },
                "arch_effect_test": {
                    "arch_statistic": float(arch_test[0]),
                    "arch_p_value": float(arch_test[1]),
                    "has_arch_effects": arch_test[1] < 0.05
                }
            },
            "descriptive_statistics": {
                "mean": float(returns.mean()),
                "std": float(returns.std()),
                "skewness": float(returns.skew()),
                "kurtosis": float(returns.kurtosis()),
                "min": float(returns.min()),
                "max": float(returns.max()),
                "count": int(len(returns))
            },
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Statistical tests failed: {e}")
        raise HTTPException(status_code=500, detail=f"Statistical tests failed: {str(e)}")

@router.get("/test")
async def test_endpoint():
    """Simple test endpoint"""
    return {"status": "working", "message": "Analytics router is operational"}

@router.get("/technical-indicators/{symbol}")
async def get_technical_indicators(symbol: str):
    """Get technical indicators for a symbol"""
    try:
        # Get historical data for calculations
        historical_data = await get_market_data(symbol.upper(), period="3m", interval="1d")
        
        if not historical_data or len(historical_data) < 20:
            # Return basic indicators if not enough data
            return {
                "rsi": 50.0,
                "macd": "NEUTRAL",
                "sma_20": 0.0,
                "bollinger_bands": "MIDDLE",
                "volume_trend": "STABLE"
            }
        
        # Convert to DataFrame for analysis
        df = pd.DataFrame(historical_data)
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df = df.sort_values('timestamp')
        
        # Simple technical indicator calculations
        prices = df['close'].values
        volumes = df['volume'].values
        
        # RSI calculation (simplified)
        gains = []
        losses = []
        for i in range(1, len(prices)):
            change = prices[i] - prices[i-1]
            if change > 0:
                gains.append(change)
                losses.append(0)
            else:
                gains.append(0)
                losses.append(abs(change))
        
        if len(gains) >= 14:
            avg_gain = sum(gains[-14:]) / 14
            avg_loss = sum(losses[-14:]) / 14
            rs = avg_gain / avg_loss if avg_loss > 0 else 0
            rsi = 100 - (100 / (1 + rs))
        else:
            rsi = 50.0
        
        # SMA calculation
        sma_20 = sum(prices[-20:]) / 20 if len(prices) >= 20 else prices[-1]
        
        # MACD signal (simplified)
        if len(prices) >= 26:
            ema_12 = sum(prices[-12:]) / 12
            ema_26 = sum(prices[-26:]) / 26
            macd_line = ema_12 - ema_26
            macd_signal = "BULLISH" if macd_line > 0 else "BEARISH"
        else:
            macd_signal = "NEUTRAL"
        
        # Bollinger Bands position
        if len(prices) >= 20:
            sma = sma_20
            std = np.std(prices[-20:])
            upper_band = sma + (2 * std)
            lower_band = sma - (2 * std)
            current_price = prices[-1]
            
            if current_price > upper_band:
                bb_position = "UPPER"
            elif current_price < lower_band:
                bb_position = "LOWER"
            else:
                bb_position = "MIDDLE"
        else:
            bb_position = "MIDDLE"
        
        # Volume trend
        if len(volumes) >= 10:
            recent_vol = sum(volumes[-5:]) / 5
            older_vol = sum(volumes[-10:-5]) / 5
            if recent_vol > older_vol * 1.2:
                volume_trend = "INCREASING"
            elif recent_vol < older_vol * 0.8:
                volume_trend = "DECREASING"
            else:
                volume_trend = "STABLE"
        else:
            volume_trend = "STABLE"
        
        return {
            "rsi": round(rsi, 1),
            "macd": macd_signal,
            "sma_20": round(sma_20, 2),
            "bollinger_bands": bb_position,
            "volume_trend": volume_trend
        }
        
    except Exception as e:
        logger.error(f"Technical indicators calculation failed for {symbol}: {e}")
        # Return default values to keep terminal working
        return {
            "rsi": 50.0,
            "macd": "NEUTRAL", 
            "sma_20": 0.0,
            "bollinger_bands": "MIDDLE",
            "volume_trend": "STABLE"
        }

@router.get("/options/greeks/{symbol}")
async def get_options_greeks(
    symbol: str,
    strike: float = Query(..., description="Strike price"),
    expiry: str = Query(..., description="Expiration date (YYYY-MM-DD)"),
    option_type: str = Query("call", description="Option type (call/put)")
):
    """Calculate options Greeks"""
    try:
        from ..services.greeks_calculator import greeks_calculator
        
        # Get current stock price
        historical_data = await get_market_data(symbol.upper(), period="1d", interval="1d")
        if not historical_data:
            raise HTTPException(status_code=404, detail=f"Price data not found for {symbol}")
        
        current_price = historical_data[-1]['close']
        
        # Calculate Greeks
        greeks = await greeks_calculator.calculate_greeks(
            symbol=symbol.upper(),
            spot_price=current_price,
            strike_price=strike,
            time_to_expiry=expiry,
            option_type=option_type
        )
        
        return greeks
        
    except Exception as e:
        logger.error(f"Options Greeks calculation failed for {symbol}: {e}")
        raise HTTPException(status_code=500, detail=f"Greeks calculation failed: {str(e)}")

@router.get("/portfolio/risk/{portfolio_id}")
async def get_portfolio_risk(portfolio_id: str):
    """Get portfolio risk metrics"""
    try:
        # This would integrate with a portfolio management system
        # For now, return sample risk metrics
        return {
            "var_1d": -0.025,  # 2.5% daily VaR
            "var_5d": -0.055,  # 5.5% weekly VaR
            "beta": 1.15,
            "sharpe_ratio": 1.85,
            "max_drawdown": -0.125,
            "volatility": 0.18,
            "correlation_spy": 0.85,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Portfolio risk calculation failed for {portfolio_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Portfolio risk calculation failed: {str(e)}")

@router.post("/backtest")
async def run_backtest(
    symbols: List[str],
    strategy: str,
    start_date: str,
    end_date: str,
    initial_capital: float = 100000.0
):
    """Run strategy backtest"""
    try:
        # This would integrate with a backtesting engine
        # For now, return sample backtest results
        return {
            "strategy": strategy,
            "symbols": symbols,
            "period": f"{start_date} to {end_date}",
            "initial_capital": initial_capital,
            "final_value": initial_capital * 1.25,  # 25% return
            "total_return": 0.25,
            "annual_return": 0.15,
            "sharpe_ratio": 1.45,
            "max_drawdown": -0.08,
            "win_rate": 0.68,
            "trades": 127,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Backtest failed: {e}")
        raise HTTPException(status_code=500, detail=f"Backtest failed: {str(e)}")

@router.get("/correlation-matrix")
async def get_correlation_matrix(symbols: str = Query(..., description="Comma-separated symbols")):
    """Get correlation matrix for symbols"""
    try:
        symbol_list = [s.strip().upper() for s in symbols.split(",")]
        
        # Get historical data for all symbols
        correlation_data = {}
        for symbol in symbol_list:
            data = await get_market_data(symbol, period="1y", interval="1d")
            if data:
                prices = [d['close'] for d in data]
                correlation_data[symbol] = prices
        
        if not correlation_data:
            raise HTTPException(status_code=404, detail="No data found for symbols")
        
        # Calculate correlation matrix
        df = pd.DataFrame(correlation_data)
        correlation_matrix = df.corr().round(3).to_dict()
        
        return {
            "symbols": symbol_list,
            "correlation_matrix": correlation_matrix,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Correlation matrix calculation failed: {e}")
        raise HTTPException(status_code=500, detail=f"Correlation calculation failed: {str(e)}")

@router.get("/momentum/{symbol}")
async def get_momentum_indicators(symbol: str):
    """Get momentum indicators for a symbol"""
    try:
        historical_data = await get_market_data(symbol.upper(), period="6m", interval="1d")
        
        if not historical_data:
            raise HTTPException(status_code=404, detail=f"Data not found for {symbol}")
        
        df = pd.DataFrame(historical_data)
        prices = df['close'].values
        
        # Calculate momentum indicators
        if len(prices) >= 20:
            # Price momentum (20-day)
            momentum_20 = (prices[-1] / prices[-20] - 1) * 100
            
            # Rate of change
            roc_10 = (prices[-1] / prices[-10] - 1) * 100 if len(prices) >= 10 else 0
            
            # Moving average convergence
            sma_10 = np.mean(prices[-10:]) if len(prices) >= 10 else prices[-1]
            sma_20 = np.mean(prices[-20:])
            ma_convergence = ((sma_10 / sma_20) - 1) * 100
        else:
            momentum_20 = 0
            roc_10 = 0
            ma_convergence = 0
        
        return {
            "symbol": symbol.upper(),
            "momentum_20d": round(momentum_20, 2),
            "roc_10d": round(roc_10, 2),
            "ma_convergence": round(ma_convergence, 2),
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Momentum calculation failed for {symbol}: {e}")
        raise HTTPException(status_code=500, detail=f"Momentum calculation failed: {str(e)}")

@router.get("/volatility/{symbol}")
async def get_volatility_metrics(symbol: str):
    """Get volatility metrics for a symbol"""
    try:
        historical_data = await get_market_data(symbol.upper(), period="1y", interval="1d")
        
        if not historical_data:
            raise HTTPException(status_code=404, detail=f"Data not found for {symbol}")
        
        df = pd.DataFrame(historical_data)
        prices = df['close'].values
        
        # Calculate returns
        returns = np.diff(np.log(prices))
        
        # Calculate volatility metrics
        daily_vol = np.std(returns) * 100  # Daily volatility %
        annual_vol = daily_vol * np.sqrt(252)  # Annualized volatility
        
        # Rolling volatility (30-day)
        if len(returns) >= 30:
            rolling_vol = np.std(returns[-30:]) * 100
        else:
            rolling_vol = daily_vol
        
        return {
            "symbol": symbol.upper(),
            "daily_volatility": round(daily_vol, 2),
            "annual_volatility": round(annual_vol, 2),
            "rolling_30d_volatility": round(rolling_vol, 2),
            "volatility_regime": "HIGH" if annual_vol > 30 else "NORMAL" if annual_vol > 15 else "LOW",
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Volatility calculation failed for {symbol}: {e}")
        raise HTTPException(status_code=500, detail=f"Volatility calculation failed: {str(e)}") 