import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
import pandas as pd
import numpy as np
import requests
import yfinance as yf
from dataclasses import dataclass
from enum import Enum
import httpx
import aiohttp
import json
from scipy import stats
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
import warnings
import os

warnings.filterwarnings('ignore')
logger = logging.getLogger(__name__)

class InvestmentType(Enum):
    HEDGE_FUND = "hedge_fund"
    PRIVATE_EQUITY = "private_equity"
    REAL_ESTATE = "real_estate"
    COMMODITY = "commodity"
    CRYPTOCURRENCY = "cryptocurrency"
    ESG = "esg"
    INFRASTRUCTURE = "infrastructure"
    ART_COLLECTIBLES = "art_collectibles"

@dataclass
class HedgeFundMetrics:
    fund_name: str
    aum: float
    strategy: str
    inception_date: str
    net_returns: List[float]
    gross_returns: List[float]
    management_fee: float
    performance_fee: float
    high_water_mark: bool
    sharpe_ratio: float
    sortino_ratio: float
    calmar_ratio: float
    max_drawdown: float
    volatility: float
    alpha: float
    beta: float
    correlation_to_market: float
    var_95: float
    cvar_95: float
    skewness: float
    kurtosis: float
    up_capture: float
    down_capture: float
    tracking_error: float
    information_ratio: float

@dataclass
class PrivateEquityMetrics:
    fund_name: str
    vintage_year: int
    fund_size: float
    committed_capital: float
    drawn_capital: float
    distributed_capital: float
    nav: float
    irr: float
    tvpi: float
    dpi: float
    rvpi: float
    pme: float
    investment_multiple: float
    realized_gains: float
    unrealized_gains: float
    management_fees_paid: float
    carried_interest: float
    fund_life: int
    investment_period: int
    harvest_period: int

@dataclass
class RealEstateMetrics:
    property_name: str
    property_type: str
    location: str
    acquisition_price: float
    current_value: float
    noi: float
    cap_rate: float
    cash_on_cash_return: float
    irr: float
    equity_multiple: float
    ltv_ratio: float
    dscr: float
    occupancy_rate: float
    lease_expiry_profile: Dict[str, float]
    rental_growth_rate: float
    expense_ratio: float
    property_appreciation: float

@dataclass
class CommodityMetrics:
    commodity_name: str
    spot_price: float
    futures_curve: List[float]
    storage_costs: float
    convenience_yield: float
    basis: float
    contango_backwardation: str
    seasonal_patterns: Dict[str, float]
    supply_demand_fundamentals: Dict[str, Any]
    inventory_levels: float
    production_data: Dict[str, float]
    consumption_data: Dict[str, float]

@dataclass
class CryptocurrencyMetrics:
    symbol: str
    price: float
    market_cap: float
    volume_24h: float
    circulating_supply: float
    total_supply: float
    max_supply: Optional[float]
    price_change_24h: float
    price_change_7d: float
    price_change_30d: float
    volatility: float
    sharpe_ratio: float
    sortino_ratio: float
    max_drawdown: float
    dominance: float
    fear_greed_index: float
    network_metrics: Dict[str, Any]
    defi_metrics: Dict[str, Any]

@dataclass
class ESGMetrics:
    company_name: str
    symbol: str
    esg_score: float
    environmental_score: float
    social_score: float
    governance_score: float
    carbon_intensity: float
    water_usage: float
    waste_generation: float
    renewable_energy_usage: float
    board_diversity: float
    gender_pay_gap: float
    employee_satisfaction: float
    community_investment: float
    controversy_score: float
    sustainable_revenue_percentage: float

@dataclass
class AlternativeInvestment:
    name: str
    category: str
    value: float
    return_rate: float
    risk_metrics: Dict[str, float]
    liquidity_score: float
    description: str

@dataclass
class CryptocurrencyData:
    symbol: str
    price: float
    market_cap: float
    volume_24h: float
    price_change_24h: float
    dominance: float
    technical_indicators: Dict[str, float]

class AlternativeInvestmentsService:
    """Complete alternative investments service with real data sources"""
    
    def __init__(self):
        self.coingecko_api_key = os.getenv("COINGECKO_API_KEY", "demo")
        self.alpha_vantage_key = os.getenv("ALPHA_VANTAGE_API_KEY", "demo")
        self.fred_api_key = os.getenv("FRED_API_KEY", "demo")
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
    
    # Private Equity & Venture Capital
    async def get_private_equity_data(self, fund_type: str = "all") -> List[AlternativeInvestment]:
        """Get private equity and venture capital fund data"""
        try:
            # This would typically connect to private equity databases
            # For now, we'll use public market proxies and industry data
            
            pe_data = []
            
            # Private Equity ETFs as proxies
            pe_etfs = {
                'PSP': 'Invesco Global Listed Private Equity ETF',
                'PEX': 'ProShares Global Listed Private Equity ETF',
                'PEY': 'Invesco High Yield Equity Dividend Achievers ETF'
            }
            
            for symbol, name in pe_etfs.items():
                try:
                    ticker = yf.Ticker(symbol)
                    info = ticker.info
                    hist = ticker.history(period="1y")
                    
                    if not hist.empty:
                        returns = hist['Close'].pct_change().dropna()
                        
                        pe_data.append(AlternativeInvestment(
                            name=name,
                            category="Private Equity",
                            value=info.get('marketCap', 0),
                            return_rate=returns.mean() * 252,  # Annualized
                            risk_metrics={
                                'volatility': returns.std() * np.sqrt(252),
                                'sharpe_ratio': (returns.mean() * 252) / (returns.std() * np.sqrt(252)) if returns.std() > 0 else 0,
                                'max_drawdown': self._calculate_max_drawdown(returns)
                            },
                            liquidity_score=0.8,  # ETFs are liquid
                            description=f"Private equity exposure through {symbol}"
                        ))
                        
                except Exception as e:
                    logger.warning(f"Private equity data for {symbol} failed: {e}")
                    continue
            
            return pe_data
            
        except Exception as e:
            logger.error(f"Private equity data fetch failed: {e}")
            return []
    
    # Real Estate
    async def get_real_estate_data(self, property_type: str = "all") -> List[AlternativeInvestment]:
        """Get REITs, real estate funds, property market data"""
        try:
            real_estate_data = []
            
            # REIT ETFs and major REITs
            reit_symbols = {
                'VNQ': 'Vanguard Real Estate ETF',
                'IYR': 'iShares U.S. Real Estate ETF',
                'SCHH': 'Schwab U.S. REIT ETF',
                'O': 'Realty Income Corporation',
                'PLD': 'Prologis Inc',
                'AMT': 'American Tower Corporation',
                'CCI': 'Crown Castle International Corp'
            }
            
            for symbol, name in reit_symbols.items():
                try:
                    ticker = yf.Ticker(symbol)
                    info = ticker.info
                    hist = ticker.history(period="1y")
                    
                    if not hist.empty:
                        returns = hist['Close'].pct_change().dropna()
                        
                        real_estate_data.append(AlternativeInvestment(
                            name=name,
                            category="Real Estate",
                            value=info.get('marketCap', 0),
                            return_rate=returns.mean() * 252,
                            risk_metrics={
                                'volatility': returns.std() * np.sqrt(252),
                                'sharpe_ratio': (returns.mean() * 252) / (returns.std() * np.sqrt(252)) if returns.std() > 0 else 0,
                                'max_drawdown': self._calculate_max_drawdown(returns),
                                'dividend_yield': info.get('dividendYield', 0)
                            },
                            liquidity_score=0.9,  # Public REITs are liquid
                            description=f"Real estate exposure through {symbol}"
                        ))
                        
                except Exception as e:
                    logger.warning(f"Real estate data for {symbol} failed: {e}")
                    continue
            
            return real_estate_data
            
        except Exception as e:
            logger.error(f"Real estate data fetch failed: {e}")
            return []
    
    # Commodities
    async def get_commodities_data(self, commodity_type: str = "all") -> List[AlternativeInvestment]:
        """Get precious metals, energy, agriculture, industrial metals"""
        try:
            commodities_data = []
            
            # Commodity ETFs and futures
            commodity_symbols = {
                'GLD': 'SPDR Gold Trust',
                'SLV': 'iShares Silver Trust',
                'USO': 'United States Oil Fund',
                'UNG': 'United States Natural Gas Fund',
                'DBA': 'Invesco DB Agriculture Fund',
                'DJP': 'iPath Bloomberg Commodity Index ETN',
                'GSG': 'iShares S&P GSCI Commodity-Indexed Trust',
                'PPLT': 'Aberdeen Standard Physical Platinum ETF',
                'PALL': 'Aberdeen Standard Physical Palladium ETF'
            }
            
            for symbol, name in commodity_symbols.items():
                try:
                    ticker = yf.Ticker(symbol)
                    info = ticker.info
                    hist = ticker.history(period="1y")
                    
                    if not hist.empty:
                        returns = hist['Close'].pct_change().dropna()
                        
                        # Determine commodity category
                        category = "Commodities"
                        if "Gold" in name or "Silver" in name or "Platinum" in name or "Palladium" in name:
                            category = "Precious Metals"
                        elif "Oil" in name or "Gas" in name:
                            category = "Energy"
                        elif "Agriculture" in name:
                            category = "Agriculture"
                        else:
                            category = "Commodities"
                        
                        commodities_data.append(AlternativeInvestment(
                            name=name,
                            category=category,
                            value=info.get('marketCap', 0),
                            return_rate=returns.mean() * 252,
                            risk_metrics={
                                'volatility': returns.std() * np.sqrt(252),
                                'sharpe_ratio': (returns.mean() * 252) / (returns.std() * np.sqrt(252)) if returns.std() > 0 else 0,
                                'max_drawdown': self._calculate_max_drawdown(returns)
                            },
                            liquidity_score=0.85,
                            description=f"Commodity exposure through {symbol}"
                        ))
                        
                except Exception as e:
                    logger.warning(f"Commodity data for {symbol} failed: {e}")
                    continue
            
            return commodities_data
            
        except Exception as e:
            logger.error(f"Commodities data fetch failed: {e}")
            return []
    
    # Hedge Funds
    async def get_hedge_fund_data(self, strategy: str = "all") -> List[AlternativeInvestment]:
        """Get hedge fund indices, fund of funds, liquid alternatives"""
        try:
            hedge_fund_data = []
            
            # Hedge fund ETFs and liquid alternatives
            hedge_fund_symbols = {
                'QAI': 'IQ Hedge Multi-Strategy Tracker ETF',
                'HDG': 'ProShares Hedge Replication ETF',
                'MNA': 'IQ Merger Arbitrage ETF',
                'CSLS': 'Direxion Daily CSI 300 China A Share Bull 2X Shares',
                'TZA': 'Direxion Daily Small Cap Bear 3X Shares',
                'SPXL': 'Direxion Daily S&P 500 Bull 3X Shares',
                'SPXS': 'Direxion Daily S&P 500 Bear 3X Shares'
            }
            
            for symbol, name in hedge_fund_symbols.items():
                try:
                    ticker = yf.Ticker(symbol)
                    info = ticker.info
                    hist = ticker.history(period="1y")
                    
                    if not hist.empty:
                        returns = hist['Close'].pct_change().dropna()
                        
                        # Determine strategy
                        strategy_type = "Multi-Strategy"
                        if "Merger" in name:
                            strategy_type = "Merger Arbitrage"
                        elif "Bear" in name or "Bull" in name:
                            strategy_type = "Leveraged/Inverse"
                        elif "Hedge" in name:
                            strategy_type = "Hedge Replication"
                        
                        hedge_fund_data.append(AlternativeInvestment(
                            name=name,
                            category=f"Hedge Fund - {strategy_type}",
                            value=info.get('marketCap', 0),
                            return_rate=returns.mean() * 252,
                            risk_metrics={
                                'volatility': returns.std() * np.sqrt(252),
                                'sharpe_ratio': (returns.mean() * 252) / (returns.std() * np.sqrt(252)) if returns.std() > 0 else 0,
                                'max_drawdown': self._calculate_max_drawdown(returns),
                                'beta': self._calculate_beta(returns)
                            },
                            liquidity_score=0.9,
                            description=f"Hedge fund exposure through {symbol}"
                        ))
                        
                except Exception as e:
                    logger.warning(f"Hedge fund data for {symbol} failed: {e}")
                    continue
            
            return hedge_fund_data
            
        except Exception as e:
            logger.error(f"Hedge fund data fetch failed: {e}")
            return []
    
    # Cryptocurrency Analysis
    async def get_cryptocurrency_data(self, symbols: List[str] = None) -> List[CryptocurrencyData]:
        """Get cryptocurrency prices, market data, technical analysis"""
        try:
            if symbols is None:
                symbols = ['bitcoin', 'ethereum', 'binancecoin', 'cardano', 'solana', 'polkadot', 'dogecoin', 'avalanche-2']
            
            crypto_data = []
            
            for symbol in symbols:
                try:
                    # Get data from CoinGecko API
                    coin_data = await self._get_coingecko_data(symbol)
                    
                    if coin_data:
                        # Get technical indicators
                        technical_indicators = await self._calculate_crypto_technical_indicators(symbol)
                        
                        crypto_data.append(CryptocurrencyData(
                            symbol=symbol.upper(),
                            price=coin_data.get('current_price', 0),
                            market_cap=coin_data.get('market_cap', 0),
                            volume_24h=coin_data.get('total_volume', 0),
                            price_change_24h=coin_data.get('price_change_percentage_24h', 0),
                            dominance=coin_data.get('market_cap_rank', 0),
                            technical_indicators=technical_indicators
                        ))
                        
                except Exception as e:
                    logger.warning(f"Cryptocurrency data for {symbol} failed: {e}")
                    continue
            
            return crypto_data
            
        except Exception as e:
            logger.error(f"Cryptocurrency data fetch failed: {e}")
            return []
    
    async def _get_coingecko_data(self, coin_id: str) -> Optional[Dict[str, Any]]:
        """Get cryptocurrency data from CoinGecko"""
        try:
            session = await self._get_session()
            url = f"https://api.coingecko.com/api/v3/coins/{coin_id}"
            
            async with session.get(url) as response:
                if response.status == 200:
                    data = await response.json()
                    
                    return {
                        'current_price': data.get('market_data', {}).get('current_price', {}).get('usd', 0),
                        'market_cap': data.get('market_data', {}).get('market_cap', {}).get('usd', 0),
                        'total_volume': data.get('market_data', {}).get('total_volume', {}).get('usd', 0),
                        'price_change_percentage_24h': data.get('market_data', {}).get('price_change_percentage_24h', 0),
                        'market_cap_rank': data.get('market_cap_rank', 0),
                        'price_change_percentage_7d': data.get('market_data', {}).get('price_change_percentage_7d', 0),
                        'price_change_percentage_30d': data.get('market_data', {}).get('price_change_percentage_30d', 0)
                    }
            
            return None
            
        except Exception as e:
            logger.error(f"CoinGecko data fetch failed for {coin_id}: {e}")
            return None
    
    async def _calculate_crypto_technical_indicators(self, coin_id: str) -> Dict[str, float]:
        """Calculate technical indicators for cryptocurrency"""
        try:
            # Get historical price data
            session = await self._get_session()
            url = f"https://api.coingecko.com/api/v3/coins/{coin_id}/market_chart"
            params = {
                'vs_currency': 'usd',
                'days': '30'
            }
            
            async with session.get(url, params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    
                    # Extract price data
                    prices = [price[1] for price in data.get('prices', [])]
                    
                    if len(prices) < 10:
                        return {}
                    
                    # Calculate technical indicators
                    indicators = {}
                    
                    # RSI
                    indicators['rsi'] = self._calculate_rsi(prices)
                    
                    # Moving averages
                    indicators['sma_7'] = np.mean(prices[-7:])
                    indicators['sma_14'] = np.mean(prices[-14:])
                    indicators['ema_12'] = self._calculate_ema(prices, 12)
                    
                    # Bollinger Bands
                    bb_data = self._calculate_bollinger_bands(prices)
                    indicators.update(bb_data)
                    
                    # MACD
                    macd_data = self._calculate_macd(prices)
                    indicators.update(macd_data)
                    
                    return indicators
            
            return {}
            
        except Exception as e:
            logger.error(f"Crypto technical indicators calculation failed for {coin_id}: {e}")
            return {}
    
    def _calculate_rsi(self, prices: List[float], period: int = 14) -> float:
        """Calculate RSI"""
        try:
            if len(prices) < period + 1:
                return 50.0
            
            deltas = np.diff(prices)
            gains = np.where(deltas > 0, deltas, 0)
            losses = np.where(deltas < 0, -deltas, 0)
            
            avg_gain = np.mean(gains[-period:])
            avg_loss = np.mean(losses[-period:])
            
            if avg_loss == 0:
                return 100.0
            
            rs = avg_gain / avg_loss
            rsi = 100 - (100 / (1 + rs))
            
            return rsi
            
        except Exception as e:
            logger.error(f"RSI calculation failed: {e}")
            return 50.0
    
    def _calculate_ema(self, prices: List[float], period: int) -> float:
        """Calculate Exponential Moving Average"""
        try:
            if len(prices) < period:
                return np.mean(prices)
            
            alpha = 2 / (period + 1)
            ema = prices[0]
            
            for price in prices[1:]:
                ema = alpha * price + (1 - alpha) * ema
            
            return ema
            
        except Exception as e:
            logger.error(f"EMA calculation failed: {e}")
            return np.mean(prices) if prices else 0.0
    
    def _calculate_bollinger_bands(self, prices: List[float], period: int = 20, std_dev: int = 2) -> Dict[str, float]:
        """Calculate Bollinger Bands"""
        try:
            if len(prices) < period:
                return {}
            
            recent_prices = prices[-period:]
            sma = np.mean(recent_prices)
            std = np.std(recent_prices)
            
            return {
                'bb_upper': sma + (std_dev * std),
                'bb_middle': sma,
                'bb_lower': sma - (std_dev * std),
                'bb_width': (std_dev * std * 2) / sma if sma > 0 else 0
            }
            
        except Exception as e:
            logger.error(f"Bollinger Bands calculation failed: {e}")
            return {}
    
    def _calculate_macd(self, prices: List[float], fast: int = 12, slow: int = 26, signal: int = 9) -> Dict[str, float]:
        """Calculate MACD"""
        try:
            if len(prices) < slow:
                return {}
            
            # Calculate EMAs
            ema_fast = self._calculate_ema(prices, fast)
            ema_slow = self._calculate_ema(prices, slow)
            
            macd_line = ema_fast - ema_slow
            
            # For signal line, we'd need more historical data
            # This is a simplified calculation
            signal_line = macd_line * 0.9  # Approximation
            
            histogram = macd_line - signal_line
            
            return {
                'macd_line': macd_line,
                'macd_signal': signal_line,
                'macd_histogram': histogram
            }
            
        except Exception as e:
            logger.error(f"MACD calculation failed: {e}")
            return {}
    
    # Alternative Investment Portfolio Analysis
    async def analyze_alternative_portfolio(self, investments: List[AlternativeInvestment]) -> Dict[str, Any]:
        """Analyze alternative investment portfolio"""
        try:
            if not investments:
                return {}
            
            analysis = {}
            
            # Portfolio composition
            total_value = sum(inv.value for inv in investments)
            category_weights = {}
            
            for inv in investments:
                weight = inv.value / total_value if total_value > 0 else 0
                if inv.category not in category_weights:
                    category_weights[inv.category] = 0
                category_weights[inv.category] += weight
            
            analysis['portfolio_composition'] = category_weights
            
            # Risk metrics
            weighted_return = sum(inv.return_rate * (inv.value / total_value) for inv in investments)
            weighted_volatility = sum(inv.risk_metrics.get('volatility', 0) * (inv.value / total_value) for inv in investments)
            
            analysis['portfolio_metrics'] = {
                'total_value': total_value,
                'weighted_return': weighted_return,
                'weighted_volatility': weighted_volatility,
                'sharpe_ratio': weighted_return / weighted_volatility if weighted_volatility > 0 else 0,
                'liquidity_score': np.mean([inv.liquidity_score for inv in investments])
            }
            
            # Risk analysis
            analysis['risk_analysis'] = {
                'high_risk_investments': [inv.name for inv in investments if inv.risk_metrics.get('volatility', 0) > 0.3],
                'low_liquidity_investments': [inv.name for inv in investments if inv.liquidity_score < 0.5],
                'diversification_score': self._calculate_diversification_score(category_weights)
            }
            
            return analysis
            
        except Exception as e:
            logger.error(f"Alternative portfolio analysis failed: {e}")
            return {}
    
    def _calculate_diversification_score(self, category_weights: Dict[str, float]) -> float:
        """Calculate portfolio diversification score"""
        try:
            if not category_weights:
                return 0.0
            
            # Herfindahl-Hirschman Index (HHI) for concentration
            hhi = sum(weight ** 2 for weight in category_weights.values())
            
            # Convert to diversification score (1 - normalized HHI)
            n_categories = len(category_weights)
            max_hhi = 1.0  # Maximum concentration
            min_hhi = 1.0 / n_categories  # Minimum concentration (equal weights)
            
            if max_hhi == min_hhi:
                return 1.0
            
            normalized_hhi = (hhi - min_hhi) / (max_hhi - min_hhi)
            diversification_score = 1.0 - normalized_hhi
            
            return max(0.0, min(1.0, diversification_score))
            
        except Exception as e:
            logger.error(f"Diversification score calculation failed: {e}")
            return 0.0
    
    def _calculate_max_drawdown(self, returns: pd.Series) -> float:
        """Calculate maximum drawdown"""
        try:
            cumulative = (1 + returns).cumprod()
            running_max = cumulative.expanding().max()
            drawdown = (cumulative - running_max) / running_max
            return drawdown.min()
        except Exception as e:
            logger.error(f"Maximum drawdown calculation failed: {e}")
            return 0.0
    
    def _calculate_beta(self, returns: pd.Series) -> float:
        """Calculate beta relative to market"""
        try:
            # Use S&P 500 as market proxy
            market_ticker = yf.Ticker('^GSPC')
            market_hist = market_ticker.history(period="1y")
            
            if not market_hist.empty:
                market_returns = market_hist['Close'].pct_change().dropna()
                
                # Align data
                aligned_data = pd.DataFrame({
                    'asset': returns,
                    'market': market_returns
                }).dropna()
                
                if len(aligned_data) > 10:
                    covariance = aligned_data['asset'].cov(aligned_data['market'])
                    market_variance = aligned_data['market'].var()
                    
                    beta = covariance / market_variance if market_variance > 0 else 1.0
                    return beta
            
            return 1.0
            
        except Exception as e:
            logger.error(f"Beta calculation failed: {e}")
            return 1.0
    
    # Market Sentiment for Alternatives
    async def get_alternative_market_sentiment(self) -> Dict[str, Any]:
        """Get market sentiment for alternative investments"""
        try:
            sentiment = {}
            
            # Fear & Greed Index (for crypto)
            sentiment['crypto_fear_greed'] = await self._get_crypto_fear_greed_index()
            
            # Commodity sentiment
            sentiment['commodity_sentiment'] = await self._get_commodity_sentiment()
            
            # Real estate sentiment
            sentiment['real_estate_sentiment'] = await self._get_real_estate_sentiment()
            
            return sentiment
            
        except Exception as e:
            logger.error(f"Alternative market sentiment failed: {e}")
            return {}
    
    async def _get_crypto_fear_greed_index(self) -> Dict[str, Any]:
        """Get crypto fear and greed index"""
        try:
            session = await self._get_session()
            url = "https://api.alternative.me/fng/"
            
            async with session.get(url) as response:
                if response.status == 200:
                    data = await response.json()
                    
                    if data.get('data'):
                        latest = data['data'][0]
                        return {
                            'value': int(latest.get('value', 50)),
                            'classification': latest.get('value_classification', 'Neutral'),
                            'timestamp': latest.get('timestamp', ''),
                            'time_until_update': latest.get('time_until_update', '')
                        }
            
            return {'value': 50, 'classification': 'Neutral'}
            
        except Exception as e:
            logger.error(f"Crypto fear and greed index failed: {e}")
            return {'value': 50, 'classification': 'Neutral'}
    
    async def _get_commodity_sentiment(self) -> Dict[str, Any]:
        """Get commodity market sentiment"""
        try:
            # Analyze commodity ETF performance
            commodity_etfs = ['GLD', 'SLV', 'USO', 'UNG', 'DBA']
            sentiment_scores = []
            
            for symbol in commodity_etfs:
                try:
                    ticker = yf.Ticker(symbol)
                    hist = ticker.history(period="1mo")
                    
                    if not hist.empty:
                        returns = hist['Close'].pct_change().dropna()
                        sentiment_score = returns.mean() * 252  # Annualized return as sentiment
                        sentiment_scores.append(sentiment_score)
                        
                except Exception as e:
                    logger.warning(f"Commodity sentiment for {symbol} failed: {e}")
                    continue
            
            avg_sentiment = np.mean(sentiment_scores) if sentiment_scores else 0
            
            return {
                'overall_sentiment': avg_sentiment,
                'sentiment_classification': 'Bullish' if avg_sentiment > 0.1 else 'Bearish' if avg_sentiment < -0.1 else 'Neutral',
                'commodity_count': len(sentiment_scores)
            }
            
        except Exception as e:
            logger.error(f"Commodity sentiment failed: {e}")
            return {'overall_sentiment': 0, 'sentiment_classification': 'Neutral'}
    
    async def _get_real_estate_sentiment(self) -> Dict[str, Any]:
        """Get real estate market sentiment"""
        try:
            # Analyze REIT performance
            reit_etfs = ['VNQ', 'IYR', 'SCHH']
            sentiment_scores = []
            
            for symbol in reit_etfs:
                try:
                    ticker = yf.Ticker(symbol)
                    hist = ticker.history(period="1mo")
                    
                    if not hist.empty:
                        returns = hist['Close'].pct_change().dropna()
                        sentiment_score = returns.mean() * 252
                        sentiment_scores.append(sentiment_score)
                        
                except Exception as e:
                    logger.warning(f"Real estate sentiment for {symbol} failed: {e}")
                    continue
            
            avg_sentiment = np.mean(sentiment_scores) if sentiment_scores else 0
            
            return {
                'overall_sentiment': avg_sentiment,
                'sentiment_classification': 'Bullish' if avg_sentiment > 0.1 else 'Bearish' if avg_sentiment < -0.1 else 'Neutral',
                'reit_count': len(sentiment_scores)
            }
            
        except Exception as e:
            logger.error(f"Real estate sentiment failed: {e}")
            return {'overall_sentiment': 0, 'sentiment_classification': 'Neutral'}

# Global service instance
alternative_investments_service = AlternativeInvestmentsService() 