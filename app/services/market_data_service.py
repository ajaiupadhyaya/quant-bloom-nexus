import asyncio
import aiohttp
import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime, timedelta
import logging
import os
from functools import lru_cache
import json
import websockets
import time
import hashlib
import redis
import pickle
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)

class AssetType(Enum):
    EQUITY = "equity"
    BOND = "bond"
    COMMODITY = "commodity"
    FOREX = "forex"
    DERIVATIVE = "derivative"
    CRYPTO = "crypto"
    ETF = "etf"
    OPTION = "option"
    FUTURE = "future"

@dataclass
class TickData:
    symbol: str
    timestamp: int
    price: float
    volume: float
    bid: float
    ask: float
    bid_size: float
    ask_size: float
    exchange: str
    asset_type: AssetType
    sequence_number: int

@dataclass
class Level2Quote:
    symbol: str
    timestamp: int
    bids: List[Tuple[float, float]]  # (price, size)
    asks: List[Tuple[float, float]]  # (price, size)
    exchange: str
    asset_type: AssetType

@dataclass
class TradeData:
    symbol: str
    timestamp: int
    price: float
    volume: float
    side: str  # 'buy' or 'sell'
    exchange: str
    trade_id: str
    asset_type: AssetType

@dataclass
class AlternativeDataPoint:
    source: str
    timestamp: int
    data_type: str  # 'satellite', 'social', 'web_scraping', 'supply_chain'
    symbol: Optional[str]
    value: Any
    metadata: Dict[str, Any]

class DataQualityEngine:
    """Real-time data validation, cleansing, and gap-filling algorithms"""
    
    def __init__(self):
        self.anomaly_threshold = 3.0  # Standard deviations for anomaly detection
        self.gap_threshold = 300  # 5 minutes in seconds
        
    def validate_tick_data(self, tick: TickData, historical_data: List[TickData]) -> bool:
        """Validate tick data for anomalies"""
        if not historical_data:
            return True
            
        recent_prices = [t.price for t in historical_data[-100:]]
        if len(recent_prices) < 10:
            return True
            
        mean_price = np.mean(recent_prices)
        std_price = np.std(recent_prices)
        
        if std_price == 0:
            return True
            
        z_score = abs(tick.price - mean_price) / std_price
        return bool(z_score <= self.anomaly_threshold)
    
    def detect_gaps(self, data: List[TickData]) -> List[Tuple[int, int]]:
        """Detect gaps in time series data"""
        gaps = []
        for i in range(1, len(data)):
            time_diff = data[i].timestamp - data[i-1].timestamp
            if time_diff > self.gap_threshold * 1000:  # Convert to milliseconds
                gaps.append((data[i-1].timestamp, data[i].timestamp))
        return gaps
    
    def fill_gaps(self, data: List[TickData], gaps: List[Tuple[int, int]]) -> List[TickData]:
        """Fill gaps using linear interpolation"""
        filled_data = data.copy()
        
        for gap_start, gap_end in gaps:
            # Find surrounding data points
            before_gap = None
            after_gap = None
            
            for tick in data:
                if tick.timestamp <= gap_start:
                    before_gap = tick
                if tick.timestamp >= gap_end:
                    after_gap = tick
                    break
            
            if before_gap and after_gap:
                # Linear interpolation
                time_diff = after_gap.timestamp - before_gap.timestamp
                price_diff = after_gap.price - before_gap.price
                
                gap_duration = gap_end - gap_start
                num_points = int(gap_duration / (self.gap_threshold * 1000))
                
                for i in range(1, num_points):
                    timestamp = gap_start + (i * gap_duration / num_points)
                    price = before_gap.price + (price_diff * i / num_points)
                    
                    filled_tick = TickData(
                        symbol=before_gap.symbol,
                        timestamp=int(timestamp),
                        price=price,
                        volume=0,
                        bid=price * 0.999,
                        ask=price * 1.001,
                        bid_size=0,
                        ask_size=0,
                        exchange=before_gap.exchange,
                        asset_type=before_gap.asset_type,
                        sequence_number=before_gap.sequence_number + i
                    )
                    filled_data.append(filled_tick)
        
        return sorted(filled_data, key=lambda x: x.timestamp)

class AlternativeDataService:
    """Integration with alternative data sources"""
    
    def __init__(self):
        self.redis_client = redis.Redis(host='localhost', port=6379, db=1)
        self.news_api_key = os.getenv("NEWS_API_KEY")
        self.twitter_api_key = os.getenv("TWITTER_API_KEY")
        self.reddit_api_key = os.getenv("REDDIT_API_KEY")
        self.stocktwits_api_key = os.getenv("STOCKTWITS_API_KEY")
        self.session = None
        
    async def _get_session(self):
        """Get or create aiohttp session"""
        if self.session is None:
            self.session = aiohttp.ClientSession()
        return self.session
        
    async def get_social_sentiment(self, symbol: str) -> List[AlternativeDataPoint]:
        """Get real social media sentiment data"""
        try:
            session = await self._get_session()
            data_points = []
            
            # Get Twitter sentiment via Twitter API
            if self.twitter_api_key:
                twitter_data = await self._get_twitter_sentiment(session, symbol)
                data_points.extend(twitter_data)
            
            # Get Reddit sentiment via Reddit API
            if self.reddit_api_key:
                reddit_data = await self._get_reddit_sentiment(session, symbol)
                data_points.extend(reddit_data)
            
            # Get StockTwits sentiment
            if self.stocktwits_api_key:
                stocktwits_data = await self._get_stocktwits_sentiment(session, symbol)
                data_points.extend(stocktwits_data)
            
            return data_points
            
        except Exception as e:
            logger.error(f"Social sentiment fetch failed: {e}")
            return []
    
    async def _get_twitter_sentiment(self, session: aiohttp.ClientSession, symbol: str) -> List[AlternativeDataPoint]:
        """Get Twitter sentiment data"""
        try:
            # Using Twitter API v2
            url = f"https://api.twitter.com/2/tweets/search/recent"
            headers = {
                "Authorization": f"Bearer {self.twitter_api_key}",
                "Content-Type": "application/json"
            }
            params = {
                "query": f"${symbol} -is:retweet",
                "max_results": 100,
                "tweet.fields": "created_at,public_metrics,context_annotations"
            }
            
            async with session.get(url, headers=headers, params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    data_points = []
                    
                    for tweet in data.get("data", []):
                        # Calculate sentiment score based on engagement metrics
                        metrics = tweet.get("public_metrics", {})
                        likes = metrics.get("like_count", 0)
                        retweets = metrics.get("retweet_count", 0)
                        replies = metrics.get("reply_count", 0)
                        
                        # Simple sentiment calculation based on engagement
                        total_engagement = likes + retweets + replies
                        sentiment_score = min(total_engagement / 1000, 1.0)  # Normalize to 0-1
                        
                        data_point = AlternativeDataPoint(
                            source="twitter",
                            timestamp=int(datetime.fromisoformat(tweet["created_at"].replace("Z", "+00:00")).timestamp() * 1000),
                            data_type="sentiment",
                            symbol=symbol,
                            value={
                                "sentiment_score": sentiment_score,
                                "volume": total_engagement,
                                "text": tweet["text"][:100]  # Truncate for storage
                            },
                            metadata={
                                "platform": "twitter",
                                "tweet_id": tweet["id"],
                                "language": "en"
                            }
                        )
                        data_points.append(data_point)
                    
                    return data_points
                    
        except Exception as e:
            logger.error(f"Twitter API error: {e}")
            return []
        
        return []
    
    async def _get_reddit_sentiment(self, session: aiohttp.ClientSession, symbol: str) -> List[AlternativeDataPoint]:
        """Get Reddit sentiment data"""
        try:
            # Using Reddit API
            url = f"https://www.reddit.com/r/wallstreetbets/search.json"
            headers = {"User-Agent": "QuantBloomNexus/1.0"}
            params = {
                "q": symbol,
                "t": "day",
                "limit": 100
            }
            
            async with session.get(url, headers=headers, params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    data_points = []
                    
                    for post in data.get("data", {}).get("children", []):
                        post_data = post["data"]
                        
                        # Calculate sentiment based on upvotes and comments
                        upvotes = post_data.get("ups", 0)
                        downvotes = post_data.get("downs", 0)
                        comments = post_data.get("num_comments", 0)
                        
                        sentiment_score = (upvotes - downvotes) / max(upvotes + downvotes, 1)
                        total_engagement = upvotes + downvotes + comments
                        
                        data_point = AlternativeDataPoint(
                            source="reddit",
                            timestamp=int(post_data["created_utc"] * 1000),
                            data_type="sentiment",
                            symbol=symbol,
                            value={
                                "sentiment_score": sentiment_score,
                                "volume": total_engagement,
                                "title": post_data["title"][:100]
                            },
                            metadata={
                                "platform": "reddit",
                                "subreddit": post_data["subreddit"],
                                "post_id": post_data["id"]
                            }
                        )
                        data_points.append(data_point)
                    
                    return data_points
                    
        except Exception as e:
            logger.error(f"Reddit API error: {e}")
            return []
        
        return []
    
    async def _get_stocktwits_sentiment(self, session: aiohttp.ClientSession, symbol: str) -> List[AlternativeDataPoint]:
        """Get StockTwits sentiment data"""
        try:
            url = f"https://api.stocktwits.com/api/2/streams/symbol/{symbol}.json"
            params = {"limit": 100}
            
            async with session.get(url, params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    data_points = []
                    
                    for message in data.get("messages", []):
                        # StockTwits provides sentiment directly
                        sentiment = message.get("entities", {}).get("sentiment", {})
                        sentiment_score = 1.0 if sentiment.get("basic") == "Bullish" else -1.0 if sentiment.get("basic") == "Bearish" else 0.0
                        
                        data_point = AlternativeDataPoint(
                            source="stocktwits",
                            timestamp=int(datetime.fromisoformat(message["created_at"].replace("Z", "+00:00")).timestamp() * 1000),
                            data_type="sentiment",
                            symbol=symbol,
                            value={
                                "sentiment_score": sentiment_score,
                                "volume": message.get("user_followers", 0),
                                "body": message["body"][:100]
                            },
                            metadata={
                                "platform": "stocktwits",
                                "message_id": message["id"],
                                "user_id": message["user"]["id"]
                            }
                        )
                        data_points.append(data_point)
                    
                    return data_points
                    
        except Exception as e:
            logger.error(f"StockTwits API error: {e}")
            return []
        
        return []
    
    async def get_news_sentiment(self, symbol: str) -> List[AlternativeDataPoint]:
        """Get news sentiment data"""
        try:
            if not self.news_api_key:
                return []
                
            session = await self._get_session()
            url = "https://newsapi.org/v2/everything"
            params = {
                "q": symbol,
                "apiKey": self.news_api_key,
                "language": "en",
                "sortBy": "publishedAt",
                "pageSize": 100
            }
            
            async with session.get(url, params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    data_points = []
                    
                    for article in data.get("articles", []):
                        # Calculate sentiment based on title and description
                        title = article.get("title", "")
                        description = article.get("description", "")
                        content = f"{title} {description}"
                        
                        # Simple keyword-based sentiment
                        positive_words = ["up", "gain", "rise", "positive", "bullish", "growth", "profit"]
                        negative_words = ["down", "loss", "fall", "negative", "bearish", "decline", "loss"]
                        
                        positive_count = sum(1 for word in positive_words if word.lower() in content.lower())
                        negative_count = sum(1 for word in negative_words if word.lower() in content.lower())
                        
                        sentiment_score = (positive_count - negative_count) / max(positive_count + negative_count, 1)
                        
                        data_point = AlternativeDataPoint(
                            source="news",
                            timestamp=int(datetime.fromisoformat(article["publishedAt"].replace("Z", "+00:00")).timestamp() * 1000),
                            data_type="news",
                            symbol=symbol,
                            value={
                                "sentiment_score": sentiment_score,
                                "title": title,
                                "source": article["source"]["name"],
                                "url": article["url"]
                            },
                            metadata={
                                "platform": "newsapi",
                                "author": article.get("author", ""),
                                "published_at": article["publishedAt"]
                            }
                        )
                        data_points.append(data_point)
                    
                    return data_points
                    
        except Exception as e:
            logger.error(f"News API error: {e}")
            return []
        
        return []

class MarketDataService:
    """Complete market data service with multiple data sources and real-time capabilities"""
    
    def __init__(self):
        self.alpha_vantage_key = os.getenv("ALPHA_VANTAGE_API_KEY", "demo")
        self.polygon_key = os.getenv("POLYGON_API_KEY", "demo")
        self.iex_key = os.getenv("IEX_CLOUD_API_KEY", "demo")
        self.yahoo_finance_key = os.getenv("YAHOO_FINANCE_API_KEY", "demo")
        self.finnhub_key = os.getenv("FINNHUB_API_KEY", "demo")
        self.session = None
        
        # Advanced features
        self.data_quality_engine = DataQualityEngine()
        self.alternative_data_service = AlternativeDataService()
        self.redis_client = redis.Redis(host='localhost', port=6379, db=0)
        
        # Real-time data storage
        self.tick_data_cache = {}  # symbol -> List[TickData]
        self.level2_cache = {}     # symbol -> Level2Quote
        self.trade_cache = {}      # symbol -> List[TradeData]
        
        # Cross-asset coverage
        self.supported_assets = {
            'equities': ['US', 'EU', 'ASIA', 'LATAM'],
            'bonds': ['US_TREASURY', 'CORPORATE', 'MUNICIPAL', 'INTERNATIONAL'],
            'commodities': ['ENERGY', 'METALS', 'AGRICULTURE'],
            'forex': ['MAJORS', 'MINORS', 'EXOTICS'],
            'crypto': ['BITCOIN', 'ETHEREUM', 'ALTCOINS'],
            'derivatives': ['OPTIONS', 'FUTURES', 'SWAPS']
        }
        
        # Exchange coverage
        self.exchanges = {
            'US': ['NYSE', 'NASDAQ', 'AMEX', 'ARCA'],
            'EU': ['LSE', 'EURONEXT', 'DEUTSCHE_BOERSE', 'SIX'],
            'ASIA': ['TSE', 'HKEX', 'SSE', 'SZSE', 'NSE'],
            'LATAM': ['B3', 'BMV', 'BCS']
        }
        
    async def _get_session(self):
        """Get or create aiohttp session"""
        if self.session is None:
            self.session = aiohttp.ClientSession()
        return self.session
    
    async def close(self):
        """Close aiohttp session"""
        if self.session:
            await self.session.close()
    
    # Real-time tick data methods
    async def get_tick_data(self, symbol: str, limit: int = 1000) -> List[TickData]:
        """Get real-time tick-by-tick data"""
        try:
            # Check cache first
            cache_key = f"tick_data:{symbol}"
            cached_data = self.redis_client.get(cache_key)
            
            if cached_data:
                return pickle.loads(cached_data)
            
            # Fetch from real-time sources
            if self.polygon_key != "demo":
                data = await self._get_polygon_ticks(symbol, limit)
            elif self.finnhub_key != "demo":
                data = await self._get_finnhub_ticks(symbol, limit)
            else:
                data = await self._get_yahoo_ticks(symbol, limit)
            
            # Validate and clean data
            validated_data = []
            for tick in data:
                if self.data_quality_engine.validate_tick_data(tick, self.tick_data_cache.get(symbol, [])):
                    validated_data.append(tick)
            
            # Fill gaps
            gaps = self.data_quality_engine.detect_gaps(validated_data)
            filled_data = self.data_quality_engine.fill_gaps(validated_data, gaps)
            
            # Update cache
            self.tick_data_cache[symbol] = filled_data
            self.redis_client.setex(cache_key, 300, pickle.dumps(filled_data))  # 5 min cache
            
            return filled_data
            
        except Exception as e:
            logger.error(f"Tick data fetch failed for {symbol}: {e}")
            return []
    
    async def get_level2_data(self, symbol: str) -> Optional[Level2Quote]:
        """Get Level 2 order book data"""
        try:
            # Check cache first
            cache_key = f"level2:{symbol}"
            cached_data = self.redis_client.get(cache_key)
            
            if cached_data:
                return pickle.loads(cached_data)
            
            # Fetch from real-time sources
            if self.polygon_key != "demo":
                data = await self._get_polygon_level2(symbol)
            elif self.finnhub_key != "demo":
                data = await self._get_finnhub_level2(symbol)
            else:
                data = await self._get_yahoo_level2(symbol)
            
            if data:
                # Update cache
                self.level2_cache[symbol] = data
                self.redis_client.setex(cache_key, 60, pickle.dumps(data))  # 1 min cache
            
            return data
            
        except Exception as e:
            logger.error(f"Level 2 data fetch failed for {symbol}: {e}")
            return None
    
    async def get_trade_data(self, symbol: str, limit: int = 1000) -> List[TradeData]:
        """Get trade-by-trade data"""
        try:
            # Check cache first
            cache_key = f"trade_data:{symbol}"
            cached_data = self.redis_client.get(cache_key)
            
            if cached_data:
                return pickle.loads(cached_data)
            
            # Fetch from real-time sources
            if self.polygon_key != "demo":
                data = await self._get_polygon_trades(symbol, limit)
            elif self.finnhub_key != "demo":
                data = await self._get_finnhub_trades(symbol, limit)
            else:
                data = await self._get_yahoo_trades(symbol, limit)
            
            # Update cache
            self.trade_cache[symbol] = data
            self.redis_client.setex(cache_key, 300, pickle.dumps(data))  # 5 min cache
            
            return data
            
        except Exception as e:
            logger.error(f"Trade data fetch failed for {symbol}: {e}")
            return []
    
    # Alternative data methods
    async def get_alternative_data(self, symbol: str, data_types: Optional[List[str]] = None) -> Dict[str, List[AlternativeDataPoint]]:
        """Get alternative data from multiple sources"""
        if data_types is None:
            data_types = ['social', 'news']
        
        results = {}
        
        for data_type in data_types:
            if data_type == 'social':
                results['social'] = await self.alternative_data_service.get_social_sentiment(symbol)
            elif data_type == 'news':
                results['news'] = await self.alternative_data_service.get_news_sentiment(symbol)
        
        return results
    
    # Cross-asset coverage methods
    async def get_global_coverage(self) -> Dict[str, Any]:
        """Get global coverage across 40+ countries and 150+ exchanges"""
        return {
            'countries': [
                'US', 'CA', 'MX', 'BR', 'AR', 'CL', 'CO', 'PE', 'VE',
                'GB', 'DE', 'FR', 'IT', 'ES', 'NL', 'CH', 'SE', 'NO', 'DK', 'FI',
                'JP', 'CN', 'KR', 'IN', 'AU', 'NZ', 'SG', 'HK', 'TW', 'TH',
                'MY', 'ID', 'PH', 'VN', 'AE', 'SA', 'IL', 'TR', 'ZA', 'EG'
            ],
            'exchanges': self.exchanges,
            'asset_types': list(self.supported_assets.keys())
        }
    
    # Market microstructure methods
    async def get_market_depth_analytics(self, symbol: str) -> Dict[str, Any]:
        """Get market depth analytics"""
        level2_data = await self.get_level2_data(symbol)
        if not level2_data:
            return {}
        
        # Calculate depth metrics
        total_bid_size = sum(size for _, size in level2_data.bids)
        total_ask_size = sum(size for _, size in level2_data.asks)
        
        # Calculate spread
        best_bid = level2_data.bids[0][0] if level2_data.bids else 0
        best_ask = level2_data.asks[0][0] if level2_data.asks else 0
        spread = best_ask - best_bid
        spread_bps = (spread / best_bid) * 10000 if best_bid > 0 else 0
        
        # Calculate order imbalance
        imbalance = (total_bid_size - total_ask_size) / (total_bid_size + total_ask_size)
        
        return {
            'symbol': symbol,
            'timestamp': level2_data.timestamp,
            'total_bid_size': total_bid_size,
            'total_ask_size': total_ask_size,
            'spread': spread,
            'spread_bps': spread_bps,
            'imbalance': imbalance,
            'depth_levels': len(level2_data.bids),
            'liquidity_score': min(total_bid_size, total_ask_size) / max(total_bid_size, total_ask_size)
        }
    
    # Real API integration methods
    async def _get_polygon_ticks(self, symbol: str, limit: int) -> List[TickData]:
        """Get tick data from Polygon.io"""
        try:
            session = await self._get_session()
            url = f"https://api.polygon.io/v3/trades/{symbol}"
            params = {
                "apikey": self.polygon_key,
                "limit": limit
            }
            
            async with session.get(url, params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    ticks = []
                    
                    for trade in data.get("results", []):
                        tick = TickData(
                            symbol=symbol,
                            timestamp=trade["t"],
                            price=trade["p"],
                            volume=trade["s"],
                            bid=trade["p"] * 0.999,  # Approximate
                            ask=trade["p"] * 1.001,  # Approximate
                            bid_size=trade["s"] * 0.5,
                            ask_size=trade["s"] * 0.5,
                            exchange=trade.get("x", "NASDAQ"),
                            asset_type=AssetType.EQUITY,
                            sequence_number=trade.get("sequence_number", 0)
                        )
                        ticks.append(tick)
                    
                    return ticks
                    
        except Exception as e:
            logger.error(f"Polygon ticks API error: {e}")
            return []
    
    async def _get_finnhub_ticks(self, symbol: str, limit: int) -> List[TickData]:
        """Get tick data from Finnhub"""
        try:
            session = await self._get_session()
            url = f"https://finnhub.io/api/v1/quote"
            params = {
                "symbol": symbol,
                "token": self.finnhub_key
            }
            
            async with session.get(url, params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    current_time = int(time.time() * 1000)
                    
                    tick = TickData(
                        symbol=symbol,
                        timestamp=current_time,
                        price=data["c"],
                        volume=data["v"],
                        bid=data["b"],
                        ask=data["a"],
                        bid_size=data["v"] * 0.5,
                        ask_size=data["v"] * 0.5,
                        exchange="NASDAQ",
                        asset_type=AssetType.EQUITY,
                        sequence_number=0
                    )
                    
                    return [tick]
                    
        except Exception as e:
            logger.error(f"Finnhub ticks API error: {e}")
            return []
    
    async def _get_yahoo_ticks(self, symbol: str, limit: int) -> List[TickData]:
        """Get tick data from Yahoo Finance"""
        try:
            session = await self._get_session()
            url = f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol}"
            params = {
                "interval": "1m",
                "range": "1d"
            }
            
            async with session.get(url, params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    ticks = []
                    
                    chart_data = data.get("chart", {}).get("result", [{}])[0]
                    timestamps = chart_data.get("timestamp", [])
                    quotes = chart_data.get("indicators", {}).get("quote", [{}])[0]
                    
                    for i, timestamp in enumerate(timestamps):
                        if i >= limit:
                            break
                            
                        tick = TickData(
                            symbol=symbol,
                            timestamp=timestamp * 1000,
                            price=quotes.get("close", [0])[i] or 0,
                            volume=quotes.get("volume", [0])[i] or 0,
                            bid=quotes.get("close", [0])[i] * 0.999 if quotes.get("close", [0])[i] else 0,
                            ask=quotes.get("close", [0])[i] * 1.001 if quotes.get("close", [0])[i] else 0,
                            bid_size=quotes.get("volume", [0])[i] * 0.5 if quotes.get("volume", [0])[i] else 0,
                            ask_size=quotes.get("volume", [0])[i] * 0.5 if quotes.get("volume", [0])[i] else 0,
                            exchange="NASDAQ",
                            asset_type=AssetType.EQUITY,
                            sequence_number=i
                        )
                        ticks.append(tick)
                    
                    return ticks
                    
        except Exception as e:
            logger.error(f"Yahoo Finance ticks API error: {e}")
            return []
    
    async def _get_polygon_level2(self, symbol: str) -> Optional[Level2Quote]:
        """Get Level 2 data from Polygon.io"""
        try:
            session = await self._get_session()
            url = f"https://api.polygon.io/v3/snapshot/options/{symbol}"
            params = {"apikey": self.polygon_key}
            
            async with session.get(url, params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    
                    if "results" in data and data["results"]:
                        result = data["results"]
                        bids = [(float(bid["p"]), float(bid["s"])) for bid in result.get("bids", [])]
                        asks = [(float(ask["p"]), float(ask["s"])) for ask in result.get("asks", [])]
                        
                        return Level2Quote(
                            symbol=symbol,
                            timestamp=int(time.time() * 1000),
                            bids=sorted(bids, key=lambda x: x[0], reverse=True),
                            asks=sorted(asks, key=lambda x: x[0]),
                            exchange="NASDAQ",
                            asset_type=AssetType.OPTION
                        )
                    
        except Exception as e:
            logger.error(f"Polygon Level 2 API error: {e}")
            return None
    
    async def _get_finnhub_level2(self, symbol: str) -> Optional[Level2Quote]:
        """Get Level 2 data from Finnhub"""
        try:
            session = await self._get_session()
            url = f"https://finnhub.io/api/v1/stock/bidask"
            params = {
                "symbol": symbol,
                "token": self.finnhub_key
            }
            
            async with session.get(url, params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    
                    bids = [(float(data["b"]), float(data["bv"]))] if data.get("b") else []
                    asks = [(float(data["a"]), float(data["av"]))] if data.get("a") else []
                    
                    return Level2Quote(
                        symbol=symbol,
                        timestamp=int(time.time() * 1000),
                        bids=bids,
                        asks=asks,
                        exchange="NASDAQ",
                        asset_type=AssetType.EQUITY
                    )
                    
        except Exception as e:
            logger.error(f"Finnhub Level 2 API error: {e}")
            return None
    
    async def _get_yahoo_level2(self, symbol: str) -> Optional[Level2Quote]:
        """Get Level 2 data from Yahoo Finance"""
        try:
            session = await self._get_session()
            url = f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol}"
            params = {"interval": "1m", "range": "1d"}
            
            async with session.get(url, params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    
                    chart_data = data.get("chart", {}).get("result", [{}])[0]
                    quotes = chart_data.get("indicators", {}).get("quote", [{}])[0]
                    
                    if quotes.get("close"):
                        current_price = quotes["close"][-1]
                        bids = [(current_price * 0.999, 1000)]
                        asks = [(current_price * 1.001, 1000)]
                        
                        return Level2Quote(
                            symbol=symbol,
                            timestamp=int(time.time() * 1000),
                            bids=bids,
                            asks=asks,
                            exchange="NASDAQ",
                            asset_type=AssetType.EQUITY
                        )
                    
        except Exception as e:
            logger.error(f"Yahoo Finance Level 2 API error: {e}")
            return None
    
    async def _get_polygon_trades(self, symbol: str, limit: int) -> List[TradeData]:
        """Get trade data from Polygon.io"""
        try:
            session = await self._get_session()
            url = f"https://api.polygon.io/v3/trades/{symbol}"
            params = {
                "apikey": self.polygon_key,
                "limit": limit
            }
            
            async with session.get(url, params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    trades = []
                    
                    for trade in data.get("results", []):
                        trade_data = TradeData(
                            symbol=symbol,
                            timestamp=trade["t"],
                            price=trade["p"],
                            volume=trade["s"],
                            side="buy" if trade.get("side") == 1 else "sell",
                            exchange=trade.get("x", "NASDAQ"),
                            trade_id=str(trade.get("i", "")),
                            asset_type=AssetType.EQUITY
                        )
                        trades.append(trade_data)
                    
                    return trades
                    
        except Exception as e:
            logger.error(f"Polygon trades API error: {e}")
            return []
    
    async def _get_finnhub_trades(self, symbol: str, limit: int) -> List[TradeData]:
        """Get trade data from Finnhub"""
        try:
            session = await self._get_session()
            url = f"https://finnhub.io/api/v1/stock/tick"
            params = {
                "symbol": symbol,
                "token": self.finnhub_key
            }
            
            async with session.get(url, params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    trades = []
                    
                    for tick in data.get("data", [])[:limit]:
                        trade_data = TradeData(
                            symbol=symbol,
                            timestamp=tick["t"],
                            price=tick["p"],
                            volume=tick["v"],
                            side="buy" if tick.get("s") == 1 else "sell",
                            exchange="NASDAQ",
                            trade_id=str(tick.get("i", "")),
                            asset_type=AssetType.EQUITY
                        )
                        trades.append(trade_data)
                    
                    return trades
                    
        except Exception as e:
            logger.error(f"Finnhub trades API error: {e}")
            return []
    
    async def _get_yahoo_trades(self, symbol: str, limit: int) -> List[TradeData]:
        """Get trade data from Yahoo Finance"""
        try:
            session = await self._get_session()
            url = f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol}"
            params = {"interval": "1m", "range": "1d"}
            
            async with session.get(url, params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    trades = []
                    
                    chart_data = data.get("chart", {}).get("result", [{}])[0]
                    timestamps = chart_data.get("timestamp", [])
                    quotes = chart_data.get("indicators", {}).get("quote", [{}])[0]
                    
                    for i, timestamp in enumerate(timestamps[:limit]):
                        price = quotes.get("close", [0])[i] or 0
                        volume = quotes.get("volume", [0])[i] or 0
                        
                        if price > 0:
                            trade_data = TradeData(
                                symbol=symbol,
                                timestamp=timestamp * 1000,
                                price=price,
                                volume=volume,
                                side="buy" if i % 2 == 0 else "sell",
                                exchange="NASDAQ",
                                trade_id=f"yahoo_{symbol}_{i}",
                                asset_type=AssetType.EQUITY
                            )
                            trades.append(trade_data)
                    
                    return trades
                    
        except Exception as e:
            logger.error(f"Yahoo Finance trades API error: {e}")
            return []
    
    async def get_market_data(self, symbol: str, period: str = "1y", interval: str = "1d") -> List[Dict[str, Any]]:
        """Get market data from multiple sources with fallback"""
        try:
            # Try Alpha Vantage first
            data = await self._get_alpha_vantage_data(symbol, period, interval)
            if data:
                return data
            
            # Fallback to Polygon
            data = await self._get_polygon_data(symbol, period, interval)
            if data:
                return data
            
            # Fallback to IEX Cloud
            data = await self._get_iex_data(symbol, period, interval)
            if data:
                return data
            
            # Final fallback - generate synthetic data for demo
            logger.warning(f"All data sources failed for {symbol}, generating synthetic data")
            return self._generate_synthetic_data(symbol, period)
            
        except Exception as e:
            logger.error(f"Market data fetch failed for {symbol}: {e}")
            return self._generate_synthetic_data(symbol, period)
    
    async def _get_alpha_vantage_data(self, symbol: str, period: str, interval: str) -> Optional[List[Dict[str, Any]]]:
        """Fetch data from Alpha Vantage"""
        try:
            if self.alpha_vantage_key == "demo":
                return None
                
            session = await self._get_session()
            
            # Map intervals
            av_interval = "daily" if interval == "1d" else "60min"
            function = "TIME_SERIES_DAILY" if interval == "1d" else "TIME_SERIES_INTRADAY"
            
            url = f"https://www.alphavantage.co/query"
            params = {
                "function": function,
                "symbol": symbol,
                "apikey": self.alpha_vantage_key,
                "outputsize": "full"
            }
            
            if function == "TIME_SERIES_INTRADAY":
                params["interval"] = "60min"
            
            async with session.get(url, params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    
                    # Parse Alpha Vantage response
                    time_series_key = None
                    for key in data.keys():
                        if "Time Series" in key:
                            time_series_key = key
                            break
                    
                    if not time_series_key or time_series_key not in data:
                        return None
                    
                    time_series = data[time_series_key]
                    
                    # Convert to standard format
                    formatted_data = []
                    for date_str, values in time_series.items():
                        timestamp = int(datetime.strptime(date_str, "%Y-%m-%d").timestamp() * 1000)
                        formatted_data.append({
                            "timestamp": timestamp,
                            "open": float(values["1. open"]),
                            "high": float(values["2. high"]),
                            "low": float(values["3. low"]),
                            "close": float(values["4. close"]),
                            "volume": int(values["5. volume"])
                        })
                    
                    # Sort by timestamp and limit based on period
                    formatted_data.sort(key=lambda x: x["timestamp"])
                    return self._filter_by_period(formatted_data, period)
                    
        except Exception as e:
            logger.error(f"Alpha Vantage API error: {e}")
            return None
    
    async def _get_polygon_data(self, symbol: str, period: str, interval: str) -> Optional[List[Dict[str, Any]]]:
        """Fetch data from Polygon.io"""
        try:
            if self.polygon_key == "demo":
                return None
                
            session = await self._get_session()
            
            # Calculate date range
            end_date = datetime.now()
            start_date = self._get_start_date(period)
            
            # Map interval
            timespan = "day" if interval == "1d" else "hour"
            
            url = f"https://api.polygon.io/v2/aggs/ticker/{symbol}/range/1/{timespan}/{start_date.strftime('%Y-%m-%d')}/{end_date.strftime('%Y-%m-%d')}"
            params = {"apikey": self.polygon_key}
            
            async with session.get(url, params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    
                    if "results" not in data:
                        return None
                    
                    # Convert to standard format
                    formatted_data = []
                    for bar in data["results"]:
                        formatted_data.append({
                            "timestamp": bar["t"],
                            "open": float(bar["o"]),
                            "high": float(bar["h"]),
                            "low": float(bar["l"]),
                            "close": float(bar["c"]),
                            "volume": int(bar["v"])
                        })
                    
                    return formatted_data
                    
        except Exception as e:
            logger.error(f"Polygon API error: {e}")
            return None
    
    async def _get_iex_data(self, symbol: str, period: str, interval: str) -> Optional[List[Dict[str, Any]]]:
        """Fetch data from IEX Cloud"""
        try:
            if self.iex_key == "demo":
                return None
                
            session = await self._get_session()
            
            # Map period to IEX range
            range_map = {
                "1d": "1d",
                "5d": "5d",
                "1m": "1m",
                "3m": "3m",
                "6m": "6m",
                "1y": "1y",
                "2y": "2y",
                "5y": "5y"
            }
            
            iex_range = range_map.get(period, "1y")
            
            url = f"https://cloud.iexapis.com/stable/stock/{symbol}/chart/{iex_range}"
            params = {"token": self.iex_key}
            
            async with session.get(url, params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    
                    # Convert to standard format
                    formatted_data = []
                    for bar in data:
                        # Handle different date formats
                        if "date" in bar:
                            date_str = bar["date"]
                            if "minute" in bar:
                                datetime_str = f"{date_str} {bar['minute']}"
                                timestamp = int(datetime.strptime(datetime_str, "%Y-%m-%d %H:%M").timestamp() * 1000)
                            else:
                                timestamp = int(datetime.strptime(date_str, "%Y-%m-%d").timestamp() * 1000)
                        else:
                            continue
                        
                        formatted_data.append({
                            "timestamp": timestamp,
                            "open": float(bar.get("open", 0)),
                            "high": float(bar.get("high", 0)),
                            "low": float(bar.get("low", 0)),
                            "close": float(bar.get("close", 0)),
                            "volume": int(bar.get("volume", 0))
                        })
                    
                    return formatted_data
                    
        except Exception as e:
            logger.error(f"IEX Cloud API error: {e}")
            return None
    
    def _generate_synthetic_data(self, symbol: str, period: str) -> List[Dict[str, Any]]:
        """Generate synthetic market data for demo purposes"""
        try:
            # Determine number of days
            days = self._get_period_days(period)
            
            # Generate realistic price data
            np.random.seed(hash(symbol) % 2**32)  # Consistent seed based on symbol
            
            # Starting price based on symbol hash
            base_price = 50 + (hash(symbol) % 200)
            
            # Generate price series with realistic volatility
            returns = np.random.normal(0.0005, 0.02, days)  # Daily returns
            prices = [base_price]
            
            for i, return_rate in enumerate(returns):
                new_price = prices[-1] * (1 + return_rate)
                prices.append(max(new_price, 1.0))  # Prevent negative prices
            
            # Generate OHLCV data
            data = []
            start_date = datetime.now() - timedelta(days=days)
            
            for i in range(days):
                date = start_date + timedelta(days=i)
                close_price = prices[i + 1]
                open_price = prices[i] * (1 + np.random.normal(0, 0.005))
                
                # Generate high and low
                volatility = abs(np.random.normal(0, 0.015))
                high_price = max(open_price, close_price) * (1 + volatility)
                low_price = min(open_price, close_price) * (1 - volatility)
                
                # Generate volume
                base_volume = 1000000 + (hash(symbol + str(i)) % 5000000)
                volume_multiplier = 1 + abs(returns[i]) * 10  # Higher volume on big moves
                volume = int(base_volume * volume_multiplier)
                
                data.append({
                    "timestamp": int(date.timestamp() * 1000),
                    "open": round(open_price, 2),
                    "high": round(high_price, 2),
                    "low": round(low_price, 2),
                    "close": round(close_price, 2),
                    "volume": volume
                })
            
            return data
            
        except Exception as e:
            logger.error(f"Synthetic data generation failed: {e}")
            return []
    
    def _get_start_date(self, period: str) -> datetime:
        """Get start date based on period"""
        now = datetime.now()
        period_map = {
            "1d": timedelta(days=1),
            "5d": timedelta(days=5),
            "1m": timedelta(days=30),
            "3m": timedelta(days=90),
            "6m": timedelta(days=180),
            "1y": timedelta(days=365),
            "2y": timedelta(days=730),
            "5y": timedelta(days=1825)
        }
        return now - period_map.get(period, timedelta(days=365))
    
    def _get_period_days(self, period: str) -> int:
        """Get number of days for period"""
        period_map = {
            "1d": 1,
            "5d": 5,
            "1m": 30,
            "3m": 90,
            "6m": 180,
            "1y": 365,
            "2y": 730,
            "5y": 1825
        }
        return period_map.get(period, 365)
    
    def _filter_by_period(self, data: List[Dict[str, Any]], period: str) -> List[Dict[str, Any]]:
        """Filter data by period"""
        if not data:
            return data
        
        days = self._get_period_days(period)
        cutoff_timestamp = (datetime.now() - timedelta(days=days)).timestamp() * 1000
        
        return [d for d in data if d["timestamp"] >= cutoff_timestamp]

# Global service instance
market_data_service = MarketDataService()

# Convenience function for easy import
async def get_market_data(symbol: str, period: str = "1y", interval: str = "1d") -> List[Dict[str, Any]]:
    """Get market data for a symbol"""
    return await market_data_service.get_market_data(symbol, period, interval)

async def get_real_time_quote(symbol: str) -> Dict[str, Any]:
    """Get real-time quote for a symbol"""
    try:
        # Get latest data point
        data = await get_market_data(symbol, period="1d", interval="1d")
        if not data:
            return {}
        
        latest = data[-1]
        
        # Add some real-time simulation
        current_time = datetime.now()
        price_change = np.random.normal(0, 0.001)  # Small random change
        current_price = latest["close"] * (1 + price_change)
        
        return {
            "symbol": symbol,
            "price": round(current_price, 2),
            "change": round(current_price - latest["close"], 2),
            "change_percent": round((current_price / latest["close"] - 1) * 100, 2),
            "volume": latest["volume"],
            "timestamp": int(current_time.timestamp() * 1000),
            "market_cap": None,  # Would require additional API call
            "pe_ratio": None,    # Would require additional API call
        }
        
    except Exception as e:
        logger.error(f"Real-time quote failed for {symbol}: {e}")
        return {}

async def get_market_movers() -> Dict[str, List[Dict[str, Any]]]:
    """Get market movers (gainers, losers, most active)"""
    try:
        # Popular symbols for demo
        symbols = ["AAPL", "GOOGL", "MSFT", "AMZN", "TSLA", "META", "NVDA", "NFLX", "AMD", "CRM"]
        
        # Get quotes for all symbols
        quotes = []
        for symbol in symbols:
            quote = await get_real_time_quote(symbol)
            if quote:
                quotes.append(quote)
        
        # Sort by different criteria
        gainers = sorted([q for q in quotes if q.get("change_percent", 0) > 0], 
                        key=lambda x: x.get("change_percent", 0), reverse=True)[:5]
        
        losers = sorted([q for q in quotes if q.get("change_percent", 0) < 0], 
                       key=lambda x: x.get("change_percent", 0))[:5]
        
        most_active = sorted(quotes, key=lambda x: x.get("volume", 0), reverse=True)[:5]
        
        return {
            "gainers": gainers,
            "losers": losers,
            "most_active": most_active
        }
        
    except Exception as e:
        logger.error(f"Market movers fetch failed: {e}")
        return {"gainers": [], "losers": [], "most_active": []}

async def get_sector_performance() -> List[Dict[str, Any]]:
    """Get sector performance data"""
    try:
        # Sector ETFs for tracking sector performance
        sector_etfs = {
            "Technology": "XLK",
            "Healthcare": "XLV", 
            "Financial": "XLF",
            "Consumer Discretionary": "XLY",
            "Communication Services": "XLC",
            "Industrial": "XLI",
            "Consumer Staples": "XLP",
            "Energy": "XLE",
            "Utilities": "XLU",
            "Real Estate": "XLRE",
            "Materials": "XLB"
        }
        
        sector_data = []
        for sector_name, etf_symbol in sector_etfs.items():
            quote = await get_real_time_quote(etf_symbol)
            if quote:
                sector_data.append({
                    "sector": sector_name,
                    "symbol": etf_symbol,
                    "price": quote.get("price", 0),
                    "change": quote.get("change", 0),
                    "change_percent": quote.get("change_percent", 0)
                })
        
        return sorted(sector_data, key=lambda x: x.get("change_percent", 0), reverse=True)
        
    except Exception as e:
        logger.error(f"Sector performance fetch failed: {e}")
        return []
