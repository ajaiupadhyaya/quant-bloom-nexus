import asyncio
import aiohttp
import logging
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
import os
import json
from newsapi import NewsApiClient
from .ai_engine import ai_engine
import yfinance as yf
import pandas as pd

logger = logging.getLogger(__name__)

class NewsService:
    """Complete news service with real-time feeds, earnings calendar, economic calendar, and event detection"""
    
    def __init__(self):
        self.news_api_key = os.getenv("NEWS_API_KEY", "demo")
        self.fmp_api_key = os.getenv("FMP_API_KEY", "demo")
        self.alpha_vantage_key = os.getenv("ALPHA_VANTAGE_API_KEY", "demo")
        self.fred_api_key = os.getenv("FRED_API_KEY", "demo")
        self.sec_api_key = os.getenv("SEC_API_KEY", "demo")
        self.session = None
        self.news_client = None
        
        if self.news_api_key != "demo":
            self.news_client = NewsApiClient(api_key=self.news_api_key)
    
    async def _get_session(self):
        """Get or create aiohttp session"""
        if self.session is None:
            self.session = aiohttp.ClientSession()
        return self.session
    
    async def close(self):
        """Close aiohttp session"""
        if self.session:
            await self.session.close()
    
    # Real-time news feeds from multiple sources
    async def get_bloomberg_news(self, limit: int = 20) -> List[Dict[str, Any]]:
        """Get Bloomberg News feed"""
        try:
            session = await self._get_session()
            url = "https://www.bloomberg.com/feed/podcast/etf-report.xml"
            
            async with session.get(url) as response:
                if response.status == 200:
                    content = await response.text()
                    return self._parse_rss_feed(content, "Bloomberg", limit)
            
            return []
            
        except Exception as e:
            logger.error(f"Bloomberg news fetch failed: {e}")
            return []
    
    async def get_reuters_news(self, limit: int = 20) -> List[Dict[str, Any]]:
        """Get Reuters news feed"""
        try:
            session = await self._get_session()
            url = "https://feeds.reuters.com/reuters/businessNews"
            
            async with session.get(url) as response:
                if response.status == 200:
                    content = await response.text()
                    return self._parse_rss_feed(content, "Reuters", limit)
            
            return []
            
        except Exception as e:
            logger.error(f"Reuters news fetch failed: {e}")
            return []
    
    async def get_dow_jones_news(self, limit: int = 20) -> List[Dict[str, Any]]:
        """Get Dow Jones news feed"""
        try:
            session = await self._get_session()
            url = "https://feeds.feedburner.com/djnews"
            
            async with session.get(url) as response:
                if response.status == 200:
                    content = await response.text()
                    return self._parse_rss_feed(content, "Dow Jones", limit)
            
            return []
            
        except Exception as e:
            logger.error(f"Dow Jones news fetch failed: {e}")
            return []
    
    def _parse_rss_feed(self, content: str, source: str, limit: int) -> List[Dict[str, Any]]:
        """Parse RSS feed content"""
        try:
            import xml.etree.ElementTree as ET
            root = ET.fromstring(content)
            
            news_items = []
            for item in root.findall('.//item')[:limit]:
                title = item.find('title')
                description = item.find('description')
                link = item.find('link')
                pub_date = item.find('pubDate')
                
                news_items.append({
                    'title': title.text if title is not None else '',
                    'description': description.text if description is not None else '',
                    'url': link.text if link is not None else '',
                    'source': source,
                    'published_at': pub_date.text if pub_date is not None else '',
                    'symbol': None
                })
            
            return news_items
            
        except Exception as e:
            logger.error(f"RSS parsing failed: {e}")
            return []
    
    # Earnings Calendar
    async def get_earnings_calendar(self, start_date: str = None, end_date: str = None) -> List[Dict[str, Any]]:
        """Get comprehensive earnings calendar with estimates and actual results"""
        try:
            if self.fmp_api_key != "demo":
                return await self._get_fmp_earnings_calendar(start_date, end_date)
            else:
                return await self._get_yahoo_earnings_calendar(start_date, end_date)
                
        except Exception as e:
            logger.error(f"Earnings calendar fetch failed: {e}")
            return []
    
    async def _get_fmp_earnings_calendar(self, start_date: str, end_date: str) -> List[Dict[str, Any]]:
        """Get earnings calendar from FMP"""
        try:
            session = await self._get_session()
            url = "https://financialmodelingprep.com/api/v3/earning_calendar"
            params = {
                'apikey': self.fmp_api_key
            }
            
            if start_date:
                params['from'] = start_date
            if end_date:
                params['to'] = end_date
            
            async with session.get(url, params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    
                    earnings = []
                    for item in data:
                        earnings.append({
                            'symbol': item.get('symbol'),
                            'company_name': item.get('companyName'),
                            'date': item.get('date'),
                            'time': item.get('time'),
                            'estimate_eps': item.get('epsEstimate'),
                            'actual_eps': item.get('epsActual'),
                            'estimate_revenue': item.get('revenueEstimate'),
                            'actual_revenue': item.get('revenueActual'),
                            'surprise_percent': item.get('surprisePercent')
                        })
                    
                    return earnings
            
            return []
            
        except Exception as e:
            logger.error(f"FMP earnings calendar failed: {e}")
            return []
    
    async def _get_yahoo_earnings_calendar(self, start_date: str, end_date: str) -> List[Dict[str, Any]]:
        """Get earnings calendar from Yahoo Finance"""
        try:
            symbols = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'META', 'NVDA', 'NFLX']
            earnings = []
            
            for symbol in symbols:
                try:
                    ticker = yf.Ticker(symbol)
                    calendar = ticker.calendar
                    
                    if calendar is not None and not calendar.empty:
                        for _, row in calendar.iterrows():
                            earnings.append({
                                'symbol': symbol,
                                'company_name': ticker.info.get('longName', symbol),
                                'date': row.name.strftime('%Y-%m-%d'),
                                'time': 'AMC',
                                'estimate_eps': row.get('Earnings Average', 0),
                                'actual_eps': None,
                                'estimate_revenue': row.get('Revenue Average', 0),
                                'actual_revenue': None,
                                'surprise_percent': None
                            })
                except Exception as e:
                    logger.warning(f"Yahoo earnings for {symbol} failed: {e}")
                    continue
            
            return earnings
            
        except Exception as e:
            logger.error(f"Yahoo earnings calendar failed: {e}")
            return []
    
    # Economic Calendar
    async def get_economic_calendar(self, start_date: str = None, end_date: str = None) -> List[Dict[str, Any]]:
        """Get economic calendar with central bank decisions, economic releases, policy announcements"""
        try:
            if self.fred_api_key != "demo":
                return await self._get_fred_economic_calendar(start_date, end_date)
            else:
                return await self._get_alpha_vantage_economic_calendar(start_date, end_date)
                
        except Exception as e:
            logger.error(f"Economic calendar fetch failed: {e}")
            return []
    
    async def _get_fred_economic_calendar(self, start_date: str, end_date: str) -> List[Dict[str, Any]]:
        """Get economic calendar from FRED"""
        try:
            session = await self._get_session()
            url = "https://api.stlouisfed.org/fred/series/search"
            params = {
                'api_key': self.fred_api_key,
                'search_text': 'GDP CPI employment unemployment',
                'file_type': 'json'
            }
            
            async with session.get(url, params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    
                    economic_events = []
                    for series in data.get('seriess', [])[:20]:
                        economic_events.append({
                            'indicator': series.get('title'),
                            'frequency': series.get('frequency'),
                            'units': series.get('units'),
                            'last_updated': series.get('last_updated'),
                            'observation_start': series.get('observation_start'),
                            'observation_end': series.get('observation_end'),
                            'notes': series.get('notes')
                        })
                    
                    return economic_events
            
            return []
            
        except Exception as e:
            logger.error(f"FRED economic calendar failed: {e}")
            return []
    
    async def _get_alpha_vantage_economic_calendar(self, start_date: str, end_date: str) -> List[Dict[str, Any]]:
        """Get economic calendar from Alpha Vantage"""
        try:
            if self.alpha_vantage_key == "demo":
                return []
                
            session = await self._get_session()
            url = "https://www.alphavantage.co/query"
            params = {
                'function': 'ECONOMIC_CALENDAR',
                'apikey': self.alpha_vantage_key
            }
            
            async with session.get(url, params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    
                    economic_events = []
                    for event in data.get('economic_calendar', []):
                        economic_events.append({
                            'indicator': event.get('event'),
                            'currency': event.get('currency'),
                            'impact': event.get('impact'),
                            'actual': event.get('actual'),
                            'forecast': event.get('forecast'),
                            'previous': event.get('previous'),
                            'date': event.get('date'),
                            'time': event.get('time')
                        })
                    
                    return economic_events
            
            return []
            
        except Exception as e:
            logger.error(f"Alpha Vantage economic calendar failed: {e}")
            return []
    
    # Corporate Actions
    async def get_corporate_actions(self, symbol: str = None) -> List[Dict[str, Any]]:
        """Get corporate actions: dividends, splits, mergers, spin-offs, rights offerings"""
        try:
            if symbol:
                return await self._get_symbol_corporate_actions(symbol)
            else:
                return await self._get_market_corporate_actions()
                
        except Exception as e:
            logger.error(f"Corporate actions fetch failed: {e}")
            return []
    
    async def _get_symbol_corporate_actions(self, symbol: str) -> List[Dict[str, Any]]:
        """Get corporate actions for specific symbol"""
        try:
            ticker = yf.Ticker(symbol)
            
            corporate_actions = []
            
            # Get dividends
            dividends = ticker.dividends
            if dividends is not None and not dividends.empty:
                for date, amount in dividends.tail(10).items():
                    corporate_actions.append({
                        'symbol': symbol,
                        'action_type': 'dividend',
                        'date': date.strftime('%Y-%m-%d'),
                        'amount': amount,
                        'description': f'Dividend payment of ${amount:.2f}'
                    })
            
            # Get stock splits
            splits = ticker.splits
            if splits is not None and not splits.empty:
                for date, ratio in splits.tail(5).items():
                    corporate_actions.append({
                        'symbol': symbol,
                        'action_type': 'split',
                        'date': date.strftime('%Y-%m-%d'),
                        'amount': ratio,
                        'description': f'Stock split {ratio}:1'
                    })
            
            return corporate_actions
            
        except Exception as e:
            logger.error(f"Symbol corporate actions failed for {symbol}: {e}")
            return []
    
    async def _get_market_corporate_actions(self) -> List[Dict[str, Any]]:
        """Get market-wide corporate actions"""
        try:
            symbols = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'META', 'NVDA', 'NFLX']
            all_actions = []
            
            for symbol in symbols:
                actions = await self._get_symbol_corporate_actions(symbol)
                all_actions.extend(actions)
            
            return all_actions
            
        except Exception as e:
            logger.error(f"Market corporate actions failed: {e}")
            return []
    
    # Event Detection
    async def detect_market_events(self, symbols: List[str] = None) -> List[Dict[str, Any]]:
        """Automated identification of market-moving events and anomalies"""
        try:
            events = []
            
            if symbols is None:
                symbols = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'META', 'NVDA', 'NFLX']
            
            for symbol in symbols:
                try:
                    # Get recent news and price data
                    news = await self.get_symbol_news(symbol, 10)
                    ticker = yf.Ticker(symbol)
                    hist = ticker.history(period="5d")
                    
                    if not hist.empty:
                        # Detect price anomalies
                        current_price = hist['Close'].iloc[-1]
                        avg_price = hist['Close'].mean()
                        price_change = (current_price - avg_price) / avg_price
                        
                        if abs(price_change) > 0.05:  # 5% price movement
                            events.append({
                                'symbol': symbol,
                                'event_type': 'price_anomaly',
                                'severity': 'high' if abs(price_change) > 0.1 else 'medium',
                                'description': f'{symbol} price moved {price_change:.2%} from average',
                                'timestamp': datetime.now().isoformat(),
                                'price_change': price_change
                            })
                        
                        # Detect volume anomalies
                        current_volume = hist['Volume'].iloc[-1]
                        avg_volume = hist['Volume'].mean()
                        volume_change = (current_volume - avg_volume) / avg_volume
                        
                        if volume_change > 2:  # 200% volume increase
                            events.append({
                                'symbol': symbol,
                                'event_type': 'volume_anomaly',
                                'severity': 'high' if volume_change > 5 else 'medium',
                                'description': f'{symbol} volume increased {volume_change:.0%}',
                                'timestamp': datetime.now().isoformat(),
                                'volume_change': volume_change
                            })
                    
                    # Detect news sentiment anomalies
                    if news:
                        sentiment_scores = [article.get('sentiment_score', 0) for article in news]
                        avg_sentiment = sum(sentiment_scores) / len(sentiment_scores)
                        
                        if abs(avg_sentiment) > 0.3:  # Strong sentiment
                            events.append({
                                'symbol': symbol,
                                'event_type': 'sentiment_anomaly',
                                'severity': 'high' if abs(avg_sentiment) > 0.5 else 'medium',
                                'description': f'{symbol} has strong {"positive" if avg_sentiment > 0 else "negative"} sentiment',
                                'timestamp': datetime.now().isoformat(),
                                'sentiment_score': avg_sentiment
                            })
                
                except Exception as e:
                    logger.warning(f"Event detection failed for {symbol}: {e}")
                    continue
            
            return events
            
        except Exception as e:
            logger.error(f"Market event detection failed: {e}")
            return []
    
    async def get_market_news(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Get general market news with sentiment analysis"""
        try:
            # Try NewsAPI first
            news_data = await self._get_newsapi_data(limit)
            if not news_data:
                # Fallback to FMP
                news_data = await self._get_fmp_news(limit)
            
            if not news_data:
                # Generate sample news for demo
                news_data = self._generate_sample_news(limit)
            
            # Add sentiment analysis
            if news_data:
                news_data = await self._add_sentiment_analysis(news_data)
            
            return news_data
            
        except Exception as e:
            logger.error(f"Market news fetch failed: {e}")
            return self._generate_sample_news(limit)
    
    async def get_symbol_news(self, symbol: str, limit: int = 20) -> List[Dict[str, Any]]:
        """Get news specific to a symbol"""
        try:
            # Try multiple sources
            news_data = []
            
            # NewsAPI with symbol query
            if self.news_client:
                try:
                    articles = self.news_client.get_everything(
                        q=f"{symbol} stock OR {symbol} earnings OR {symbol} financial",
                        language='en',
                        sort_by='publishedAt',
                        page_size=limit
                    )
                    
                    if articles.get('articles'):
                        for article in articles['articles'][:limit]:
                            news_data.append({
                                'title': article.get('title', ''),
                                'description': article.get('description', ''),
                                'url': article.get('url', ''),
                                'source': article.get('source', {}).get('name', 'Unknown'),
                                'published_at': article.get('publishedAt', ''),
                                'symbol': symbol
                            })
                except Exception as e:
                    logger.warning(f"NewsAPI symbol search failed: {e}")
            
            # FMP symbol news
            fmp_news = await self._get_fmp_symbol_news(symbol, limit)
            if fmp_news:
                news_data.extend(fmp_news)
            
            # If no real news, generate sample
            if not news_data:
                news_data = self._generate_symbol_news(symbol, limit)
            
            # Add sentiment analysis
            news_data = await self._add_sentiment_analysis(news_data)
            
            return news_data[:limit]
            
        except Exception as e:
            logger.error(f"Symbol news fetch failed for {symbol}: {e}")
            return self._generate_symbol_news(symbol, limit)
    
    async def _get_newsapi_data(self, limit: int) -> List[Dict[str, Any]]:
        """Fetch news from NewsAPI"""
        try:
            if not self.news_client:
                return []
            
            # Get business headlines
            headlines = self.news_client.get_top_headlines(
                category='business',
                language='en',
                page_size=limit
            )
            
            if not headlines.get('articles'):
                return []
            
            news_data = []
            for article in headlines['articles']:
                news_data.append({
                    'title': article.get('title', ''),
                    'description': article.get('description', ''),
                    'url': article.get('url', ''),
                    'source': article.get('source', {}).get('name', 'Unknown'),
                    'published_at': article.get('publishedAt', ''),
                    'symbol': None
                })
            
            return news_data
            
        except Exception as e:
            logger.error(f"NewsAPI fetch failed: {e}")
            return []
    
    async def _get_fmp_news(self, limit: int) -> List[Dict[str, Any]]:
        """Fetch news from Financial Modeling Prep"""
        try:
            if self.fmp_api_key == "demo":
                return []
            
            session = await self._get_session()
            url = f"https://financialmodelingprep.com/api/v3/stock_news"
            params = {
                'apikey': self.fmp_api_key,
                'limit': limit
            }
            
            async with session.get(url, params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    
                    news_data = []
                    for article in data[:limit]:
                        news_data.append({
                            'title': article.get('title', ''),
                            'description': article.get('text', '')[:200] + '...',
                            'url': article.get('url', ''),
                            'source': article.get('site', 'FMP'),
                            'published_at': article.get('publishedDate', ''),
                            'symbol': article.get('symbol')
                        })
                    
                    return news_data
            
            return []
            
        except Exception as e:
            logger.error(f"FMP news fetch failed: {e}")
            return []
    
    async def _get_fmp_symbol_news(self, symbol: str, limit: int) -> List[Dict[str, Any]]:
        """Fetch symbol-specific news from FMP"""
        try:
            if self.fmp_api_key == "demo":
                return []
            
            session = await self._get_session()
            url = f"https://financialmodelingprep.com/api/v3/stock_news"
            params = {
                'apikey': self.fmp_api_key,
                'tickers': symbol,
                'limit': limit
            }
            
            async with session.get(url, params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    
                    news_data = []
                    for article in data[:limit]:
                        news_data.append({
                            'title': article.get('title', ''),
                            'description': article.get('text', '')[:200] + '...',
                            'url': article.get('url', ''),
                            'source': article.get('site', 'FMP'),
                            'published_at': article.get('publishedDate', ''),
                            'symbol': symbol
                        })
                    
                    return news_data
            
            return []
            
        except Exception as e:
            logger.error(f"FMP symbol news fetch failed: {e}")
            return []
    
    def _generate_sample_news(self, limit: int) -> List[Dict[str, Any]]:
        """Generate sample news for demo"""
        sample_headlines = [
            "Federal Reserve Signals Potential Rate Changes Amid Economic Uncertainty",
            "Technology Stocks Rally on Strong Earnings Reports",
            "Oil Prices Surge Following Geopolitical Tensions",
            "Banking Sector Shows Resilience Despite Market Volatility",
            "Cryptocurrency Market Experiences Significant Movement",
            "Manufacturing Data Indicates Economic Recovery Momentum",
            "Healthcare Stocks Gain on Breakthrough Research News",
            "Energy Transition Investments Drive Clean Tech Surge",
            "Consumer Spending Patterns Shift in Post-Pandemic Economy",
            "Global Supply Chain Disruptions Impact Multiple Sectors",
            "AI and Machine Learning Companies Attract Major Investments",
            "Real Estate Market Shows Mixed Signals Across Regions",
            "Automotive Industry Accelerates Electric Vehicle Adoption",
            "Biotech Sector Advances with New Drug Approvals",
            "Financial Services Embrace Digital Transformation Trends"
        ]
        
        news_data = []
        for i in range(min(limit, len(sample_headlines))):
            published_time = datetime.now() - timedelta(hours=i)
            news_data.append({
                'title': sample_headlines[i],
                'description': f"Market analysis and insights on {sample_headlines[i].lower()}. This represents current market sentiment and potential impacts on trading strategies.",
                'url': f"https://example.com/news/{i}",
                'source': 'Market Analysis',
                'published_at': published_time.isoformat(),
                'symbol': None
            })
        
        return news_data
    
    def _generate_symbol_news(self, symbol: str, limit: int) -> List[Dict[str, Any]]:
        """Generate sample symbol-specific news"""
        news_templates = [
            f"{symbol} Reports Strong Quarterly Earnings Beat Expectations",
            f"Analysts Upgrade {symbol} Rating Following Strategic Announcement",
            f"{symbol} Announces New Product Launch with Market Implications",
            f"Institutional Investors Increase {symbol} Holdings",
            f"{symbol} Management Provides Optimistic Forward Guidance",
            f"Market Volatility Creates Opportunity in {symbol} Stock",
            f"{symbol} Expands Operations with Strategic Acquisition",
            f"Technical Analysis Suggests Bullish Pattern for {symbol}",
            f"{symbol} Dividend Announcement Attracts Income Investors",
            f"Options Activity Surges in {symbol} Ahead of Earnings"
        ]
        
        news_data = []
        for i in range(min(limit, len(news_templates))):
            published_time = datetime.now() - timedelta(hours=i)
            news_data.append({
                'title': news_templates[i],
                'description': f"Detailed analysis of recent developments affecting {symbol} stock performance and future outlook.",
                'url': f"https://example.com/news/{symbol.lower()}/{i}",
                'source': 'Financial Analysis',
                'published_at': published_time.isoformat(),
                'symbol': symbol
            })
        
        return news_data
    
    async def _add_sentiment_analysis(self, news_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Add sentiment analysis to news articles"""
        try:
            if not news_data:
                return news_data
            
            # Extract texts for sentiment analysis
            texts = []
            for article in news_data:
                # Combine title and description for better sentiment analysis
                text = f"{article.get('title', '')} {article.get('description', '')}"
                texts.append(text)
            
            # Analyze sentiment
            sentiment_results = ai_engine.sentiment_analyzer.analyze_sentiment(texts)
            
            # Add sentiment to each article
            for i, article in enumerate(news_data):
                if i < len(sentiment_results):
                    sentiment = sentiment_results[i]
                    article.update({
                        'sentiment_score': sentiment.get('sentiment_score', 0.0),
                        'sentiment_label': sentiment.get('label', 'neutral'),
                        'sentiment_confidence': sentiment.get('confidence', 0.0)
                    })
                else:
                    article.update({
                        'sentiment_score': 0.0,
                        'sentiment_label': 'neutral',
                        'sentiment_confidence': 0.0
                    })
            
            return news_data
            
        except Exception as e:
            logger.error(f"Sentiment analysis failed: {e}")
            # Return news without sentiment if analysis fails
            for article in news_data:
                article.update({
                    'sentiment_score': 0.0,
                    'sentiment_label': 'neutral',
                    'sentiment_confidence': 0.0
                })
            return news_data
    
    async def get_news_sentiment_summary(self, symbol: str = None) -> Dict[str, Any]:
        """Get sentiment summary for market or specific symbol"""
        try:
            if symbol:
                news_data = await self.get_symbol_news(symbol, limit=50)
            else:
                news_data = await self.get_market_news(limit=100)
            
            if not news_data:
                return {
                    'overall_sentiment': 0.0,
                    'sentiment_distribution': {'positive': 0, 'negative': 0, 'neutral': 0},
                    'confidence': 0.0,
                    'total_articles': 0
                }
            
            # Calculate sentiment metrics
            sentiment_scores = [article.get('sentiment_score', 0.0) for article in news_data]
            sentiment_labels = [article.get('sentiment_label', 'neutral') for article in news_data]
            
            overall_sentiment = sum(sentiment_scores) / len(sentiment_scores) if sentiment_scores else 0.0
            
            sentiment_distribution = {
                'positive': len([s for s in sentiment_labels if s in ['positive', 'bullish']]),
                'negative': len([s for s in sentiment_labels if s in ['negative', 'bearish']]),
                'neutral': len([s for s in sentiment_labels if s == 'neutral'])
            }
            
            # Calculate confidence based on agreement
            confidence_scores = [article.get('sentiment_confidence', 0.0) for article in news_data]
            average_confidence = sum(confidence_scores) / len(confidence_scores) if confidence_scores else 0.0
            
            return {
                'overall_sentiment': round(overall_sentiment, 3),
                'sentiment_distribution': sentiment_distribution,
                'confidence': round(average_confidence, 3),
                'total_articles': len(news_data),
                'symbol': symbol,
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Sentiment summary failed: {e}")
            return {
                'overall_sentiment': 0.0,
                'sentiment_distribution': {'positive': 0, 'negative': 0, 'neutral': 0},
                'confidence': 0.0,
                'total_articles': 0
            }

# Global service instance
news_service = NewsService() 