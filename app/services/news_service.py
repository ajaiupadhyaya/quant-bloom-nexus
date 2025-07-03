import asyncio
import aiohttp
import logging
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
import os
import json
from newsapi import NewsApiClient
from .ai_engine import ai_engine

logger = logging.getLogger(__name__)

class NewsService:
    """Complete news service with sentiment analysis"""
    
    def __init__(self):
        self.news_api_key = os.getenv("NEWS_API_KEY", "demo")
        self.fmp_api_key = os.getenv("FMP_API_KEY", "demo")
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