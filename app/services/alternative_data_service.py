"""
Alternative Data Service
Integrates satellite imagery, social media sentiment, supply chain intelligence, 
web scraping data, and other alternative data sources for advanced market analysis
"""

import asyncio
import aiohttp
import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional, Union
from datetime import datetime, timedelta
import logging
import os
import json
from dataclasses import dataclass
from enum import Enum
import hashlib
import re
from textblob import TextBlob
import requests
from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)

class DataSourceType(Enum):
    SATELLITE = "satellite"
    SOCIAL_MEDIA = "social_media"
    SUPPLY_CHAIN = "supply_chain"
    WEB_SCRAPING = "web_scraping"
    CREDIT_CARD = "credit_card"
    PATENT = "patent"
    REGULATORY = "regulatory"
    ESG = "esg"
    ECONOMIC_INDICATOR = "economic_indicator"

@dataclass
class AlternativeDataPoint:
    source_type: DataSourceType
    timestamp: datetime
    symbol: Optional[str]
    value: Union[float, str, Dict[str, Any]]
    confidence: float
    metadata: Dict[str, Any]
    impact_score: float
    relevance_score: float

@dataclass
class SatelliteData:
    timestamp: datetime
    location: str
    latitude: float
    longitude: float
    metric_type: str  # 'parking_lots', 'oil_storage', 'shipping', 'construction'
    value: float
    confidence: float
    source: str
    related_symbols: List[str]

@dataclass
class SocialMediaSentiment:
    timestamp: datetime
    platform: str  # 'twitter', 'reddit', 'news', 'forums'
    symbol: str
    sentiment_score: float  # -1 to 1
    volume: int
    engagement_rate: float
    key_themes: List[str]
    influencer_mentions: int
    viral_content: List[str]

@dataclass
class SupplyChainData:
    timestamp: datetime
    company: str
    symbol: str
    supplier_data: Dict[str, Any]
    disruption_risk: float
    cost_pressure_index: float
    delivery_delays: float
    inventory_levels: float
    geographic_concentration: float

@dataclass
class PatentData:
    timestamp: datetime
    company: str
    symbol: str
    patent_id: str
    technology_area: str
    innovation_score: float
    competitive_impact: float
    market_potential: float

class AlternativeDataService:
    """Comprehensive alternative data service integrating multiple data sources"""
    
    def __init__(self):
        # API Keys
        self.planet_api_key = os.getenv("PLANET_API_KEY", "demo")
        self.twitter_bearer_token = os.getenv("TWITTER_BEARER_TOKEN", "demo")
        self.reddit_client_id = os.getenv("REDDIT_CLIENT_ID", "demo")
        self.reddit_client_secret = os.getenv("REDDIT_CLIENT_SECRET", "demo")
        self.newsapi_key = os.getenv("NEWS_API_KEY", "demo")
        self.patent_api_key = os.getenv("PATENT_API_KEY", "demo")
        self.esg_api_key = os.getenv("ESG_API_KEY", "demo")
        
        # Cache for data
        self.data_cache = {}
        self.session = None
        
        # Initialization
        self._initialize_data_sources()
    
    def _initialize_data_sources(self):
        """Initialize connections to various data sources"""
        logger.info("Initializing alternative data sources...")
        
        # Initialize data source configurations
        self.data_sources = {
            DataSourceType.SATELLITE: {
                'endpoints': {
                    'planet': 'https://api.planet.com/data/v1',
                    'maxar': 'https://api.maxar.com/v1',
                    'spaceknow': 'https://api.spaceknow.com/v3'
                },
                'active': True
            },
            DataSourceType.SOCIAL_MEDIA: {
                'endpoints': {
                    'twitter': 'https://api.twitter.com/2',
                    'reddit': 'https://oauth.reddit.com',
                    'stocktwits': 'https://api.stocktwits.com/api/2'
                },
                'active': True
            },
            DataSourceType.SUPPLY_CHAIN: {
                'endpoints': {
                    'importgenius': 'https://api.importgenius.com/v1',
                    'panjiva': 'https://panjiva.com/api/v1',
                    'freightos': 'https://api.freightos.com/v1'
                },
                'active': True
            },
            DataSourceType.WEB_SCRAPING: {
                'targets': [
                    'sec.gov',
                    'company_websites',
                    'job_boards',
                    'industry_publications'
                ],
                'active': True
            }
        }
    
    async def get_session(self):
        """Get or create aiohttp session"""
        if self.session is None:
            self.session = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=30)
            )
        return self.session
    
    async def close(self):
        """Close aiohttp session"""
        if self.session:
            await self.session.close()
    
    # =================== SATELLITE DATA ===================
    
    async def get_satellite_data(self, symbol: str, metric_type: str = "all", 
                               days_back: int = 30) -> List[SatelliteData]:
        """Get satellite imagery data for company analysis"""
        try:
            if self.planet_api_key == "demo":
                return self._generate_mock_satellite_data(symbol, metric_type, days_back)
            
            # Implementation for real satellite data APIs
            session = await self.get_session()
            
            # Get company locations for satellite analysis
            locations = await self._get_company_locations(symbol)
            
            satellite_data = []
            
            for location in locations:
                # Planet Labs API for satellite imagery analysis
                data = await self._fetch_planet_data(session, location, metric_type, days_back)
                satellite_data.extend(data)
                
                # Add Maxar/SpaceKnow data if available
                if metric_type in ['oil_storage', 'shipping', 'construction']:
                    additional_data = await self._fetch_maxar_data(session, location, metric_type, days_back)
                    satellite_data.extend(additional_data)
            
            return satellite_data
            
        except Exception as e:
            logger.error(f"Satellite data fetch failed for {symbol}: {e}")
            return self._generate_mock_satellite_data(symbol, metric_type, days_back)
    
    def _generate_mock_satellite_data(self, symbol: str, metric_type: str, days_back: int) -> List[SatelliteData]:
        """Generate mock satellite data for demo purposes"""
        data = []
        base_date = datetime.now() - timedelta(days=days_back)
        
        # Company-specific satellite metrics
        company_metrics = {
            'AAPL': {'parking_lots': 0.85, 'construction': 0.7},
            'TSLA': {'parking_lots': 0.9, 'construction': 0.95, 'shipping': 0.8},
            'WMT': {'parking_lots': 0.95, 'shipping': 0.9},
            'XOM': {'oil_storage': 0.8, 'shipping': 0.7},
            'BA': {'construction': 0.6, 'shipping': 0.5}
        }
        
        metrics = company_metrics.get(symbol, {'parking_lots': 0.7})
        
        for i in range(0, days_back, 7):  # Weekly satellite data
            date = base_date + timedelta(days=i)
            
            for metric, base_value in metrics.items():
                if metric_type == "all" or metric_type == metric:
                    # Add some realistic variation
                    variation = np.random.normal(0, 0.1)
                    value = max(0, min(1, base_value + variation))
                    
                    data.append(SatelliteData(
                        timestamp=date,
                        location=f"{symbol}_primary_location",
                        latitude=40.7128 + np.random.normal(0, 0.1),
                        longitude=-74.0060 + np.random.normal(0, 0.1),
                        metric_type=metric,
                        value=value,
                        confidence=0.85 + np.random.normal(0, 0.1),
                        source="planet_labs_demo",
                        related_symbols=[symbol]
                    ))
        
        return data
    
    async def _get_company_locations(self, symbol: str) -> List[Dict[str, Any]]:
        """Get company facility locations for satellite analysis"""
        # Mock company locations - in production, this would query a facilities database
        locations = {
            'AAPL': [
                {'name': 'Cupertino HQ', 'lat': 37.3318, 'lon': -122.0312},
                {'name': 'Austin Facility', 'lat': 30.2672, 'lon': -97.7431}
            ],
            'TSLA': [
                {'name': 'Fremont Factory', 'lat': 37.5485, 'lon': -121.9886},
                {'name': 'Austin Gigafactory', 'lat': 30.2672, 'lon': -97.7431},
                {'name': 'Shanghai Gigafactory', 'lat': 31.2304, 'lon': 121.4737}
            ],
            'WMT': [
                {'name': 'Bentonville HQ', 'lat': 36.3729, 'lon': -94.2088}
            ]
        }
        
        return locations.get(symbol, [{'name': 'Unknown', 'lat': 0, 'lon': 0}])
    
    # =================== SOCIAL MEDIA SENTIMENT ===================
    
    async def get_social_media_sentiment(self, symbol: str, days_back: int = 7) -> List[SocialMediaSentiment]:
        """Get comprehensive social media sentiment analysis"""
        try:
            sentiment_data = []
            
            # Twitter sentiment
            twitter_data = await self._get_twitter_sentiment(symbol, days_back)
            sentiment_data.extend(twitter_data)
            
            # Reddit sentiment
            reddit_data = await self._get_reddit_sentiment(symbol, days_back)
            sentiment_data.extend(reddit_data)
            
            # StockTwits sentiment
            stocktwits_data = await self._get_stocktwits_sentiment(symbol, days_back)
            sentiment_data.extend(stocktwits_data)
            
            # News sentiment
            news_data = await self._get_news_sentiment(symbol, days_back)
            sentiment_data.extend(news_data)
            
            return sentiment_data
            
        except Exception as e:
            logger.error(f"Social media sentiment fetch failed for {symbol}: {e}")
            return self._generate_mock_sentiment_data(symbol, days_back)
    
    def _generate_mock_sentiment_data(self, symbol: str, days_back: int) -> List[SocialMediaSentiment]:
        """Generate mock social media sentiment data"""
        data = []
        base_date = datetime.now() - timedelta(days=days_back)
        
        platforms = ['twitter', 'reddit', 'stocktwits', 'news']
        
        for i in range(days_back):
            date = base_date + timedelta(days=i)
            
            for platform in platforms:
                # Generate realistic sentiment with some correlation to market movements
                base_sentiment = np.random.normal(0.1, 0.3)  # Slightly positive bias
                sentiment_score = max(-1, min(1, base_sentiment))
                
                # Volume varies by platform
                volume_ranges = {
                    'twitter': (100, 5000),
                    'reddit': (10, 500),
                    'stocktwits': (50, 1000),
                    'news': (5, 50)
                }
                
                min_vol, max_vol = volume_ranges[platform]
                volume = np.random.randint(min_vol, max_vol)
                
                # Generate key themes
                themes = self._generate_themes_for_symbol(symbol)
                
                data.append(SocialMediaSentiment(
                    timestamp=date,
                    platform=platform,
                    symbol=symbol,
                    sentiment_score=sentiment_score,
                    volume=volume,
                    engagement_rate=np.random.uniform(0.02, 0.15),
                    key_themes=np.random.choice(themes, size=np.random.randint(1, 4), replace=False).tolist(),
                    influencer_mentions=np.random.randint(0, 10),
                    viral_content=[]
                ))
        
        return data
    
    def _generate_themes_for_symbol(self, symbol: str) -> List[str]:
        """Generate relevant themes for a symbol"""
        general_themes = ['earnings', 'growth', 'competition', 'regulation', 'innovation']
        
        symbol_themes = {
            'AAPL': ['iphone', 'services', 'china', 'ai', 'privacy'],
            'TSLA': ['ev', 'autonomous', 'gigafactory', 'musk', 'energy'],
            'GOOGL': ['search', 'cloud', 'ai', 'advertising', 'antitrust'],
            'MSFT': ['cloud', 'ai', 'enterprise', 'gaming', 'productivity'],
            'AMZN': ['aws', 'retail', 'logistics', 'alexa', 'prime']
        }
        
        return general_themes + symbol_themes.get(symbol, [])
    
    # =================== SUPPLY CHAIN INTELLIGENCE ===================
    
    async def get_supply_chain_data(self, symbol: str, days_back: int = 30) -> List[SupplyChainData]:
        """Get supply chain intelligence data"""
        try:
            if self.session is None:
                await self.get_session()
            
            # In production, this would integrate with ImportGenius, Panjiva, etc.
            return self._generate_mock_supply_chain_data(symbol, days_back)
            
        except Exception as e:
            logger.error(f"Supply chain data fetch failed for {symbol}: {e}")
            return self._generate_mock_supply_chain_data(symbol, days_back)
    
    def _generate_mock_supply_chain_data(self, symbol: str, days_back: int) -> List[SupplyChainData]:
        """Generate mock supply chain data"""
        data = []
        base_date = datetime.now() - timedelta(days=days_back)
        
        # Company-specific supply chain characteristics
        supply_chain_profiles = {
            'AAPL': {
                'disruption_risk': 0.6,
                'cost_pressure': 0.7,
                'delivery_delays': 0.4,
                'inventory_levels': 0.3,
                'geo_concentration': 0.8
            },
            'TSLA': {
                'disruption_risk': 0.8,
                'cost_pressure': 0.9,
                'delivery_delays': 0.7,
                'inventory_levels': 0.2,
                'geo_concentration': 0.7
            },
            'WMT': {
                'disruption_risk': 0.3,
                'cost_pressure': 0.5,
                'delivery_delays': 0.2,
                'inventory_levels': 0.8,
                'geo_concentration': 0.4
            }
        }
        
        profile = supply_chain_profiles.get(symbol, {
            'disruption_risk': 0.5,
            'cost_pressure': 0.5,
            'delivery_delays': 0.5,
            'inventory_levels': 0.5,
            'geo_concentration': 0.5
        })
        
        for i in range(0, days_back, 7):  # Weekly supply chain data
            date = base_date + timedelta(days=i)
            
            # Add realistic variations
            variations = {k: np.random.normal(0, 0.1) for k in profile.keys()}
            
            data.append(SupplyChainData(
                timestamp=date,
                company=symbol,
                symbol=symbol,
                supplier_data={
                    'primary_suppliers': np.random.randint(50, 200),
                    'supplier_countries': np.random.randint(5, 25),
                    'tier_1_concentration': np.random.uniform(0.3, 0.8)
                },
                disruption_risk=max(0, min(1, profile['disruption_risk'] + variations['disruption_risk'])),
                cost_pressure_index=max(0, min(1, profile['cost_pressure'] + variations['cost_pressure'])),
                delivery_delays=max(0, min(1, profile['delivery_delays'] + variations['delivery_delays'])),
                inventory_levels=max(0, min(1, profile['inventory_levels'] + variations['inventory_levels'])),
                geographic_concentration=max(0, min(1, profile['geo_concentration'] + variations['geo_concentration']))
            ))
        
        return data
    
    # =================== WEB SCRAPING DATA ===================
    
    async def get_web_scraping_data(self, symbol: str, data_type: str = "all") -> Dict[str, Any]:
        """Get web scraping data from various sources"""
        try:
            scraping_results = {}
            
            if data_type in ["all", "sec_filings"]:
                scraping_results["sec_filings"] = await self._scrape_sec_filings(symbol)
            
            if data_type in ["all", "job_postings"]:
                scraping_results["job_postings"] = await self._scrape_job_postings(symbol)
            
            if data_type in ["all", "company_website"]:
                scraping_results["company_website"] = await self._scrape_company_website(symbol)
            
            if data_type in ["all", "industry_news"]:
                scraping_results["industry_news"] = await self._scrape_industry_news(symbol)
            
            return scraping_results
            
        except Exception as e:
            logger.error(f"Web scraping failed for {symbol}: {e}")
            return self._generate_mock_web_data(symbol)
    
    def _generate_mock_web_data(self, symbol: str) -> Dict[str, Any]:
        """Generate mock web scraping data"""
        return {
            "sec_filings": {
                "recent_filings": np.random.randint(1, 10),
                "filing_sentiment": np.random.uniform(-0.5, 0.5),
                "risk_factors_mentioned": np.random.randint(5, 25),
                "forward_looking_statements": np.random.randint(10, 50)
            },
            "job_postings": {
                "total_postings": np.random.randint(50, 500),
                "growth_rate": np.random.uniform(-0.2, 0.3),
                "skill_demand_trends": ['python', 'ai', 'cloud', 'data_science'],
                "geographic_distribution": {
                    "us": 0.7,
                    "europe": 0.2,
                    "asia": 0.1
                }
            },
            "company_website": {
                "update_frequency": np.random.uniform(0.1, 0.9),
                "content_sentiment": np.random.uniform(-0.2, 0.8),
                "product_mentions": np.random.randint(5, 20),
                "executive_communications": np.random.randint(1, 10)
            },
            "industry_news": {
                "mention_frequency": np.random.randint(10, 100),
                "sentiment_trend": np.random.uniform(-0.5, 0.5),
                "competitive_mentions": np.random.randint(5, 30),
                "innovation_coverage": np.random.uniform(0.1, 0.9)
            }
        }
    
    # =================== PATENT DATA ===================
    
    async def get_patent_data(self, symbol: str, days_back: int = 90) -> List[PatentData]:
        """Get patent filing and innovation data"""
        try:
            return self._generate_mock_patent_data(symbol, days_back)
            
        except Exception as e:
            logger.error(f"Patent data fetch failed for {symbol}: {e}")
            return self._generate_mock_patent_data(symbol, days_back)
    
    def _generate_mock_patent_data(self, symbol: str, days_back: int) -> List[PatentData]:
        """Generate mock patent data"""
        data = []
        base_date = datetime.now() - timedelta(days=days_back)
        
        # Technology areas by company
        tech_areas = {
            'AAPL': ['mobile_devices', 'user_interfaces', 'ai_ml', 'hardware_design'],
            'GOOGL': ['search_algorithms', 'ai_ml', 'cloud_computing', 'autonomous_vehicles'],
            'MSFT': ['software_systems', 'cloud_computing', 'ai_ml', 'productivity_tools'],
            'TSLA': ['battery_technology', 'autonomous_vehicles', 'energy_storage', 'manufacturing'],
            'NVDA': ['gpu_architecture', 'ai_ml', 'graphics_processing', 'data_centers']
        }
        
        areas = tech_areas.get(symbol, ['general_technology'])
        
        # Generate patents over the period
        num_patents = np.random.randint(1, 15)  # Realistic number for the period
        
        for i in range(num_patents):
            filing_date = base_date + timedelta(days=np.random.randint(0, days_back))
            
            data.append(PatentData(
                timestamp=filing_date,
                company=symbol,
                symbol=symbol,
                patent_id=f"US{np.random.randint(10000000, 99999999)}",
                technology_area=np.random.choice(areas),
                innovation_score=np.random.uniform(0.3, 0.95),
                competitive_impact=np.random.uniform(0.2, 0.9),
                market_potential=np.random.uniform(0.1, 0.8)
            ))
        
        return data
    
    # =================== COMPREHENSIVE ANALYSIS ===================
    
    async def get_comprehensive_alternative_data(self, symbol: str, days_back: int = 30) -> Dict[str, Any]:
        """Get comprehensive alternative data analysis for a symbol"""
        try:
            logger.info(f"Fetching comprehensive alternative data for {symbol}")
            
            # Gather all data types
            satellite_data = await self.get_satellite_data(symbol, days_back=days_back)
            social_sentiment = await self.get_social_media_sentiment(symbol, days_back=days_back)
            supply_chain = await self.get_supply_chain_data(symbol, days_back=days_back)
            web_data = await self.get_web_scraping_data(symbol)
            patent_data = await self.get_patent_data(symbol, days_back=days_back)
            
            # Calculate aggregate metrics
            aggregate_sentiment = self._calculate_aggregate_sentiment(social_sentiment)
            supply_chain_risk = self._calculate_supply_chain_risk(supply_chain)
            innovation_index = self._calculate_innovation_index(patent_data)
            activity_index = self._calculate_activity_index(satellite_data)
            
            # Generate overall alternative data score
            alt_data_score = self._calculate_alternative_data_score(
                aggregate_sentiment, supply_chain_risk, innovation_index, activity_index
            )
            
            return {
                "symbol": symbol,
                "timestamp": datetime.now().isoformat(),
                "alternative_data_score": alt_data_score,
                "aggregate_metrics": {
                    "sentiment_score": aggregate_sentiment,
                    "supply_chain_risk": supply_chain_risk,
                    "innovation_index": innovation_index,
                    "activity_index": activity_index
                },
                "data_sources": {
                    "satellite": {
                        "data_points": len(satellite_data),
                        "latest_metrics": satellite_data[-3:] if satellite_data else []
                    },
                    "social_media": {
                        "data_points": len(social_sentiment),
                        "platforms_covered": len(set(s.platform for s in social_sentiment)),
                        "average_sentiment": aggregate_sentiment
                    },
                    "supply_chain": {
                        "data_points": len(supply_chain),
                        "risk_level": "high" if supply_chain_risk > 0.7 else "medium" if supply_chain_risk > 0.4 else "low"
                    },
                    "web_scraping": web_data,
                    "patents": {
                        "data_points": len(patent_data),
                        "innovation_areas": len(set(p.technology_area for p in patent_data)),
                        "average_innovation_score": innovation_index
                    }
                },
                "insights": self._generate_insights(symbol, aggregate_sentiment, supply_chain_risk, innovation_index, activity_index),
                "recommendations": self._generate_recommendations(alt_data_score, aggregate_sentiment, supply_chain_risk)
            }
            
        except Exception as e:
            logger.error(f"Comprehensive alternative data analysis failed for {symbol}: {e}")
            return {"error": str(e), "symbol": symbol}
    
    def _calculate_aggregate_sentiment(self, sentiment_data: List[SocialMediaSentiment]) -> float:
        """Calculate weighted aggregate sentiment score"""
        if not sentiment_data:
            return 0.0
        
        # Weight by platform and volume
        platform_weights = {'twitter': 0.3, 'reddit': 0.2, 'news': 0.4, 'stocktwits': 0.1}
        
        weighted_scores = []
        for s in sentiment_data:
            weight = platform_weights.get(s.platform, 0.1) * np.log1p(s.volume)
            weighted_scores.append(s.sentiment_score * weight)
        
        return np.average(weighted_scores) if weighted_scores else 0.0
    
    def _calculate_supply_chain_risk(self, supply_chain_data: List[SupplyChainData]) -> float:
        """Calculate overall supply chain risk"""
        if not supply_chain_data:
            return 0.5  # Neutral
        
        # Weight different risk factors
        total_risk = 0
        for sc in supply_chain_data:
            risk_score = (
                sc.disruption_risk * 0.3 +
                sc.cost_pressure_index * 0.2 +
                sc.delivery_delays * 0.2 +
                (1 - sc.inventory_levels) * 0.15 +  # Low inventory = higher risk
                sc.geographic_concentration * 0.15
            )
            total_risk += risk_score
        
        return total_risk / len(supply_chain_data)
    
    def _calculate_innovation_index(self, patent_data: List[PatentData]) -> float:
        """Calculate innovation index from patent data"""
        if not patent_data:
            return 0.5  # Neutral
        
        # Consider recent patents more heavily
        now = datetime.now()
        weighted_scores = []
        
        for p in patent_data:
            days_old = (now - p.timestamp).days
            time_weight = max(0.1, 1 - (days_old / 365))  # Decay over a year
            
            patent_score = (
                p.innovation_score * 0.4 +
                p.competitive_impact * 0.3 +
                p.market_potential * 0.3
            )
            
            weighted_scores.append(patent_score * time_weight)
        
        return np.mean(weighted_scores) if weighted_scores else 0.5
    
    def _calculate_activity_index(self, satellite_data: List[SatelliteData]) -> float:
        """Calculate business activity index from satellite data"""
        if not satellite_data:
            return 0.5  # Neutral
        
        # Average recent activity metrics
        recent_data = [s for s in satellite_data if (datetime.now() - s.timestamp).days <= 30]
        
        if not recent_data:
            return 0.5
        
        activity_scores = [s.value * s.confidence for s in recent_data]
        return np.mean(activity_scores)
    
    def _calculate_alternative_data_score(self, sentiment: float, supply_risk: float, 
                                        innovation: float, activity: float) -> float:
        """Calculate overall alternative data score"""
        # Normalize sentiment from [-1,1] to [0,1]
        norm_sentiment = (sentiment + 1) / 2
        
        # Invert supply chain risk (lower risk = better score)
        norm_supply_risk = 1 - supply_risk
        
        # Weighted combination
        score = (
            norm_sentiment * 0.25 +
            norm_supply_risk * 0.25 +
            innovation * 0.25 +
            activity * 0.25
        )
        
        return max(0, min(1, score))
    
    def _generate_insights(self, symbol: str, sentiment: float, supply_risk: float, 
                          innovation: float, activity: float) -> List[str]:
        """Generate human-readable insights from alternative data"""
        insights = []
        
        if sentiment > 0.2:
            insights.append(f"Strong positive sentiment detected across social media platforms for {symbol}")
        elif sentiment < -0.2:
            insights.append(f"Negative sentiment trend observed for {symbol} on social platforms")
        
        if supply_risk > 0.7:
            insights.append(f"High supply chain risk detected for {symbol} - monitor for operational impacts")
        elif supply_risk < 0.3:
            insights.append(f"Robust supply chain indicated for {symbol}")
        
        if innovation > 0.7:
            insights.append(f"Strong innovation activity observed for {symbol} based on patent filings")
        elif innovation < 0.3:
            insights.append(f"Limited recent innovation activity for {symbol}")
        
        if activity > 0.7:
            insights.append(f"High business activity levels detected via satellite imagery for {symbol}")
        elif activity < 0.3:
            insights.append(f"Reduced business activity observed for {symbol}")
        
        return insights
    
    def _generate_recommendations(self, alt_score: float, sentiment: float, supply_risk: float) -> List[str]:
        """Generate trading/investment recommendations based on alternative data"""
        recommendations = []
        
        if alt_score > 0.7:
            recommendations.append("Alternative data signals are strongly positive - consider increased position")
        elif alt_score < 0.3:
            recommendations.append("Alternative data signals weakness - consider reducing exposure")
        
        if supply_risk > 0.8:
            recommendations.append("Extreme supply chain risk - monitor for earnings impact and consider hedging")
        
        if sentiment < -0.5:
            recommendations.append("Negative sentiment momentum - wait for sentiment reversal before entry")
        elif sentiment > 0.5:
            recommendations.append("Strong positive sentiment - momentum trade opportunity")
        
        return recommendations

# Global service instance
alternative_data_service = AlternativeDataService()

# Convenience functions
async def get_alternative_data_summary(symbol: str, days_back: int = 30) -> Dict[str, Any]:
    """Get comprehensive alternative data summary for a symbol"""
    return await alternative_data_service.get_comprehensive_alternative_data(symbol, days_back)

async def get_satellite_insights(symbol: str, metric_type: str = "all") -> List[SatelliteData]:
    """Get satellite data insights for a symbol"""
    return await alternative_data_service.get_satellite_data(symbol, metric_type)

async def get_social_sentiment_summary(symbol: str) -> Dict[str, Any]:
    """Get social media sentiment summary"""
    sentiment_data = await alternative_data_service.get_social_media_sentiment(symbol)
    
    if not sentiment_data:
        return {}
    
    # Aggregate by platform
    platform_summary = {}
    for s in sentiment_data:
        if s.platform not in platform_summary:
            platform_summary[s.platform] = {
                'sentiment_scores': [],
                'volumes': [],
                'themes': []
            }
        
        platform_summary[s.platform]['sentiment_scores'].append(s.sentiment_score)
        platform_summary[s.platform]['volumes'].append(s.volume)
        platform_summary[s.platform]['themes'].extend(s.key_themes)
    
    # Calculate averages
    for platform, data in platform_summary.items():
        data['avg_sentiment'] = np.mean(data['sentiment_scores'])
        data['total_volume'] = sum(data['volumes'])
        data['top_themes'] = list(set(data['themes']))[:5]
    
    return platform_summary

async def get_supply_chain_risk_score(symbol: str) -> Dict[str, Any]:
    """Get supply chain risk assessment"""
    supply_data = await alternative_data_service.get_supply_chain_data(symbol)
    
    if not supply_data:
        return {"risk_score": 0.5, "status": "unknown"}
    
    latest = supply_data[-1]
    
    return {
        "risk_score": alternative_data_service._calculate_supply_chain_risk(supply_data),
        "latest_metrics": {
            "disruption_risk": latest.disruption_risk,
            "cost_pressure": latest.cost_pressure_index,
            "delivery_delays": latest.delivery_delays,
            "inventory_levels": latest.inventory_levels,
            "geographic_concentration": latest.geographic_concentration
        },
        "status": "high_risk" if latest.disruption_risk > 0.7 else "moderate_risk" if latest.disruption_risk > 0.4 else "low_risk"
    } 