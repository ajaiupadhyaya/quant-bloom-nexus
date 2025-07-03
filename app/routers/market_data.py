from fastapi import APIRouter, Query, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
from datetime import date, datetime
import logging
import yfinance as yf

from ..services.market_data_service import (
    get_market_data, 
    get_real_time_quote, 
    get_market_movers,
    get_sector_performance
)
from ..services.news_service import news_service

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/market-data", tags=["Market Data"])

# Pydantic model for the response data
class HistoricalData(BaseModel):
    date: date
    open: float
    high: float
    low: float
    close: float
    volume: int

@router.get("/quote/{symbol}")
async def get_quote(symbol: str):
    """Get real-time quote for a symbol"""
    try:
        quote = await get_real_time_quote(symbol.upper())
        if not quote:
            raise HTTPException(status_code=404, detail=f"Quote not found for {symbol}")
        
        # Return quote directly as expected by frontend
        return quote
        
    except Exception as e:
        logger.error(f"Quote fetch failed for {symbol}: {e}")
        raise HTTPException(status_code=500, detail=f"Quote fetch failed: {str(e)}")

@router.get("/historical/{symbol}")
async def get_historical_data(
    symbol: str,
    period: str = Query("1y", description="Time period (1d, 5d, 1m, 3m, 6m, 1y, 2y, 5y)"),
    interval: str = Query("1d", description="Data interval (1d, 1h, 5m)")
):
    """Get historical market data for a symbol"""
    try:
        data = await get_market_data(symbol.upper(), period, interval)
        if not data:
            raise HTTPException(status_code=404, detail=f"Historical data not found for {symbol}")
        
        return data
        
    except Exception as e:
        logger.error(f"Historical data fetch failed for {symbol}: {e}")
        raise HTTPException(status_code=500, detail=f"Historical data fetch failed: {str(e)}")

@router.get("/movers")
async def get_movers():
    """Get market movers (gainers, losers, most active)"""
    try:
        movers = await get_market_movers()
        return movers
        
    except Exception as e:
        logger.error(f"Market movers fetch failed: {e}")
        raise HTTPException(status_code=500, detail=f"Market movers fetch failed: {str(e)}")

@router.get("/sectors")
async def get_sectors():
    """Get sector performance data"""
    try:
        sectors = await get_sector_performance()
        return sectors
        
    except Exception as e:
        logger.error(f"Sector performance fetch failed: {e}")
        raise HTTPException(status_code=500, detail=f"Sector performance fetch failed: {str(e)}")

@router.get("/news")
async def get_market_news(limit: int = Query(50, ge=1, le=100)):
    """Get market news with sentiment analysis"""
    try:
        news = await news_service.get_market_news(limit)
        return news
        
    except Exception as e:
        logger.error(f"Market news fetch failed: {e}")
        # Return empty list instead of error to keep terminal working
        return []

@router.get("/news/{symbol}")
async def get_symbol_news(symbol: str, limit: int = Query(20, ge=1, le=50)):
    """Get news for a specific symbol with sentiment analysis"""
    try:
        news = await news_service.get_symbol_news(symbol.upper(), limit)
        return news
        
    except Exception as e:
        logger.error(f"Symbol news fetch failed for {symbol}: {e}")
        return []

@router.get("/sentiment")
async def get_market_sentiment():
    """Get overall market sentiment analysis"""
    try:
        sentiment = await news_service.get_news_sentiment_summary()
        return sentiment
        
    except Exception as e:
        logger.error(f"Market sentiment fetch failed: {e}")
        return {"overall_sentiment": 0.0, "confidence": 0.0}

@router.get("/sentiment/{symbol}")
async def get_symbol_sentiment(symbol: str):
    """Get sentiment analysis for a specific symbol"""
    try:
        sentiment = await news_service.get_news_sentiment_summary(symbol.upper())
        return sentiment
        
    except Exception as e:
        logger.error(f"Symbol sentiment fetch failed for {symbol}: {e}")
        return {"overall_sentiment": 0.0, "confidence": 0.0}

@router.get("/watchlist")
async def get_watchlist_data(symbols: str = Query(..., description="Comma-separated list of symbols")):
    """Get data for multiple symbols (watchlist)"""
    try:
        symbol_list = [s.strip().upper() for s in symbols.split(",") if s.strip()]
        if not symbol_list:
            raise HTTPException(status_code=400, detail="No symbols provided")
        
        # Get quotes for all symbols
        watchlist_data = []
        for symbol in symbol_list:
            try:
                quote = await get_real_time_quote(symbol)
                if quote:
                    watchlist_data.append(quote)
            except Exception as e:
                logger.warning(f"Failed to get quote for {symbol}: {e}")
                continue
        
        return watchlist_data
        
    except Exception as e:
        logger.error(f"Watchlist data fetch failed: {e}")
        raise HTTPException(status_code=500, detail=f"Watchlist data fetch failed: {str(e)}")

@router.get("/overview")
async def get_market_overview():
    """Get comprehensive market overview"""
    try:
        # Get market data concurrently
        import asyncio
        
        movers, sectors, sentiment = await asyncio.gather(
            get_market_movers(),
            get_sector_performance(),
            news_service.get_news_sentiment_summary(),
            return_exceptions=True
        )
        
        # Handle exceptions
        if isinstance(movers, Exception):
            movers = {"gainers": [], "losers": [], "most_active": []}
        if isinstance(sectors, Exception):
            sectors = []
        if isinstance(sentiment, Exception):
            sentiment = {"overall_sentiment": 0.0, "confidence": 0.0}
        
        return {
            "market_movers": movers,
            "sector_performance": sectors,
            "market_sentiment": sentiment,
        }
        
    except Exception as e:
        logger.error(f"Market overview fetch failed: {e}")
        raise HTTPException(status_code=500, detail=f"Market overview fetch failed: {str(e)}")

@router.get("/search")
async def search_symbols(query: str = Query(..., min_length=1)):
    """Search for symbols/companies"""
    try:
        # Simple symbol search - in production, this would use a proper search API
        common_symbols = {
            "AAPL": "Apple Inc.",
            "GOOGL": "Alphabet Inc.",
            "MSFT": "Microsoft Corporation",
            "AMZN": "Amazon.com Inc.",
            "TSLA": "Tesla Inc.",
            "META": "Meta Platforms Inc.",
            "NVDA": "NVIDIA Corporation",
            "NFLX": "Netflix Inc.",
            "AMD": "Advanced Micro Devices",
            "CRM": "Salesforce Inc.",
            "UBER": "Uber Technologies",
            "SNAP": "Snap Inc.",
            "TWTR": "Twitter Inc.",
            "ZOOM": "Zoom Video Communications",
            "SHOP": "Shopify Inc.",
            "SQ": "Block Inc.",
            "PYPL": "PayPal Holdings",
            "ADBE": "Adobe Inc.",
            "ORCL": "Oracle Corporation"
        }
        
        query_upper = query.upper()
        results = []
        
        for symbol, name in common_symbols.items():
            if (query_upper in symbol or 
                query_upper in name.upper() or 
                symbol.startswith(query_upper)):
                results.append({
                    "symbol": symbol,
                    "name": name,
                    "type": "stock"
                })
        
        return results[:10]  # Limit to 10 results
        
    except Exception as e:
        logger.error(f"Symbol search failed for {query}: {e}")
        raise HTTPException(status_code=500, detail=f"Symbol search failed: {str(e)}")