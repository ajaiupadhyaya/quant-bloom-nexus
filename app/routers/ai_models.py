
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Dict
from app.services.sentiment_service import SentimentService
from newsapi import NewsApiClient
import redis
import json
from app.core.config import settings

router = APIRouter()
sentiment_service = SentimentService() # Initialize the sentiment service

# Initialize NewsAPI client
newsapi = NewsApiClient(api_key=settings.NEWS_API_KEY)

# Initialize Redis client
redis_client = redis.Redis(host=settings.REDIS_HOST, port=settings.REDIS_PORT, db=0)

class HeadlineRequest(BaseModel):
    headlines: List[str]

class SentimentResponse(BaseModel):
    headline: str
    sentiment: str

@router.get("/ai_models/predict/{symbol}")
def predict(symbol: str):
    # Your logic for AI model prediction
    return {"symbol": symbol, "prediction": "buy"}

@router.post("/sentiment/analyze", response_model=List[SentimentResponse])
def analyze_news_sentiment(request: HeadlineRequest):
    """
    Analyzes the sentiment of a list of news headlines.
    """
    try:
        results = sentiment_service.analyze_headlines(request.headlines)
        return results
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Sentiment analysis failed: {e}")

@router.get("/news/sentiment", response_model=List[SentimentResponse])
def get_news_with_sentiment():
    """
    Fetches the latest news headlines, analyzes their sentiment, and returns the results.
    Results are cached in Redis for 5 minutes.
    """
    cache_key = "latest_news_sentiment"
    cached_data = redis_client.get(cache_key)

    if cached_data:
        print("Returning news sentiment from cache.")
        return json.loads(cached_data)

    try:
        # Fetch top headlines from NewsAPI
        # You might want to filter by category, language, etc., based on your needs
        top_headlines = newsapi.get_top_headlines(language='en', page_size=50)
        
        if not top_headlines or top_headlines['status'] != 'ok' or not top_headlines['articles']:
            raise HTTPException(status_code=500, detail="Failed to fetch headlines from NewsAPI.")

        headlines = [article['title'] for article in top_headlines['articles'] if article['title']]
        
        if not headlines:
            return []

        # Analyze sentiment
        sentiment_results = sentiment_service.analyze_headlines(headlines)

        # Cache results in Redis with a 5-minute expiration (300 seconds)
        redis_client.setex(cache_key, 300, json.dumps([r.dict() for r in sentiment_results]))

        return sentiment_results

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching or analyzing news sentiment: {e}")
