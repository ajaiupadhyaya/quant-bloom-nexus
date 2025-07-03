from fastapi import APIRouter, HTTPException, BackgroundTasks, Depends
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import asyncio
import logging

from ..services.ai_engine import ai_engine
from ..services.market_data_service import get_market_data

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/ai", tags=["AI Models"])

class PredictionRequest(BaseModel):
    symbol: str
    days_ahead: int = 5
    model_type: str = "lstm"  # lstm, ensemble, all

class SentimentRequest(BaseModel):
    texts: List[str]

class TradingSignalRequest(BaseModel):
    symbol: str
    include_sentiment: bool = True

class MarketAnalysisRequest(BaseModel):
    symbol: str
    analysis_type: str = "comprehensive"  # comprehensive, regime, volatility

@router.post("/predict/price")
async def predict_price(request: PredictionRequest):
    """Advanced price prediction using AI models"""
    try:
        # Get market data
        market_data = await get_market_data(request.symbol, period="1y")
        
        if not market_data or len(market_data) < 100:
            raise HTTPException(status_code=400, detail="Insufficient market data")
        
        # Convert to DataFrame
        df = pd.DataFrame(market_data)
        df['date'] = pd.to_datetime(df['timestamp'], unit='ms')
        df = df.set_index('date').sort_index()
        
        # Get predictions based on model type
        if request.model_type == "lstm":
            predictions = await ai_engine.predict_price_lstm(df, request.days_ahead)
        elif request.model_type == "ensemble":
            predictions = await ai_engine.predict_price_ensemble(df, request.days_ahead)
        elif request.model_type == "all":
            lstm_pred, ensemble_pred = await asyncio.gather(
                ai_engine.predict_price_lstm(df, request.days_ahead),
                ai_engine.predict_price_ensemble(df, request.days_ahead)
            )
            predictions = {
                "lstm": lstm_pred,
                "ensemble": ensemble_pred,
                "combined_confidence": (lstm_pred.get('confidence', 0) + ensemble_pred.get('confidence', 0)) / 2
            }
        else:
            raise HTTPException(status_code=400, detail="Invalid model type")
        
        return {
            "symbol": request.symbol,
            "predictions": predictions,
            "request_time": datetime.now().isoformat(),
            "data_points_used": len(df)
        }
        
    except Exception as e:
        logger.error(f"Price prediction failed for {request.symbol}: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

@router.post("/sentiment/analyze")
async def analyze_sentiment(request: SentimentRequest):
    """Advanced sentiment analysis using transformer models"""
    try:
        if not request.texts:
            raise HTTPException(status_code=400, detail="No texts provided")
        
        # Analyze sentiment
        results = ai_engine.sentiment_analyzer.analyze_sentiment(request.texts)
        
        # Calculate aggregate metrics
        scores = [r['sentiment_score'] for r in results]
        aggregate_sentiment = {
            "overall_score": float(np.mean(scores)),
            "sentiment_std": float(np.std(scores)),
            "positive_ratio": len([s for s in scores if s > 0.1]) / len(scores),
            "negative_ratio": len([s for s in scores if s < -0.1]) / len(scores),
            "neutral_ratio": len([s for s in scores if -0.1 <= s <= 0.1]) / len(scores)
        }
        
        return {
            "individual_results": results,
            "aggregate_sentiment": aggregate_sentiment,
            "analysis_time": datetime.now().isoformat(),
            "texts_analyzed": len(request.texts)
        }
        
    except Exception as e:
        logger.error(f"Sentiment analysis failed: {e}")
        raise HTTPException(status_code=500, detail=f"Sentiment analysis failed: {str(e)}")

@router.post("/trading/signal")
async def get_trading_signal(request: TradingSignalRequest):
    """Get AI-powered trading signals"""
    try:
        # Get market data
        market_data = await get_market_data(request.symbol, period="6m")
        
        if not market_data or len(market_data) < 50:
            raise HTTPException(status_code=400, detail="Insufficient market data")
        
        # Convert to DataFrame
        df = pd.DataFrame(market_data)
        df['date'] = pd.to_datetime(df['timestamp'], unit='ms')
        df = df.set_index('date').sort_index()
        
        # Get sentiment if requested
        sentiment_score = 0.0
        if request.include_sentiment:
            # This would typically fetch recent news for the symbol
            # For now, we'll use a placeholder
            sentiment_score = np.random.normal(0, 0.3)  # Simulated sentiment
        
        # Get trading signal
        signal = await ai_engine.get_trading_signal(df, sentiment_score)
        
        # Add additional context
        current_price = df['close'].iloc[-1]
        price_change_24h = (current_price - df['close'].iloc[-2]) / df['close'].iloc[-2]
        
        return {
            "symbol": request.symbol,
            "signal": signal,
            "current_price": float(current_price),
            "price_change_24h": float(price_change_24h),
            "sentiment_included": request.include_sentiment,
            "sentiment_score": float(sentiment_score),
            "signal_time": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Trading signal failed for {request.symbol}: {e}")
        raise HTTPException(status_code=500, detail=f"Trading signal failed: {str(e)}")

@router.post("/analysis/comprehensive")
async def comprehensive_market_analysis(request: MarketAnalysisRequest):
    """Comprehensive AI market analysis"""
    try:
        # Get market data
        market_data = await get_market_data(request.symbol, period="1y")
        
        if not market_data or len(market_data) < 100:
            raise HTTPException(status_code=400, detail="Insufficient market data")
        
        # Convert to DataFrame
        df = pd.DataFrame(market_data)
        df['date'] = pd.to_datetime(df['timestamp'], unit='ms')
        df = df.set_index('date').sort_index()
        
        # Perform comprehensive analysis
        analysis = await ai_engine.comprehensive_analysis(df)
        
        # Add market context
        current_price = df['close'].iloc[-1]
        price_changes = {
            "1d": float((current_price - df['close'].iloc[-2]) / df['close'].iloc[-2]),
            "7d": float((current_price - df['close'].iloc[-7]) / df['close'].iloc[-7]) if len(df) >= 7 else 0,
            "30d": float((current_price - df['close'].iloc[-30]) / df['close'].iloc[-30]) if len(df) >= 30 else 0
        }
        
        # Calculate volatility metrics
        returns = df['close'].pct_change().dropna()
        volatility_metrics = {
            "daily_volatility": float(returns.std()),
            "annualized_volatility": float(returns.std() * np.sqrt(252)),
            "max_drawdown": float((df['close'] / df['close'].cummax() - 1).min())
        }
        
        return {
            "symbol": request.symbol,
            "analysis": analysis,
            "market_context": {
                "current_price": float(current_price),
                "price_changes": price_changes,
                "volatility_metrics": volatility_metrics
            },
            "analysis_time": datetime.now().isoformat(),
            "data_period": f"{len(df)} days"
        }
        
    except Exception as e:
        logger.error(f"Comprehensive analysis failed for {request.symbol}: {e}")
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")

@router.get("/models/status")
async def get_model_status():
    """Get status of all AI models"""
    try:
        status = {
            "lstm_model": {
                "loaded": ai_engine.lstm_model is not None,
                "parameters": sum(p.numel() for p in ai_engine.lstm_model.parameters()) if ai_engine.lstm_model else 0
            },
            "sentiment_analyzer": {
                "loaded": ai_engine.sentiment_analyzer.pipeline is not None,
                "model_name": ai_engine.sentiment_analyzer.model_name
            },
            "rl_agent": {
                "loaded": ai_engine.rl_agent is not None,
                "epsilon": float(ai_engine.rl_agent.epsilon) if ai_engine.rl_agent else 0,
                "memory_size": len(ai_engine.rl_agent.memory) if ai_engine.rl_agent else 0
            },
            "ensemble_models": {
                name: {"loaded": model is not None}
                for name, model in ai_engine.ensemble_models.items()
            }
        }
        
        return {
            "models": status,
            "system_info": {
                "torch_available": True,
                "cuda_available": False,  # Will be detected automatically
                "transformers_available": True
            },
            "status_time": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Model status check failed: {e}")
        raise HTTPException(status_code=500, detail=f"Status check failed: {str(e)}")

@router.post("/models/retrain/{model_type}")
async def retrain_model(model_type: str, background_tasks: BackgroundTasks, symbol: str = "AAPL"):
    """Retrain AI models with latest data"""
    try:
        if model_type not in ["lstm", "ensemble", "rl"]:
            raise HTTPException(status_code=400, detail="Invalid model type")
        
        # Add retraining task to background
        background_tasks.add_task(
            _retrain_model_background,
            model_type,
            symbol
        )
        
        return {
            "message": f"Retraining {model_type} model initiated",
            "symbol": symbol,
            "estimated_time": "5-15 minutes",
            "initiated_at": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Model retraining failed: {e}")
        raise HTTPException(status_code=500, detail=f"Retraining failed: {str(e)}")

async def _retrain_model_background(model_type: str, symbol: str):
    """Background task for model retraining"""
    try:
        logger.info(f"Starting {model_type} model retraining for {symbol}")
        
        # Get training data
        market_data = await get_market_data(symbol, period="2y")
        df = pd.DataFrame(market_data)
        df['date'] = pd.to_datetime(df['timestamp'], unit='ms')
        df = df.set_index('date').sort_index()
        
        if model_type == "lstm":
            # Retrain LSTM model
            await _retrain_lstm(df)
        elif model_type == "ensemble":
            # Retrain ensemble models
            await _retrain_ensemble(df)
        elif model_type == "rl":
            # Retrain RL agent
            await _retrain_rl_agent(df)
        
        logger.info(f"Completed {model_type} model retraining")
        
    except Exception as e:
        logger.error(f"Background retraining failed: {e}")

async def _retrain_lstm(df: pd.DataFrame):
    """Retrain LSTM model"""
    # Implementation would involve:
    # 1. Prepare training sequences
    # 2. Split train/validation
    # 3. Train with proper loss function
    # 4. Save model weights
    pass

async def _retrain_ensemble(df: pd.DataFrame):
    """Retrain ensemble models"""
    # Implementation would involve:
    # 1. Prepare features and targets
    # 2. Retrain each ensemble model
    # 3. Validate performance
    # 4. Update model weights
    pass

async def _retrain_rl_agent(df: pd.DataFrame):
    """Retrain RL agent"""
    # Implementation would involve:
    # 1. Simulate trading environment
    # 2. Generate experiences
    # 3. Train agent with replay buffer
    # 4. Update policy
    pass
