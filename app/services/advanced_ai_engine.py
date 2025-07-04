# Advanced AI Engine with Latest Research-Based Components
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.ensemble import IsolationForest
from transformers import AutoTokenizer, AutoModel
import logging
from typing import List, Dict, Tuple, Optional, Any, Union
from datetime import datetime, timedelta
import asyncio
import warnings
from dataclasses import dataclass
from abc import ABC, abstractmethod
import json
import math
from collections import deque
import random
import aiohttp
import os
import yfinance as yf
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, accuracy_score, classification_report
import pickle

warnings.filterwarnings('ignore')
logger = logging.getLogger(__name__)

@dataclass
class MarketRegime:
    """Market regime classification"""
    regime_type: str  # 'bull', 'bear', 'sideways', 'volatile'
    confidence: float
    volatility_level: str  # 'low', 'medium', 'high', 'extreme'
    trend_strength: float
    momentum: float
    timestamp: datetime

@dataclass
class AlphaSignal:
    """Alpha signal from formulaic discovery"""
    formula: str
    predicted_return: float
    confidence: float
    risk_adjusted_score: float
    time_horizon: str  # 'short', 'medium', 'long'
    validity_period: int  # hours

@dataclass
class PredictionResult:
    symbol: str
    prediction: float
    confidence: float
    model_type: str
    features_used: List[str]
    timestamp: str

@dataclass
class TradingSignal:
    symbol: str
    signal_type: str  # 'buy', 'sell', 'hold'
    confidence: float
    price_target: float
    stop_loss: float
    reasoning: str
    timestamp: str

@dataclass
class SentimentAnalysis:
    text: str
    sentiment_score: float
    sentiment_label: str
    confidence: float
    entities: List[str]
    topics: List[str]

class HigherOrderTransformer(nn.Module):
    """Higher-order transformer for multimodal time-series financial data"""
    
    def __init__(self, input_dim: int = 64, d_model: int = 512, nhead: int = 8, 
                 num_layers: int = 6, dropout: float = 0.1, max_seq_len: int = 1000):
        super(HigherOrderTransformer, self).__init__()
        
        self.d_model = d_model
        self.input_projection = nn.Linear(input_dim, d_model)
        self.positional_encoding = PositionalEncoding(d_model, dropout, max_seq_len)
        
        # Higher-order attention with low-rank decomposition
        self.transformer_layers = nn.ModuleList([
            HigherOrderTransformerLayer(d_model, nhead, dropout)
            for _ in range(num_layers)
        ])
        
        self.output_projection = nn.Linear(d_model, 1)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, mask=None):
        # x shape: (batch_size, seq_len, input_dim)
        x = self.input_projection(x)
        x = self.positional_encoding(x)
        
        for layer in self.transformer_layers:
            x = layer(x, mask)
            
        return self.output_projection(x)

class HigherOrderTransformerLayer(nn.Module):
    """Higher-order transformer layer with tensor decomposition"""
    
    def __init__(self, d_model: int, nhead: int, dropout: float = 0.1):
        super().__init__()
        
        self.self_attn = HigherOrderMultiHeadAttention(d_model, nhead, dropout)
        self.linear1 = nn.Linear(d_model, d_model * 4)
        self.linear2 = nn.Linear(d_model * 4, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, mask=None):
        # Higher-order self-attention
        attn_output = self.self_attn(x, x, x, mask)
        x = self.norm1(x + self.dropout(attn_output))
        
        # Feed-forward network
        ff_output = self.linear2(F.relu(self.linear1(x)))
        x = self.norm2(x + self.dropout(ff_output))
        
        return x

class HigherOrderMultiHeadAttention(nn.Module):
    """Higher-order multi-head attention with low-rank tensor decomposition"""
    
    def __init__(self, d_model: int, nhead: int, dropout: float = 0.1, rank: int = 32):
        super().__init__()
        
        self.d_model = d_model
        self.nhead = nhead
        self.head_dim = d_model // nhead
        self.rank = rank
        
        # Low-rank decomposition for computational efficiency
        self.q_proj_u = nn.Linear(d_model, rank * nhead, bias=False)
        self.q_proj_v = nn.Linear(rank, self.head_dim, bias=False)
        
        self.k_proj_u = nn.Linear(d_model, rank * nhead, bias=False)
        self.k_proj_v = nn.Linear(rank, self.head_dim, bias=False)
        
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        self.scale = 1.0 / math.sqrt(self.head_dim)
        
    def forward(self, query, key, value, mask=None):
        batch_size, seq_len, _ = query.shape
        
        # Low-rank projections
        q_u = self.q_proj_u(query).view(batch_size, seq_len, self.nhead, self.rank)
        q = self.q_proj_v(q_u).transpose(1, 2)  # (batch, nhead, seq_len, head_dim)
        
        k_u = self.k_proj_u(key).view(batch_size, seq_len, self.nhead, self.rank)
        k = self.k_proj_v(k_u).transpose(1, 2)
        
        v = self.v_proj(value).view(batch_size, seq_len, self.nhead, self.head_dim).transpose(1, 2)
        
        # Scaled dot-product attention
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        
        if mask is not None:
            attn_weights = attn_weights.masked_fill(mask == 0, -1e9)
            
        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        attn_output = torch.matmul(attn_weights, v)
        attn_output = attn_output.transpose(1, 2).contiguous().view(
            batch_size, seq_len, self.d_model
        )
        
        return self.out_proj(attn_output)

class PositionalEncoding(nn.Module):
    """Positional encoding for transformer"""
    
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)

class MultiAgentTradingSystem:
    """Multi-agent system where each agent specializes in different strategies"""
    
    def __init__(self, num_agents: int = 5, state_dim: int = 100, action_dim: int = 3):
        self.num_agents = num_agents
        self.agents = []
        
        # Create specialized agents
        for i in range(num_agents):
            agent = TradingAgent(
                state_dim=state_dim,
                action_dim=action_dim,
                specialization=self._get_specialization(i)
            )
            self.agents.append(agent)
            
        self.consensus_module = ConsensusModule(num_agents)
        
    def _get_specialization(self, agent_id: int) -> str:
        """Assign specialization to each agent"""
        specializations = ['momentum', 'mean_reversion', 'volatility', 'sentiment', 'technical']
        return specializations[agent_id % len(specializations)]
    
    async def get_collective_decision(self, market_state: np.ndarray) -> Dict[str, Any]:
        """Get collective trading decision from all agents"""
        agent_decisions = []
        
        for agent in self.agents:
            decision = await agent.get_decision(market_state)
            agent_decisions.append(decision)
            
        # Use consensus module to aggregate decisions
        collective_decision = self.consensus_module.aggregate_decisions(agent_decisions)
        
        return collective_decision

class TradingAgent:
    """Individual trading agent with specialization"""
    
    def __init__(self, state_dim: int, action_dim: int, specialization: str):
        self.specialization = specialization
        self.policy_network = PolicyNetwork(state_dim, action_dim)
        self.value_network = ValueNetwork(state_dim)
        self.memory = deque(maxlen=10000)
        self.optimizer_policy = optim.Adam(self.policy_network.parameters(), lr=3e-4)
        self.optimizer_value = optim.Adam(self.value_network.parameters(), lr=3e-4)
        
    async def get_decision(self, state: np.ndarray) -> Dict[str, Any]:
        """Get trading decision based on specialization"""
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        
        with torch.no_grad():
            action_probs = self.policy_network(state_tensor)
            value = self.value_network(state_tensor)
            
        action_dist = Categorical(action_probs)
        action = action_dist.sample()
        
        return {
            'action': action.item(),
            'confidence': action_probs.max().item(),
            'value_estimate': value.item(),
            'specialization': self.specialization
        }

class PolicyNetwork(nn.Module):
    """Policy network for trading decisions"""
    
    def __init__(self, state_dim: int, action_dim: int):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim),
            nn.Softmax(dim=-1)
        )
        
    def forward(self, x):
        return self.network(x)

class ValueNetwork(nn.Module):
    """Value network for state value estimation"""
    
    def __init__(self, state_dim: int):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
        
    def forward(self, x):
        return self.network(x)

class ConsensusModule:
    """Consensus mechanism for multi-agent decisions"""
    
    def __init__(self, num_agents: int):
        self.num_agents = num_agents
        self.agent_weights = torch.ones(num_agents) / num_agents
        
    def aggregate_decisions(self, decisions: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Aggregate decisions from multiple agents"""
        # Weighted voting based on agent confidence and past performance
        actions = torch.tensor([d['action'] for d in decisions])
        confidences = torch.tensor([d['confidence'] for d in decisions])
        
        # Weight votes by confidence
        weighted_votes = torch.zeros(3)  # Assuming 3 actions: hold, buy, sell
        
        for i, decision in enumerate(decisions):
            weighted_votes[decision['action']] += confidences[i] * self.agent_weights[i]
            
        final_action = weighted_votes.argmax().item()
        final_confidence = weighted_votes.max().item() / weighted_votes.sum().item()
        
        return {
            'action': final_action,
            'confidence': final_confidence,
            'agent_votes': decisions,
            'consensus_strength': self._calculate_consensus_strength(decisions)
        }
    
    def _calculate_consensus_strength(self, decisions: List[Dict[str, Any]]) -> float:
        """Calculate how much agents agree"""
        actions = [d['action'] for d in decisions]
        most_common_action = max(set(actions), key=actions.count)
        agreement_ratio = actions.count(most_common_action) / len(actions)
        return agreement_ratio

class AdvancedAIEngine:
    """Complete advanced AI engine integrating all cutting-edge components"""
    
    def __init__(self):
        self.transformer_model = HigherOrderTransformer()
        self.multi_agent_system = MultiAgentTradingSystem()
        
        self.performance_tracker = deque(maxlen=1000)
        self.model_cache = {}
        
        self.models = {}
        self.scalers = {}
        self.session = None
        self.model_path = "models/"
        
        # Create models directory if it doesn't exist
        os.makedirs(self.model_path, exist_ok=True)
        
        logger.info("Advanced AI Engine initialized with cutting-edge components")
    
    async def comprehensive_market_analysis(self, market_data: pd.DataFrame, 
                                          news_context: str = "") -> Dict[str, Any]:
        """Comprehensive analysis using all AI components"""
        try:
            # Get multi-agent consensus
            market_state = self._prepare_market_state(market_data)
            agent_decision = await self.multi_agent_system.get_collective_decision(market_state)
            
            # Transformer prediction
            transformer_features = self._prepare_transformer_features(market_data)
            transformer_pred = await self._get_transformer_prediction(transformer_features)
            
            # Combine signals
            final_analysis = self._combine_signals(agent_decision, transformer_pred)
            
            return final_analysis
            
        except Exception as e:
            logger.error(f"Comprehensive analysis failed: {e}")
            return self._fallback_analysis()
    
    def _prepare_market_state(self, market_data: pd.DataFrame) -> np.ndarray:
        """Prepare market state for multi-agent system"""
        features = []
        
        # Price features
        if 'close' in market_data.columns:
            returns = market_data['close'].pct_change().fillna(0)
            features.extend([
                returns.iloc[-1] if len(returns) > 0 else 0,
                returns.rolling(5).mean().iloc[-1] if len(returns) >= 5 else 0,
                returns.rolling(20).std().iloc[-1] if len(returns) >= 20 else 0
            ])
        
        # Pad to required size
        while len(features) < 100:
            features.append(0.0)
            
        return np.array(features[:100])
    
    def _prepare_transformer_features(self, market_data: pd.DataFrame) -> torch.Tensor:
        """Prepare features for transformer model"""
        features = []
        
        # Use the last 50 time steps
        window_size = min(50, len(market_data))
        
        for i in range(window_size):
            row_features = []
            
            # OHLCV features
            if 'open' in market_data.columns:
                row_features.append(market_data['open'].iloc[-(window_size-i)])
            if 'high' in market_data.columns:
                row_features.append(market_data['high'].iloc[-(window_size-i)])
            if 'low' in market_data.columns:
                row_features.append(market_data['low'].iloc[-(window_size-i)])
            if 'close' in market_data.columns:
                row_features.append(market_data['close'].iloc[-(window_size-i)])
            if 'volume' in market_data.columns:
                row_features.append(market_data['volume'].iloc[-(window_size-i)])
                
            # Pad to 64 features
            while len(row_features) < 64:
                row_features.append(0.0)
                
            features.append(row_features[:64])
        
        return torch.tensor(features, dtype=torch.float32).unsqueeze(0)
    
    async def _get_transformer_prediction(self, features: torch.Tensor) -> Dict[str, Any]:
        """Get prediction from transformer model"""
        try:
            with torch.no_grad():
                output = self.transformer_model(features)
                prediction = output.squeeze().mean().item()
                
            return {
                'prediction': prediction,
                'confidence': min(1.0, abs(prediction)),
                'direction': 1 if prediction > 0 else -1
            }
        except Exception as e:
            logger.error(f"Transformer prediction failed: {e}")
            return {'prediction': 0.0, 'confidence': 0.0, 'direction': 0}
    
    def _combine_signals(self, agent_decision: Dict[str, Any], 
                        transformer_pred: Dict[str, Any]) -> Dict[str, Any]:
        """Combine all signals into final analysis"""
        
        # Weight signals
        agent_score = (agent_decision['action'] - 1) * agent_decision['confidence']  # Convert to -1, 0, 1 scale
        transformer_score = transformer_pred['prediction']
        
        final_score = 0.5 * agent_score + 0.5 * transformer_score
        
        # Determine final action
        if final_score > 0.1:
            final_action = 'BUY'
        elif final_score < -0.1:
            final_action = 'SELL'
        else:
            final_action = 'HOLD'
        
        return {
            'action': final_action,
            'confidence': min(1.0, abs(final_score)),
            'score': final_score,
            'agent_consensus': agent_decision,
            'transformer_prediction': transformer_pred,
            'timestamp': datetime.now().isoformat()
        }
    
    def _fallback_analysis(self) -> Dict[str, Any]:
        """Fallback analysis when main analysis fails"""
        return {
            'action': 'HOLD',
            'confidence': 0.0,
            'score': 0.0,
            'agent_consensus': {'action': 1, 'confidence': 0.0},
            'transformer_prediction': {'prediction': 0.0},
            'timestamp': datetime.now().isoformat(),
            'error': 'Analysis failed, using fallback'
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
    
    # Machine Learning Models
    async def train_price_prediction_model(self, symbol: str, model_type: str = "random_forest") -> Dict[str, Any]:
        """Train machine learning models for price prediction"""
        try:
            # Get historical data
            ticker = yf.Ticker(symbol)
            hist = ticker.history(period="2y")
            
            if hist.empty or len(hist) < 100:
                return {"error": "Insufficient data for training"}
            
            # Prepare features
            features = self._prepare_price_features(hist)
            target = hist['Close'].pct_change().shift(-1).dropna()  # Next day return
            
            # Align features and target
            features = features[:-1]  # Remove last row since we don't have target for it
            target = target.dropna()
            
            if len(features) != len(target):
                min_len = min(len(features), len(target))
                features = features[:min_len]
                target = target[:min_len]
            
            if len(features) < 50:
                return {"error": "Insufficient aligned data for training"}
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                features, target, test_size=0.2, random_state=42
            )
            
            # Scale features
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            # Train model
            if model_type == "random_forest":
                model = RandomForestRegressor(n_estimators=100, random_state=42)
            elif model_type == "linear":
                model = LinearRegression()
            else:
                model = RandomForestRegressor(n_estimators=100, random_state=42)
            
            model.fit(X_train_scaled, y_train)
            
            # Evaluate model
            y_pred = model.predict(X_test_scaled)
            mse = mean_squared_error(y_test, y_pred)
            r2 = model.score(X_test_scaled, y_test)
            
            # Save model
            model_key = f"{symbol}_{model_type}"
            self.models[model_key] = model
            self.scalers[model_key] = scaler
            
            # Save to disk
            await self._save_model(model_key, model, scaler)
            
            return {
                "symbol": symbol,
                "model_type": model_type,
                "mse": mse,
                "r2": r2,
                "training_samples": len(X_train),
                "test_samples": len(X_test),
                "feature_importance": self._get_feature_importance(model, features.columns) if hasattr(model, 'feature_importances_') else {}
            }
            
        except Exception as e:
            logger.error(f"Price prediction model training failed for {symbol}: {e}")
            return {"error": str(e)}
    
    def _prepare_price_features(self, hist: pd.DataFrame) -> pd.DataFrame:
        """Prepare features for price prediction"""
        try:
            features = pd.DataFrame()
            
            # Price-based features
            features['price'] = hist['Close']
            features['returns'] = hist['Close'].pct_change()
            features['log_returns'] = np.log(hist['Close'] / hist['Close'].shift(1))
            
            # Moving averages
            for period in [5, 10, 20, 50]:
                features[f'sma_{period}'] = hist['Close'].rolling(window=period).mean()
                features[f'ema_{period}'] = hist['Close'].ewm(span=period).mean()
                features[f'price_sma_{period}_ratio'] = hist['Close'] / features[f'sma_{period}']
            
            # Volatility features
            features['volatility_5'] = hist['Close'].rolling(window=5).std()
            features['volatility_20'] = hist['Close'].rolling(window=20).std()
            features['volatility_50'] = hist['Close'].rolling(window=50).std()
            
            # Volume features
            features['volume'] = hist['Volume']
            features['volume_sma_20'] = hist['Volume'].rolling(window=20).mean()
            features['volume_ratio'] = hist['Volume'] / features['volume_sma_20']
            
            # Technical indicators
            features['rsi'] = self._calculate_rsi(hist['Close'])
            features['macd'] = self._calculate_macd(hist['Close'])
            features['bollinger_position'] = self._calculate_bollinger_position(hist['Close'])
            
            # Time-based features
            features['day_of_week'] = hist.index.dayofweek
            features['month'] = hist.index.month
            features['quarter'] = hist.index.quarter
            
            # Lagged features
            for lag in [1, 2, 3, 5]:
                features[f'returns_lag_{lag}'] = features['returns'].shift(lag)
                features[f'volume_lag_{lag}'] = features['volume'].shift(lag)
            
            # Drop NaN values
            features = features.dropna()
            
            return features
            
        except Exception as e:
            logger.error(f"Feature preparation failed: {e}")
            return pd.DataFrame()
    
    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """Calculate RSI"""
        try:
            delta = prices.diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
            return rsi
        except Exception as e:
            logger.error(f"RSI calculation failed: {e}")
            return pd.Series(index=prices.index)
    
    def _calculate_macd(self, prices: pd.Series, fast: int = 12, slow: int = 26) -> pd.Series:
        """Calculate MACD"""
        try:
            ema_fast = prices.ewm(span=fast).mean()
            ema_slow = prices.ewm(span=slow).mean()
            macd = ema_fast - ema_slow
            return macd
        except Exception as e:
            logger.error(f"MACD calculation failed: {e}")
            return pd.Series(index=prices.index)
    
    def _calculate_bollinger_position(self, prices: pd.Series, period: int = 20, std_dev: int = 2) -> pd.Series:
        """Calculate position within Bollinger Bands"""
        try:
            sma = prices.rolling(window=period).mean()
            std = prices.rolling(window=period).std()
            upper_band = sma + (std_dev * std)
            lower_band = sma - (std_dev * std)
            position = (prices - lower_band) / (upper_band - lower_band)
            return position
        except Exception as e:
            logger.error(f"Bollinger position calculation failed: {e}")
            return pd.Series(index=prices.index)
    
    def _get_feature_importance(self, model, feature_names) -> Dict[str, float]:
        """Get feature importance from model"""
        try:
            if hasattr(model, 'feature_importances_'):
                importance_dict = dict(zip(feature_names, model.feature_importances_))
                return dict(sorted(importance_dict.items(), key=lambda x: x[1], reverse=True))
            return {}
        except Exception as e:
            logger.error(f"Feature importance extraction failed: {e}")
            return {}
    
    async def _save_model(self, model_key: str, model, scaler):
        """Save model to disk"""
        try:
            model_data = {
                'model': model,
                'scaler': scaler,
                'timestamp': datetime.now().isoformat()
            }
            
            with open(f"{self.model_path}{model_key}.pkl", 'wb') as f:
                pickle.dump(model_data, f)
                
        except Exception as e:
            logger.error(f"Model saving failed: {e}")
    
    async def load_model(self, model_key: str):
        """Load model from disk"""
        try:
            with open(f"{self.model_path}{model_key}.pkl", 'rb') as f:
                model_data = pickle.load(f)
                
            self.models[model_key] = model_data['model']
            self.scalers[model_key] = model_data['scaler']
            
            return True
            
        except Exception as e:
            logger.error(f"Model loading failed: {e}")
            return False
    
    # Price Predictions
    async def predict_price_movement(self, symbol: str, model_type: str = "random_forest") -> PredictionResult:
        """Predict price movement using trained models"""
        try:
            model_key = f"{symbol}_{model_type}"
            
            # Load model if not in memory
            if model_key not in self.models:
                loaded = await self.load_model(model_key)
                if not loaded:
                    return None
            
            # Get latest data
            ticker = yf.Ticker(symbol)
            hist = ticker.history(period="60d")
            
            if hist.empty:
                return None
            
            # Prepare features
            features = self._prepare_price_features(hist)
            
            if features.empty:
                return None
            
            # Get latest feature vector
            latest_features = features.iloc[-1:].values
            
            # Scale features
            scaler = self.scalers[model_key]
            scaled_features = scaler.transform(latest_features)
            
            # Make prediction
            model = self.models[model_key]
            prediction = model.predict(scaled_features)[0]
            
            # Calculate confidence (simplified)
            confidence = 0.7  # In practice, you'd calculate this based on model uncertainty
            
            return PredictionResult(
                symbol=symbol,
                prediction=prediction,
                confidence=confidence,
                model_type=model_type,
                features_used=features.columns.tolist(),
                timestamp=datetime.now().isoformat()
            )
            
        except Exception as e:
            logger.error(f"Price prediction failed for {symbol}: {e}")
            return None
    
    # Trading Signal Generation
    async def generate_trading_signals(self, symbol: str, strategy: str = "ml_enhanced") -> List[TradingSignal]:
        """Generate automated trading signals"""
        try:
            signals = []
            
            # Get current market data
            ticker = yf.Ticker(symbol)
            hist = ticker.history(period="30d")
            
            if hist.empty:
                return signals
            
            # Get current price
            current_price = hist['Close'].iloc[-1]
            
            if strategy == "ml_enhanced":
                # Use ML prediction
                prediction = await self.predict_price_movement(symbol)
                
                if prediction:
                    # Generate signal based on prediction
                    if prediction.prediction > 0.02:  # 2% expected return
                        signals.append(TradingSignal(
                            symbol=symbol,
                            signal_type="buy",
                            confidence=prediction.confidence,
                            price_target=current_price * (1 + prediction.prediction),
                            stop_loss=current_price * 0.95,  # 5% stop loss
                            reasoning=f"ML model predicts {prediction.prediction:.2%} return",
                            timestamp=datetime.now().isoformat()
                        ))
                    elif prediction.prediction < -0.02:  # -2% expected return
                        signals.append(TradingSignal(
                            symbol=symbol,
                            signal_type="sell",
                            confidence=prediction.confidence,
                            price_target=current_price * (1 + prediction.prediction),
                            stop_loss=current_price * 1.05,  # 5% stop loss
                            reasoning=f"ML model predicts {prediction.prediction:.2%} return",
                            timestamp=datetime.now().isoformat()
                        ))
                    else:
                        signals.append(TradingSignal(
                            symbol=symbol,
                            signal_type="hold",
                            confidence=prediction.confidence,
                            price_target=current_price,
                            stop_loss=current_price,
                            reasoning="ML model predicts minimal movement",
                            timestamp=datetime.now().isoformat()
                        ))
            
            elif strategy == "technical":
                # Technical analysis signals
                signals.extend(self._generate_technical_signals(hist, symbol))
            
            return signals
            
        except Exception as e:
            logger.error(f"Trading signal generation failed for {symbol}: {e}")
            return []
    
    def _generate_technical_signals(self, hist: pd.DataFrame, symbol: str) -> List[TradingSignal]:
        """Generate technical analysis signals"""
        try:
            signals = []
            current_price = hist['Close'].iloc[-1]
            
            # RSI signals
            rsi = self._calculate_rsi(hist['Close'])
            if not rsi.empty:
                current_rsi = rsi.iloc[-1]
                
                if current_rsi < 30:
                    signals.append(TradingSignal(
                        symbol=symbol,
                        signal_type="buy",
                        confidence=0.6,
                        price_target=current_price * 1.05,
                        stop_loss=current_price * 0.95,
                        reasoning=f"RSI oversold ({current_rsi:.1f})",
                        timestamp=datetime.now().isoformat()
                    ))
                elif current_rsi > 70:
                    signals.append(TradingSignal(
                        symbol=symbol,
                        signal_type="sell",
                        confidence=0.6,
                        price_target=current_price * 0.95,
                        stop_loss=current_price * 1.05,
                        reasoning=f"RSI overbought ({current_rsi:.1f})",
                        timestamp=datetime.now().isoformat()
                    ))
            
            # Moving average crossover
            sma_20 = hist['Close'].rolling(window=20).mean()
            sma_50 = hist['Close'].rolling(window=50).mean()
            
            if len(sma_20) > 0 and len(sma_50) > 0:
                current_sma_20 = sma_20.iloc[-1]
                current_sma_50 = sma_50.iloc[-1]
                prev_sma_20 = sma_20.iloc[-2] if len(sma_20) > 1 else current_sma_20
                prev_sma_50 = sma_50.iloc[-2] if len(sma_50) > 1 else current_sma_50
                
                # Golden cross
                if prev_sma_20 <= prev_sma_50 and current_sma_20 > current_sma_50:
                    signals.append(TradingSignal(
                        symbol=symbol,
                        signal_type="buy",
                        confidence=0.7,
                        price_target=current_price * 1.08,
                        stop_loss=current_price * 0.95,
                        reasoning="Golden cross (SMA 20 > SMA 50)",
                        timestamp=datetime.now().isoformat()
                    ))
                
                # Death cross
                elif prev_sma_20 >= prev_sma_50 and current_sma_20 < current_sma_50:
                    signals.append(TradingSignal(
                        symbol=symbol,
                        signal_type="sell",
                        confidence=0.7,
                        price_target=current_price * 0.92,
                        stop_loss=current_price * 1.05,
                        reasoning="Death cross (SMA 20 < SMA 50)",
                        timestamp=datetime.now().isoformat()
                    ))
            
            return signals
            
        except Exception as e:
            logger.error(f"Technical signal generation failed: {e}")
            return []
    
    # Sentiment Analysis
    async def analyze_sentiment(self, text: str) -> SentimentAnalysis:
        """Analyze sentiment of text using NLP"""
        try:
            # Simple sentiment analysis (in practice, you'd use more sophisticated models)
            sentiment_score = self._simple_sentiment_analysis(text)
            
            # Determine sentiment label
            if sentiment_score > 0.1:
                sentiment_label = "positive"
            elif sentiment_score < -0.1:
                sentiment_label = "negative"
            else:
                sentiment_label = "neutral"
            
            # Extract entities (simplified)
            entities = self._extract_entities(text)
            
            # Extract topics (simplified)
            topics = self._extract_topics(text)
            
            return SentimentAnalysis(
                text=text,
                sentiment_score=sentiment_score,
                sentiment_label=sentiment_label,
                confidence=abs(sentiment_score),
                entities=entities,
                topics=topics
            )
            
        except Exception as e:
            logger.error(f"Sentiment analysis failed: {e}")
            return None
    
    def _simple_sentiment_analysis(self, text: str) -> float:
        """Simple sentiment analysis using keyword matching"""
        try:
            positive_words = [
                'bullish', 'positive', 'growth', 'profit', 'gain', 'rise', 'up', 'strong',
                'excellent', 'great', 'good', 'buy', 'outperform', 'beat', 'surge', 'rally'
            ]
            
            negative_words = [
                'bearish', 'negative', 'decline', 'loss', 'fall', 'down', 'weak',
                'poor', 'bad', 'sell', 'underperform', 'miss', 'drop', 'crash', 'plunge'
            ]
            
            text_lower = text.lower()
            
            positive_count = sum(1 for word in positive_words if word in text_lower)
            negative_count = sum(1 for word in negative_words if word in text_lower)
            
            total_words = len(text.split())
            if total_words == 0:
                return 0.0
            
            sentiment_score = (positive_count - negative_count) / total_words
            
            return max(-1.0, min(1.0, sentiment_score))
            
        except Exception as e:
            logger.error(f"Simple sentiment analysis failed: {e}")
            return 0.0
    
    def _extract_entities(self, text: str) -> List[str]:
        """Extract entities from text (simplified)"""
        try:
            # Simple entity extraction using keyword matching
            entities = []
            
            # Stock symbols (simple pattern)
            import re
            stock_pattern = r'\b[A-Z]{1,5}\b'
            stock_matches = re.findall(stock_pattern, text)
            entities.extend(stock_matches)
            
            # Company names (common ones)
            company_names = ['Apple', 'Microsoft', 'Google', 'Amazon', 'Tesla', 'Meta', 'Netflix']
            for company in company_names:
                if company.lower() in text.lower():
                    entities.append(company)
            
            return list(set(entities))
            
        except Exception as e:
            logger.error(f"Entity extraction failed: {e}")
            return []
    
    def _extract_topics(self, text: str) -> List[str]:
        """Extract topics from text (simplified)"""
        try:
            topics = []
            text_lower = text.lower()
            
            # Define topic keywords
            topic_keywords = {
                'earnings': ['earnings', 'revenue', 'profit', 'quarterly', 'annual'],
                'market': ['market', 'trading', 'stock', 'price', 'volume'],
                'economy': ['economy', 'economic', 'gdp', 'inflation', 'interest'],
                'technology': ['tech', 'technology', 'software', 'hardware', 'digital'],
                'finance': ['finance', 'financial', 'banking', 'investment', 'fund']
            }
            
            for topic, keywords in topic_keywords.items():
                if any(keyword in text_lower for keyword in keywords):
                    topics.append(topic)
            
            return topics
            
        except Exception as e:
            logger.error(f"Topic extraction failed: {e}")
            return []
    
    # Market Analysis
    async def analyze_market_sentiment(self, symbols: List[str] = None) -> Dict[str, Any]:
        """Analyze overall market sentiment"""
        try:
            if symbols is None:
                symbols = ['^GSPC', '^DJI', '^IXIC']  # S&P 500, Dow Jones, NASDAQ
            
            market_analysis = {}
            
            for symbol in symbols:
                try:
                    ticker = yf.Ticker(symbol)
                    hist = ticker.history(period="30d")
                    
                    if not hist.empty:
                        returns = hist['Close'].pct_change().dropna()
                        
                        market_analysis[symbol] = {
                            'current_price': hist['Close'].iloc[-1],
                            'return_30d': (hist['Close'].iloc[-1] / hist['Close'].iloc[0] - 1),
                            'volatility': returns.std() * np.sqrt(252),
                            'trend': 'bullish' if returns.mean() > 0 else 'bearish',
                            'strength': abs(returns.mean()) / returns.std() if returns.std() > 0 else 0
                        }
                        
                except Exception as e:
                    logger.warning(f"Market analysis for {symbol} failed: {e}")
                    continue
            
            return market_analysis
            
        except Exception as e:
            logger.error(f"Market sentiment analysis failed: {e}")
            return {}
    
    # Portfolio Optimization with AI
    async def optimize_portfolio_ai(self, symbols: List[str], target_return: float = None) -> Dict[str, Any]:
        """AI-enhanced portfolio optimization"""
        try:
            # Get historical data for all symbols
            returns_data = {}
            
            for symbol in symbols:
                try:
                    ticker = yf.Ticker(symbol)
                    hist = ticker.history(period="1y")
                    
                    if not hist.empty:
                        returns = hist['Close'].pct_change().dropna()
                        returns_data[symbol] = returns
                        
                except Exception as e:
                    logger.warning(f"Data fetch for {symbol} failed: {e}")
                    continue
            
            if len(returns_data) < 2:
                return {"error": "Insufficient data for optimization"}
            
            # Create returns DataFrame
            returns_df = pd.DataFrame(returns_data)
            returns_df = returns_df.dropna()
            
            if len(returns_df) < 50:
                return {"error": "Insufficient aligned data"}
            
            # Calculate expected returns using ML predictions
            expected_returns = {}
            for symbol in returns_df.columns:
                prediction = await self.predict_price_movement(symbol)
                if prediction:
                    expected_returns[symbol] = prediction.prediction * 252  # Annualized
                else:
                    expected_returns[symbol] = returns_df[symbol].mean() * 252
            
            # Calculate covariance matrix
            cov_matrix = returns_df.cov() * 252
            
            # Optimize portfolio
            weights = self._optimize_portfolio_weights(expected_returns, cov_matrix, target_return)
            
            if weights is None:
                return {"error": "Optimization failed"}
            
            # Calculate portfolio metrics
            portfolio_return = sum(expected_returns[symbol] * weights[symbol] for symbol in weights)
            portfolio_vol = np.sqrt(sum(weights[symbol] * weights[symbol2] * cov_matrix.loc[symbol, symbol2] 
                                       for symbol in weights for symbol2 in weights))
            
            return {
                "optimal_weights": weights,
                "expected_return": portfolio_return,
                "volatility": portfolio_vol,
                "sharpe_ratio": portfolio_return / portfolio_vol if portfolio_vol > 0 else 0,
                "symbols": list(weights.keys())
            }
            
        except Exception as e:
            logger.error(f"AI portfolio optimization failed: {e}")
            return {"error": str(e)}
    
    def _optimize_portfolio_weights(self, expected_returns: Dict[str, float], 
                                  cov_matrix: pd.DataFrame, target_return: float = None) -> Optional[Dict[str, float]]:
        """Optimize portfolio weights"""
        try:
            from scipy.optimize import minimize
            
            symbols = list(expected_returns.keys())
            n_assets = len(symbols)
            
            # Objective function: minimize portfolio variance
            def portfolio_variance(weights):
                return sum(weights[i] * weights[j] * cov_matrix.iloc[i, j] 
                          for i in range(n_assets) for j in range(n_assets))
            
            # Constraints
            constraints = [{'type': 'eq', 'fun': lambda x: sum(x) - 1}]  # Weights sum to 1
            
            if target_return is not None:
                constraints.append({
                    'type': 'eq', 
                    'fun': lambda x: sum(expected_returns[symbols[i]] * x[i] for i in range(n_assets)) - target_return
                })
            
            # Bounds: weights between 0 and 1
            bounds = [(0, 1) for _ in range(n_assets)]
            
            # Initial guess: equal weights
            initial_weights = [1.0 / n_assets] * n_assets
            
            # Optimize
            result = minimize(
                portfolio_variance,
                initial_weights,
                method='SLSQP',
                bounds=bounds,
                constraints=constraints
            )
            
            if result.success:
                return dict(zip(symbols, result.x))
            else:
                return None
                
        except Exception as e:
            logger.error(f"Portfolio weight optimization failed: {e}")
            return None

# Global instance
advanced_ai_engine = AdvancedAIEngine() 