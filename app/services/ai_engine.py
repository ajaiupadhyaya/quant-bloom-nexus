import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
import joblib
import logging
from typing import List, Dict, Tuple, Optional, Any
from datetime import datetime, timedelta
import asyncio
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)

class LSTMPricePredictor(nn.Module):
    """Advanced LSTM model for price prediction"""
    
    def __init__(self, input_size: int = 5, hidden_size: int = 128, num_layers: int = 3, dropout: float = 0.2):
        super(LSTMPricePredictor, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            batch_first=True,
            bidirectional=True
        )
        
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_size * 2,
            num_heads=8,
            dropout=dropout
        )
        
        self.fc_layers = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1)
        )
        
    def forward(self, x):
        # LSTM layers
        lstm_out, _ = self.lstm(x)
        
        # Attention mechanism
        attn_out, _ = self.attention(lstm_out, lstm_out, lstm_out)
        
        # Take the last output
        final_out = attn_out[:, -1, :]
        
        # Fully connected layers
        prediction = self.fc_layers(final_out)
        
        return prediction

class TransformerSentimentAnalyzer:
    """Advanced transformer model for financial sentiment analysis"""
    
    def __init__(self):
        self.model_name = "ProsusAI/finbert"
        self.tokenizer = None
        self.model = None
        self.pipeline = None
        self._initialize_model()
        
    def _initialize_model(self):
        try:
            self.pipeline = pipeline(
                "sentiment-analysis",
                model=self.model_name,
                tokenizer=self.model_name,
                device=0 if torch.cuda.is_available() else -1
            )
            logger.info("FinBERT sentiment model loaded successfully")
        except Exception as e:
            logger.warning(f"Failed to load FinBERT, using fallback: {e}")
            self.pipeline = pipeline("sentiment-analysis", device=-1)
    
    def analyze_sentiment(self, texts: List[str]) -> List[Dict[str, Any]]:
        """Analyze sentiment of financial texts"""
        if not texts:
            return []
            
        try:
            results = []
            for text in texts:
                # Truncate text to model's max length
                truncated_text = text[:512]
                
                sentiment_result = self.pipeline(truncated_text)[0]
                
                # Convert to financial sentiment scale
                label = sentiment_result['label'].lower()
                score = sentiment_result['score']
                
                if label in ['positive', 'bullish']:
                    sentiment_score = score
                elif label in ['negative', 'bearish']:
                    sentiment_score = -score
                else:  # neutral
                    sentiment_score = 0.0
                
                results.append({
                    'text': text,
                    'sentiment_score': sentiment_score,
                    'confidence': score,
                    'label': label,
                    'timestamp': datetime.now().isoformat()
                })
                
            return results
            
        except Exception as e:
            logger.error(f"Sentiment analysis failed: {e}")
            return [{'sentiment_score': 0.0, 'confidence': 0.0, 'label': 'neutral'} for _ in texts]

class ReinforcementLearningAgent:
    """Deep Q-Network (DQN) for trading decisions"""
    
    def __init__(self, state_size: int = 20, action_size: int = 3, learning_rate: float = 0.001):
        self.state_size = state_size
        self.action_size = action_size  # 0: Hold, 1: Buy, 2: Sell
        self.memory = []
        self.memory_size = 10000
        self.epsilon = 1.0  # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = learning_rate
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Neural networks
        self.q_network = self._build_network().to(self.device)
        self.target_network = self._build_network().to(self.device)
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=learning_rate)
        
        # Update target network
        self.update_target_network()
        
    def _build_network(self) -> nn.Module:
        """Build the Q-network"""
        return nn.Sequential(
            nn.Linear(self.state_size, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, self.action_size)
        )
    
    def remember(self, state, action, reward, next_state, done):
        """Store experience in replay memory"""
        if len(self.memory) >= self.memory_size:
            self.memory.pop(0)
        self.memory.append((state, action, reward, next_state, done))
    
    def act(self, state: np.ndarray) -> int:
        """Choose action using epsilon-greedy policy"""
        if np.random.random() <= self.epsilon:
            return np.random.choice(self.action_size)
        
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        q_values = self.q_network(state_tensor)
        return q_values.argmax().item()
    
    def replay(self, batch_size: int = 32):
        """Train the network on a batch of experiences"""
        if len(self.memory) < batch_size:
            return
        
        batch = np.random.choice(len(self.memory), batch_size, replace=False)
        states = torch.FloatTensor([self.memory[i][0] for i in batch]).to(self.device)
        actions = torch.LongTensor([self.memory[i][1] for i in batch]).to(self.device)
        rewards = torch.FloatTensor([self.memory[i][2] for i in batch]).to(self.device)
        next_states = torch.FloatTensor([self.memory[i][3] for i in batch]).to(self.device)
        dones = torch.BoolTensor([self.memory[i][4] for i in batch]).to(self.device)
        
        current_q_values = self.q_network(states).gather(1, actions.unsqueeze(1))
        next_q_values = self.target_network(next_states).max(1)[0].detach()
        target_q_values = rewards + (0.99 * next_q_values * ~dones)
        
        loss = nn.MSELoss()(current_q_values.squeeze(), target_q_values)
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
    
    def update_target_network(self):
        """Update target network with current network weights"""
        self.target_network.load_state_dict(self.q_network.state_dict())

class AdvancedAIEngine:
    """Complete AI/ML engine for trading terminal"""
    
    def __init__(self):
        self.lstm_model = None
        self.sentiment_analyzer = TransformerSentimentAnalyzer()
        self.rl_agent = None
        self.price_scaler = MinMaxScaler()
        self.feature_scaler = StandardScaler()
        self.ensemble_models = {}
        self.model_cache = {}
        
        # Initialize models
        self._initialize_models()
        
    def _initialize_models(self):
        """Initialize all AI models"""
        try:
            # LSTM Model
            self.lstm_model = LSTMPricePredictor()
            
            # RL Agent
            self.rl_agent = ReinforcementLearningAgent()
            
            # Ensemble models
            self.ensemble_models = {
                'random_forest': RandomForestRegressor(n_estimators=100, random_state=42),
                'gradient_boost': GradientBoostingRegressor(n_estimators=100, random_state=42),
            }
            
            logger.info("All AI models initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize AI models: {e}")
    
    def prepare_features(self, data: pd.DataFrame) -> np.ndarray:
        """Prepare features for ML models"""
        try:
            features = []
            
            # Price features
            features.extend([
                data['close'].pct_change().fillna(0),
                data['high'].pct_change().fillna(0),
                data['low'].pct_change().fillna(0),
                data['volume'].pct_change().fillna(0)
            ])
            
            # Technical indicators
            # Simple Moving Averages
            for window in [5, 10, 20, 50]:
                sma = data['close'].rolling(window=window).mean()
                features.append((data['close'] / sma - 1).fillna(0))
            
            # Exponential Moving Averages
            for span in [12, 26]:
                ema = data['close'].ewm(span=span).mean()
                features.append((data['close'] / ema - 1).fillna(0))
            
            # RSI
            delta = data['close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
            features.append((rsi / 100).fillna(0.5))
            
            # MACD
            ema12 = data['close'].ewm(span=12).mean()
            ema26 = data['close'].ewm(span=26).mean()
            macd = ema12 - ema26
            signal = macd.ewm(span=9).mean()
            features.append((macd / data['close']).fillna(0))
            features.append((signal / data['close']).fillna(0))
            
            # Bollinger Bands
            sma20 = data['close'].rolling(window=20).mean()
            std20 = data['close'].rolling(window=20).std()
            bb_upper = sma20 + (std20 * 2)
            bb_lower = sma20 - (std20 * 2)
            bb_position = (data['close'] - bb_lower) / (bb_upper - bb_lower)
            features.append(bb_position.fillna(0.5))
            
            # Volatility
            volatility = data['close'].rolling(window=20).std() / data['close'].rolling(window=20).mean()
            features.append(volatility.fillna(0))
            
            # Volume indicators
            volume_sma = data['volume'].rolling(window=20).mean()
            volume_ratio = data['volume'] / volume_sma
            features.append(volume_ratio.fillna(1))
            
            return np.column_stack(features)
            
        except Exception as e:
            logger.error(f"Feature preparation failed: {e}")
            return np.zeros((len(data), 20))
    
    async def predict_price_lstm(self, data: pd.DataFrame, days_ahead: int = 5) -> Dict[str, Any]:
        """Predict future prices using LSTM"""
        try:
            features = self.prepare_features(data)
            
            # Prepare sequences for LSTM
            sequence_length = 60
            if len(features) < sequence_length:
                return {'predictions': [], 'confidence': 0.0}
            
            # Scale features
            scaled_features = self.feature_scaler.fit_transform(features)
            
            # Create sequences
            sequences = []
            for i in range(sequence_length, len(scaled_features)):
                sequences.append(scaled_features[i-sequence_length:i])
            
            if not sequences:
                return {'predictions': [], 'confidence': 0.0}
            
            # Convert to tensor
            X = torch.FloatTensor(np.array(sequences))
            
            # Make predictions
            self.lstm_model.eval()
            with torch.no_grad():
                predictions = []
                current_sequence = X[-1:].clone()
                
                for _ in range(days_ahead):
                    pred = self.lstm_model(current_sequence)
                    predictions.append(pred.item())
                    
                    # Update sequence for next prediction
                    # This is a simplified approach - in practice, you'd update with actual features
                    new_features = current_sequence[:, -1:, :].clone()
                    new_features[:, :, 0] = pred  # Update price feature
                    current_sequence = torch.cat([current_sequence[:, 1:, :], new_features], dim=1)
            
            # Convert predictions back to actual prices
            last_price = data['close'].iloc[-1]
            actual_predictions = []
            for i, pred in enumerate(predictions):
                predicted_price = last_price * (1 + pred)
                actual_predictions.append({
                    'date': (datetime.now() + timedelta(days=i+1)).isoformat(),
                    'predicted_price': float(predicted_price),
                    'confidence': max(0.1, 1.0 - (i * 0.1))  # Decreasing confidence
                })
                last_price = predicted_price
            
            return {
                'predictions': actual_predictions,
                'model': 'LSTM',
                'confidence': 0.8,
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"LSTM prediction failed: {e}")
            return {'predictions': [], 'confidence': 0.0}
    
    async def predict_price_ensemble(self, data: pd.DataFrame, days_ahead: int = 5) -> Dict[str, Any]:
        """Predict prices using ensemble methods"""
        try:
            features = self.prepare_features(data)
            
            if len(features) < 50:
                return {'predictions': [], 'confidence': 0.0}
            
            # Prepare target variable (next day returns)
            returns = data['close'].pct_change().shift(-1).fillna(0)
            
            # Split data for training
            train_size = int(len(features) * 0.8)
            X_train = features[:train_size]
            y_train = returns[:train_size]
            
            # Train ensemble models
            ensemble_predictions = {}
            for name, model in self.ensemble_models.items():
                try:
                    model.fit(X_train, y_train)
                    
                    # Predict future returns
                    last_features = features[-1].reshape(1, -1)
                    pred_return = model.predict(last_features)[0]
                    ensemble_predictions[name] = pred_return
                    
                except Exception as e:
                    logger.warning(f"Model {name} failed: {e}")
                    ensemble_predictions[name] = 0.0
            
            # Combine predictions
            avg_return = np.mean(list(ensemble_predictions.values()))
            
            # Generate future predictions
            last_price = data['close'].iloc[-1]
            predictions = []
            
            for i in range(days_ahead):
                predicted_price = last_price * (1 + avg_return)
                predictions.append({
                    'date': (datetime.now() + timedelta(days=i+1)).isoformat(),
                    'predicted_price': float(predicted_price),
                    'confidence': max(0.2, 0.9 - (i * 0.1))
                })
                last_price = predicted_price
                avg_return *= 0.8  # Decay the return prediction
            
            return {
                'predictions': predictions,
                'model': 'Ensemble',
                'ensemble_predictions': ensemble_predictions,
                'confidence': 0.7,
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Ensemble prediction failed: {e}")
            return {'predictions': [], 'confidence': 0.0}
    
    async def get_trading_signal(self, data: pd.DataFrame, news_sentiment: float = 0.0) -> Dict[str, Any]:
        """Get trading signal from RL agent"""
        try:
            features = self.prepare_features(data)
            
            if len(features) < 20:
                return {'action': 'hold', 'confidence': 0.0}
            
            # Prepare state for RL agent
            state = np.concatenate([
                features[-1],  # Latest technical features
                [news_sentiment],  # Sentiment score
            ])
            
            # Ensure state has correct size
            if len(state) < self.rl_agent.state_size:
                state = np.pad(state, (0, self.rl_agent.state_size - len(state)), 'constant')
            elif len(state) > self.rl_agent.state_size:
                state = state[:self.rl_agent.state_size]
            
            # Get action from RL agent
            action = self.rl_agent.act(state)
            
            action_map = {0: 'hold', 1: 'buy', 2: 'sell'}
            confidence = 1.0 - self.rl_agent.epsilon  # Higher confidence as agent learns
            
            return {
                'action': action_map[action],
                'confidence': float(confidence),
                'state_features': state.tolist(),
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Trading signal generation failed: {e}")
            return {'action': 'hold', 'confidence': 0.0}
    
    async def analyze_market_regime(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Analyze current market regime"""
        try:
            if len(data) < 100:
                return {'regime': 'unknown', 'confidence': 0.0}
            
            # Calculate various market indicators
            returns = data['close'].pct_change().dropna()
            
            # Volatility regime
            short_vol = returns.rolling(20).std()
            long_vol = returns.rolling(60).std()
            vol_ratio = short_vol.iloc[-1] / long_vol.iloc[-1]
            
            # Trend regime
            sma_50 = data['close'].rolling(50).mean()
            sma_200 = data['close'].rolling(200).mean()
            
            current_price = data['close'].iloc[-1]
            trend_strength = (current_price - sma_200.iloc[-1]) / sma_200.iloc[-1]
            
            # Momentum regime
            momentum = returns.rolling(20).mean().iloc[-1]
            
            # Determine regime
            if vol_ratio > 1.5:
                regime = 'high_volatility'
                confidence = min(vol_ratio / 2.0, 1.0)
            elif trend_strength > 0.1:
                regime = 'bull_market'
                confidence = min(trend_strength * 5, 1.0)
            elif trend_strength < -0.1:
                regime = 'bear_market'
                confidence = min(abs(trend_strength) * 5, 1.0)
            elif abs(momentum) < 0.001:
                regime = 'sideways'
                confidence = 0.7
            else:
                regime = 'transitional'
                confidence = 0.5
            
            return {
                'regime': regime,
                'confidence': float(confidence),
                'volatility_ratio': float(vol_ratio),
                'trend_strength': float(trend_strength),
                'momentum': float(momentum),
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Market regime analysis failed: {e}")
            return {'regime': 'unknown', 'confidence': 0.0}
    
    async def comprehensive_analysis(self, data: pd.DataFrame, news_texts: List[str] = None) -> Dict[str, Any]:
        """Perform comprehensive AI analysis"""
        try:
            # Sentiment analysis
            sentiment_score = 0.0
            if news_texts:
                sentiment_results = self.sentiment_analyzer.analyze_sentiment(news_texts)
                sentiment_score = np.mean([r['sentiment_score'] for r in sentiment_results])
            
            # Run all analyses concurrently
            lstm_pred, ensemble_pred, trading_signal, market_regime = await asyncio.gather(
                self.predict_price_lstm(data),
                self.predict_price_ensemble(data),
                self.get_trading_signal(data, sentiment_score),
                self.analyze_market_regime(data)
            )
            
            return {
                'lstm_predictions': lstm_pred,
                'ensemble_predictions': ensemble_pred,
                'trading_signal': trading_signal,
                'market_regime': market_regime,
                'sentiment_analysis': {
                    'score': float(sentiment_score),
                    'interpretation': 'bullish' if sentiment_score > 0.1 else 'bearish' if sentiment_score < -0.1 else 'neutral'
                },
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Comprehensive analysis failed: {e}")
            return {
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }

# Global AI engine instance
ai_engine = AdvancedAIEngine() 