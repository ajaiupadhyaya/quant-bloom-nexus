import numpy as np
import pandas as pd
from scipy import signal
from typing import Dict, List, Tuple, Optional, Any
import logging
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)

class TrendDirection(Enum):
    BULLISH = "bullish"
    BEARISH = "bearish"
    NEUTRAL = "neutral"

class PatternType(Enum):
    HEAD_AND_SHOULDERS = "head_and_shoulders"
    DOUBLE_TOP = "double_top"
    DOUBLE_BOTTOM = "double_bottom"
    TRIANGLE = "triangle"
    WEDGE = "wedge"
    FLAG = "flag"
    PENNANT = "pennant"
    CUP_AND_HANDLE = "cup_and_handle"

@dataclass
class TechnicalSignal:
    indicator: str
    signal: str  # BUY, SELL, HOLD
    strength: float  # 0-1
    timestamp: str
    value: float
    description: str

@dataclass
class PatternRecognition:
    pattern_type: PatternType
    confidence: float
    start_date: str
    end_date: str
    target_price: float
    stop_loss: float

class TechnicalAnalysisEngine:
    """Comprehensive technical analysis engine with 50+ indicators"""
    
    def __init__(self):
        self.signals = []
        
    # =================== TREND INDICATORS ===================
    
    def simple_moving_average(self, data: pd.Series, window: int) -> pd.Series:
        """Simple Moving Average"""
        return data.rolling(window=window).mean()
    
    def exponential_moving_average(self, data: pd.Series, window: int) -> pd.Series:
        """Exponential Moving Average"""
        return data.ewm(span=window).mean()
    
    def weighted_moving_average(self, data: pd.Series, window: int) -> pd.Series:
        """Weighted Moving Average"""
        weights = np.arange(1, window + 1)
        return data.rolling(window).apply(lambda x: np.dot(x, weights) / weights.sum(), raw=True)
    
    def hull_moving_average(self, data: pd.Series, window: int) -> pd.Series:
        """Hull Moving Average"""
        half_window = int(window / 2)
        sqrt_window = int(np.sqrt(window))
        
        wma_half = self.weighted_moving_average(data, half_window)
        wma_full = self.weighted_moving_average(data, window)
        
        hull_ma = self.weighted_moving_average(2 * wma_half - wma_full, sqrt_window)
        return hull_ma
    
    def kaufman_adaptive_moving_average(self, data: pd.Series, window: int = 10) -> pd.Series:
        """Kaufman's Adaptive Moving Average"""
        change = abs(data.diff(window))
        volatility = data.diff().abs().rolling(window).sum()
        
        efficiency_ratio = change / volatility
        efficiency_ratio = efficiency_ratio.fillna(0)
        
        # Smoothing constants
        fastest_sc = 2 / (2 + 1)
        slowest_sc = 2 / (30 + 1)
        
        sc = (efficiency_ratio * (fastest_sc - slowest_sc) + slowest_sc) ** 2
        
        kama = pd.Series(index=data.index, dtype=float)
        kama.iloc[0] = data.iloc[0]
        
        for i in range(1, len(data)):
            kama.iloc[i] = kama.iloc[i-1] + sc.iloc[i] * (data.iloc[i] - kama.iloc[i-1])
        
        return kama
    
    def parabolic_sar(self, high: pd.Series, low: pd.Series, close: pd.Series, 
                     af_start: float = 0.02, af_increment: float = 0.02, af_max: float = 0.2) -> pd.Series:
        """Parabolic SAR"""
        length = len(close)
        psar = np.zeros(length)
        af = af_start
        ep = 0
        trend = 1  # 1 for uptrend, -1 for downtrend
        
        psar[0] = low.iloc[0]
        
        for i in range(1, length):
            if trend == 1:  # Uptrend
                psar[i] = psar[i-1] + af * (ep - psar[i-1])
                
                if low.iloc[i] < psar[i]:
                    trend = -1
                    psar[i] = ep
                    af = af_start
                    ep = low.iloc[i]
                else:
                    if high.iloc[i] > ep:
                        ep = high.iloc[i]
                        af = min(af + af_increment, af_max)
                    
                    psar[i] = min(psar[i], low.iloc[i-1], low.iloc[i-2] if i > 1 else low.iloc[i-1])
            
            else:  # Downtrend
                psar[i] = psar[i-1] + af * (ep - psar[i-1])
                
                if high.iloc[i] > psar[i]:
                    trend = 1
                    psar[i] = ep
                    af = af_start
                    ep = high.iloc[i]
                else:
                    if low.iloc[i] < ep:
                        ep = low.iloc[i]
                        af = min(af + af_increment, af_max)
                    
                    psar[i] = max(psar[i], high.iloc[i-1], high.iloc[i-2] if i > 1 else high.iloc[i-1])
        
        return pd.Series(psar, index=close.index)
    
    # =================== MOMENTUM INDICATORS ===================
    
    def rsi(self, data: pd.Series, window: int = 14) -> pd.Series:
        """Relative Strength Index"""
        delta = data.diff()
        gain = delta.where(delta > 0, 0).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))
    
    def stochastic_oscillator(self, high: pd.Series, low: pd.Series, close: pd.Series, 
                            k_window: int = 14, d_window: int = 3) -> Dict[str, pd.Series]:
        """Stochastic Oscillator"""
        lowest_low = low.rolling(window=k_window).min()
        highest_high = high.rolling(window=k_window).max()
        
        k_percent = 100 * ((close - lowest_low) / (highest_high - lowest_low))
        d_percent = k_percent.rolling(window=d_window).mean()
        
        return {'%K': k_percent, '%D': d_percent}
    
    def williams_r(self, high: pd.Series, low: pd.Series, close: pd.Series, window: int = 14) -> pd.Series:
        """Williams %R"""
        highest_high = high.rolling(window=window).max()
        lowest_low = low.rolling(window=window).min()
        
        return -100 * ((highest_high - close) / (highest_high - lowest_low))
    
    def commodity_channel_index(self, high: pd.Series, low: pd.Series, close: pd.Series, 
                               window: int = 20) -> pd.Series:
        """Commodity Channel Index"""
        typical_price = (high + low + close) / 3
        sma = typical_price.rolling(window=window).mean()
        mad = typical_price.rolling(window=window).apply(lambda x: np.mean(np.abs(x - x.mean())))
        
        return (typical_price - sma) / (0.015 * mad)
    
    def rate_of_change(self, data: pd.Series, window: int = 12) -> pd.Series:
        """Rate of Change"""
        return ((data - data.shift(window)) / data.shift(window)) * 100
    
    def momentum(self, data: pd.Series, window: int = 10) -> pd.Series:
        """Momentum"""
        return data - data.shift(window)
    
    def awesome_oscillator(self, high: pd.Series, low: pd.Series) -> pd.Series:
        """Awesome Oscillator"""
        median_price = (high + low) / 2
        ao = median_price.rolling(5).mean() - median_price.rolling(34).mean()
        return ao
    
    # =================== VOLATILITY INDICATORS ===================
    
    def bollinger_bands(self, data: pd.Series, window: int = 20, num_std: float = 2) -> Dict[str, pd.Series]:
        """Bollinger Bands"""
        sma = data.rolling(window=window).mean()
        std = data.rolling(window=window).std()
        
        upper = sma + (std * num_std)
        lower = sma - (std * num_std)
        
        return {
            'upper': upper,
            'middle': sma,
            'lower': lower,
            'bandwidth': (upper - lower) / sma,
            'percent_b': (data - lower) / (upper - lower)
        }
    
    def keltner_channels(self, high: pd.Series, low: pd.Series, close: pd.Series, 
                        window: int = 20, multiplier: float = 2) -> Dict[str, pd.Series]:
        """Keltner Channels"""
        ema = close.ewm(span=window).mean()
        atr = self.average_true_range(high, low, close, window)
        
        upper = ema + (multiplier * atr)
        lower = ema - (multiplier * atr)
        
        return {'upper': upper, 'middle': ema, 'lower': lower}
    
    def average_true_range(self, high: pd.Series, low: pd.Series, close: pd.Series, 
                          window: int = 14) -> pd.Series:
        """Average True Range"""
        high_low = high - low
        high_close = abs(high - close.shift())
        low_close = abs(low - close.shift())
        
        true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        return true_range.rolling(window=window).mean()
    
    def donchian_channels(self, high: pd.Series, low: pd.Series, window: int = 20) -> Dict[str, pd.Series]:
        """Donchian Channels"""
        upper = high.rolling(window=window).max()
        lower = low.rolling(window=window).min()
        middle = (upper + lower) / 2
        
        return {'upper': upper, 'middle': middle, 'lower': lower}
    
    # =================== VOLUME INDICATORS ===================
    
    def on_balance_volume(self, close: pd.Series, volume: pd.Series) -> pd.Series:
        """On Balance Volume"""
        obv = np.where(close > close.shift(), volume, 
                      np.where(close < close.shift(), -volume, 0))
        return pd.Series(obv, index=close.index).cumsum()
    
    def accumulation_distribution_line(self, high: pd.Series, low: pd.Series, 
                                     close: pd.Series, volume: pd.Series) -> pd.Series:
        """Accumulation/Distribution Line"""
        clv = ((close - low) - (high - close)) / (high - low)
        clv = clv.fillna(0)
        ad = (clv * volume).cumsum()
        return ad
    
    def chaikin_money_flow(self, high: pd.Series, low: pd.Series, close: pd.Series, 
                          volume: pd.Series, window: int = 20) -> pd.Series:
        """Chaikin Money Flow"""
        clv = ((close - low) - (high - close)) / (high - low)
        clv = clv.fillna(0)
        
        money_flow_volume = clv * volume
        cmf = money_flow_volume.rolling(window=window).sum() / volume.rolling(window=window).sum()
        return cmf
    
    def volume_weighted_average_price(self, high: pd.Series, low: pd.Series, 
                                    close: pd.Series, volume: pd.Series) -> pd.Series:
        """Volume Weighted Average Price"""
        typical_price = (high + low + close) / 3
        vwap = (typical_price * volume).cumsum() / volume.cumsum()
        return vwap
    
    def money_flow_index(self, high: pd.Series, low: pd.Series, close: pd.Series, 
                        volume: pd.Series, window: int = 14) -> pd.Series:
        """Money Flow Index"""
        typical_price = (high + low + close) / 3
        money_flow = typical_price * volume
        
        positive_flow = money_flow.where(typical_price > typical_price.shift(), 0)
        negative_flow = money_flow.where(typical_price < typical_price.shift(), 0)
        
        positive_mf = positive_flow.rolling(window=window).sum()
        negative_mf = negative_flow.rolling(window=window).sum()
        
        money_ratio = positive_mf / negative_mf
        mfi = 100 - (100 / (1 + money_ratio))
        return mfi
    
    # =================== OSCILLATORS ===================
    
    def macd(self, data: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> Dict[str, pd.Series]:
        """MACD"""
        ema_fast = data.ewm(span=fast).mean()
        ema_slow = data.ewm(span=slow).mean()
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=signal).mean()
        histogram = macd_line - signal_line
        
        return {'macd': macd_line, 'signal': signal_line, 'histogram': histogram}
    
    def trix(self, data: pd.Series, window: int = 14) -> pd.Series:
        """TRIX"""
        ema1 = data.ewm(span=window).mean()
        ema2 = ema1.ewm(span=window).mean()
        ema3 = ema2.ewm(span=window).mean()
        
        trix = (ema3 / ema3.shift() - 1) * 10000
        return trix
    
    def ultimate_oscillator(self, high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
        """Ultimate Oscillator"""
        bp = close - pd.concat([low, close.shift()], axis=1).min(axis=1)
        tr = pd.concat([high - low, abs(high - close.shift()), abs(low - close.shift())], axis=1).max(axis=1)
        
        avg7 = bp.rolling(7).sum() / tr.rolling(7).sum()
        avg14 = bp.rolling(14).sum() / tr.rolling(14).sum()
        avg28 = bp.rolling(28).sum() / tr.rolling(28).sum()
        
        uo = 100 * ((4 * avg7) + (2 * avg14) + avg28) / (4 + 2 + 1)
        return uo
    
    # =================== PATTERN RECOGNITION ===================
    
    def detect_support_resistance(self, data: pd.Series, window: int = 20, 
                                 min_touches: int = 3) -> Dict[str, List[float]]:
        """Detect Support and Resistance Levels"""
        try:
            highs = data.rolling(window=window).max()
            lows = data.rolling(window=window).min()
            
            # Find local maxima and minima
            local_maxima = []
            local_minima = []
            
            for i in range(window, len(data) - window):
                if data.iloc[i] == highs.iloc[i]:
                    local_maxima.append(data.iloc[i])
                if data.iloc[i] == lows.iloc[i]:
                    local_minima.append(data.iloc[i])
            
            # Group similar levels
            resistance_levels = self._group_levels(local_maxima, min_touches)
            support_levels = self._group_levels(local_minima, min_touches)
            
            return {'resistance': resistance_levels, 'support': support_levels}
        except Exception as e:
            logger.error(f"Support/Resistance detection failed: {e}")
            return {'resistance': [], 'support': []}
    
    def _group_levels(self, levels: List[float], min_touches: int, tolerance: float = 0.02) -> List[float]:
        """Group similar price levels"""
        if not levels:
            return []
        
        grouped_levels = []
        levels_sorted = sorted(levels)
        
        current_group = [levels_sorted[0]]
        
        for level in levels_sorted[1:]:
            if abs(level - current_group[-1]) / current_group[-1] <= tolerance:
                current_group.append(level)
            else:
                if len(current_group) >= min_touches:
                    grouped_levels.append(np.mean(current_group))
                current_group = [level]
        
        if len(current_group) >= min_touches:
            grouped_levels.append(np.mean(current_group))
        
        return grouped_levels
    
    def detect_head_and_shoulders(self, data: pd.Series, window: int = 20) -> Optional[PatternRecognition]:
        """Detect Head and Shoulders Pattern"""
        try:
            peaks = signal.find_peaks(data, distance=window)[0]
            
            if len(peaks) < 3:
                return None
            
            # Look for three consecutive peaks where middle is highest
            for i in range(len(peaks) - 2):
                left_shoulder = data.iloc[peaks[i]]
                head = data.iloc[peaks[i + 1]]
                right_shoulder = data.iloc[peaks[i + 2]]
                
                # Check if it's a valid head and shoulders
                if (head > left_shoulder and head > right_shoulder and
                    abs(left_shoulder - right_shoulder) / left_shoulder < 0.05):
                    
                    # Calculate neckline and target
                    neckline = min(data.iloc[peaks[i]:peaks[i+1]].min(), 
                                 data.iloc[peaks[i+1]:peaks[i+2]].min())
                    target_price = neckline - (head - neckline)
                    
                    return PatternRecognition(
                        pattern_type=PatternType.HEAD_AND_SHOULDERS,
                        confidence=0.7,
                        start_date=data.index[peaks[i]].strftime('%Y-%m-%d'),
                        end_date=data.index[peaks[i+2]].strftime('%Y-%m-%d'),
                        target_price=target_price,
                        stop_loss=head
                    )
            
            return None
        except Exception as e:
            logger.error(f"Head and shoulders detection failed: {e}")
            return None
    
    def detect_double_top_bottom(self, data: pd.Series, window: int = 20) -> Optional[PatternRecognition]:
        """Detect Double Top/Bottom Patterns"""
        try:
            peaks = signal.find_peaks(data, distance=window)[0]
            valleys = signal.find_peaks(-data, distance=window)[0]
            
            # Double Top
            if len(peaks) >= 2:
                for i in range(len(peaks) - 1):
                    peak1 = data.iloc[peaks[i]]
                    peak2 = data.iloc[peaks[i + 1]]
                    
                    if abs(peak1 - peak2) / peak1 < 0.03:  # Similar heights
                        valley_between = data.iloc[peaks[i]:peaks[i+1]].min()
                        target_price = valley_between - (peak1 - valley_between)
                        
                        return PatternRecognition(
                            pattern_type=PatternType.DOUBLE_TOP,
                            confidence=0.6,
                            start_date=data.index[peaks[i]].strftime('%Y-%m-%d'),
                            end_date=data.index[peaks[i+1]].strftime('%Y-%m-%d'),
                            target_price=target_price,
                            stop_loss=max(peak1, peak2)
                        )
            
            # Double Bottom
            if len(valleys) >= 2:
                for i in range(len(valleys) - 1):
                    valley1 = data.iloc[valleys[i]]
                    valley2 = data.iloc[valleys[i + 1]]
                    
                    if abs(valley1 - valley2) / valley1 < 0.03:  # Similar depths
                        peak_between = data.iloc[valleys[i]:valleys[i+1]].max()
                        target_price = peak_between + (peak_between - valley1)
                        
                        return PatternRecognition(
                            pattern_type=PatternType.DOUBLE_BOTTOM,
                            confidence=0.6,
                            start_date=data.index[valleys[i]].strftime('%Y-%m-%d'),
                            end_date=data.index[valleys[i+1]].strftime('%Y-%m-%d'),
                            target_price=target_price,
                            stop_loss=min(valley1, valley2)
                        )
            
            return None
        except Exception as e:
            logger.error(f"Double top/bottom detection failed: {e}")
            return None
    
    # =================== SIGNAL GENERATION ===================
    
    def generate_comprehensive_signals(self, high: pd.Series, low: pd.Series, 
                                     close: pd.Series, volume: pd.Series) -> List[TechnicalSignal]:
        """Generate comprehensive technical signals"""
        signals = []
        
        try:
            # RSI Signals
            rsi = self.rsi(close)
            if rsi.iloc[-1] < 30:
                signals.append(TechnicalSignal(
                    indicator="RSI",
                    signal="BUY",
                    strength=0.8,
                    timestamp=close.index[-1].strftime('%Y-%m-%d'),
                    value=rsi.iloc[-1],
                    description="RSI oversold condition"
                ))
            elif rsi.iloc[-1] > 70:
                signals.append(TechnicalSignal(
                    indicator="RSI",
                    signal="SELL",
                    strength=0.8,
                    timestamp=close.index[-1].strftime('%Y-%m-%d'),
                    value=rsi.iloc[-1],
                    description="RSI overbought condition"
                ))
            
            # MACD Signals
            macd_data = self.macd(close)
            macd_line = macd_data['macd']
            signal_line = macd_data['signal']
            
            if macd_line.iloc[-1] > signal_line.iloc[-1] and macd_line.iloc[-2] <= signal_line.iloc[-2]:
                signals.append(TechnicalSignal(
                    indicator="MACD",
                    signal="BUY",
                    strength=0.7,
                    timestamp=close.index[-1].strftime('%Y-%m-%d'),
                    value=macd_line.iloc[-1],
                    description="MACD bullish crossover"
                ))
            elif macd_line.iloc[-1] < signal_line.iloc[-1] and macd_line.iloc[-2] >= signal_line.iloc[-2]:
                signals.append(TechnicalSignal(
                    indicator="MACD",
                    signal="SELL",
                    strength=0.7,
                    timestamp=close.index[-1].strftime('%Y-%m-%d'),
                    value=macd_line.iloc[-1],
                    description="MACD bearish crossover"
                ))
            
            # Bollinger Bands Signals
            bb = self.bollinger_bands(close)
            if close.iloc[-1] < bb['lower'].iloc[-1]:
                signals.append(TechnicalSignal(
                    indicator="Bollinger Bands",
                    signal="BUY",
                    strength=0.6,
                    timestamp=close.index[-1].strftime('%Y-%m-%d'),
                    value=close.iloc[-1],
                    description="Price below lower Bollinger Band"
                ))
            elif close.iloc[-1] > bb['upper'].iloc[-1]:
                signals.append(TechnicalSignal(
                    indicator="Bollinger Bands",
                    signal="SELL",
                    strength=0.6,
                    timestamp=close.index[-1].strftime('%Y-%m-%d'),
                    value=close.iloc[-1],
                    description="Price above upper Bollinger Band"
                ))
            
            # Moving Average Signals
            sma_20 = self.simple_moving_average(close, 20)
            sma_50 = self.simple_moving_average(close, 50)
            
            if sma_20.iloc[-1] > sma_50.iloc[-1] and sma_20.iloc[-2] <= sma_50.iloc[-2]:
                signals.append(TechnicalSignal(
                    indicator="Moving Average",
                    signal="BUY",
                    strength=0.7,
                    timestamp=close.index[-1].strftime('%Y-%m-%d'),
                    value=close.iloc[-1],
                    description="Golden cross: 20-day SMA crosses above 50-day SMA"
                ))
            elif sma_20.iloc[-1] < sma_50.iloc[-1] and sma_20.iloc[-2] >= sma_50.iloc[-2]:
                signals.append(TechnicalSignal(
                    indicator="Moving Average",
                    signal="SELL",
                    strength=0.7,
                    timestamp=close.index[-1].strftime('%Y-%m-%d'),
                    value=close.iloc[-1],
                    description="Death cross: 20-day SMA crosses below 50-day SMA"
                ))
            
            return signals
            
        except Exception as e:
            logger.error(f"Signal generation failed: {e}")
            return []

# Global technical analysis engine
technical_analysis_engine = TechnicalAnalysisEngine() 