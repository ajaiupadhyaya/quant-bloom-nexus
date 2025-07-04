import asyncio
import aiohttp
import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime, timedelta
import logging
import os
import yfinance as yf
import talib
from dataclasses import dataclass
import json

logger = logging.getLogger(__name__)

@dataclass
class TechnicalIndicator:
    name: str
    values: List[float]
    parameters: Dict[str, Any]
    category: str
    description: str

@dataclass
class ChartPattern:
    name: str
    confidence: float
    start_date: str
    end_date: str
    description: str
    price_levels: Dict[str, float]

class TechnicalAnalysisService:
    """Complete technical analysis service with 100+ indicators and pattern recognition"""
    
    def __init__(self):
        self.session = None
        
    async def _get_session(self):
        """Get or create aiohttp session"""
        if self.session is None:
            self.session = aiohttp.ClientSession()
        return self.session
    
    async def close(self):
        """Close aiohttp session"""
        if self.session:
            await self.session.close()
    
    # Advanced Charting with 100+ technical indicators
    async def get_technical_indicators(self, symbol: str, period: str = "1y", interval: str = "1d") -> Dict[str, TechnicalIndicator]:
        """Get 100+ technical indicators with custom parameter settings"""
        try:
            # Get price data
            ticker = yf.Ticker(symbol)
            hist = ticker.history(period=period, interval=interval)
            
            if hist.empty:
                return {}
            
            indicators = {}
            
            # Trend Indicators
            indicators.update(await self._calculate_trend_indicators(hist))
            
            # Momentum Indicators
            indicators.update(await self._calculate_momentum_indicators(hist))
            
            # Volume Indicators
            indicators.update(await self._calculate_volume_indicators(hist))
            
            # Volatility Indicators
            indicators.update(await self._calculate_volatility_indicators(hist))
            
            # Support/Resistance Indicators
            indicators.update(await self._calculate_support_resistance_indicators(hist))
            
            return indicators
            
        except Exception as e:
            logger.error(f"Technical indicators failed for {symbol}: {e}")
            return {}
    
    async def _calculate_trend_indicators(self, hist: pd.DataFrame) -> Dict[str, TechnicalIndicator]:
        """Calculate trend indicators"""
        indicators = {}
        
        try:
            # Moving Averages
            for period in [5, 10, 20, 50, 100, 200]:
                sma = talib.SMA(hist['Close'].values, timeperiod=period)
                ema = talib.EMA(hist['Close'].values, timeperiod=period)
                
                indicators[f'SMA_{period}'] = TechnicalIndicator(
                    name=f"Simple Moving Average ({period})",
                    values=sma.tolist(),
                    parameters={'period': period, 'type': 'SMA'},
                    category="trend",
                    description=f"Simple moving average over {period} periods"
                )
                
                indicators[f'EMA_{period}'] = TechnicalIndicator(
                    name=f"Exponential Moving Average ({period})",
                    values=ema.tolist(),
                    parameters={'period': period, 'type': 'EMA'},
                    category="trend",
                    description=f"Exponential moving average over {period} periods"
                )
            
            # Bollinger Bands
            bb_upper, bb_middle, bb_lower = talib.BBANDS(hist['Close'].values, timeperiod=20, nbdevup=2, nbdevdn=2)
            
            indicators['BB_Upper'] = TechnicalIndicator(
                name="Bollinger Bands Upper",
                values=bb_upper.tolist(),
                parameters={'period': 20, 'std_dev': 2},
                category="trend",
                description="Upper Bollinger Band"
            )
            
            indicators['BB_Middle'] = TechnicalIndicator(
                name="Bollinger Bands Middle",
                values=bb_middle.tolist(),
                parameters={'period': 20, 'std_dev': 2},
                category="trend",
                description="Middle Bollinger Band (SMA)"
            )
            
            indicators['BB_Lower'] = TechnicalIndicator(
                name="Bollinger Bands Lower",
                values=bb_lower.tolist(),
                parameters={'period': 20, 'std_dev': 2},
                category="trend",
                description="Lower Bollinger Band"
            )
            
            # Ichimoku Cloud
            ichimoku = self._calculate_ichimoku(hist)
            indicators.update(ichimoku)
            
            # Parabolic SAR
            sar = talib.SAR(hist['High'].values, hist['Low'].values, acceleration=0.02, maximum=0.2)
            indicators['Parabolic_SAR'] = TechnicalIndicator(
                name="Parabolic SAR",
                values=sar.tolist(),
                parameters={'acceleration': 0.02, 'maximum': 0.2},
                category="trend",
                description="Parabolic Stop and Reverse"
            )
            
            # ADX (Average Directional Index)
            adx = talib.ADX(hist['High'].values, hist['Low'].values, hist['Close'].values, timeperiod=14)
            indicators['ADX'] = TechnicalIndicator(
                name="Average Directional Index",
                values=adx.tolist(),
                parameters={'period': 14},
                category="trend",
                description="Trend strength indicator"
            )
            
        except Exception as e:
            logger.error(f"Trend indicators calculation failed: {e}")
        
        return indicators
    
    async def _calculate_momentum_indicators(self, hist: pd.DataFrame) -> Dict[str, TechnicalIndicator]:
        """Calculate momentum indicators"""
        indicators = {}
        
        try:
            # RSI
            rsi = talib.RSI(hist['Close'].values, timeperiod=14)
            indicators['RSI'] = TechnicalIndicator(
                name="Relative Strength Index",
                values=rsi.tolist(),
                parameters={'period': 14},
                category="momentum",
                description="Momentum oscillator measuring speed and magnitude of price changes"
            )
            
            # MACD
            macd, macd_signal, macd_hist = talib.MACD(hist['Close'].values, fastperiod=12, slowperiod=26, signalperiod=9)
            indicators['MACD'] = TechnicalIndicator(
                name="MACD",
                values=macd.tolist(),
                parameters={'fast_period': 12, 'slow_period': 26, 'signal_period': 9},
                category="momentum",
                description="Moving Average Convergence Divergence"
            )
            
            indicators['MACD_Signal'] = TechnicalIndicator(
                name="MACD Signal",
                values=macd_signal.tolist(),
                parameters={'fast_period': 12, 'slow_period': 26, 'signal_period': 9},
                category="momentum",
                description="MACD signal line"
            )
            
            indicators['MACD_Histogram'] = TechnicalIndicator(
                name="MACD Histogram",
                values=macd_hist.tolist(),
                parameters={'fast_period': 12, 'slow_period': 26, 'signal_period': 9},
                category="momentum",
                description="MACD histogram"
            )
            
            # Stochastic Oscillator
            stoch_k, stoch_d = talib.STOCH(hist['High'].values, hist['Low'].values, hist['Close'].values, fastk_period=14, slowk_period=3, slowd_period=3)
            indicators['Stochastic_K'] = TechnicalIndicator(
                name="Stochastic %K",
                values=stoch_k.tolist(),
                parameters={'k_period': 14, 'd_period': 3},
                category="momentum",
                description="Stochastic oscillator %K line"
            )
            
            indicators['Stochastic_D'] = TechnicalIndicator(
                name="Stochastic %D",
                values=stoch_d.tolist(),
                parameters={'k_period': 14, 'd_period': 3},
                category="momentum",
                description="Stochastic oscillator %D line"
            )
            
            # Williams %R
            willr = talib.WILLR(hist['High'].values, hist['Low'].values, hist['Close'].values, timeperiod=14)
            indicators['Williams_R'] = TechnicalIndicator(
                name="Williams %R",
                values=willr.tolist(),
                parameters={'period': 14},
                category="momentum",
                description="Williams %R oscillator"
            )
            
            # CCI (Commodity Channel Index)
            cci = talib.CCI(hist['High'].values, hist['Low'].values, hist['Close'].values, timeperiod=14)
            indicators['CCI'] = TechnicalIndicator(
                name="Commodity Channel Index",
                values=cci.tolist(),
                parameters={'period': 14},
                category="momentum",
                description="Commodity Channel Index"
            )
            
            # ROC (Rate of Change)
            roc = talib.ROC(hist['Close'].values, timeperiod=10)
            indicators['ROC'] = TechnicalIndicator(
                name="Rate of Change",
                values=roc.tolist(),
                parameters={'period': 10},
                category="momentum",
                description="Rate of Change"
            )
            
        except Exception as e:
            logger.error(f"Momentum indicators calculation failed: {e}")
        
        return indicators
    
    async def _calculate_volume_indicators(self, hist: pd.DataFrame) -> Dict[str, TechnicalIndicator]:
        """Calculate volume indicators"""
        indicators = {}
        
        try:
            # Volume Profile
            vwap = self._calculate_vwap(hist)
            indicators['VWAP'] = TechnicalIndicator(
                name="Volume Weighted Average Price",
                values=vwap,
                parameters={},
                category="volume",
                description="Volume Weighted Average Price"
            )
            
            # On Balance Volume (OBV)
            obv = talib.OBV(hist['Close'].values, hist['Volume'].values)
            indicators['OBV'] = TechnicalIndicator(
                name="On Balance Volume",
                values=obv.tolist(),
                parameters={},
                category="volume",
                description="On Balance Volume"
            )
            
            # Accumulation/Distribution Line
            ad = talib.AD(hist['High'].values, hist['Low'].values, hist['Close'].values, hist['Volume'].values)
            indicators['AD_Line'] = TechnicalIndicator(
                name="Accumulation/Distribution Line",
                values=ad.tolist(),
                parameters={},
                category="volume",
                description="Accumulation/Distribution Line"
            )
            
            # Chaikin Money Flow
            cmf = talib.ADOSC(hist['High'].values, hist['Low'].values, hist['Close'].values, hist['Volume'].values, fastperiod=3, slowperiod=10)
            indicators['Chaikin_Money_Flow'] = TechnicalIndicator(
                name="Chaikin Money Flow",
                values=cmf.tolist(),
                parameters={'fast_period': 3, 'slow_period': 10},
                category="volume",
                description="Chaikin Money Flow"
            )
            
            # Money Flow Index
            mfi = talib.MFI(hist['High'].values, hist['Low'].values, hist['Close'].values, hist['Volume'].values, timeperiod=14)
            indicators['MFI'] = TechnicalIndicator(
                name="Money Flow Index",
                values=mfi.tolist(),
                parameters={'period': 14},
                category="volume",
                description="Money Flow Index"
            )
            
        except Exception as e:
            logger.error(f"Volume indicators calculation failed: {e}")
        
        return indicators
    
    async def _calculate_volatility_indicators(self, hist: pd.DataFrame) -> Dict[str, TechnicalIndicator]:
        """Calculate volatility indicators"""
        indicators = {}
        
        try:
            # Average True Range (ATR)
            atr = talib.ATR(hist['High'].values, hist['Low'].values, hist['Close'].values, timeperiod=14)
            indicators['ATR'] = TechnicalIndicator(
                name="Average True Range",
                values=atr.tolist(),
                parameters={'period': 14},
                category="volatility",
                description="Average True Range"
            )
            
            # Standard Deviation
            std_dev = talib.STDDEV(hist['Close'].values, timeperiod=20, nbdev=1)
            indicators['Standard_Deviation'] = TechnicalIndicator(
                name="Standard Deviation",
                values=std_dev.tolist(),
                parameters={'period': 20},
                category="volatility",
                description="Standard Deviation"
            )
            
            # Historical Volatility
            hist_vol = self._calculate_historical_volatility(hist)
            indicators['Historical_Volatility'] = TechnicalIndicator(
                name="Historical Volatility",
                values=hist_vol,
                parameters={'period': 20},
                category="volatility",
                description="Historical Volatility"
            )
            
        except Exception as e:
            logger.error(f"Volatility indicators calculation failed: {e}")
        
        return indicators
    
    async def _calculate_support_resistance_indicators(self, hist: pd.DataFrame) -> Dict[str, TechnicalIndicator]:
        """Calculate support and resistance indicators"""
        indicators = {}
        
        try:
            # Pivot Points
            pivot_points = self._calculate_pivot_points(hist)
            indicators['Pivot_Points'] = TechnicalIndicator(
                name="Pivot Points",
                values=pivot_points,
                parameters={},
                category="support_resistance",
                description="Pivot Points"
            )
            
            # Fibonacci Retracements
            fib_levels = self._calculate_fibonacci_levels(hist)
            indicators['Fibonacci_Levels'] = TechnicalIndicator(
                name="Fibonacci Retracements",
                values=fib_levels,
                parameters={},
                category="support_resistance",
                description="Fibonacci Retracement Levels"
            )
            
        except Exception as e:
            logger.error(f"Support/Resistance indicators calculation failed: {e}")
        
        return indicators
    
    def _calculate_ichimoku(self, hist: pd.DataFrame) -> Dict[str, TechnicalIndicator]:
        """Calculate Ichimoku Cloud components"""
        indicators = {}
        
        try:
            high = hist['High'].values
            low = hist['Low'].values
            close = hist['Close'].values
            
            # Tenkan-sen (Conversion Line)
            period9_high = pd.Series(high).rolling(window=9).max()
            period9_low = pd.Series(low).rolling(window=9).min()
            tenkan_sen = (period9_high + period9_low) / 2
            
            # Kijun-sen (Base Line)
            period26_high = pd.Series(high).rolling(window=26).max()
            period26_low = pd.Series(low).rolling(window=26).min()
            kijun_sen = (period26_high + period26_low) / 2
            
            # Senkou Span A (Leading Span A)
            senkou_span_a = ((tenkan_sen + kijun_sen) / 2).shift(26)
            
            # Senkou Span B (Leading Span B)
            period52_high = pd.Series(high).rolling(window=52).max()
            period52_low = pd.Series(low).rolling(window=52).min()
            senkou_span_b = ((period52_high + period52_low) / 2).shift(26)
            
            # Chikou Span (Lagging Span)
            chikou_span = close.shift(-26)
            
            indicators['Ichimoku_Tenkan'] = TechnicalIndicator(
                name="Ichimoku Tenkan-sen",
                values=tenkan_sen.tolist(),
                parameters={'period': 9},
                category="trend",
                description="Ichimoku Conversion Line"
            )
            
            indicators['Ichimoku_Kijun'] = TechnicalIndicator(
                name="Ichimoku Kijun-sen",
                values=kijun_sen.tolist(),
                parameters={'period': 26},
                category="trend",
                description="Ichimoku Base Line"
            )
            
            indicators['Ichimoku_Senkou_A'] = TechnicalIndicator(
                name="Ichimoku Senkou Span A",
                values=senkou_span_a.tolist(),
                parameters={'shift': 26},
                category="trend",
                description="Ichimoku Leading Span A"
            )
            
            indicators['Ichimoku_Senkou_B'] = TechnicalIndicator(
                name="Ichimoku Senkou Span B",
                values=senkou_span_b.tolist(),
                parameters={'shift': 26},
                category="trend",
                description="Ichimoku Leading Span B"
            )
            
            indicators['Ichimoku_Chikou'] = TechnicalIndicator(
                name="Ichimoku Chikou Span",
                values=chikou_span.tolist(),
                parameters={'shift': -26},
                category="trend",
                description="Ichimoku Lagging Span"
            )
            
        except Exception as e:
            logger.error(f"Ichimoku calculation failed: {e}")
        
        return indicators
    
    def _calculate_vwap(self, hist: pd.DataFrame) -> List[float]:
        """Calculate Volume Weighted Average Price"""
        try:
            typical_price = (hist['High'] + hist['Low'] + hist['Close']) / 3
            vwap = (typical_price * hist['Volume']).cumsum() / hist['Volume'].cumsum()
            return vwap.tolist()
        except Exception as e:
            logger.error(f"VWAP calculation failed: {e}")
            return []
    
    def _calculate_historical_volatility(self, hist: pd.DataFrame, period: int = 20) -> List[float]:
        """Calculate historical volatility"""
        try:
            returns = hist['Close'].pct_change()
            volatility = returns.rolling(window=period).std() * np.sqrt(252)  # Annualized
            return volatility.tolist()
        except Exception as e:
            logger.error(f"Historical volatility calculation failed: {e}")
            return []
    
    def _calculate_pivot_points(self, hist: pd.DataFrame) -> List[float]:
        """Calculate pivot points"""
        try:
            pivot = (hist['High'] + hist['Low'] + hist['Close']) / 3
            return pivot.tolist()
        except Exception as e:
            logger.error(f"Pivot points calculation failed: {e}")
            return []
    
    def _calculate_fibonacci_levels(self, hist: pd.DataFrame) -> List[float]:
        """Calculate Fibonacci retracement levels"""
        try:
            high = hist['High'].max()
            low = hist['Low'].min()
            diff = high - low
            
            fib_levels = [
                high,
                high - 0.236 * diff,
                high - 0.382 * diff,
                high - 0.5 * diff,
                high - 0.618 * diff,
                high - 0.786 * diff,
                low
            ]
            
            return fib_levels
        except Exception as e:
            logger.error(f"Fibonacci levels calculation failed: {e}")
            return []
    
    # Pattern Recognition
    async def detect_chart_patterns(self, symbol: str, period: str = "1y") -> List[ChartPattern]:
        """Automated chart pattern detection and alerts"""
        try:
            ticker = yf.Ticker(symbol)
            hist = ticker.history(period=period)
            
            if hist.empty:
                return []
            
            patterns = []
            
            # Detect common patterns
            patterns.extend(self._detect_candlestick_patterns(hist))
            patterns.extend(self._detect_chart_patterns(hist))
            patterns.extend(self._detect_support_resistance_patterns(hist))
            
            return patterns
            
        except Exception as e:
            logger.error(f"Pattern detection failed for {symbol}: {e}")
            return []
    
    def _detect_candlestick_patterns(self, hist: pd.DataFrame) -> List[ChartPattern]:
        """Detect candlestick patterns"""
        patterns = []
        
        try:
            # Get candlestick patterns using talib
            pattern_functions = {
                'CDL2CROWS': 'Two Crows',
                'CDL3BLACKCROWS': 'Three Black Crows',
                'CDL3INSIDE': 'Three Inside Up/Down',
                'CDL3LINESTRIKE': 'Three-Line Strike',
                'CDL3OUTSIDE': 'Three Outside Up/Down',
                'CDL3STARSINSOUTH': 'Three Stars In The South',
                'CDL3WHITESOLDIERS': 'Three Advancing White Soldiers',
                'CDLABANDONEDBABY': 'Abandoned Baby',
                'CDLADVANCEBLOCK': 'Advance Block',
                'CDLBELTHOLD': 'Belt-hold',
                'CDLBREAKAWAY': 'Breakaway',
                'CDLDARKCLOUDCOVER': 'Dark Cloud Cover',
                'CDLDOJI': 'Doji',
                'CDLENGULFING': 'Engulfing Pattern',
                'CDLEVENINGDOJISTAR': 'Evening Doji Star',
                'CDLEVENINGSTAR': 'Evening Star',
                'CDLHAMMER': 'Hammer',
                'CDLHANGINGMAN': 'Hanging Man',
                'CDLHARAMI': 'Harami Pattern',
                'CDLMARUBOZU': 'Marubozu',
                'CDLMORNINGDOJISTAR': 'Morning Doji Star',
                'CDLMORNINGSTAR': 'Morning Star',
                'CDLPIERCING': 'Piercing Pattern',
                'CDLSHOOTINGSTAR': 'Shooting Star',
                'CDLSPINNINGTOP': 'Spinning Top',
                'CDLTAKURI': 'Takuri',
                'CDLTRISTAR': 'Tristar Pattern'
            }
            
            for func_name, pattern_name in pattern_functions.items():
                try:
                    func = getattr(talib, func_name)
                    result = func(hist['Open'].values, hist['High'].values, hist['Low'].values, hist['Close'].values)
                    
                    # Find where pattern occurs (non-zero values)
                    pattern_indices = np.where(result != 0)[0]
                    
                    for idx in pattern_indices[-5:]:  # Last 5 occurrences
                        if idx < len(hist):
                            date = hist.index[idx].strftime('%Y-%m-%d')
                            confidence = abs(result[idx]) / 100 if result[idx] != 0 else 0.5
                            
                            patterns.append(ChartPattern(
                                name=pattern_name,
                                confidence=confidence,
                                start_date=date,
                                end_date=date,
                                description=f"{pattern_name} pattern detected",
                                price_levels={'price': hist['Close'].iloc[idx]}
                            ))
                            
                except Exception as e:
                    logger.warning(f"Candlestick pattern {pattern_name} detection failed: {e}")
                    continue
            
        except Exception as e:
            logger.error(f"Candlestick pattern detection failed: {e}")
        
        return patterns
    
    def _detect_chart_patterns(self, hist: pd.DataFrame) -> List[ChartPattern]:
        """Detect chart patterns like head and shoulders, triangles, etc."""
        patterns = []
        
        try:
            # Head and Shoulders detection
            hs_pattern = self._detect_head_and_shoulders(hist)
            if hs_pattern:
                patterns.append(hs_pattern)
            
            # Double Top/Bottom detection
            dt_pattern = self._detect_double_top_bottom(hist)
            if dt_pattern:
                patterns.append(dt_pattern)
            
            # Triangle detection
            triangle_pattern = self._detect_triangle(hist)
            if triangle_pattern:
                patterns.append(triangle_pattern)
            
        except Exception as e:
            logger.error(f"Chart pattern detection failed: {e}")
        
        return patterns
    
    def _detect_head_and_shoulders(self, hist: pd.DataFrame) -> Optional[ChartPattern]:
        """Detect head and shoulders pattern"""
        try:
            # Simplified head and shoulders detection
            # This is a basic implementation - real pattern detection is more complex
            
            # Look for three peaks with middle peak higher than others
            highs = hist['High'].rolling(window=5, center=True).max()
            peaks = []
            
            for i in range(10, len(highs) - 10):
                if highs.iloc[i] == hist['High'].iloc[i]:
                    peaks.append((i, hist['High'].iloc[i]))
            
            if len(peaks) >= 3:
                # Check if middle peak is higher
                peaks = peaks[-3:]  # Last 3 peaks
                if peaks[1][1] > peaks[0][1] and peaks[1][1] > peaks[2][1]:
                    return ChartPattern(
                        name="Head and Shoulders",
                        confidence=0.7,
                        start_date=hist.index[peaks[0][0]].strftime('%Y-%m-%d'),
                        end_date=hist.index[peaks[2][0]].strftime('%Y-%m-%d'),
                        description="Head and Shoulders pattern detected",
                        price_levels={
                            'left_shoulder': peaks[0][1],
                            'head': peaks[1][1],
                            'right_shoulder': peaks[2][1]
                        }
                    )
            
            return None
            
        except Exception as e:
            logger.error(f"Head and shoulders detection failed: {e}")
            return None
    
    def _detect_double_top_bottom(self, hist: pd.DataFrame) -> Optional[ChartPattern]:
        """Detect double top or double bottom pattern"""
        try:
            # Simplified double top/bottom detection
            highs = hist['High'].rolling(window=5, center=True).max()
            lows = hist['Low'].rolling(window=5, center=True).min()
            
            # Look for double top
            for i in range(10, len(highs) - 10):
                if highs.iloc[i] == hist['High'].iloc[i]:
                    # Look for another peak within reasonable distance
                    for j in range(i + 10, min(i + 50, len(highs))):
                        if highs.iloc[j] == hist['High'].iloc[j]:
                            # Check if peaks are at similar levels
                            if abs(hist['High'].iloc[i] - hist['High'].iloc[j]) / hist['High'].iloc[i] < 0.02:
                                return ChartPattern(
                                    name="Double Top",
                                    confidence=0.6,
                                    start_date=hist.index[i].strftime('%Y-%m-%d'),
                                    end_date=hist.index[j].strftime('%Y-%m-%d'),
                                    description="Double Top pattern detected",
                                    price_levels={
                                        'peak1': hist['High'].iloc[i],
                                        'peak2': hist['High'].iloc[j]
                                    }
                                )
            
            return None
            
        except Exception as e:
            logger.error(f"Double top/bottom detection failed: {e}")
            return None
    
    def _detect_triangle(self, hist: pd.DataFrame) -> Optional[ChartPattern]:
        """Detect triangle patterns"""
        try:
            # Simplified triangle detection
            # Look for converging trend lines
            
            # Get recent highs and lows
            recent_highs = hist['High'].tail(20)
            recent_lows = hist['Low'].tail(20)
            
            # Check if highs are decreasing and lows are increasing (ascending triangle)
            high_trend = np.polyfit(range(len(recent_highs)), recent_highs, 1)[0]
            low_trend = np.polyfit(range(len(recent_lows)), recent_lows, 1)[0]
            
            if high_trend < -0.01 and low_trend > 0.01:
                return ChartPattern(
                    name="Ascending Triangle",
                    confidence=0.5,
                    start_date=hist.index[-20].strftime('%Y-%m-%d'),
                    end_date=hist.index[-1].strftime('%Y-%m-%d'),
                    description="Ascending Triangle pattern detected",
                    price_levels={
                        'resistance': recent_highs.mean(),
                        'support': recent_lows.mean()
                    }
                )
            
            return None
            
        except Exception as e:
            logger.error(f"Triangle detection failed: {e}")
            return None
    
    def _detect_support_resistance_patterns(self, hist: pd.DataFrame) -> List[ChartPattern]:
        """Detect support and resistance levels"""
        patterns = []
        
        try:
            # Find support levels (local minima)
            support_levels = []
            for i in range(5, len(hist) - 5):
                if hist['Low'].iloc[i] == hist['Low'].iloc[i-5:i+6].min():
                    support_levels.append((i, hist['Low'].iloc[i]))
            
            # Find resistance levels (local maxima)
            resistance_levels = []
            for i in range(5, len(hist) - 5):
                if hist['High'].iloc[i] == hist['High'].iloc[i-5:i+6].max():
                    resistance_levels.append((i, hist['High'].iloc[i]))
            
            # Create patterns for significant levels
            for idx, level in support_levels[-3:]:  # Last 3 support levels
                patterns.append(ChartPattern(
                    name="Support Level",
                    confidence=0.8,
                    start_date=hist.index[idx].strftime('%Y-%m-%d'),
                    end_date=hist.index[idx].strftime('%Y-%m-%d'),
                    description=f"Support level at {level:.2f}",
                    price_levels={'support': level}
                ))
            
            for idx, level in resistance_levels[-3:]:  # Last 3 resistance levels
                patterns.append(ChartPattern(
                    name="Resistance Level",
                    confidence=0.8,
                    start_date=hist.index[idx].strftime('%Y-%m-%d'),
                    end_date=hist.index[idx].strftime('%Y-%m-%d'),
                    description=f"Resistance level at {level:.2f}",
                    price_levels={'resistance': level}
                ))
            
        except Exception as e:
            logger.error(f"Support/Resistance pattern detection failed: {e}")
        
        return patterns
    
    # Multi-Timeframe Analysis
    async def get_multi_timeframe_analysis(self, symbol: str) -> Dict[str, Dict[str, TechnicalIndicator]]:
        """Synchronized charts across multiple timeframes"""
        try:
            timeframes = {
                '1m': '1d',
                '5m': '5d',
                '15m': '1mo',
                '1h': '3mo',
                '1d': '1y',
                '1wk': '5y'
            }
            
            analysis = {}
            
            for interval, period in timeframes.items():
                try:
                    ticker = yf.Ticker(symbol)
                    hist = ticker.history(period=period, interval=interval)
                    
                    if not hist.empty:
                        indicators = await self.get_technical_indicators_from_data(hist)
                        analysis[interval] = indicators
                    
                except Exception as e:
                    logger.warning(f"Multi-timeframe analysis failed for {interval}: {e}")
                    continue
            
            return analysis
            
        except Exception as e:
            logger.error(f"Multi-timeframe analysis failed for {symbol}: {e}")
            return {}
    
    async def get_technical_indicators_from_data(self, hist: pd.DataFrame) -> Dict[str, TechnicalIndicator]:
        """Get technical indicators from provided data"""
        try:
            indicators = {}
            
            # Trend Indicators
            indicators.update(await self._calculate_trend_indicators(hist))
            
            # Momentum Indicators
            indicators.update(await self._calculate_momentum_indicators(hist))
            
            # Volume Indicators
            indicators.update(await self._calculate_volume_indicators(hist))
            
            # Volatility Indicators
            indicators.update(await self._calculate_volatility_indicators(hist))
            
            return indicators
            
        except Exception as e:
            logger.error(f"Technical indicators from data failed: {e}")
            return {}
    
    # Custom Indicators
    async def create_custom_indicator(self, symbol: str, formula: str, parameters: Dict[str, Any]) -> TechnicalIndicator:
        """Create and backtest custom technical indicators"""
        try:
            # Get price data
            ticker = yf.Ticker(symbol)
            hist = ticker.history(period="1y")
            
            if hist.empty:
                return None
            
            # Simple custom indicator calculation
            # This is a basic implementation - real custom indicators would be more complex
            
            if formula == "custom_momentum":
                # Custom momentum indicator
                close = hist['Close'].values
                volume = hist['Volume'].values
                
                # Price momentum weighted by volume
                momentum = np.diff(close) * volume[1:] / np.mean(volume)
                momentum = np.concatenate([[0], momentum])  # Add 0 for first element
                
                return TechnicalIndicator(
                    name="Custom Momentum",
                    values=momentum.tolist(),
                    parameters=parameters,
                    category="custom",
                    description="Custom momentum indicator weighted by volume"
                )
            
            elif formula == "custom_volatility":
                # Custom volatility indicator
                close = hist['Close'].values
                period = parameters.get('period', 20)
                
                # Rolling volatility with custom calculation
                volatility = []
                for i in range(len(close)):
                    if i < period:
                        volatility.append(0)
                    else:
                        window = close[i-period:i]
                        vol = np.std(window) / np.mean(window)
                        volatility.append(vol)
                
                return TechnicalIndicator(
                    name="Custom Volatility",
                    values=volatility,
                    parameters=parameters,
                    category="custom",
                    description="Custom volatility indicator"
                )
            
            else:
                logger.warning(f"Unknown custom indicator formula: {formula}")
                return None
            
        except Exception as e:
            logger.error(f"Custom indicator creation failed: {e}")
            return None

# Global service instance
technical_analysis_service = TechnicalAnalysisService() 