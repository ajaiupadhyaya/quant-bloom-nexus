import { HistoricalDataPoint } from '../data/RealTimeDataProvider';

export interface IndicatorResult {
  timestamp: number;
  value: number;
  signal?: 'buy' | 'sell' | 'hold';
  confidence?: number;
}

export interface MACDResult {
  timestamp: number;
  macd: number;
  signal: number;
  histogram: number;
  crossover?: 'bullish' | 'bearish';
}

export interface BollingerBandsResult {
  timestamp: number;
  upperBand: number;
  middleBand: number;
  lowerBand: number;
  bandwidth: number;
  percentB: number;
  squeeze?: boolean;
}

export interface StochasticResult {
  timestamp: number;
  k: number;
  d: number;
  signal?: 'buy' | 'sell' | 'hold';
}

export interface RSIResult {
  timestamp: number;
  rsi: number;
  signal?: 'buy' | 'sell' | 'hold';
  overbought: boolean;
  oversold: boolean;
}

export interface IchimokuResult {
  timestamp: number;
  tenkanSen: number;
  kijunSen: number;
  senkouSpanA: number;
  senkouSpanB: number;
  chikouSpan: number;
  cloudColor: 'bullish' | 'bearish';
  signal?: 'buy' | 'sell' | 'hold';
}

export class TechnicalIndicators {
  
  // ==================== TREND INDICATORS ====================

  /**
   * Simple Moving Average (SMA)
   */
  public static calculateSMA(data: HistoricalDataPoint[], period: number): IndicatorResult[] {
    const results: IndicatorResult[] = [];
    
    for (let i = period - 1; i < data.length; i++) {
      const sum = data.slice(i - period + 1, i + 1)
        .reduce((acc, point) => acc + point.close, 0);
      
      results.push({
        timestamp: data[i].timestamp,
        value: sum / period
      });
    }
    
    return results;
  }

  /**
   * Exponential Moving Average (EMA)
   */
  public static calculateEMA(data: HistoricalDataPoint[], period: number): IndicatorResult[] {
    const results: IndicatorResult[] = [];
    const multiplier = 2 / (period + 1);
    
    let ema = data.slice(0, period).reduce((acc, point) => acc + point.close, 0) / period;
    
    for (let i = period - 1; i < data.length; i++) {
      if (i === period - 1) {
        results.push({
          timestamp: data[i].timestamp,
          value: ema
        });
      } else {
        ema = (data[i].close * multiplier) + (ema * (1 - multiplier));
        results.push({
          timestamp: data[i].timestamp,
          value: ema
        });
      }
    }
    
    return results;
  }

  /**
   * Moving Average Convergence Divergence (MACD)
   */
  public static calculateMACD(
    data: HistoricalDataPoint[], 
    fastPeriod: number = 12, 
    slowPeriod: number = 26, 
    signalPeriod: number = 9
  ): MACDResult[] {
    const fastEMA = this.calculateEMA(data, fastPeriod);
    const slowEMA = this.calculateEMA(data, slowPeriod);
    const results: MACDResult[] = [];
    
    const macdLine: IndicatorResult[] = [];
    for (let i = 0; i < Math.min(fastEMA.length, slowEMA.length); i++) {
      macdLine.push({
        timestamp: fastEMA[i].timestamp,
        value: fastEMA[i].value - slowEMA[i].value
      });
    }
    
    const signalEMA = this.calculateEMA(
      macdLine.map(m => ({ 
        timestamp: m.timestamp, 
        open: m.value, 
        high: m.value, 
        low: m.value, 
        close: m.value, 
        volume: 0 
      })), 
      signalPeriod
    );
    
    for (let i = 0; i < Math.min(macdLine.length, signalEMA.length); i++) {
      const macd = macdLine[i + (macdLine.length - signalEMA.length)];
      const signal = signalEMA[i];
      const histogram = macd.value - signal.value;
      
      let crossover: 'bullish' | 'bearish' | undefined;
      if (i > 0) {
        const prevMacd = macdLine[i - 1 + (macdLine.length - signalEMA.length)];
        const prevSignal = signalEMA[i - 1];
        const prevHistogram = prevMacd.value - prevSignal.value;
        
        if (prevHistogram <= 0 && histogram > 0) {
          crossover = 'bullish';
        } else if (prevHistogram >= 0 && histogram < 0) {
          crossover = 'bearish';
        }
      }
      
      results.push({
        timestamp: macd.timestamp,
        macd: macd.value,
        signal: signal.value,
        histogram: histogram,
        crossover
      });
    }
    
    return results;
  }

  /**
   * Average Directional Index (ADX)
   */
  public static calculateADX(data: HistoricalDataPoint[], period: number = 14): IndicatorResult[] {
    const results: IndicatorResult[] = [];
    const trueRanges: number[] = [];
    const plusDMs: number[] = [];
    const minusDMs: number[] = [];
    
    // Calculate True Range, +DM, -DM
    for (let i = 1; i < data.length; i++) {
      const high = data[i].high;
      const low = data[i].low;
      const prevClose = data[i - 1].close;
      const prevHigh = data[i - 1].high;
      const prevLow = data[i - 1].low;
      
      const tr = Math.max(
        high - low,
        Math.abs(high - prevClose),
        Math.abs(low - prevClose)
      );
      
      const plusDM = (high - prevHigh > prevLow - low) ? Math.max(high - prevHigh, 0) : 0;
      const minusDM = (prevLow - low > high - prevHigh) ? Math.max(prevLow - low, 0) : 0;
      
      trueRanges.push(tr);
      plusDMs.push(plusDM);
      minusDMs.push(minusDM);
    }
    
    // Calculate smoothed versions
    if (trueRanges.length >= period) {
      for (let i = period - 1; i < trueRanges.length; i++) {
        const atr = trueRanges.slice(i - period + 1, i + 1).reduce((a, b) => a + b) / period;
        const plusDI = (plusDMs.slice(i - period + 1, i + 1).reduce((a, b) => a + b) / period) / atr * 100;
        const minusDI = (minusDMs.slice(i - period + 1, i + 1).reduce((a, b) => a + b) / period) / atr * 100;
        
        const dx = Math.abs(plusDI - minusDI) / (plusDI + minusDI) * 100;
        
        results.push({
          timestamp: data[i + 1].timestamp,
          value: dx
        });
      }
    }
    
    return results;
  }

  // ==================== MOMENTUM INDICATORS ====================

  /**
   * Relative Strength Index (RSI)
   */
  public static calculateRSI(data: HistoricalDataPoint[], period: number = 14): RSIResult[] {
    const results: RSIResult[] = [];
    const gains: number[] = [];
    const losses: number[] = [];
    
    for (let i = 1; i < data.length; i++) {
      const change = data[i].close - data[i - 1].close;
      gains.push(change > 0 ? change : 0);
      losses.push(change < 0 ? Math.abs(change) : 0);
    }
    
    for (let i = period - 1; i < gains.length; i++) {
      const avgGain = gains.slice(i - period + 1, i + 1).reduce((a, b) => a + b) / period;
      const avgLoss = losses.slice(i - period + 1, i + 1).reduce((a, b) => a + b) / period;
      
      const rs = avgGain / (avgLoss || 0.0001);
      const rsi = 100 - (100 / (1 + rs));
      
      const overbought = rsi > 70;
      const oversold = rsi < 30;
      
      let signal: 'buy' | 'sell' | 'hold' = 'hold';
      if (oversold) signal = 'buy';
      else if (overbought) signal = 'sell';
      
      results.push({
        timestamp: data[i + 1].timestamp,
        rsi,
        signal,
        overbought,
        oversold
      });
    }
    
    return results;
  }

  /**
   * Stochastic Oscillator
   */
  public static calculateStochastic(
    data: HistoricalDataPoint[], 
    kPeriod: number = 14, 
    dPeriod: number = 3
  ): StochasticResult[] {
    const results: StochasticResult[] = [];
    const kValues: number[] = [];
    
    // Calculate %K
    for (let i = kPeriod - 1; i < data.length; i++) {
      const period = data.slice(i - kPeriod + 1, i + 1);
      const lowest = Math.min(...period.map(p => p.low));
      const highest = Math.max(...period.map(p => p.high));
      const current = data[i].close;
      
      const k = ((current - lowest) / (highest - lowest)) * 100;
      kValues.push(k);
    }
    
    // Calculate %D (SMA of %K)
    for (let i = dPeriod - 1; i < kValues.length; i++) {
      const k = kValues[i];
      const d = kValues.slice(i - dPeriod + 1, i + 1).reduce((a, b) => a + b) / dPeriod;
      
      let signal: 'buy' | 'sell' | 'hold' = 'hold';
      if (k < 20 && d < 20) signal = 'buy';
      else if (k > 80 && d > 80) signal = 'sell';
      
      results.push({
        timestamp: data[kPeriod - 1 + i].timestamp,
        k,
        d,
        signal
      });
    }
    
    return results;
  }

  /**
   * Williams %R
   */
  public static calculateWilliamsR(data: HistoricalDataPoint[], period: number = 14): IndicatorResult[] {
    const results: IndicatorResult[] = [];
    
    for (let i = period - 1; i < data.length; i++) {
      const periodData = data.slice(i - period + 1, i + 1);
      const highest = Math.max(...periodData.map(p => p.high));
      const lowest = Math.min(...periodData.map(p => p.low));
      const close = data[i].close;
      
      const williamsR = ((highest - close) / (highest - lowest)) * -100;
      
      let signal: 'buy' | 'sell' | 'hold' = 'hold';
      if (williamsR < -80) signal = 'buy';
      else if (williamsR > -20) signal = 'sell';
      
      results.push({
        timestamp: data[i].timestamp,
        value: williamsR,
        signal
      });
    }
    
    return results;
  }

  // ==================== VOLATILITY INDICATORS ====================

  /**
   * Bollinger Bands
   */
  public static calculateBollingerBands(
    data: HistoricalDataPoint[], 
    period: number = 20, 
    stdDev: number = 2
  ): BollingerBandsResult[] {
    const results: BollingerBandsResult[] = [];
    const sma = this.calculateSMA(data, period);
    
    for (let i = period - 1; i < data.length; i++) {
      const periodData = data.slice(i - period + 1, i + 1);
      const mean = sma[i - period + 1].value;
      
      const variance = periodData.reduce((acc, point) => acc + Math.pow(point.close - mean, 2), 0) / period;
      const standardDeviation = Math.sqrt(variance);
      
      const upperBand = mean + (stdDev * standardDeviation);
      const lowerBand = mean - (stdDev * standardDeviation);
      const bandwidth = (upperBand - lowerBand) / mean * 100;
      const percentB = (data[i].close - lowerBand) / (upperBand - lowerBand);
      
      const squeeze = bandwidth < 10;
      
      results.push({
        timestamp: data[i].timestamp,
        upperBand,
        middleBand: mean,
        lowerBand,
        bandwidth,
        percentB,
        squeeze
      });
    }
    
    return results;
  }

  /**
   * Average True Range (ATR)
   */
  public static calculateATR(data: HistoricalDataPoint[], period: number = 14): IndicatorResult[] {
    const results: IndicatorResult[] = [];
    const trueRanges: number[] = [];
    
    // Calculate True Range
    for (let i = 1; i < data.length; i++) {
      const high = data[i].high;
      const low = data[i].low;
      const prevClose = data[i - 1].close;
      
      const tr = Math.max(
        high - low,
        Math.abs(high - prevClose),
        Math.abs(low - prevClose)
      );
      
      trueRanges.push(tr);
    }
    
    // Calculate ATR
    for (let i = period - 1; i < trueRanges.length; i++) {
      const atr = trueRanges.slice(i - period + 1, i + 1).reduce((a, b) => a + b) / period;
      
      results.push({
        timestamp: data[i + 1].timestamp,
        value: atr
      });
    }
    
    return results;
  }

  // ==================== VOLUME INDICATORS ====================

  /**
   * On-Balance Volume (OBV)
   */
  public static calculateOBV(data: HistoricalDataPoint[]): IndicatorResult[] {
    const results: IndicatorResult[] = [];
    let obv = 0;
    
    for (let i = 1; i < data.length; i++) {
      const currentClose = data[i].close;
      const prevClose = data[i - 1].close;
      
      if (currentClose > prevClose) {
        obv += data[i].volume;
      } else if (currentClose < prevClose) {
        obv -= data[i].volume;
      }
      
      results.push({
        timestamp: data[i].timestamp,
        value: obv
      });
    }
    
    return results;
  }

  /**
   * Volume Weighted Average Price (VWAP)
   */
  public static calculateVWAP(data: HistoricalDataPoint[]): IndicatorResult[] {
    const results: IndicatorResult[] = [];
    let cumulativeVolume = 0;
    let cumulativeVolumePrice = 0;
    
    for (let i = 0; i < data.length; i++) {
      const typicalPrice = (data[i].high + data[i].low + data[i].close) / 3;
      const volumePrice = typicalPrice * data[i].volume;
      
      cumulativeVolumePrice += volumePrice;
      cumulativeVolume += data[i].volume;
      
      const vwap = cumulativeVolumePrice / cumulativeVolume;
      
      results.push({
        timestamp: data[i].timestamp,
        value: vwap
      });
    }
    
    return results;
  }

  // ==================== ADVANCED INDICATORS ====================

  /**
   * Ichimoku Kinko Hyo
   */
  public static calculateIchimoku(
    data: HistoricalDataPoint[],
    tenkanPeriod: number = 9,
    kijunPeriod: number = 26,
    senkouPeriod: number = 52
  ): IchimokuResult[] {
    const results: IchimokuResult[] = [];
    
    for (let i = Math.max(tenkanPeriod, kijunPeriod, senkouPeriod) - 1; i < data.length; i++) {
      // Tenkan-sen (Conversion Line)
      const tenkanData = data.slice(i - tenkanPeriod + 1, i + 1);
      const tenkanHigh = Math.max(...tenkanData.map(d => d.high));
      const tenkanLow = Math.min(...tenkanData.map(d => d.low));
      const tenkanSen = (tenkanHigh + tenkanLow) / 2;
      
      // Kijun-sen (Base Line)
      const kijunData = data.slice(i - kijunPeriod + 1, i + 1);
      const kijunHigh = Math.max(...kijunData.map(d => d.high));
      const kijunLow = Math.min(...kijunData.map(d => d.low));
      const kijunSen = (kijunHigh + kijunLow) / 2;
      
      // Senkou Span A (Leading Span A)
      const senkouSpanA = (tenkanSen + kijunSen) / 2;
      
      // Senkou Span B (Leading Span B)
      const senkouData = data.slice(i - senkouPeriod + 1, i + 1);
      const senkouHigh = Math.max(...senkouData.map(d => d.high));
      const senkouLow = Math.min(...senkouData.map(d => d.low));
      const senkouSpanB = (senkouHigh + senkouLow) / 2;
      
      // Chikou Span (Lagging Span) - current close plotted 26 periods back
      const chikouSpan = data[i].close;
      
      // Cloud analysis
      const cloudColor: 'bullish' | 'bearish' = senkouSpanA > senkouSpanB ? 'bullish' : 'bearish';
      
      // Generate signals
      let signal: 'buy' | 'sell' | 'hold' = 'hold';
      const price = data[i].close;
      
      if (price > Math.max(senkouSpanA, senkouSpanB) && tenkanSen > kijunSen) {
        signal = 'buy';
      } else if (price < Math.min(senkouSpanA, senkouSpanB) && tenkanSen < kijunSen) {
        signal = 'sell';
      }
      
      results.push({
        timestamp: data[i].timestamp,
        tenkanSen,
        kijunSen,
        senkouSpanA,
        senkouSpanB,
        chikouSpan,
        cloudColor,
        signal
      });
    }
    
    return results;
  }

  /**
   * Parabolic SAR
   */
  public static calculateParabolicSAR(
    data: HistoricalDataPoint[],
    accelerationFactor: number = 0.02,
    maxAcceleration: number = 0.2
  ): IndicatorResult[] {
    const results: IndicatorResult[] = [];
    
    if (data.length < 2) return results;
    
    let sar = data[0].low;
    let af = accelerationFactor;
    let ep = data[0].high; // Extreme Point
    let isUptrend = true;
    
    for (let i = 1; i < data.length; i++) {
      const prevSar = sar;
      
      // Calculate new SAR
      sar = prevSar + af * (ep - prevSar);
      
      if (isUptrend) {
        // Uptrend
        if (data[i].low <= sar) {
          // Trend reversal
          isUptrend = false;
          sar = ep;
          ep = data[i].low;
          af = accelerationFactor;
        } else {
          // Continue uptrend
          if (data[i].high > ep) {
            ep = data[i].high;
            af = Math.min(af + accelerationFactor, maxAcceleration);
          }
          // SAR cannot be above previous two periods' lows
          sar = Math.min(sar, data[i - 1].low);
          if (i > 1) {
            sar = Math.min(sar, data[i - 2].low);
          }
        }
      } else {
        // Downtrend
        if (data[i].high >= sar) {
          // Trend reversal
          isUptrend = true;
          sar = ep;
          ep = data[i].high;
          af = accelerationFactor;
        } else {
          // Continue downtrend
          if (data[i].low < ep) {
            ep = data[i].low;
            af = Math.min(af + accelerationFactor, maxAcceleration);
          }
          // SAR cannot be below previous two periods' highs
          sar = Math.max(sar, data[i - 1].high);
          if (i > 1) {
            sar = Math.max(sar, data[i - 2].high);
          }
        }
      }
      
      const signal: 'buy' | 'sell' | 'hold' = 'hold';
      // Signal generation based on trend changes would require tracking previous trend state
      
      results.push({
        timestamp: data[i].timestamp,
        value: sar,
        signal
      });
    }
    
    return results;
  }

  // ==================== UTILITY METHODS ====================

  /**
   * Calculate multiple indicators at once for efficiency
   */
  public static calculateAllIndicators(data: HistoricalDataPoint[]): {
    sma20: IndicatorResult[];
    sma50: IndicatorResult[];
    ema12: IndicatorResult[];
    ema26: IndicatorResult[];
    macd: MACDResult[];
    rsi: RSIResult[];
    stochastic: StochasticResult[];
    bollinger: BollingerBandsResult[];
    atr: IndicatorResult[];
    obv: IndicatorResult[];
    vwap: IndicatorResult[];
    ichimoku: IchimokuResult[];
  } {
    return {
      sma20: this.calculateSMA(data, 20),
      sma50: this.calculateSMA(data, 50),
      ema12: this.calculateEMA(data, 12),
      ema26: this.calculateEMA(data, 26),
      macd: this.calculateMACD(data),
      rsi: this.calculateRSI(data),
      stochastic: this.calculateStochastic(data),
      bollinger: this.calculateBollingerBands(data),
      atr: this.calculateATR(data),
      obv: this.calculateOBV(data),
      vwap: this.calculateVWAP(data),
      ichimoku: this.calculateIchimoku(data)
    };
  }

  /**
   * Generate trading signals based on multiple indicators
   */
  public static generateCompositeSignal(
    data: HistoricalDataPoint[],
    indicators: ReturnType<typeof TechnicalIndicators.calculateAllIndicators>
  ): { signal: 'buy' | 'sell' | 'hold'; confidence: number; reasons: string[] }[] {
    const results: { signal: 'buy' | 'sell' | 'hold'; confidence: number; reasons: string[] }[] = [];
    
    // Get the minimum length to avoid index errors
    const minLength = Math.min(
      indicators.rsi.length,
      indicators.macd.length,
      indicators.stochastic.length,
      indicators.bollinger.length
    );
    
    for (let i = 0; i < minLength; i++) {
      const reasons: string[] = [];
      let buySignals = 0;
      let sellSignals = 0;
      
      // RSI signals
      if (indicators.rsi[i].signal === 'buy') {
        buySignals++;
        reasons.push('RSI oversold');
      } else if (indicators.rsi[i].signal === 'sell') {
        sellSignals++;
        reasons.push('RSI overbought');
      }
      
      // MACD signals
      if (indicators.macd[i].crossover === 'bullish') {
        buySignals++;
        reasons.push('MACD bullish crossover');
      } else if (indicators.macd[i].crossover === 'bearish') {
        sellSignals++;
        reasons.push('MACD bearish crossover');
      }
      
      // Stochastic signals
      if (indicators.stochastic[i].signal === 'buy') {
        buySignals++;
        reasons.push('Stochastic oversold');
      } else if (indicators.stochastic[i].signal === 'sell') {
        sellSignals++;
        reasons.push('Stochastic overbought');
      }
      
      // Bollinger Bands signals
      if (indicators.bollinger[i].percentB < 0) {
        buySignals++;
        reasons.push('Price below Bollinger lower band');
      } else if (indicators.bollinger[i].percentB > 1) {
        sellSignals++;
        reasons.push('Price above Bollinger upper band');
      }
      
      // Determine overall signal
      let signal: 'buy' | 'sell' | 'hold' = 'hold';
      let confidence = 0;
      
      if (buySignals > sellSignals) {
        signal = 'buy';
        confidence = (buySignals / (buySignals + sellSignals)) * 100;
      } else if (sellSignals > buySignals) {
        signal = 'sell';
        confidence = (sellSignals / (buySignals + sellSignals)) * 100;
      } else {
        confidence = 50;
      }
      
      results.push({
        signal,
        confidence,
        reasons
      });
    }
    
    return results;
  }
} 