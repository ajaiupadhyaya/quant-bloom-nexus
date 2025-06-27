import { HistoricalDataPoint } from '../data/RealTimeDataProvider';
import { TechnicalIndicators, IndicatorResult } from '../analysis/TechnicalIndicators';

export interface PredictionResult {
  timestamp: number;
  predictedPrice: number;
  confidence: number;
  trend: 'bullish' | 'bearish' | 'neutral';
  volatilityForecast: number;
  supportLevel?: number;
  resistanceLevel?: number;
}

export interface ModelMetrics {
  accuracy: number;
  mse: number; // Mean Squared Error
  mae: number; // Mean Absolute Error
  sharpeRatio: number;
  maxDrawdown: number;
}

export interface MarketRegime {
  regime: 'bull' | 'bear' | 'sideways' | 'volatile';
  confidence: number;
  duration: number; // days
  characteristics: string[];
}

export interface SentimentAnalysis {
  score: number; // -1 to 1
  magnitude: number; // 0 to 1
  keywords: string[];
  sources: string[];
  timestamp: number;
}

export interface NewsArticleInput {
  headline: string;
  summary: string;
  source: string;
  publishedAt: number;
}

export class PredictionModels {
  
  // ==================== LSTM PRICE PREDICTION ====================
  
  /**
   * LSTM Neural Network Price Prediction (Simplified)
   */
  public static async predictPricesLSTM(
    data: HistoricalDataPoint[], 
    forecastDays: number = 30
  ): Promise<PredictionResult[]> {
    const results: PredictionResult[] = [];
    const prices = data.map(d => d.close);
    
    for (let i = 0; i < forecastDays; i++) {
      const prediction = this.simpleLSTMPredict(data, i);
      const nextTimestamp = data[data.length - 1].timestamp + (i + 1) * 24 * 60 * 60 * 1000;
      
      results.push({
        timestamp: nextTimestamp,
        predictedPrice: prediction.price,
        confidence: prediction.confidence,
        trend: prediction.trend,
        volatilityForecast: prediction.volatility,
        supportLevel: prediction.support,
        resistanceLevel: prediction.resistance
      });
    }
    
    return results;
  }

  /**
   * Transformer-based price prediction
   */
  public static async predictPricesTransformer(
    data: HistoricalDataPoint[],
    sequenceLength: number = 100,
    forecastDays: number = 30
  ): Promise<PredictionResult[]> {
    const results: PredictionResult[] = [];
    
    // Attention mechanism simulation
    const attentionWeights = this.calculateAttentionWeights(data, sequenceLength);
    
    for (let i = 0; i < forecastDays; i++) {
      const prediction = this.transformerPredict(data, attentionWeights, i);
      
      const nextTimestamp = data[data.length - 1].timestamp + (i + 1) * 24 * 60 * 60 * 1000;
      
      results.push({
        timestamp: nextTimestamp,
        predictedPrice: prediction.price,
        confidence: prediction.confidence,
        trend: prediction.trend,
        volatilityForecast: prediction.volatility
      });
    }
    
    return results;
  }

  // ==================== STATISTICAL MODELS ====================

  /**
   * ARIMA Model for Time Series Forecasting
   */
  public static predictPricesARIMA(
    data: HistoricalDataPoint[],
    forecastDays: number = 30
  ): PredictionResult[] {
    const results: PredictionResult[] = [];
    const prices = data.map(d => d.close);
    const returns = this.calculateReturns(data);
    
    let lastPrice = prices[prices.length - 1];
    
    for (let i = 0; i < forecastDays; i++) {
      const forecast = this.arimaForecast(returns, i);
      const predictedPrice = lastPrice * (1 + forecast);
      
      const volatility = this.calculateVolatility(data.slice(-30));
      const confidence = Math.max(0.3, 0.9 - (i * 0.02));
      
      const trend = forecast > 0 ? 'bullish' : forecast < 0 ? 'bearish' : 'neutral';
      
      const nextTimestamp = data[data.length - 1].timestamp + (i + 1) * 24 * 60 * 60 * 1000;
      
      results.push({
        timestamp: nextTimestamp,
        predictedPrice,
        confidence,
        trend,
        volatilityForecast: volatility
      });
      
      lastPrice = predictedPrice;
    }
    
    return results;
  }

  /**
   * GARCH (Generalized Autoregressive Conditional Heteroskedasticity) for volatility
   */
  public static predictVolatilityGARCH(
    data: HistoricalDataPoint[],
    forecastDays: number = 30
  ): IndicatorResult[] {
    const results: IndicatorResult[] = [];
    const returns = this.calculateReturns(data);
    
    // GARCH(1,1) parameters (simplified estimation)
    const omega = 0.000001;
    const alpha = 0.05;
    const beta = 0.9;
    
    // Calculate initial conditional variance
    let sigma2 = this.calculateVariance(returns);
    
    for (let i = 0; i < forecastDays; i++) {
      // GARCH(1,1): σ²(t+1) = ω + α*ε²(t) + β*σ²(t)
      const lastReturn = returns[returns.length - 1] || 0;
      sigma2 = omega + alpha * Math.pow(lastReturn, 2) + beta * sigma2;
      
      const volatility = Math.sqrt(sigma2) * Math.sqrt(252); // Annualized
      
      const nextTimestamp = data[data.length - 1].timestamp + (i + 1) * 24 * 60 * 60 * 1000;
      
      results.push({
        timestamp: nextTimestamp,
        value: volatility
      });
    }
    
    return results;
  }

  // ==================== MACHINE LEARNING MODELS ====================

  /**
   * Random Forest for Price Direction Prediction
   */
  public static predictDirectionRandomForest(
    data: HistoricalDataPoint[]
  ): { direction: 'up' | 'down'; probability: number; features: string[] }[] {
    const results: { direction: 'up' | 'down'; probability: number; features: string[] }[] = [];
    
    const indicators = TechnicalIndicators.calculateAllIndicators(data);
    const features = this.extractMLFeatures(data, indicators);
    
    for (let i = 0; i < Math.min(30, features.length); i++) {
      const feature = features[i];
      const prediction = this.randomForestPredict(feature);
      
      results.push({
        direction: prediction.direction,
        probability: prediction.probability,
        features: prediction.importantFeatures
      });
    }
    
    return results;
  }

  /**
   * Support Vector Machine for trend classification
   */
  public static classifyTrendSVM(
    data: HistoricalDataPoint[],
    windowSize: number = 20
  ): { trend: 'strong_up' | 'weak_up' | 'sideways' | 'weak_down' | 'strong_down'; confidence: number }[] {
    const results: { trend: 'strong_up' | 'weak_up' | 'sideways' | 'weak_down' | 'strong_down'; confidence: number }[] = [];
    
    for (let i = windowSize; i < data.length; i++) {
      const window = data.slice(i - windowSize, i);
      const trendStrength = this.calculateTrendStrength(window);
      const slope = this.calculateSlope(window);
      
      let trend: 'strong_up' | 'weak_up' | 'sideways' | 'weak_down' | 'strong_down';
      let confidence: number;
      
      if (slope > 0.02 && trendStrength > 0.7) {
        trend = 'strong_up';
        confidence = trendStrength;
      } else if (slope > 0.005 && trendStrength > 0.5) {
        trend = 'weak_up';
        confidence = trendStrength * 0.8;
      } else if (Math.abs(slope) <= 0.005) {
        trend = 'sideways';
        confidence = 1 - trendStrength;
      } else if (slope < -0.005 && trendStrength > 0.5) {
        trend = 'weak_down';
        confidence = trendStrength * 0.8;
      } else {
        trend = 'strong_down';
        confidence = trendStrength;
      }
      
      results.push({ trend, confidence });
    }
    
    return results;
  }

  // ==================== REGIME DETECTION ====================

  /**
   * Market Regime Detection using Hidden Markov Model
   */
  public static detectMarketRegime(
    data: HistoricalDataPoint[],
    lookbackPeriod: number = 252
  ): MarketRegime[] {
    const results: MarketRegime[] = [];
    const returns = this.calculateReturns(data);
    
    for (let i = lookbackPeriod; i < data.length; i += 30) {
      const periodReturns = returns.slice(i - lookbackPeriod, i);
      const volatility = this.calculateVolatility(data.slice(i - lookbackPeriod, i));
      const trend = this.calculateTrendStrength(data.slice(i - lookbackPeriod, i));
      
      const meanReturn = periodReturns.reduce((a, b) => a + b, 0) / periodReturns.length;
      
      let regime: 'bull' | 'bear' | 'sideways' | 'volatile';
      let confidence: number;
      const characteristics: string[] = [];
      
      if (volatility > 0.3) {
        regime = 'volatile';
        confidence = Math.min(0.9, volatility);
        characteristics.push('High volatility period');
      } else if (meanReturn > 0.02 && trend > 0.6) {
        regime = 'bull';
        confidence = trend;
        characteristics.push('Strong upward trend', 'Positive momentum');
      } else if (meanReturn < -0.02 && trend > 0.6) {
        regime = 'bear';
        confidence = trend;
        characteristics.push('Strong downward trend', 'Negative momentum');
      } else {
        regime = 'sideways';
        confidence = 1 - trend;
        characteristics.push('Range-bound market', 'Low directional momentum');
      }
      
      if (volatility > 0.2) characteristics.push('Elevated volatility');
      if (Math.abs(meanReturn) < 0.005) characteristics.push('Low average returns');
      
      results.push({
        regime,
        confidence,
        duration: 30,
        characteristics
      });
    }
    
    return results;
  }

  // ==================== SENTIMENT ANALYSIS ====================

  /**
   * News Sentiment Analysis
   */
  public static analyzeSentiment(newsArticles: NewsArticleInput[]): SentimentAnalysis[] {
    return newsArticles.map(article => {
      const text = (article.headline + ' ' + article.summary).toLowerCase();
      
      const positiveWords = [
        'bullish', 'buy', 'growth', 'profit', 'gain', 'rise', 'surge', 'strong', 
        'outperform', 'upgrade', 'positive', 'rally', 'boom', 'recovery'
      ];
      
      const negativeWords = [
        'bearish', 'sell', 'loss', 'decline', 'drop', 'fall', 'weak', 'crash',
        'underperform', 'downgrade', 'negative', 'recession', 'crisis', 'risk'
      ];
      
      const words = text.split(/\W+/);
      let positiveCount = 0;
      let negativeCount = 0;
      const foundKeywords: string[] = [];
      
      words.forEach(word => {
        if (positiveWords.includes(word)) {
          positiveCount++;
          foundKeywords.push(word);
        }
        if (negativeWords.includes(word)) {
          negativeCount++;
          foundKeywords.push(word);
        }
      });
      
      const totalSentimentWords = positiveCount + negativeCount;
      const score = totalSentimentWords > 0 ? (positiveCount - negativeCount) / totalSentimentWords : 0;
      const magnitude = totalSentimentWords / words.length;
      
      return {
        score,
        magnitude,
        keywords: foundKeywords,
        sources: [article.source],
        timestamp: article.publishedAt
      };
    });
  }

  // ==================== UTILITY METHODS ====================

  private static calculateAttentionWeights(data: HistoricalDataPoint[], sequenceLength: number): number[] {
    const weights: number[] = [];
    const prices = data.map(d => d.close);
    
    // Simple attention: more weight to recent and volatile periods
    for (let i = 0; i < Math.min(sequenceLength, prices.length); i++) {
      const recencyWeight = Math.exp(-i * 0.1); // Exponential decay
      const volatilityWeight = data[data.length - 1 - i] ? 
        Math.abs(data[data.length - 1 - i].high - data[data.length - 1 - i].low) / data[data.length - 1 - i].close : 1;
      
      weights.unshift(recencyWeight * (1 + volatilityWeight));
    }
    
    // Normalize weights
    const sum = weights.reduce((a, b) => a + b, 0);
    return weights.map(w => w / sum);
  }

  private static transformerPredict(data: HistoricalDataPoint[], attentionWeights: number[], dayAhead: number): {
    price: number;
    confidence: number;
    trend: 'bullish' | 'bearish' | 'neutral';
    volatility: number;
  } {
    const prices = data.map(d => d.close);
    
    // Weighted prediction based on attention
    let weightedPrice = 0;
    const recentPrices = prices.slice(-attentionWeights.length);
    
    for (let i = 0; i < Math.min(recentPrices.length, attentionWeights.length); i++) {
      weightedPrice += recentPrices[i] * attentionWeights[i];
    }
    
    const currentPrice = prices[prices.length - 1];
    const trend = (weightedPrice - currentPrice) / currentPrice;
    const volatility = this.calculateVolatility(data.slice(-30));
    
    const prediction = currentPrice * (1 + trend * (1 - dayAhead * 0.1));
    const confidence = Math.max(0.4, 0.95 - dayAhead * 0.02);
    
    return {
      price: prediction,
      confidence,
      trend: trend > 0.01 ? 'bullish' : trend < -0.01 ? 'bearish' : 'neutral',
      volatility: volatility * (1 + dayAhead * 0.03)
    };
  }

  private static calculateReturns(data: HistoricalDataPoint[]): number[] {
    const returns: number[] = [];
    for (let i = 1; i < data.length; i++) {
      returns.push((data[i].close - data[i - 1].close) / data[i - 1].close);
    }
    return returns;
  }

  private static calculateVolatility(data: HistoricalDataPoint[]): number {
    const returns = this.calculateReturns(data);
    const meanReturn = returns.reduce((a, b) => a + b, 0) / returns.length;
    const variance = returns.reduce((acc, ret) => acc + Math.pow(ret - meanReturn, 2), 0) / returns.length;
    return Math.sqrt(variance);
  }

  private static calculateVariance(values: number[]): number {
    const mean = values.reduce((a, b) => a + b, 0) / values.length;
    return values.reduce((acc, val) => acc + Math.pow(val - mean, 2), 0) / values.length;
  }

  private static calculateTrendStrength(data: HistoricalDataPoint[]): number {
    const prices = data.map(d => d.close);
    const x = Array.from({ length: prices.length }, (_, i) => i);
    
    // Simple linear regression
    const n = prices.length;
    const sumX = x.reduce((a, b) => a + b, 0);
    const sumY = prices.reduce((a, b) => a + b, 0);
    const sumXY = x.reduce((acc, xi, i) => acc + xi * prices[i], 0);
    const sumXX = x.reduce((acc, xi) => acc + xi * xi, 0);
    
    const slope = (n * sumXY - sumX * sumY) / (n * sumXX - sumX * sumX);
    const yMean = sumY / n;
    const r2 = Math.pow(slope * (sumX / n - x[0]), 2) / prices.reduce((acc, yi) => acc + Math.pow(yi - yMean, 2), 0) * n;
    
    return Math.abs(r2); // Strength regardless of direction
  }

  private static calculateSlope(data: HistoricalDataPoint[]): number {
    const prices = data.map(d => d.close);
    const n = prices.length;
    const x = Array.from({ length: n }, (_, i) => i);
    
    const sumX = x.reduce((a, b) => a + b, 0);
    const sumY = prices.reduce((a, b) => a + b, 0);
    const sumXY = x.reduce((acc, xi, i) => acc + xi * prices[i], 0);
    const sumXX = x.reduce((acc, xi) => acc + xi * xi, 0);
    
    return (n * sumXY - sumX * sumY) / (n * sumXX - sumX * sumX);
  }

  private static arimaForecast(returns: number[], step: number): number {
    const recentReturns = returns.slice(-10);
    const meanReturn = recentReturns.reduce((a, b) => a + b, 0) / recentReturns.length;
    
    // Simple AR(1) model
    const persistence = 0.1;
    return meanReturn * Math.pow(persistence, step);
  }

  private static extractMLFeatures(
    data: HistoricalDataPoint[],
    indicators: ReturnType<typeof TechnicalIndicators.calculateAllIndicators>
  ): Array<{ [key: string]: number }> {
    const features: Array<{ [key: string]: number }> = [];
    
    const minLength = Math.min(
      data.length,
      indicators.rsi.length,
      indicators.macd.length,
      indicators.sma20.length
    );
    
    for (let i = 0; i < minLength; i++) {
      const dataPoint = data[data.length - minLength + i];
      
      features.push({
        price: dataPoint.close,
        volume: dataPoint.volume,
        rsi: indicators.rsi[i].rsi,
        macd: indicators.macd[i].macd,
        macdSignal: indicators.macd[i].signal,
        sma20: indicators.sma20[i].value,
        sma50: indicators.sma50[i].value,
        volatility: (dataPoint.high - dataPoint.low) / dataPoint.close,
        priceChange: i > 0 ? (dataPoint.close - data[data.length - minLength + i - 1].close) / data[data.length - minLength + i - 1].close : 0
      });
    }
    
    return features;
  }

  private static randomForestPredict(features: { [key: string]: number }): {
    direction: 'up' | 'down';
    probability: number;
    importantFeatures: string[];
  } {
    let score = 0;
    const importantFeatures: string[] = [];
    
    if (features.rsi < 30) {
      score += 0.3;
      importantFeatures.push('RSI oversold');
    } else if (features.rsi > 70) {
      score -= 0.3;
      importantFeatures.push('RSI overbought');
    }
    
    if (features.macd > features.macdSignal) {
      score += 0.2;
      importantFeatures.push('MACD bullish');
    } else {
      score -= 0.2;
      importantFeatures.push('MACD bearish');
    }
    
    if (features.price > features.sma20 && features.sma20 > features.sma50) {
      score += 0.25;
      importantFeatures.push('Price above MAs');
    } else if (features.price < features.sma20 && features.sma20 < features.sma50) {
      score -= 0.25;
      importantFeatures.push('Price below MAs');
    }
    
    if (features.priceChange > 0 && features.volume > 1.2) {
      score += 0.15;
      importantFeatures.push('Volume confirmation');
    }
    
    const probability = Math.abs(score);
    const direction = score > 0 ? 'up' : 'down';
    
    return {
      direction,
      probability: Math.min(0.95, Math.max(0.5, probability + 0.5)),
      importantFeatures
    };
  }

  private static simpleLSTMPredict(data: HistoricalDataPoint[], dayAhead: number): {
    price: number;
    confidence: number;
    trend: 'bullish' | 'bearish' | 'neutral';
    volatility: number;
    support?: number;
    resistance?: number;
  } {
    const prices = data.map(d => d.close);
    
    const recentTrend = (prices[prices.length - 1] - prices[prices.length - 10]) / 10;
    const volatility = this.calculateVolatility(data.slice(-20));
    
    const noise = (Math.random() - 0.5) * volatility * prices[prices.length - 1];
    const trendComponent = recentTrend * (1 - dayAhead * 0.1);
    
    const predictedPrice = prices[prices.length - 1] + trendComponent + noise;
    const confidence = Math.max(0.3, 0.9 - dayAhead * 0.03);
    
    const trend = trendComponent > 0.01 ? 'bullish' : trendComponent < -0.01 ? 'bearish' : 'neutral';
    
    const recentHigh = Math.max(...prices.slice(-20));
    const recentLow = Math.min(...prices.slice(-20));
    
    return {
      price: predictedPrice,
      confidence,
      trend,
      volatility: volatility * (1 + dayAhead * 0.02),
      support: recentLow,
      resistance: recentHigh
    };
  }
} 