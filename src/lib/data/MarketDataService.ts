// @ts-ignore - These packages will be available after npm install
import { EventEmitter } from 'eventemitter3';
// @ts-ignore
import io, { Socket } from 'socket.io-client';
// @ts-ignore
import { v4 as uuidv4 } from 'uuid';
// @ts-ignore
import moment from 'moment-timezone';
// @ts-ignore
import Big from 'big.js';
import axios, { AxiosInstance, AxiosResponse } from 'axios';

export interface MarketDataPoint {
  symbol: string;
  timestamp: number;
  bid: number;
  ask: number;
  last: number;
  volume: number;
  open: number;
  high: number;
  low: number;
  change: number;
  changePercent: number;
  vwap?: number;
  marketCap?: number;
  pe?: number;
  beta?: number;
}

export interface Level2Quote {
  symbol: string;
  timestamp: number;
  bids: Array<[number, number]>; // [price, size]
  asks: Array<[number, number]>; // [price, size]
  sequence: number;
}

export interface TradeData {
  symbol: string;
  timestamp: number;
  price: number;
  size: number;
  side: 'buy' | 'sell';
  tradeId: string;
  exchange: string;
}

export interface NewsItem {
  id: string;
  headline: string;
  summary: string;
  timestamp: number;
  source: string;
  symbols: string[];
  sentiment: number; // -1 to 1
  relevance: number; // 0 to 1
  category: string;
  url?: string;
}

export interface EconomicIndicator {
  indicator: string;
  country: string;
  timestamp: number;
  actual?: number;
  forecast?: number;
  previous?: number;
  importance: 'low' | 'medium' | 'high';
  impact: 'negative' | 'neutral' | 'positive';
}

export interface Quote {
  symbol: string;
  price: number;
  change: number;
  change_percent: number;
  volume: number;
  timestamp: number;
  market_cap?: number;
  pe_ratio?: number;
}

export interface HistoricalDataPoint {
  timestamp: number;
  open: number;
  high: number;
  low: number;
  close: number;
  volume: number;
}

export interface NewsArticle {
  title: string;
  description: string;
  url: string;
  source: string;
  published_at: string;
  symbol?: string;
  sentiment_score: number;
  sentiment_label: string;
  sentiment_confidence: number;
}

export interface MarketMovers {
  gainers: Quote[];
  losers: Quote[];
  most_active: Quote[];
}

export interface SectorPerformance {
  sector: string;
  symbol: string;
  price: number;
  change: number;
  change_percent: number;
}

export interface SentimentSummary {
  overall_sentiment: number;
  sentiment_distribution: {
    positive: number;
    negative: number;
    neutral: number;
  };
  confidence: number;
  total_articles: number;
  symbol?: string;
  timestamp: string;
}

export interface AIAnalysis {
  lstm_predictions: {
    predictions: Array<{
      date: string;
      predicted_price: number;
      confidence: number;
    }>;
    model: string;
    confidence: number;
  };
  ensemble_predictions: {
    predictions: Array<{
      date: string;
      predicted_price: number;
      confidence: number;
    }>;
    model: string;
    confidence: number;
  };
  trading_signal: {
    action: 'buy' | 'sell' | 'hold';
    confidence: number;
    timestamp: string;
  };
  market_regime: {
    regime: string;
    confidence: number;
    volatility_ratio: number;
    trend_strength: number;
    momentum: number;
  };
  sentiment_analysis: {
    score: number;
    interpretation: 'bullish' | 'bearish' | 'neutral';
  };
}

export class MarketDataService extends (EventEmitter as any) {
  private socket: Socket | null = null;
  private subscriptions: Set<string> = new Set();
  private dataCache: Map<string, MarketDataPoint> = new Map();
  private level2Cache: Map<string, Level2Quote> = new Map();
  private newsCache: NewsItem[] = [];
  private reconnectAttempts = 0;
  private maxReconnectAttempts = 10;
  private reconnectDelay = 1000;
  private isConnected = false;
  
  // Performance monitoring
  private latencyStats = {
    min: Infinity,
    max: 0,
    avg: 0,
    count: 0,
    sum: 0
  };

  // Data providers configuration
  private providers = {
    primary: 'bloomberg', // bloomberg, refinitiv, iex, alpha_vantage
    fallback: ['yahoo', 'iex', 'alpha_vantage'],
    endpoints: {
      bloomberg: process.env.BLOOMBERG_API_URL || 'wss://api.bloomberg.com/v1',
      refinitiv: process.env.REFINITIV_API_URL || 'wss://api.refinitiv.com/v1',
      iex: process.env.IEX_API_URL || 'wss://cloud-sse.iexapis.com/stable',
      yahoo: 'wss://streamer.finance.yahoo.com',
      alpha_vantage: process.env.ALPHA_VANTAGE_URL || 'wss://ws.alphavantage.co'
    }
  };

  private api: AxiosInstance;
  private cache: Map<string, { data: any; timestamp: number; ttl: number }>;
  private readonly CACHE_TTL = 30000; // 30 seconds default
  private readonly BASE_URL = process.env.NODE_ENV === 'production' 
    ? 'https://your-domain.com/api/v1' 
    : 'http://localhost:8000/api/v1';

  constructor() {
    super();
    this.initializeConnection();
    this.setupPerformanceMonitoring();

    this.api = axios.create({
      baseURL: this.BASE_URL,
      timeout: 10000,
      headers: {
        'Content-Type': 'application/json',
      },
    });

    this.cache = new Map();

    // Request interceptor for logging
    this.api.interceptors.request.use((config) => {
      console.log(`🚀 API Request: ${config.method?.toUpperCase()} ${config.url}`);
      return config;
    });

    // Response interceptor for error handling
    this.api.interceptors.response.use(
      (response) => response,
      (error) => {
        console.error('❌ API Error:', error.response?.data || error.message);
        return Promise.reject(error);
      }
    );
  }

  private async initializeConnection(): Promise<void> {
    try {
      const endpoint = this.providers.endpoints[this.providers.primary];
      
      this.socket = io(endpoint, {
        transports: ['websocket'],
        timeout: 5000,
        reconnection: true,
        reconnectionAttempts: this.maxReconnectAttempts,
        reconnectionDelay: this.reconnectDelay,
        auth: {
          token: process.env.MARKET_DATA_API_KEY,
          timestamp: Date.now()
        }
      });

      this.socket.on('connect', () => {
        console.log('Connected to market data feed');
        this.isConnected = true;
        this.reconnectAttempts = 0;
        this.emit('connected');
      });

      this.socket.on('disconnect', () => {
        console.log('Disconnected from market data feed');
        this.isConnected = false;
        this.emit('disconnected');
      });

      this.socket.on('quote', (data: MarketDataPoint) => {
        this.handleQuoteData(data);
      });

      this.socket.on('level2', (data: Level2Quote) => {
        this.handleLevel2Data(data);
      });

      this.socket.on('trade', (data: TradeData) => {
        this.handleTradeData(data);
      });

      this.socket.on('news', (data: NewsItem) => {
        this.handleNewsData(data);
      });

      this.socket.on('economic', (data: EconomicIndicator) => {
        this.handleEconomicData(data);
      });

      this.socket.on('error', (error: any) => {
        console.error('Market data feed error:', error);
        this.emit('error', error);
      });

    } catch (error) {
      console.error('Failed to initialize market data connection:', error);
      setTimeout(() => this.initializeConnection(), this.reconnectDelay);
    }
  }

  private handleQuoteData(data: MarketDataPoint): void {
    const startTime = performance.now();
    
    // Data validation and enrichment
    const enrichedData = this.enrichMarketData(data);
    this.dataCache.set(data.symbol, enrichedData);
    
    // Calculate latency
    const latency = performance.now() - startTime;
    this.updateLatencyStats(latency);
    
    this.emit('quote', enrichedData);
    this.emit(`quote:${data.symbol}`, enrichedData);
  }

  private handleLevel2Data(data: Level2Quote): void {
    this.level2Cache.set(data.symbol, data);
    this.emit('level2', data);
    this.emit(`level2:${data.symbol}`, data);
  }

  private handleTradeData(data: TradeData): void {
    this.emit('trade', data);
    this.emit(`trade:${data.symbol}`, data);
  }

  private handleNewsData(data: NewsItem): void {
    this.newsCache.unshift(data);
    if (this.newsCache.length > 1000) {
      this.newsCache = this.newsCache.slice(0, 1000);
    }
    this.emit('news', data);
  }

  private handleEconomicData(data: EconomicIndicator): void {
    this.emit('economic', data);
  }

  private enrichMarketData(data: MarketDataPoint): MarketDataPoint {
    const cached = this.dataCache.get(data.symbol);
    
    return {
      ...data,
      timestamp: data.timestamp || Date.now(),
      change: data.last - (cached?.last || data.open),
      changePercent: cached?.last ? 
        ((data.last - cached.last) / cached.last) * 100 : 0,
      vwap: this.calculateVWAP(data.symbol, data),
    };
  }

  private calculateVWAP(symbol: string, data: MarketDataPoint): number {
    // Simplified VWAP calculation - in production, this would use tick data
    return (data.high + data.low + data.last) / 3;
  }

  private updateLatencyStats(latency: number): void {
    this.latencyStats.count++;
    this.latencyStats.sum += latency;
    this.latencyStats.min = Math.min(this.latencyStats.min, latency);
    this.latencyStats.max = Math.max(this.latencyStats.max, latency);
    this.latencyStats.avg = this.latencyStats.sum / this.latencyStats.count;
  }

  private setupPerformanceMonitoring(): void {
    setInterval(() => {
      this.emit('performance', {
        ...this.latencyStats,
        cacheSize: this.dataCache.size,
        subscriptions: this.subscriptions.size,
        connected: this.isConnected,
        timestamp: Date.now()
      });
    }, 5000);
  }

  // Public API Methods
  public subscribe(symbols: string | string[]): void {
    const symbolsArray = Array.isArray(symbols) ? symbols : [symbols];
    
    symbolsArray.forEach(symbol => {
      if (!this.subscriptions.has(symbol)) {
        this.subscriptions.add(symbol);
        this.socket?.emit('subscribe', { symbol, type: 'quote' });
        this.socket?.emit('subscribe', { symbol, type: 'level2' });
        this.socket?.emit('subscribe', { symbol, type: 'trade' });
      }
    });
  }

  public unsubscribe(symbols: string | string[]): void {
    const symbolsArray = Array.isArray(symbols) ? symbols : [symbols];
    
    symbolsArray.forEach(symbol => {
      if (this.subscriptions.has(symbol)) {
        this.subscriptions.delete(symbol);
        this.socket?.emit('unsubscribe', { symbol });
        this.dataCache.delete(symbol);
        this.level2Cache.delete(symbol);
      }
    });
  }

  public getQuote(symbol: string): MarketDataPoint | null {
    return this.dataCache.get(symbol) || null;
  }

  public getLevel2(symbol: string): Level2Quote | null {
    return this.level2Cache.get(symbol) || null;
  }

  public getNews(limit = 50): NewsItem[] {
    return this.newsCache.slice(0, limit);
  }

  public getNewsForSymbol(symbol: string, limit = 10): NewsItem[] {
    return this.newsCache
      .filter(news => news.symbols.includes(symbol))
      .slice(0, limit);
  }

  public async getHistoricalMarketData(
    symbol: string, 
    period: string = '1D',
    interval: string = '1m'
  ): Promise<MarketDataPoint[]> {
    try {
      // Implementation would connect to historical data API
      const response = await fetch(
        `/api/historical/${symbol}?period=${period}&interval=${interval}`
      );
      return await response.json();
    } catch (error) {
      console.error('Failed to fetch historical data:', error);
      return [];
    }
  }

  public getPerformanceStats() {
    return {
      ...this.latencyStats,
      cacheSize: this.dataCache.size,
      subscriptions: Array.from(this.subscriptions),
      connected: this.isConnected
    };
  }

  public disconnect(): void {
    if (this.socket) {
      this.socket.disconnect();
      this.socket = null;
    }
    this.subscriptions.clear();
    this.dataCache.clear();
    this.level2Cache.clear();
  }

  // Advanced market data methods
  public async getMarketDepth(symbol: string, depth = 10): Promise<Level2Quote | null> {
    const level2 = this.getLevel2(symbol);
    if (!level2) return null;

    return {
      ...level2,
      bids: level2.bids.slice(0, depth),
      asks: level2.asks.slice(0, depth)
    };
  }

  public calculateSpread(symbol: string): number | null {
    const quote = this.getQuote(symbol);
    if (!quote) return null;
    
    return Big(quote.ask).minus(quote.bid).toNumber();
  }

  public calculateSpreadBasisPoints(symbol: string): number | null {
    const quote = this.getQuote(symbol);
    if (!quote) return null;
    
    const spread = this.calculateSpread(symbol);
    if (spread === null) return null;
    
    const midPrice = (quote.bid + quote.ask) / 2;
    return (spread / midPrice) * 10000; // Convert to basis points
  }

  public getMarketStatus(): {
    status: 'open' | 'closed' | 'pre-market' | 'after-hours';
    nextOpen?: number;
    nextClose?: number;
  } {
    const now = moment().tz('America/New_York');
    const marketOpen = moment().tz('America/New_York').hour(9).minute(30).second(0);
    const marketClose = moment().tz('America/New_York').hour(16).minute(0).second(0);
    
    if (now.isBefore(marketOpen)) {
      return {
        status: 'pre-market',
        nextOpen: marketOpen.valueOf()
      };
    } else if (now.isAfter(marketClose)) {
      return {
        status: 'after-hours',
        nextOpen: marketOpen.add(1, 'day').valueOf()
      };
    } else if (now.day() === 0 || now.day() === 6) {
      return {
        status: 'closed',
        nextOpen: moment().tz('America/New_York')
          .day(1).hour(9).minute(30).second(0).valueOf()
      };
    } else {
      return {
        status: 'open',
        nextClose: marketClose.valueOf()
      };
    }
  }

  private getCacheKey(endpoint: string, params?: any): string {
    return `${endpoint}_${JSON.stringify(params || {})}`;
  }

  private getFromCache<T>(key: string): T | null {
    const cached = this.cache.get(key);
    if (cached && Date.now() - cached.timestamp < cached.ttl) {
      return cached.data as T;
    }
    return null;
  }

  private setCache<T>(key: string, data: T, ttl: number = this.CACHE_TTL): void {
    this.cache.set(key, {
      data,
      timestamp: Date.now(),
      ttl,
    });
  }

  // Market Data Methods
  async getQuoteData(symbol: string): Promise<Quote> {
    const cacheKey = this.getCacheKey('quote', { symbol });
    const cached = this.getFromCache<Quote>(cacheKey);
    if (cached) return cached;

    try {
      const response: AxiosResponse = await this.api.get(`/market/quote/${symbol}`);
      const quote = response.data.data;
      this.setCache(cacheKey, quote, 5000); // 5 second cache for quotes
      return quote;
    } catch (error) {
      console.error(`Failed to fetch quote for ${symbol}:`, error);
      throw new Error(`Failed to fetch quote for ${symbol}`);
    }
  }

  async getHistoricalDataService(
    symbol: string,
    period: string = '1y',
    interval: string = '1d'
  ): Promise<HistoricalDataPoint[]> {
    const cacheKey = this.getCacheKey('historical', { symbol, period, interval });
    const cached = this.getFromCache<HistoricalDataPoint[]>(cacheKey);
    if (cached) return cached;

    try {
      const response: AxiosResponse = await this.api.get(`/market/historical/${symbol}`, {
        params: { period, interval },
      });
      const data = response.data.data;
      this.setCache(cacheKey, data, 60000); // 1 minute cache for historical data
      return data;
    } catch (error) {
      console.error(`Failed to fetch historical data for ${symbol}:`, error);
      throw new Error(`Failed to fetch historical data for ${symbol}`);
    }
  }

  async getMarketMovers(): Promise<MarketMovers> {
    const cacheKey = this.getCacheKey('movers');
    const cached = this.getFromCache<MarketMovers>(cacheKey);
    if (cached) return cached;

    try {
      const response: AxiosResponse = await this.api.get('/market/movers');
      const movers = response.data.movers;
      this.setCache(cacheKey, movers, 30000); // 30 second cache
      return movers;
    } catch (error) {
      console.error('Failed to fetch market movers:', error);
      throw new Error('Failed to fetch market movers');
    }
  }

  async getSectorPerformance(): Promise<SectorPerformance[]> {
    const cacheKey = this.getCacheKey('sectors');
    const cached = this.getFromCache<SectorPerformance[]>(cacheKey);
    if (cached) return cached;

    try {
      const response: AxiosResponse = await this.api.get('/market/sectors');
      const sectors = response.data.sectors;
      this.setCache(cacheKey, sectors, 60000); // 1 minute cache
      return sectors;
    } catch (error) {
      console.error('Failed to fetch sector performance:', error);
      throw new Error('Failed to fetch sector performance');
    }
  }

  async getWatchlistData(symbols: string[]): Promise<Quote[]> {
    if (symbols.length === 0) return [];

    const symbolsParam = symbols.join(',');
    const cacheKey = this.getCacheKey('watchlist', { symbols: symbolsParam });
    const cached = this.getFromCache<Quote[]>(cacheKey);
    if (cached) return cached;

    try {
      const response: AxiosResponse = await this.api.get('/market/watchlist', {
        params: { symbols: symbolsParam },
      });
      const watchlist = response.data.watchlist;
      this.setCache(cacheKey, watchlist, 5000); // 5 second cache
      return watchlist;
    } catch (error) {
      console.error('Failed to fetch watchlist data:', error);
      throw new Error('Failed to fetch watchlist data');
    }
  }

  // News Methods
  async getMarketNews(limit: number = 50): Promise<NewsArticle[]> {
    const cacheKey = this.getCacheKey('market_news', { limit });
    const cached = this.getFromCache<NewsArticle[]>(cacheKey);
    if (cached) return cached;

    try {
      const response: AxiosResponse = await this.api.get('/market/news', {
        params: { limit },
      });
      const news = response.data.news;
      this.setCache(cacheKey, news, 300000); // 5 minute cache for news
      return news;
    } catch (error) {
      console.error('Failed to fetch market news:', error);
      throw new Error('Failed to fetch market news');
    }
  }

  async getSymbolNews(symbol: string, limit: number = 20): Promise<NewsArticle[]> {
    const cacheKey = this.getCacheKey('symbol_news', { symbol, limit });
    const cached = this.getFromCache<NewsArticle[]>(cacheKey);
    if (cached) return cached;

    try {
      const response: AxiosResponse = await this.api.get(`/market/news/${symbol}`, {
        params: { limit },
      });
      const news = response.data.news;
      this.setCache(cacheKey, news, 300000); // 5 minute cache
      return news;
    } catch (error) {
      console.error(`Failed to fetch news for ${symbol}:`, error);
      throw new Error(`Failed to fetch news for ${symbol}`);
    }
  }

  async getMarketSentiment(): Promise<SentimentSummary> {
    const cacheKey = this.getCacheKey('market_sentiment');
    const cached = this.getFromCache<SentimentSummary>(cacheKey);
    if (cached) return cached;

    try {
      const response: AxiosResponse = await this.api.get('/market/sentiment');
      const sentiment = response.data.sentiment;
      this.setCache(cacheKey, sentiment, 300000); // 5 minute cache
      return sentiment;
    } catch (error) {
      console.error('Failed to fetch market sentiment:', error);
      throw new Error('Failed to fetch market sentiment');
    }
  }

  async getSymbolSentiment(symbol: string): Promise<SentimentSummary> {
    const cacheKey = this.getCacheKey('symbol_sentiment', { symbol });
    const cached = this.getFromCache<SentimentSummary>(cacheKey);
    if (cached) return cached;

    try {
      const response: AxiosResponse = await this.api.get(`/market/sentiment/${symbol}`);
      const sentiment = response.data.sentiment;
      this.setCache(cacheKey, sentiment, 300000); // 5 minute cache
      return sentiment;
    } catch (error) {
      console.error(`Failed to fetch sentiment for ${symbol}:`, error);
      throw new Error(`Failed to fetch sentiment for ${symbol}`);
    }
  }

  // AI/ML Methods
  async getPricePrediction(
    symbol: string,
    daysAhead: number = 5,
    modelType: 'lstm' | 'ensemble' | 'all' = 'all'
  ): Promise<any> {
    const cacheKey = this.getCacheKey('price_prediction', { symbol, daysAhead, modelType });
    const cached = this.getFromCache<any>(cacheKey);
    if (cached) return cached;

    try {
      const response: AxiosResponse = await this.api.post('/ai/predict/price', {
        symbol,
        days_ahead: daysAhead,
        model_type: modelType,
      });
      const predictions = response.data.predictions;
      this.setCache(cacheKey, predictions, 600000); // 10 minute cache
      return predictions;
    } catch (error) {
      console.error(`Failed to get price prediction for ${symbol}:`, error);
      throw new Error(`Failed to get price prediction for ${symbol}`);
    }
  }

  async getTradingSignal(symbol: string, includeSentiment: boolean = true): Promise<any> {
    const cacheKey = this.getCacheKey('trading_signal', { symbol, includeSentiment });
    const cached = this.getFromCache<any>(cacheKey);
    if (cached) return cached;

    try {
      const response: AxiosResponse = await this.api.post('/ai/trading/signal', {
        symbol,
        include_sentiment: includeSentiment,
      });
      const signal = response.data.signal;
      this.setCache(cacheKey, signal, 60000); // 1 minute cache
      return signal;
    } catch (error) {
      console.error(`Failed to get trading signal for ${symbol}:`, error);
      throw new Error(`Failed to get trading signal for ${symbol}`);
    }
  }

  async getComprehensiveAnalysis(symbol: string): Promise<AIAnalysis> {
    const cacheKey = this.getCacheKey('comprehensive_analysis', { symbol });
    const cached = this.getFromCache<AIAnalysis>(cacheKey);
    if (cached) return cached;

    try {
      const response: AxiosResponse = await this.api.post('/ai/analysis/comprehensive', {
        symbol,
        analysis_type: 'comprehensive',
      });
      const analysis = response.data.analysis;
      this.setCache(cacheKey, analysis, 600000); // 10 minute cache
      return analysis;
    } catch (error) {
      console.error(`Failed to get comprehensive analysis for ${symbol}:`, error);
      throw new Error(`Failed to get comprehensive analysis for ${symbol}`);
    }
  }

  async analyzeSentiment(texts: string[]): Promise<any> {
    try {
      const response: AxiosResponse = await this.api.post('/ai/sentiment/analyze', {
        texts,
      });
      return response.data;
    } catch (error) {
      console.error('Failed to analyze sentiment:', error);
      throw new Error('Failed to analyze sentiment');
    }
  }

  // Search Methods
  async searchSymbols(query: string): Promise<any[]> {
    if (query.length < 1) return [];

    const cacheKey = this.getCacheKey('search', { query });
    const cached = this.getFromCache<any[]>(cacheKey);
    if (cached) return cached;

    try {
      const response: AxiosResponse = await this.api.get('/market/search', {
        params: { query },
      });
      const results = response.data.results;
      this.setCache(cacheKey, results, 300000); // 5 minute cache
      return results;
    } catch (error) {
      console.error(`Failed to search symbols for ${query}:`, error);
      throw new Error(`Failed to search symbols for ${query}`);
    }
  }

  // Utility Methods
  async getMarketOverview(): Promise<any> {
    const cacheKey = this.getCacheKey('market_overview');
    const cached = this.getFromCache<any>(cacheKey);
    if (cached) return cached;

    try {
      const response: AxiosResponse = await this.api.get('/market/overview');
      const overview = response.data.overview;
      this.setCache(cacheKey, overview, 60000); // 1 minute cache
      return overview;
    } catch (error) {
      console.error('Failed to fetch market overview:', error);
      throw new Error('Failed to fetch market overview');
    }
  }

  async getAIModelStatus(): Promise<any> {
    try {
      const response: AxiosResponse = await this.api.get('/ai/models/status');
      return response.data;
    } catch (error) {
      console.error('Failed to get AI model status:', error);
      throw new Error('Failed to get AI model status');
    }
  }

  // Cache management
  clearCache(): void {
    this.cache.clear();
  }

  getCacheStats(): { size: number; keys: string[] } {
    return {
      size: this.cache.size,
      keys: Array.from(this.cache.keys()),
    };
  }
}

// Singleton instance
export const marketDataService = new MarketDataService();
export default marketDataService; 