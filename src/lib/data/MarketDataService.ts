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

  constructor() {
    super();
    this.initializeConnection();
    this.setupPerformanceMonitoring();
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

  public async getHistoricalData(
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
}

// Singleton instance
export const marketDataService = new MarketDataService(); 