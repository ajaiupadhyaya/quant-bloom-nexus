import { EventEmitter } from 'eventemitter3';

export interface APIConfiguration {
  alphaVantage: {
    apiKey: string;
    baseUrl: string;
    rateLimitPerMinute: number;
  };
  iexCloud: {
    apiKey: string;
    baseUrl: string;
    sandboxMode: boolean;
  };
  polygon: {
    apiKey: string;
    baseUrl: string;
    tier: 'basic' | 'starter' | 'developer' | 'advanced';
  };
  yahooFinance: {
    baseUrl: string;
    backup: boolean;
  };
  fredAPI: {
    apiKey: string;
    baseUrl: string;
  };
  newsAPI: {
    apiKey: string;
    baseUrl: string;
  };
}

export interface SecurityData {
  symbol: string;
  name: string;
  exchange: string;
  type: 'stock' | 'etf' | 'option' | 'future' | 'forex' | 'crypto' | 'bond';
  sector?: string;
  industry?: string;
  marketCap?: number;
  sharesOutstanding?: number;
  currency: string;
  country: string;
}

export interface RealTimeQuote {
  symbol: string;
  timestamp: number;
  price: number;
  bid: number;
  ask: number;
  volume: number;
  dayOpen: number;
  dayHigh: number;
  dayLow: number;
  previousClose: number;
  change: number;
  changePercent: number;
  marketCap?: number;
  pe?: number;
  exchange: string;
  marketStatus: 'open' | 'closed' | 'pre_market' | 'after_hours';
}

export interface HistoricalDataPoint {
  timestamp: number;
  open: number;
  high: number;
  low: number;
  close: number;
  volume: number;
}

export interface OptionsChain {
  symbol: string;
  expirationDates: string[];
  strikes: number[];
  options: {
    [expiration: string]: {
      [strike: number]: {
        call?: OptionContract;
        put?: OptionContract;
      };
    };
  };
}

export interface OptionContract {
  symbol: string;
  strike: number;
  expiration: string;
  type: 'call' | 'put';
  bid: number;
  ask: number;
  lastPrice: number;
  volume: number;
  openInterest: number;
  impliedVolatility: number;
  delta: number;
  gamma: number;
  theta: number;
  vega: number;
  rho: number;
  intrinsicValue: number;
  timeValue: number;
}

export interface EconomicIndicator {
  seriesId: string;
  title: string;
  frequency: string;
  units: string;
  lastUpdated: number;
  observations: Array<{
    date: string;
    value: number;
  }>;
}

export interface NewsArticle {
  id: string;
  headline: string;
  summary: string;
  publishedAt: number;
  source: string;
  url: string;
  symbols: string[];
  sentiment: number;
  relevanceScore: number;
}

export class RealTimeDataProvider extends EventEmitter {
  private subscriptions: Set<string> = new Set();
  private cachedData: Map<string, unknown> = new Map();

  constructor() {
    super();
  }

  // Alpha Vantage API simulation
  public async getQuote(symbol: string): Promise<RealTimeQuote | null> {
    try {
      // For demo purposes, generate realistic market data
      const basePrice = this.getBasePriceForSymbol(symbol);
      const variation = (Math.random() - 0.5) * 0.04; // ±2% variation
      const currentPrice = basePrice * (1 + variation);
      
      const dayOpen = basePrice * (1 + (Math.random() - 0.5) * 0.02);
      const dayHigh = Math.max(dayOpen, currentPrice) * (1 + Math.random() * 0.015);
      const dayLow = Math.min(dayOpen, currentPrice) * (1 - Math.random() * 0.015);
      const previousClose = basePrice * (1 + (Math.random() - 0.5) * 0.01);
      
      return {
        symbol,
        timestamp: Date.now(),
        price: Number(currentPrice.toFixed(2)),
        bid: Number((currentPrice - 0.01).toFixed(2)),
        ask: Number((currentPrice + 0.01).toFixed(2)),
        volume: Math.floor(Math.random() * 10000000) + 100000,
        dayOpen: Number(dayOpen.toFixed(2)),
        dayHigh: Number(dayHigh.toFixed(2)),
        dayLow: Number(dayLow.toFixed(2)),
        previousClose: Number(previousClose.toFixed(2)),
        change: Number((currentPrice - previousClose).toFixed(2)),
        changePercent: Number(((currentPrice - previousClose) / previousClose * 100).toFixed(2)),
        marketCap: this.getMarketCapForSymbol(symbol),
        pe: 15 + Math.random() * 20,
        exchange: 'NASDAQ',
        marketStatus: this.getMarketStatus()
      };
    } catch (error) {
      console.error('Quote API error:', error);
      return null;
    }
  }

  public async getHistoricalData(
    symbol: string,
    period: string = '1y',
    interval: string = 'daily'
  ): Promise<HistoricalDataPoint[]> {
    try {
      const data: HistoricalDataPoint[] = [];
      const days = this.getPeriodDays(period);
      const basePrice = this.getBasePriceForSymbol(symbol);
      
      let currentPrice = basePrice;
      const startDate = new Date();
      startDate.setDate(startDate.getDate() - days);
      
      for (let i = 0; i < days; i++) {
        const date = new Date(startDate);
        date.setDate(date.getDate() + i);
        
        // Skip weekends for stocks
        if (date.getDay() === 0 || date.getDay() === 6) continue;
        
        const dailyVariation = (Math.random() - 0.5) * 0.06; // ±3% daily variation
        const open = currentPrice;
        const close = open * (1 + dailyVariation);
        const high = Math.max(open, close) * (1 + Math.random() * 0.02);
        const low = Math.min(open, close) * (1 - Math.random() * 0.02);
        
        data.push({
          timestamp: date.getTime(),
          open: Number(open.toFixed(2)),
          high: Number(high.toFixed(2)),
          low: Number(low.toFixed(2)),
          close: Number(close.toFixed(2)),
          volume: Math.floor(Math.random() * 5000000) + 100000
        });
        
        currentPrice = close;
      }
      
      return data;
    } catch (error) {
      console.error('Historical data error:', error);
      return [];
    }
  }

  public async getMarketNews(symbols?: string[]): Promise<NewsArticle[]> {
    try {
      // Generate realistic news data
      const newsTemplates = [
        'Federal Reserve signals potential rate changes in upcoming meeting',
        'Tech sector shows strong quarterly earnings growth',
        'Oil prices fluctuate amid geopolitical tensions',
        'Cryptocurrency market experiences increased volatility',
        'Banking sector faces regulatory scrutiny',
        'Renewable energy stocks gain momentum',
        'Supply chain disruptions impact manufacturing',
        'Consumer confidence reaches new highs',
        'AI companies attract significant investment',
        'International trade agreements boost markets'
      ];

      const sources = ['Reuters', 'Bloomberg', 'WSJ', 'Financial Times', 'MarketWatch'];
      
      return newsTemplates.map((headline, index) => ({
        id: `news_${index}_${Date.now()}`,
        headline,
        summary: `${headline}. Market analysts provide insights on potential impact...`,
        publishedAt: Date.now() - (index * 3600000), // Stagger by hours
        source: sources[index % sources.length],
        url: `https://example.com/news/${index}`,
        symbols: symbols || ['AAPL', 'MSFT', 'GOOGL'],
        sentiment: (Math.random() - 0.5) * 2, // -1 to 1
        relevanceScore: 0.5 + Math.random() * 0.5
      }));
    } catch (error) {
      console.error('News API error:', error);
      return [];
    }
  }

  public subscribeToRealTimeData(symbols: string[]): void {
    symbols.forEach(symbol => {
      if (!this.subscriptions.has(symbol)) {
        this.subscriptions.add(symbol);
        this.startRealTimeUpdates(symbol);
      }
    });
  }

  public unsubscribeFromRealTimeData(symbols: string[]): void {
    symbols.forEach(symbol => {
      this.subscriptions.delete(symbol);
    });
  }

  private startRealTimeUpdates(symbol: string): void {
    // Simulate real-time price updates every 1-5 seconds
    const updateInterval = setInterval(async () => {
      if (!this.subscriptions.has(symbol)) {
        clearInterval(updateInterval);
        return;
      }

      const quote = await this.getQuote(symbol);
      if (quote) {
        this.emit('quote', quote);
        this.emit(`quote:${symbol}`, quote);
      }
    }, 1000 + Math.random() * 4000);
  }

  private getBasePriceForSymbol(symbol: string): number {
    const priceMap: { [key: string]: number } = {
      'AAPL': 150.25,
      'MSFT': 335.50,
      'GOOGL': 2750.00,
      'AMZN': 3100.00,
      'TSLA': 245.75,
      'NVDA': 220.30,
      'META': 315.60,
      'NFLX': 445.20,
      'SPY': 445.50,
      'QQQ': 375.80,
      'DIA': 350.20,
      'IWM': 195.40
    };
    
    return priceMap[symbol] || 100 + Math.random() * 200;
  }

  private getMarketCapForSymbol(symbol: string): number {
    const marketCapMap: { [key: string]: number } = {
      'AAPL': 2400000000000,
      'MSFT': 2300000000000,
      'GOOGL': 1600000000000,
      'AMZN': 1200000000000,
      'TSLA': 800000000000,
      'NVDA': 550000000000,
      'META': 500000000000,
      'NFLX': 200000000000
    };
    
    return marketCapMap[symbol] || Math.floor(Math.random() * 100000000000);
  }

  private getPeriodDays(period: string): number {
    const periodMap: { [key: string]: number } = {
      '1d': 1,
      '5d': 5,
      '1m': 30,
      '3m': 90,
      '6m': 180,
      '1y': 365,
      '2y': 730,
      '5y': 1825
    };
    
    return periodMap[period] || 365;
  }

  private getMarketStatus(): 'open' | 'closed' | 'pre_market' | 'after_hours' {
    const now = new Date();
    const easternTime = new Date(now.toLocaleString("en-US", {timeZone: "America/New_York"}));
    const hour = easternTime.getHours();
    const day = easternTime.getDay();

    if (day === 0 || day === 6) return 'closed';
    if (hour >= 4 && hour < 9.5) return 'pre_market';
    if (hour >= 9.5 && hour < 16) return 'open';
    if (hour >= 16 && hour < 20) return 'after_hours';
    return 'closed';
  }

  public dispose(): void {
    this.subscriptions.clear();
    this.cachedData.clear();
  }
}

// Default configuration for demo/development
export const defaultAPIConfig: APIConfiguration = {
  alphaVantage: {
    apiKey: process.env.ALPHA_VANTAGE_API_KEY || 'demo',
    baseUrl: 'https://www.alphavantage.co',
    rateLimitPerMinute: 5
  },
  iexCloud: {
    apiKey: process.env.IEX_CLOUD_API_KEY || 'demo',
    baseUrl: 'https://cloud.iexapis.com/stable',
    sandboxMode: true
  },
  polygon: {
    apiKey: process.env.POLYGON_API_KEY || 'demo',
    baseUrl: 'https://api.polygon.io',
    tier: 'basic'
  },
  yahooFinance: {
    baseUrl: 'https://query1.finance.yahoo.com',
    backup: true
  },
  fredAPI: {
    apiKey: process.env.FRED_API_KEY || 'demo',
    baseUrl: 'https://api.stlouisfed.org/fred'
  },
  newsAPI: {
    apiKey: process.env.NEWS_API_KEY || 'demo',
    baseUrl: 'https://newsapi.org/v2'
  }
};

// Singleton instance
export const realTimeDataProvider = new RealTimeDataProvider(); 