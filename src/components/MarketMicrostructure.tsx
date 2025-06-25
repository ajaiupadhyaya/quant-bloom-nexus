
import React, { useState, useEffect } from 'react';
import { BarChart, Bar, XAxis, YAxis, ResponsiveContainer, LineChart, Line, ScatterChart, Scatter, Cell } from 'recharts';
import { Activity, Zap, Target, TrendingUp, TrendingDown, Circle } from 'lucide-react';

interface OrderBookLevel {
  price: number;
  size: number;
  orders: number;
  side: 'bid' | 'ask';
}

interface TradeData {
  time: string;
  price: number;
  size: number;
  side: 'buy' | 'sell';
  venue: string;
  aggressor: boolean;
}

interface LiquidityMetrics {
  effectiveSpread: number;
  realizedSpread: number;
  priceImpact: number;
  marketDepth: number;
  resilience: number;
}

interface VenueAnalysis {
  venue: string;
  volume: number;
  marketShare: number;
  avgSpread: number;
  fillRate: number;
  latency: number;
}

export const MarketMicrostructure = () => {
  const [orderBook, setOrderBook] = useState<OrderBookLevel[]>([]);
  const [trades, setTrades] = useState<TradeData[]>([]);
  const [liquidityMetrics, setLiquidityMetrics] = useState<LiquidityMetrics>({
    effectiveSpread: 0.08,
    realizedSpread: 0.06,
    priceImpact: 0.12,
    marketDepth: 2.5,
    resilience: 0.85
  });

  const [venueAnalysis, setVenueAnalysis] = useState<VenueAnalysis[]>([
    { venue: 'NYSE', volume: 1250000, marketShare: 0.35, avgSpread: 0.08, fillRate: 0.98, latency: 2.1 },
    { venue: 'NASDAQ', volume: 980000, marketShare: 0.28, avgSpread: 0.07, fillRate: 0.97, latency: 1.8 },
    { venue: 'BATS', volume: 650000, marketShare: 0.18, avgSpread: 0.09, fillRate: 0.96, latency: 2.3 },
    { venue: 'IEX', volume: 420000, marketShare: 0.12, avgSpread: 0.10, fillRate: 0.94, latency: 3.2 },
    { venue: 'EDGX', volume: 245000, marketShare: 0.07, avgSpread: 0.11, fillRate: 0.95, latency: 2.8 }
  ]);

  const [spreadAnalysis, setSpreadAnalysis] = useState([
    { time: '09:30', bid: 179.98, ask: 180.02, spread: 0.04, volume: 125000 },
    { time: '09:35', bid: 180.15, ask: 180.19, spread: 0.04, volume: 98000 },
    { time: '09:40', bid: 180.22, ask: 180.28, spread: 0.06, volume: 145000 },
    { time: '09:45', bid: 180.31, ask: 180.35, spread: 0.04, volume: 112000 },
    { time: '09:50', bid: 180.28, ask: 180.34, spread: 0.06, volume: 167000 }
  ]);

  useEffect(() => {
    const generateOrderBook = () => {
      const book: OrderBookLevel[] = [];
      const midPrice = 180.25;
      
      // Generate bids
      for (let i = 1; i <= 10; i++) {
        book.push({
          price: midPrice - (i * 0.01),
          size: Math.floor(Math.random() * 5000) + 1000,
          orders: Math.floor(Math.random() * 20) + 5,
          side: 'bid'
        });
      }
      
      // Generate asks
      for (let i = 1; i <= 10; i++) {
        book.push({
          price: midPrice + (i * 0.01),
          size: Math.floor(Math.random() * 5000) + 1000,
          orders: Math.floor(Math.random() * 20) + 5,
          side: 'ask'
        });
      }
      
      return book;
    };

    const generateTrades = () => {
      const newTrades: TradeData[] = [];
      const venues = ['NYSE', 'NASDAQ', 'BATS', 'IEX', 'EDGX'];
      
      for (let i = 0; i < 20; i++) {
        const time = new Date(Date.now() - (20 - i) * 1000);
        newTrades.push({
          time: time.toLocaleTimeString(),
          price: 180 + Math.random() * 0.5,
          size: Math.floor(Math.random() * 1000) + 100,
          side: Math.random() > 0.5 ? 'buy' : 'sell',
          venue: venues[Math.floor(Math.random() * venues.length)],
          aggressor: Math.random() > 0.3
        });
      }
      
      return newTrades;
    };

    setOrderBook(generateOrderBook());
    setTrades(generateTrades());
    
    const interval = setInterval(() => {
      setOrderBook(generateOrderBook());
      setTrades(generateTrades());
      
      setLiquidityMetrics(prev => ({
        effectiveSpread: prev.effectiveSpread + (Math.random() - 0.5) * 0.01,
        realizedSpread: prev.realizedSpread + (Math.random() - 0.5) * 0.01,
        priceImpact: prev.priceImpact + (Math.random() - 0.5) * 0.02,
        marketDepth: prev.marketDepth + (Math.random() - 0.5) * 0.1,
        resilience: Math.min(1, Math.max(0, prev.resilience + (Math.random() - 0.5) * 0.05))
      }));
    }, 2000);

    return () => clearInterval(interval);
  }, []);

  return (
    <div className="terminal-panel h-full flex flex-col">
      <div className="border-b border-terminal-border p-3">
        <div className="flex items-center justify-between">
          <div className="flex items-center space-x-2">
            <Activity className="w-4 h-4 text-terminal-orange" />
            <h2 className="text-sm font-semibold text-terminal-orange">MARKET MICROSTRUCTURE</h2>
          </div>
          <div className="text-xs text-terminal-muted">Real-time L2 Data</div>
        </div>
      </div>

      <div className="flex-1 overflow-y-auto">
        {/* Liquidity Metrics */}
        <div className="p-3 border-b border-terminal-border/50">
          <h3 className="text-xs font-medium text-terminal-cyan mb-3">LIQUIDITY METRICS</h3>
          <div className="grid grid-cols-3 gap-4">
            <div>
              <div className="text-xs text-terminal-muted">Effective Spread</div>
              <div className="text-terminal-red font-semibold financial-number">
                {(liquidityMetrics.effectiveSpread * 100).toFixed(2)} bps
              </div>
            </div>
            <div>
              <div className="text-xs text-terminal-muted">Market Depth</div>
              <div className="text-terminal-cyan font-semibold financial-number">
                ${liquidityMetrics.marketDepth.toFixed(1)}M
              </div>
            </div>
            <div>
              <div className="text-xs text-terminal-muted">Price Impact</div>
              <div className="text-terminal-amber font-semibold financial-number">
                {(liquidityMetrics.priceImpact * 100).toFixed(1)} bps
              </div>
            </div>
          </div>
        </div>

        {/* Order Book Depth */}
        <div className="p-3 border-b border-terminal-border/50">
          <h3 className="text-xs font-medium text-terminal-cyan mb-3">ORDER BOOK DEPTH</h3>
          <div className="h-32">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={[
                ...orderBook.filter(level => level.side === 'bid').slice(0, 5).reverse(),
                ...orderBook.filter(level => level.side === 'ask').slice(0, 5)
              ]}>
                <XAxis 
                  dataKey="price" 
                  axisLine={false}
                  tick={{ fontSize: 9, fill: '#888888' }}
                />
                <YAxis 
                  axisLine={false}
                  tick={{ fontSize: 9, fill: '#888888' }}
                />
                <Bar dataKey="size">
                  {orderBook.map((entry, index) => (
                    <Cell 
                      key={`cell-${index}`} 
                      fill={entry.side === 'bid' ? '#00ff88' : '#ff4757'} 
                      opacity={0.7}
                    />
                  ))}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </div>
        </div>

        {/* Venue Analysis */}
        <div className="p-3 border-b border-terminal-border/50">
          <h3 className="text-xs font-medium text-terminal-cyan mb-3">VENUE ANALYSIS</h3>
          <div className="space-y-2">
            {venueAnalysis.map((venue, index) => (
              <div key={index} className="flex items-center justify-between bg-terminal-bg/30 p-2 rounded">
                <div className="flex items-center space-x-2">
                  <div className="w-2 h-2 bg-terminal-cyan rounded-full" />
                  <span className="text-xs font-medium text-terminal-text">{venue.venue}</span>
                </div>
                <div className="flex space-x-4 text-xs">
                  <div>
                    <div className="text-terminal-muted">Volume</div>
                    <div className="text-terminal-green font-semibold financial-number">
                      {(venue.volume / 1000).toFixed(0)}K
                    </div>
                  </div>
                  <div>
                    <div className="text-terminal-muted">Share</div>
                    <div className="text-terminal-cyan font-semibold financial-number">
                      {(venue.marketShare * 100).toFixed(0)}%
                    </div>
                  </div>
                  <div>
                    <div className="text-terminal-muted">Spread</div>
                    <div className="text-terminal-amber font-semibold financial-number">
                      {(venue.avgSpread * 100).toFixed(1)} bps
                    </div>
                  </div>
                  <div>
                    <div className="text-terminal-muted">Latency</div>
                    <div className="text-terminal-text font-semibold financial-number">
                      {venue.latency.toFixed(1)}ms
                    </div>
                  </div>
                </div>
              </div>
            ))}
          </div>
        </div>

        {/* Recent Trades */}
        <div className="p-3">
          <h3 className="text-xs font-medium text-terminal-cyan mb-3">TRADE TAPE</h3>
          <div className="space-y-1 max-h-24 overflow-y-auto">
            {trades.slice(-8).map((trade, index) => (
              <div key={index} className="flex items-center justify-between text-xs">
                <div className="flex items-center space-x-2">
                  <Circle className={`w-2 h-2 ${
                    trade.side === 'buy' ? 'text-terminal-green' : 'text-terminal-red'
                  }`} />
                  <span className="text-terminal-text font-mono">
                    {trade.price.toFixed(2)}
                  </span>
                  <span className="text-terminal-muted">
                    {trade.size.toLocaleString()}
                  </span>
                </div>
                <div className="flex items-center space-x-2">
                  <span className="text-terminal-muted">{trade.venue}</span>
                  <span className="text-terminal-muted">{trade.time.slice(-8)}</span>
                </div>
              </div>
            ))}
          </div>
        </div>
      </div>
    </div>
  );
};
